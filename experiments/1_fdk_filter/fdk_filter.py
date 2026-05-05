from typing import Literal, Union, Any, TextIO, overload
import numpy as np
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

@dataclass
class Param:
    val: Any
    recompile_flag: bool = True

    def __repr__(self):
        return f"Param(val={self.val}, recompile_flag={self.recompile_flag})"
    
ParamNames = Literal[
    'geometry_type', 'file_format', 'sinogram_shape', 'delta_det_channel',
    'delta_det_row', 'det_row_offset', 'det_channel_offset', 'sigma_y',
    'alu_unit', 'alu_value', 'recon_shape', 'delta_voxel', 'sigma_x', 'sigma_prox',
    'p', 'q', 'T', 'qggmrf_nbr_wts',
    'auto_regularize_flag', 'positivity_flag', 'snr_db', 'sharpness',
    'granularity', 'partition_sequence', 'verbose', 'use_gpu',
]

DIRECT_RECON_VIEW_BATCH_SIZE = 100  # This is set here due to a bug in jax.vmap when the batch size is too large.

class FDK:

    def __init__(self, sinogram_shape, source_detector_dist=None, source_iso_dist=None):

        cpus = jax.devices('cpu')
        # gpus = jax.devices('gpu')

        # devices = np.array(gpus).reshape((-1, 1))
        # mesh = Mesh(devices, ('views', 'rows'))

        self.main_device = cpus[0]
        self.sinogram_device = self.main_device
        self.replicated_device = self.main_device

        self.entries_per_cylinder_batch = 100

        num_views, num_det_rows, num_det_channels = sinogram_shape
        if source_detector_dist is None:
            source_detector_dist = 4 * num_det_channels
        if source_iso_dist is None:
            source_iso_dist = source_detector_dist

        delta_det_channel = 1.0
        magnification = source_detector_dist / source_iso_dist
        delta_voxel = delta_det_channel / magnification

        self.params = {
            'geometry_type': Param(None, False),  # The geometry type should never change during a recon.
            'sinogram_shape': Param(sinogram_shape, True),
            'source_detector_dist': Param(source_detector_dist, True),
            'source_iso_dist': Param(source_iso_dist, True),
            'delta_det_channel': Param(delta_det_channel, True),
            'delta_det_row': Param(1.0, True),
            'det_row_offset': Param(0.0, True),
            'det_channel_offset': Param(0.0, True),
            'delta_voxel': Param(delta_voxel, True),
            'sigma_y': Param(1.0, False),
            'alu_unit': Param(None, False),
            'alu_value': Param(1.0, False),
        }

        # self.sinogram_device = NamedSharding(mesh, P('views'))
        # self.replicated_device = NamedSharding(mesh, P())

        pass

    def get_params(self, parameter_names: Union[ParamNames, list[ParamNames]]) -> Any:
        """
        Get the values of the listed parameter names from the internal parameter dictionary.

        This method retrieves the current values of one or more parameters managed by the model.

        Args:
            parameter_names (str or list of str): Name of a parameter, or a list of parameter names.

        Returns:
            Any or list: Single parameter value if a string is passed, or a list of values if a list is passed.

        Raises:
            NameError: If any of the provided parameter names are not recognized.

        Example:
            >>> sharpness = model.get_params('sharpness')
            >>> recon_shape, sharpness = model.get_params(['recon_shape', 'sharpness'])
        """
        param_values = self.get_params_from_dict(self.params, parameter_names)
        return param_values

    @staticmethod
    def get_params_from_dict(param_dict, parameter_names: Union[str, list[str]]):
        """
        Get the values of the listed parameter names from the supplied dict.
        Raises an exception if a parameter name is not defined in parameters.

        Args:
            param_dict (dict): The dictionary of parameters
            parameter_names (str or list of str): String or list of strings

        Returns:
            Single value or list of values
        """
        if isinstance(parameter_names, str):
            if parameter_names in param_dict.keys():
                value = param_dict[parameter_names].val
            else:
                raise NameError('"{}" is not a recognized argument'.format(parameter_names))
            return value
        values = []
        for name in parameter_names:
            if name in param_dict.keys():
                values.append(param_dict[name].val)
            else:
                raise NameError('"{}" is not a recognized argument'.format(name))
        return values
    
    @staticmethod
    @jax.jit
    def detector_mn_to_uv(m, n, delta_det_channel, delta_det_row, det_channel_offset, det_row_offset, num_det_rows,
                      num_det_channels):
        """
        Convert fractional detector grid indices (m, n) into detector coordinates (u, v).

        Parameters:
            m: Fractional row index on the detector grid (vertical direction).
            n: Fractional channel index on the detector grid (horizontal direction).
            delta_det_channel: Spacing (pitch) of the detector channels (horizontal direction).
            delta_det_row: Spacing (pitch) of the detector rows (vertical direction).
            det_channel_offset: Offset in the detector channel (horizontal) direction.
            det_row_offset: Offset in the detector row (vertical) direction.
            num_det_rows: Total number of rows in the detector.
            num_det_channels: Total number of channels in the detector.

        Returns:
            u: Physical detector coordinate in the channel direction.
            v: Physical detector coordinate in the row direction.
        """
        # Calculate the center of the detector grid
        det_center_row = (num_det_rows - 1) / 2.0
        det_center_channel = (num_det_channels - 1) / 2.0

        # Compute detector coordinates (u, v)
        v = (m - det_center_row) * delta_det_row - det_row_offset
        u = (n - det_center_channel) * delta_det_channel - det_channel_offset

        return u, v
    
    @staticmethod
    def generate_direct_recon_filter(num_channels, filter_name="ramp"):
        """
        Creates the specified space domain filter of size (2*num_channels - 1).

        Currently supported filters include: \"ramp\", which corresponds to a ramp in frequency domain.

        Args:
            num_channels (int): Number of detector channels in the sinogram.
            filter_name (string, optional): Name of the filter to be generated. Defaults to "ramp."

        Returns:
            filter (jnp): The computed filter (filter.size = 2*num_channels + 1).
        """

        # If you want to add a new filter, place its name into supported_filters, and ...
        # ... create a new if statement with the filter math.
        # TODO:  Anyone who adds a second filter will need to address how to document the set of available filters
        # in a way that is easy to maintain and appears correctly on readthedocs.  Also, any new filters will need
        # to have the proper scaling.
        supported_filters = ["ramp"]

        # Raise error if filter is not supported.
        if filter_name not in supported_filters:
            raise ValueError(f"Unsupported filter. Supported filters are: {', '.join(supported_filters)}.")

        n = jnp.arange(-num_channels + 1, num_channels)  # ex: num_channels = 3, -> n = [-2, -1, 0, 1, 2]

        recon_filter = 0
        if filter_name == "ramp":
            recon_filter = (1 / 2) * jnp.sinc(n) - (1 / 4) * (jnp.sinc(n / 2)) ** 2

        return recon_filter

    def get_magnification(self):
        """
        Returns the magnification for the cone beam geometry.

        Returns:
            magnification = source_detector_dist / source_iso_dist
        """
        source_detector_dist, source_iso_dist = self.get_params(['source_detector_dist', 'source_iso_dist'])
        if jnp.isinf(source_detector_dist):
            magnification = 1
        else:
            magnification = source_detector_dist / source_iso_dist
        return magnification

    def fdk_filter(self, sinogram, filter_name="ramp", view_batch_size=DIRECT_RECON_VIEW_BATCH_SIZE):
        """
        Perform FDK filtering on the given sinogram.

        Args:
            sinogram (jax array): The input sinogram with shape (num_views, num_rows, num_channels).
            filter_name (string, optional): Name of the filter to be used. Defaults to "ramp"
            view_batch_size (int, optional):  Size of view batches (used to limit memory use)

        Returns:
            filtered_sinogram (jax array): The sinogram after FDK filtering.
        """
        # Get parameters
        num_views, num_rows, num_channels = sinogram.shape
        source_detector_dist, source_iso_dist = self.get_params(['source_detector_dist', 'source_iso_dist'])
        delta_voxel, delta_det_row, delta_det_channel = self.get_params(['delta_voxel', 'delta_det_row', 'delta_det_channel'])
        det_row_offset, det_channel_offset = self.get_params(['det_row_offset', 'det_channel_offset'])

        if view_batch_size is None:
            view_batch_size = self.view_batch_size_for_vmap
            max_view_batch_size = 128  # Limit the view batch size here and ParallelBeam due to https://github.com/jax-ml/jax/issues/27591
            view_batch_size = min(view_batch_size, max_view_batch_size)

        # Magnification factor M_0 = Source-Detector Distance / Source-Isocenter Distance
        M_0 = self.get_magnification()

        # Define the index arrays for channels and rows
        m = jnp.arange(num_rows)  # Column vector for rows
        n = jnp.arange(num_channels)  # Row vector for channels
        m_grid, n_grid = jnp.meshgrid(m, n, indexing='ij')

        # Coordinate transformation to physical distances:
        u_grid, v_grid = self.detector_mn_to_uv(m_grid, n_grid, delta_det_channel, delta_det_row,
                                                det_channel_offset, det_row_offset, num_rows, num_channels)

        # Compute the weight
        weight_map = source_detector_dist / jnp.sqrt(source_detector_dist ** 2 + u_grid**2 + v_grid**2)

        # Apply the pre-weighting factor to the sinogram
        weight_map = jax.device_put(weight_map, self.replicated_device)
        weighted_sinogram = sinogram * weight_map[None, :, :]
        del weight_map

        # Compute the scaled filter
        # Scaling factor alpha adjusts the filter to account for voxel size, ensuring consistent reconstruction.
        # For a detailed theoretical derivation of this scaling factor, please refer to the zip file linked at
        # https://mbirjax.readthedocs.io/en/latest/theory.html
        recon_filter = self.generate_direct_recon_filter(num_channels, filter_name=filter_name)
        alpha = delta_det_row / (delta_voxel**3 * M_0)
        recon_filter = alpha * recon_filter
        recon_filter = jax.device_put(recon_filter, self.replicated_device)

        # Define convolution for a single row (across its channels)
        def convolve_row(row):
            return jax.scipy.signal.fftconvolve(row, recon_filter, mode="valid")

        # Apply above convolve func across each row of a view, batching rows to bound peak memory
        row_batch_size = min(num_rows, self.entries_per_cylinder_batch) 

        def apply_convolution_to_view(view):
            return jax.lax.map(convolve_row, view, batch_size=row_batch_size)

        # Apply convolution across the channels of the weighted sinogram per each fixed view & row
        num_views = sinogram.shape[0]

        num_devices = 1 # self.sinogram_device.mesh.devices.size
        filtered_sinogram = jax.lax.map(apply_convolution_to_view, weighted_sinogram, batch_size=num_devices)
        filtered_sinogram.block_until_ready()
        del weighted_sinogram
        filtered_sinogram *= jnp.pi / num_views

        return filtered_sinogram
    
if __name__ == "__main__":
    
    sinogram_shape = (16, 16, 16)
    fdk_obj = FDK(sinogram_shape)

    # create sinogram object, place on gpus
    sinogram = jnp.ones(sinogram_shape)
    sinogram = jax.device_put(sinogram, fdk_obj.sinogram_device)

    fdk_obj.fdk_filter(sinogram)