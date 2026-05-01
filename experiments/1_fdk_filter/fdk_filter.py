import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

DIRECT_RECON_VIEW_BATCH_SIZE = 100  # This is set here due to a bug in jax.vmap when the batch size is too large.

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
    recon_filter = generate_direct_recon_filter(num_channels, filter_name=filter_name)
    alpha = delta_det_row / (delta_voxel**3 * M_0)
    recon_filter = alpha * recon_filter
    recon_filter = jax.device_put(recon_filter, self.replicated_device)

    # Define convolution for a single row (across its channels)
    def convolve_row(row):
        return jax.scipy.signal.fftconvolve(row, recon_filter, mode="valid")

    # Apply above convolve func across each row of a view, batching rows to bound peak memory
    row_batch_size = min(num_rows, self.entries_per_cylinder_batch) # TODO:CADEN use different min value

    def apply_convolution_to_view(view):
        return jax.lax.map(convolve_row, view, batch_size=row_batch_size)

    # Apply convolution across the channels of the weighted sinogram per each fixed view & row
    num_views = sinogram.shape[0]

    if self.use_gpu == 'sharding':
        num_devices = self.sinogram_device.mesh.devices.size
        filtered_sinogram = jax.lax.map(apply_convolution_to_view, weighted_sinogram, batch_size=num_devices)
        filtered_sinogram.block_until_ready()
        del weighted_sinogram
        filtered_sinogram *= jnp.pi / num_views
    else:
        filtered_sino_list = []
        for i in range(0, num_views, view_batch_size):
            sino_batch = jax.device_put(weighted_sinogram[i:min(i + view_batch_size, num_views)], self.worker)
            filtered_sinogram_batch = jax.lax.map(apply_convolution_to_view, sino_batch, batch_size=view_batch_size)
            filtered_sinogram_batch.block_until_ready()
            filtered_sino_list.append(jax.device_put(filtered_sinogram_batch, self.sinogram_device))
        filtered_sinogram = jnp.concatenate(filtered_sino_list, axis=0)
        filtered_sinogram *= jnp.pi / num_views
    return filtered_sinogram