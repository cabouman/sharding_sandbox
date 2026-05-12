import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

DIRECT_RECON_VIEW_BATCH_SIZE = 100

geometry_type = None
sinogram_shape = None
source_detector_dist = None
source_iso_dist = None
delta_det_channel = 1.0
delta_det_row = 1.0
det_row_offset = 0.0
det_channel_offset = 0.0
delta_voxel = None
sigma_y = 1.0
alu_unit = None
alu_value = 1.0


class FDK:

    def __init__(self, input_sinogram_shape, input_source_detector_dist=None, input_source_iso_dist=None):
        global sinogram_shape, source_detector_dist, source_iso_dist, delta_det_channel, delta_voxel

        cpus = jax.devices('cpu')
        gpus = jax.devices('gpu')

        devices = np.array(gpus).reshape((-1, 1))
        mesh = Mesh(devices, ('views', 'rows'))

        self.main_device = cpus[0]
        self.sinogram_device = NamedSharding(mesh, P('views'))
        self.replicated_device = NamedSharding(mesh, P())
        self.entries_per_cylinder_batch = 100

        num_views, num_det_rows, num_det_channels = input_sinogram_shape
        if input_source_detector_dist is None:
            input_source_detector_dist = 4 * num_det_channels
        if input_source_iso_dist is None:
            input_source_iso_dist = input_source_detector_dist

        magnification = input_source_detector_dist / input_source_iso_dist

        sinogram_shape = input_sinogram_shape
        source_detector_dist = input_source_detector_dist
        source_iso_dist = input_source_iso_dist
        delta_det_channel = 1.0
        delta_voxel = 1.0 / magnification

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
        det_center_row = (num_det_rows - 1) / 2.0
        det_center_channel = (num_det_channels - 1) / 2.0

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
        supported_filters = ["ramp"]

        if filter_name not in supported_filters:
            raise ValueError(f"Unsupported filter. Supported filters are: {', '.join(supported_filters)}.")

        n = jnp.arange(-num_channels + 1, num_channels)

        recon_filter = 0
        if filter_name == "ramp":
            recon_filter = (1 / 2) * jnp.sinc(n) - (1 / 4) * (jnp.sinc(n / 2)) ** 2

        return recon_filter

    def get_magnification(self):
        global source_detector_dist
        if jnp.isinf(source_detector_dist):
            return 1
        return source_detector_dist / source_iso_dist

    def fdk_filter(self, sinogram, filter_name="ramp", view_batch_size=DIRECT_RECON_VIEW_BATCH_SIZE):

        num_views, num_rows, num_channels = sinogram.shape

        M_0 = self.get_magnification()

        m = jnp.arange(num_rows)
        n = jnp.arange(num_channels)
        m_grid, n_grid = jnp.meshgrid(m, n, indexing='ij')

        u_grid, v_grid = self.detector_mn_to_uv(m_grid, n_grid, delta_det_channel, delta_det_row,
                                                det_channel_offset, det_row_offset, num_rows, num_channels)

        weight_map = source_detector_dist / jnp.sqrt(source_detector_dist ** 2 + u_grid**2 + v_grid**2)

        weight_map = jax.device_put(weight_map, self.replicated_device)

        recon_filter = self.generate_direct_recon_filter(num_channels, filter_name=filter_name)
        alpha = delta_det_row / (delta_voxel**3 * M_0)
        recon_filter = alpha * recon_filter
        recon_filter = jax.device_put(recon_filter, self.replicated_device)

        def convolve_row(row):
            return jax.scipy.signal.fftconvolve(row, recon_filter, mode="valid")

        row_batch_size = min(num_rows, self.entries_per_cylinder_batch)
        row_batch_size = 10

        def apply_weight_and_convolve(view):
            weighted_view = view * weight_map
            return jax.lax.map(convolve_row, weighted_view, batch_size=row_batch_size)

        filtered_sinogram = jax.lax.map(apply_weight_and_convolve, sinogram, batch_size=1)
        filtered_sinogram.block_until_ready()
        filtered_sinogram *= jnp.pi / num_views

        return filtered_sinogram

def viewer():
    num_views = 256
    num_det_rows = 256
    num_det_channels = 256

    output_directory = f"/scratch/gautschi/ncardel/recon_mem"
    h5_path = f"{output_directory}/cone_{num_views}_{num_det_rows}_{num_det_channels}_projection_data.h5"
    import h5py
    with h5py.File(h5_path, "r") as f:
        sinogram = f["sinogram"][:]

    fdk_obj = FDK(sinogram.shape)
    sinogram = jax.device_put(sinogram, fdk_obj.sinogram_device)
    filtered_sinogram = fdk_obj.fdk_filter(sinogram)

    try:
        import mbirjax as mj
        mj.slice_viewer(sinogram, title='Un-filtered sinogram.')
        mj.slice_viewer(filtered_sinogram, title='FDK filtered sinogram.')
    finally:
        pass

if __name__ == "__main__":

    # viewer for verifying sinogram is filtered right
    # viewer()

    # testing

    # sinogram_shape = (16, 16, 16)
    sinogram_shape = (1792, 1792, 1792)
    # sinogram_shape = (2048, 2048, 2048)
    fdk_obj = FDK(sinogram_shape)
    sinogram = jnp.ones(sinogram_shape)
    sinogram = jax.device_put(sinogram, fdk_obj.sinogram_device)
    filtered_sinogram = fdk_obj.fdk_filter(sinogram)

    print("complete")
