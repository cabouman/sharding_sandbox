import functools
import time
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

    def fdk_filter(self, sinogram, filter_name="ramp", view_chunk_size=None):
        # sinogram may be a numpy array or a JAX array on any device

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
        recon_filter = alpha * jax.device_put(recon_filter, self.replicated_device)

        row_batch_size = 50 #min(num_rows, self.entries_per_cylinder_batch)

        num_gpus = len(jax.devices('gpu'))
        if view_chunk_size is None:
            view_chunk_size = num_gpus * 32
        # Round up to multiple of num_gpus so the chunk shards evenly
        view_chunk_size = max(num_gpus, ((view_chunk_size + num_gpus - 1) // num_gpus) * num_gpus)

        @jax.jit
        def process_chunk(chunk):
            def convolve_row(row):
                return jax.scipy.signal.fftconvolve(row, recon_filter, mode="valid")
            def apply_weight_and_convolve(view):
                return jax.lax.map(convolve_row, view * weight_map, batch_size=row_batch_size)
            return jax.lax.map(apply_weight_and_convolve, chunk, batch_size=4)

        # donate_argnums=(0,) lets XLA reuse the output buffer in-place — no full copy
        @functools.partial(jax.jit, donate_argnums=(0,))
        def write_chunk(output, update, start):
            return jax.lax.dynamic_update_slice(output, update, (start, 0, 0))

        @functools.partial(jax.jit, donate_argnums=(0,))
        def scale(arr, factor):
            return arr * factor

        # Operate in-place on the sinogram buffer: extract each chunk as a copy,
        # process it, then donate the buffer back for the write — avoids a second
        # full-array allocation and halves peak GPU memory vs a separate output buffer.
        arr = sinogram

        for start in range(0, num_views, view_chunk_size):
            end = min(start + view_chunk_size, num_views)
            chunk_views = end - start

            # Slice creates an independent copy; arr's buffer remains intact until write_chunk
            chunk = arr[start:end]
            if chunk_views < view_chunk_size:
                chunk = jnp.pad(chunk, ((0, view_chunk_size - chunk_views), (0, 0), (0, 0)))

            result = process_chunk(chunk)
            result.block_until_ready()
            del chunk

            update = result if chunk_views == view_chunk_size else result[:chunk_views]
            arr = write_chunk(arr, update, jnp.array(start, dtype=jnp.int32))
            del result

        return scale(arr, jnp.pi / num_views)

def viewer():
    import mbirjax as mj
    import h5py

    num_views = 2048
    num_det_rows = 2048
    num_det_channels = 2048

    output_directory = f"/scratch/gautschi/ncardel/recon_mem"
    h5_path = f"{output_directory}/cone_{num_views}_{num_det_rows}_{num_det_channels}_projection_data.h5"
    with h5py.File(h5_path, "r") as f:
        sinogram = f["sinogram"][:]

    with h5py.File(h5_path, "r") as f:
        sinogram2 = f["sinogram"][:]

    fdk_obj = FDK(sinogram.shape)
    sinogram2 = jax.device_put(sinogram2, fdk_obj.sinogram_device)
    sinogram = jax.device_put(sinogram, fdk_obj.sinogram_device)

    t0 = time.perf_counter()
    filtered_sinogram = fdk_obj.fdk_filter(sinogram)
    filtered_sinogram.block_until_ready()
    print(f"fdk_filter: {time.perf_counter() - t0:.2f}s")

    mj.slice_viewer(sinogram2, title='Un-filtered sinogram.')
    mj.slice_viewer(filtered_sinogram, title='FDK filtered sinogram.')


if __name__ == "__main__":

    # viewer for verifying sinogram is filtered right
    viewer()

    # testing

    # sinogram_shape = (16, 16, 16)
    sinogram_shape = (1792, 1792, 1792)
    sinogram_shape = (2048, 2048, 2048)
    fdk_obj = FDK(sinogram_shape)
    sinogram = jnp.ones(sinogram_shape, dtype=jnp.float32)
    sinogram = jax.device_put(sinogram, fdk_obj.sinogram_device)
    t0 = time.perf_counter()
    filtered_sinogram = fdk_obj.fdk_filter(sinogram)
    filtered_sinogram.block_until_ready()
    print(f"fdk_filter: {time.perf_counter() - t0:.2f}s")

    print("complete")
