"""
gpu_utils.py
CuPy-based GPU utility layer.

Provides:
  - parallel reduction (sum, min)
  - in-place weight normalisation
  - Gaussian random number generation
  - inclusive prefix scan (for systematic resampling)
  - grid/block dimension computation

All methods operate on cp.ndarray objects. No CPU-GPU transfer happens here
unless explicitly requested (e.g. parallel_reduce_sum returns a Python float).

Design note
-----------
This class is stateless beyond a non-blocking CUDA stream and the
threads_per_block setting. The stream is exposed so callers can record
CUDA events around kernel launches for timing. All CuPy built-in calls
(cp.sum, cp.cumsum, cp.random.normal) are implicitly submitted to
CuPy's default stream unless the stream context is activated — callers
that need strict ordering should use `with self.stream:`.
"""

import cupy as cp


class GPUUtils:
    """
    Thin wrapper around CuPy primitives used by ParticleFilter and MPPI.

    Parameters
    ----------
    config : Config
        Must have fields: threads_per_block (int), device_id (int).
    """

    def __init__(self, config):
        self.threads      = config.threads_per_block
        self.device_id    = config.device_id

        # Select device
        cp.cuda.Device(self.device_id).use()

        # Non-blocking stream — exposed so callers can synchronize or
        # record CUDA events for fine-grained timing.
        self.stream = cp.cuda.Stream(non_blocking=True)

    # ------------------------------------------------------------------ #
    # Reductions
    # ------------------------------------------------------------------ #

    def parallel_reduce_sum(self, arr: cp.ndarray) -> float:
        """
        Sum all elements of a 1-D GPU array.

        Returns a Python float (triggers a single device-to-host transfer).
        """
        return float(cp.sum(arr))

    def parallel_reduce_min(self, arr: cp.ndarray) -> float:
        """
        Minimum of all elements in a 1-D GPU array.

        Returns a Python float.
        """
        return float(cp.min(arr))

    def parallel_reduce_max(self, arr: cp.ndarray) -> float:
        """Maximum of all elements in a 1-D GPU array."""
        return float(cp.max(arr))

    # ------------------------------------------------------------------ #
    # Weight normalisation
    # ------------------------------------------------------------------ #

    def parallel_normalize(self, weights: cp.ndarray):
        """
        In-place normalise weights so they sum to 1.

        If the total is zero or negative (numerical underflow), weights are
        reset to a uniform distribution to avoid NaN propagation.

        Parameters
        ----------
        weights : cp.ndarray, shape (N,), dtype float32
            Modified in place.
        """
        total = cp.sum(weights)
        if float(total) > 0.0:
            weights /= total
        else:
            # Fallback: uniform weights to prevent NaN
            weights[:] = 1.0 / weights.shape[0]

    # ------------------------------------------------------------------ #
    # Random number generation
    # ------------------------------------------------------------------ #

    def generate_normal(
        self,
        shape,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> cp.ndarray:
        """
        Generate i.i.d. Gaussian samples on the GPU.

        Parameters
        ----------
        shape : tuple[int, ...]
            Output shape.
        mean  : float
        std   : float

        Returns
        -------
        cp.ndarray, dtype float32
        """
        return cp.random.normal(mean, std, shape, dtype=cp.float32)

    def generate_uniform(
        self,
        shape,
        low: float = 0.0,
        high: float = 1.0,
    ) -> cp.ndarray:
        """Uniform random samples on the GPU, dtype float32."""
        return cp.random.uniform(low, high, shape).astype(cp.float32)

    # ------------------------------------------------------------------ #
    # Prefix scan
    # ------------------------------------------------------------------ #

    def inclusive_scan(self, arr: cp.ndarray) -> cp.ndarray:
        """
        Inclusive cumulative sum (prefix scan) for systematic resampling.

        Parameters
        ----------
        arr : cp.ndarray, shape (N,), dtype float32
            Typically the normalised weight vector.

        Returns
        -------
        cp.ndarray, shape (N,), dtype float32
            CDF array — last element equals 1.0 (modulo float32 rounding).
        """
        return cp.cumsum(arr)

    # ------------------------------------------------------------------ #
    # Grid / block helpers
    # ------------------------------------------------------------------ #

    def get_grid_block(self, N: int):
        """
        Return (grid, block) tuples for launching a 1-D CUDA kernel with
        exactly N threads (rounded up to the next multiple of threads_per_block).

        Parameters
        ----------
        N : int
            Total number of threads required.

        Returns
        -------
        grid  : tuple[int]   — number of blocks
        block : tuple[int]   — threads per block
        """
        block = (self.threads,)
        grid  = ((N + self.threads - 1) // self.threads,)
        return grid, block

    # ------------------------------------------------------------------ #
    # Memory helpers
    # ------------------------------------------------------------------ #

    def zeros(self, shape, dtype=cp.float32) -> cp.ndarray:
        """Allocate a zero-filled GPU array."""
        return cp.zeros(shape, dtype=dtype)

    def asarray(self, x, dtype=cp.float32) -> cp.ndarray:
        """Transfer a numpy array (or scalar) to the GPU."""
        return cp.asarray(x, dtype=dtype)

    def synchronize(self):
        """Block until all GPU work on the selected device is complete."""
        cp.cuda.Device(self.device_id).synchronize()
