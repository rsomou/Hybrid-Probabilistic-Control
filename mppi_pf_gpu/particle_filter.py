"""
particle_filter.py
GPU-resident bootstrap particle filter.

All arrays (particles, weights) live on the GPU for the entire episode.
Only three things cross the bus per step:
  CPU → GPU : pf_obs (16 floats: q, qdot, obj_xy) via update()
  CPU → GPU : action vector     (7 floats)  via propagate()
  GPU → CPU : weighted-mean state estimate  via estimate()
              ESS scalar                    via effective_sample_size()

Kernel compilation happens once in __init__ — not per step.

Architecture note
-----------------
The CUDA source compiled here = dynamics device code + PF kernel code.
Kernels are launched on gpu.stream for non-blocking execution; explicit
synchronization is performed in runner.py before taking wall-clock timestamps.
"""

import numpy as np
import cupy as cp

from kernels.pusher_kernels import ALL_PF_KERNELS


class ParticleFilter:
    """
    Bootstrap particle filter with GPU-accelerated propagation and
    weight update.

    Parameters
    ----------
    dynamics : AnalyticalDynamics
        Provides dynamics CUDA source and state/obs dimensions.
    config   : Config
    gpu      : GPUUtils
    """

    def __init__(self, dynamics, config, gpu):
        self.dynamics = dynamics
        self.N        = config.N
        self.gpu      = gpu
        self.config   = config

        # GPU-resident state (allocated at initialize())
        self.particles: cp.ndarray = None   # (N, state_dim) float32
        self.weights:   cp.ndarray = None   # (N,)           float32

        # ---- Compile kernels once ----------------------------------------
        # Use RawModule so the source is compiled exactly once and both
        # kernel functions are loaded from the same compiled binary.
        cuda_src = dynamics.get_cuda_dynamics_code() + ALL_PF_KERNELS
        compile_opts = ("--use_fast_math",)

        _module = cp.RawModule(code=cuda_src, options=compile_opts)
        self._propagate_kernel = _module.get_function("pf_propagate")
        self._weight_kernel    = _module.get_function("pf_weight_update")

    # ------------------------------------------------------------------ #
    # Episode initialisation
    # ------------------------------------------------------------------ #

    def initialize(self, obs: np.ndarray):
        """
        Bootstrap particle set from the first environment observation.

        Parameters
        ----------
        obs : (obs_dim,) numpy array — first obs from env.reset()
        """
        particles_cpu   = self.dynamics.sample_initial_particles(obs, self.N)
        self.particles  = cp.asarray(particles_cpu, dtype=cp.float32)
        self.weights    = cp.ones(self.N, dtype=cp.float32) / self.N

    # ------------------------------------------------------------------ #
    # Propagation (prior update)
    # ------------------------------------------------------------------ #

    def propagate(self, action: np.ndarray):
        """
        Apply dynamics to every particle and add Gaussian process noise.

        Parameters
        ----------
        action : (action_dim,) numpy array — action applied at this step

        This should be called *after* env.step() so that the particle cloud
        tracks the true state trajectory.
        """
        action_gpu = cp.asarray(action, dtype=cp.float32)
        noise      = self.gpu.generate_normal(
            (self.N, self.dynamics.state_dim),
            std=1.0,    # kernel multiplies by process_noise_std internally
        )

        grid, block = self.gpu.get_grid_block(self.N)

        self._propagate_kernel(
            grid, block,
            (
                self.particles,
                action_gpu,
                noise,
                cp.float32(self.config.process_noise_std),
                cp.float32(self.config.dt),
                np.int32(self.N),
            ),
        )

    # ------------------------------------------------------------------ #
    # Weight update (likelihood)
    # ------------------------------------------------------------------ #

    def update(self, obs: np.ndarray):
        """
        Multiply each particle's weight by its observation likelihood.

        Parameters
        ----------
        obs : (obs_dim,) numpy array — current observation from env

        After multiplication weights are renormalised via GPUUtils.
        """
        # Convert raw 23-dim gym obs → 16-dim PF obs: [q, qdot, obj_x, obj_y]
        obs_gpu     = cp.asarray(self.dynamics.gym_obs_to_pf_obs(obs), dtype=cp.float32)
        grid, block = self.gpu.get_grid_block(self.N)

        self._weight_kernel(
            grid, block,
            (
                self.particles,
                obs_gpu,
                self.weights,
                cp.float32(self.config.obs_noise_std),
                cp.float32(self.config.obs_noise_std_obj),
                np.int32(self.N),
            ),
        )

        self.gpu.parallel_normalize(self.weights)

    # ------------------------------------------------------------------ #
    # Systematic resampling
    # ------------------------------------------------------------------ #

    def resample(self):
        """
        Systematic resampling: replace degenerate particles using the
        CDF of the weight distribution.

        Implemented entirely on GPU via cp.cumsum + cp.searchsorted.
        No custom kernel required — CuPy handles this efficiently.
        """
        cdf  = self.gpu.inclusive_scan(self.weights)     # (N,) float32

        # Stratified starting point, then N equally spaced points
        u0   = float(cp.random.uniform(0.0, 1.0 / self.N))
        u    = cp.arange(self.N, dtype=cp.float32) / self.N + u0

        # Map each u value to the index of the particle it selects
        indices = cp.searchsorted(cdf, u, side="left")
        indices = cp.clip(indices, 0, self.N - 1)

        # Resample (fancy-index copies data on GPU)
        self.particles = self.particles[indices].copy()
        self.weights   = cp.ones(self.N, dtype=cp.float32) / self.N

    # ------------------------------------------------------------------ #
    # State extraction
    # ------------------------------------------------------------------ #

    def estimate(self) -> np.ndarray:
        """
        Compute the weighted mean state estimate.

        Returns
        -------
        mean_state : (state_dim,) numpy float32 array — on CPU
        """
        mean = cp.average(self.particles, axis=0, weights=self.weights)
        return cp.asnumpy(mean).astype(np.float32)

    def sample(self, K: int) -> cp.ndarray:
        """
        Draw K state samples proportional to current weights.

        Used to initialise MPPI rollouts so that the planning distribution
        matches the current belief distribution.

        Parameters
        ----------
        K : int — number of samples (typically config.K)

        Returns
        -------
        cp.ndarray, shape (K, state_dim), dtype float32 — on GPU
        """
        indices = cp.random.choice(
            self.N,
            size=K,
            replace=True,
            p=self.weights,
        )
        return self.particles[indices].copy()

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def effective_sample_size(self) -> float:
        """
        Effective sample size: ESS = 1 / sum(w_i^2).

        Range: [1, N]. Values close to N indicate healthy diversity;
        values close to 1 indicate degeneracy and trigger resampling.

        The future deadline-aware scheduler reads this to modulate K.

        Returns
        -------
        float — ESS scalar on CPU
        """
        return float(1.0 / cp.sum(self.weights ** 2))
