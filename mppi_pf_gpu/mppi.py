"""
mppi.py
GPU-resident Model Predictive Path Integral (MPPI) controller.

All planning arrays live on the GPU between steps.
Per-step CPU-GPU transfers:
  GPU → CPU : u_bar[0]  (7 floats) — the action to apply
  CPU → GPU : (none during compute_action; target set once per episode)

The K parameter in compute_action() accepts an override so the future
deadline-aware scheduler can pass a dynamically chosen K_next without
touching any kernel code.

Kernel compilation happens once in __init__.
"""

import numpy as np
import cupy as cp

from kernels.pusher_kernels import ALL_MPPI_KERNELS


class MPPI:
    """
    GPU-accelerated MPPI controller.

    Parameters
    ----------
    dynamics : AnalyticalDynamics
    config   : Config
    gpu      : GPUUtils
    """

    def __init__(self, dynamics, config, gpu):
        self.dynamics = dynamics
        self.config   = config
        self.gpu      = gpu
        self.K        = config.K
        self.H        = config.H

        state_dim  = dynamics.state_dim
        action_dim = dynamics.action_dim

        # GPU-resident nominal control sequence — reset each episode
        self.u_bar: cp.ndarray = cp.zeros(
            (self.H, action_dim), dtype=cp.float32
        )

        # GPU-resident action bounds (never change)
        self.action_low  = cp.asarray(dynamics.action_low,  dtype=cp.float32)
        self.action_high = cp.asarray(dynamics.action_high, dtype=cp.float32)

        # Target position set once per episode via set_target()
        self.target_gpu: cp.ndarray = None

        # ---- Compile all MPPI kernels once --------------------------------
        # Use RawModule so the source is compiled exactly once and all three
        # kernel functions are loaded from the same compiled binary.
        cuda_src = dynamics.get_cuda_dynamics_code() + ALL_MPPI_KERNELS
        compile_opts = ("--use_fast_math",)

        _module = cp.RawModule(code=cuda_src, options=compile_opts)
        self._rollout_kernel = _module.get_function("mppi_rollout")
        self._weight_kernel  = _module.get_function("compute_importance_weights")
        self._update_kernel  = _module.get_function("weighted_eps_update")

        # ---- Pre-allocate GPU arrays to K_max to avoid reallocation -------
        # Variable-K calls slice into the first K rows.
        K_max      = config.K_max
        self._eps_buf    = cp.zeros(
            (K_max, self.H, action_dim), dtype=cp.float32
        )
        self._costs_buf  = cp.zeros(K_max, dtype=cp.float32)
        self._w_buf      = cp.zeros(K_max, dtype=cp.float32)

        # Active slices (updated in compute_action if K changes)
        self._K_active = config.K
        self._eps      = self._eps_buf[: self._K_active]
        self._costs    = self._costs_buf[: self._K_active]
        self._weights  = self._w_buf[: self._K_active]

    # ------------------------------------------------------------------ #
    # Episode-level API
    # ------------------------------------------------------------------ #

    def set_target(self, target: np.ndarray):
        """
        Upload target position for the current episode.

        Parameters
        ----------
        target : (2,) array — 2-D target (x, y) in world frame
        """
        self.target_gpu = cp.asarray(target, dtype=cp.float32)

    def reset(self):
        """Reset nominal control sequence for a new episode."""
        self.u_bar = cp.zeros(
            (self.H, self.dynamics.action_dim), dtype=cp.float32
        )

    # ------------------------------------------------------------------ #
    # Core planning step
    # ------------------------------------------------------------------ #

    def compute_action(
        self,
        initial_states: cp.ndarray,
        K: int = None,
    ):
        """
        Run one MPPI update and return the next action.

        Parameters
        ----------
        initial_states : cp.ndarray, shape (K, state_dim), dtype float32
            Particle-sampled initial states for rollouts (already on GPU).
        K : int or None
            Optional override for number of trajectories.
            If None, uses self.K (from config).
            The future adaptive scheduler passes K_next here.

        Returns
        -------
        action  : np.ndarray, shape (action_dim,) — selected action on CPU
        timing  : dict — timing placeholders for the future scheduler.
                  Keys: 'rollout_ms', 'weight_ms', 'update_ms' (all 0.0 for
                  now; populate by wrapping each launch in CUDA events when
                  fine-grained per-kernel timing is needed).
        """
        if K is None:
            K = self.K

        # Update active slices if K changed
        if K != self._K_active:
            self._K_active = K
            self._eps      = self._eps_buf[:K]
            self._costs    = self._costs_buf[:K]
            self._weights  = self._w_buf[:K]

        H          = self.H
        action_dim = self.dynamics.action_dim

        timing = {"rollout_ms": 0.0, "weight_ms": 0.0, "update_ms": 0.0}

        # 1. Sample perturbations ε ~ N(0, σ²I) on GPU
        # cp.random.normal has no 'out' parameter; assign into the pre-allocated
        # view so we still avoid allocating a fresh (K_max, H, action_dim) buffer.
        self._eps[:] = cp.random.normal(
            0.0, self.config.sigma,
            (K, H, action_dim),
            dtype=cp.float32,
        )

        # 2. Rollout kernel — K threads, each rolls out H steps
        grid_k, block = self.gpu.get_grid_block(K)
        self._rollout_kernel(
            grid_k, block,
            (
                initial_states,
                self.u_bar,
                self._eps,
                self.action_low,
                self.action_high,
                self.target_gpu,
                self._costs,
                cp.float32(self.config.dt),
                np.int32(K),
                np.int32(H),
            ),
        )

        # 3. Convert costs → unnormalised importance weights
        #    Numerically stable: subtract min_cost before exp.
        #    cp.min() returns a 0-dim cp.ndarray. CuPy treats any cp.ndarray
        #    as a device pointer in kernel argument tuples, so a 0-dim array
        #    would pass the raw GPU address as the float value — wrong.
        #    Explicitly convert to a numpy scalar to get pass-by-value.
        min_cost = np.float32(float(cp.min(self._costs)))
        self._weight_kernel(
            grid_k, block,
            (
                self._costs,
                self._weights,
                np.float32(self.config.lambda_),
                min_cost,
                np.int32(K),
            ),
        )
        self.gpu.parallel_normalize(self._weights)

        # 4. Weighted accumulation of ε → u_bar delta
        #    Each thread handles one (t, a) pair; inner loop over K
        u_bar_delta = cp.zeros((H, action_dim), dtype=cp.float32)
        n_ta        = H * action_dim
        grid_u, _   = self.gpu.get_grid_block(n_ta)
        self._update_kernel(
            grid_u, block,
            (
                self._weights,
                self._eps,
                u_bar_delta,
                np.int32(K),
                np.int32(H),
            ),
        )

        # 5. Update u_bar and clip to action bounds
        self.u_bar = cp.clip(
            self.u_bar + u_bar_delta,
            self.action_low,
            self.action_high,
        )

        # 6. Extract first action, shift horizon (receding horizon).
        #    cp.roll creates a new array — safe from overlapping-write issues
        #    that in-place slice assignment can cause on GPU arrays.
        action     = cp.asnumpy(self.u_bar[0].copy())
        self.u_bar = cp.roll(self.u_bar, -1, axis=0)
        self.u_bar[-1] = 0.0

        return action, timing
