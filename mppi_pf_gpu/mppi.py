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

        # Per-joint sigma vector — shape (action_dim,), stays on GPU
        # Effective std for joint j = sigma * sigma_joint_weights[j]
        self.sigma_vec: cp.ndarray = cp.asarray(
            [w * config.sigma for w in config.sigma_joint_weights],
            dtype=cp.float32,
        )

        # Target position set once per episode via set_target()
        self.target_gpu: cp.ndarray = None

        # Action smoothing: EMA of output actions across steps
        self._prev_action: np.ndarray = np.zeros(action_dim, dtype=np.float32)

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

        # ---- Spline control-point interpolation tables (precomputed) ------
        # Noise is sampled as (K, N_CP, action_dim) then linearly interpolated
        # to (K, H, action_dim).  This reduces effective sample dimension from
        # H*action_dim=350 to N_CP*action_dim=42 and guarantees the per-rollout
        # action trace is band-limited (no intra-step sign reversals).
        N_CP = config.N_CP
        assert 2 <= N_CP <= self.H, "N_CP must be in [2, H]"
        t_cp  = np.linspace(0, N_CP - 1, self.H, dtype=np.float32)  # (H,)
        cp_lo = np.minimum(t_cp.astype(np.int32), N_CP - 2)          # floor, clamp to [0, N_CP-2]
        cp_fr = (t_cp - cp_lo.astype(np.float32))                    # fractional part in [0, 1]
        # Upload to GPU; shapes broadcast over (K, H, action_dim)
        self._cp_lo   = cp.asarray(cp_lo)                   # (H,)    int32
        self._cp_hi   = cp.asarray(cp_lo + 1)               # (H,)    int32 — safe: max = N_CP-1
        self._cp_frac = cp.asarray(cp_fr[None, :, None])    # (1,H,1) float32

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
        """Reset nominal control sequence and smoothing state for a new episode."""
        self.u_bar = cp.zeros(
            (self.H, self.dynamics.action_dim), dtype=cp.float32
        )
        self._prev_action = np.zeros(self.dynamics.action_dim, dtype=np.float32)
        self._step = 0

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

        # 1. Sample spline control-point (CP) perturbations, then interpolate
        #    to the full H-step action sequence.
        #
        #    Sampling in CP space (K, N_CP, 7) instead of (K, H, 7) reduces
        #    effective noise dimension by ~H/N_CP ≈ 8x.  Linear interpolation
        #    from N_CP=6 CPs to H=50 steps enforces band-limited, smooth traces
        #    — no intra-step sign reversals.  AR(1) is then applied in CP space
        #    for additional inter-CP smoothness.
        beta = self.config.noise_beta
        N_CP = self.config.N_CP

        white_cp = cp.random.normal(
            0.0, 1.0, (K, N_CP, action_dim), dtype=cp.float32,
        ) * self.sigma_vec  # (K, N_CP, action_dim)

        # AR(1) in CP space
        if beta > 0 and N_CP > 1:
            scale = float(np.sqrt(1.0 - beta * beta))
            cp_eps = cp.empty((K, N_CP, action_dim), dtype=cp.float32)
            cp_eps[:, 0, :] = white_cp[:, 0, :]
            for t in range(1, N_CP):
                cp_eps[:, t, :] = (beta * cp_eps[:, t - 1, :]
                                   + scale * white_cp[:, t, :])
        else:
            cp_eps = white_cp

        # Linear interpolation: (K, N_CP, action_dim) → (K, H, action_dim)
        # _cp_lo / _cp_hi are precomputed integer index arrays of shape (H,);
        # _cp_frac has shape (1, H, 1) for broadcasting over (K, H, action_dim).
        lo = cp_eps[:, self._cp_lo, :]   # (K, H, action_dim)
        hi = cp_eps[:, self._cp_hi, :]   # (K, H, action_dim)
        self._eps[:K] = lo + self._cp_frac * (hi - lo)

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

        # Elite filtering: zero out weights for the worst (1-elite_frac)*K
        # trajectories before normalisation.  Only the top elite_frac*K
        # lowest-cost rollouts contribute to u_bar — sharpens the update
        # signal and prevents bad rollouts from polluting the nominal plan.
        if self.config.elite_frac < 1.0:
            elite_k = max(1, int(K * self.config.elite_frac))
            # argsort costs ascending; mask everything outside the elite set
            sorted_idx   = cp.argsort(self._costs)          # indices low→high cost
            elite_idx    = sorted_idx[:elite_k]             # best elite_k
            mask         = cp.zeros(K, dtype=cp.float32)
            mask[elite_idx] = 1.0
            self._weights *= mask

        self.gpu.parallel_normalize(self._weights)

        # ---- MPPI diagnostics (every 10 steps) ----------------------------
        if hasattr(self, '_step') and self._step % 10 == 0:
            c = cp.asnumpy(self._costs[:K])
            w = cp.asnumpy(self._weights[:K])
            ubar_cpu = cp.asnumpy(self.u_bar)
            eps_cpu  = cp.asnumpy(self._eps[:K])

            # Cost landscape
            c_mean, c_std = float(c.mean()), float(c.std())
            c_min, c_max  = float(c.min()),  float(c.max())
            c_range       = c_max - c_min
            c_cv          = c_std / (c_mean + 1e-8)  # coefficient of variation

            # Weight concentration (effective K)
            w_eff = 1.0 / (float((w ** 2).sum()) + 1e-12)
            w_max = float(w.max())

            # u_bar magnitude (how much nominal plan has developed)
            ubar_rms = float(np.sqrt((ubar_cpu[:, :6] ** 2).mean()))

            # eps magnitude across rollouts
            eps_rms = float(np.sqrt((eps_cpu[:, :, :6] ** 2).mean()))

            # Per-joint eps spread (std across K rollouts at t=0)
            eps_t0_std = eps_cpu[:, 0, :].std(axis=0)

            print(
                f"  MPPI step {self._step}: "
                f"cost mean={c_mean:.1f} std={c_std:.2f} "
                f"range={c_range:.2f} CV={c_cv:.4f} | "
                f"w_eff={w_eff:.0f}/{K} w_max={w_max:.4f} | "
                f"ubar_rms={ubar_rms:.3f} eps_rms={eps_rms:.3f}"
            )
            print(
                f"           "
                f"eps_t0_std/joint=[{', '.join(f'{s:.3f}' for s in eps_t0_std)}] | "
                f"cost p5={np.percentile(c,5):.1f} p50={np.percentile(c,50):.1f} "
                f"p95={np.percentile(c,95):.1f}"
            )
        if hasattr(self, '_step'):
            self._step += 1

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

        # 5. Update u_bar and clip to action bounds.
        #    Pin column 6 (wrist_roll) to zero: J_6=0 always so the joint
        #    cannot transfer force to the object, and leaving it free causes
        #    the planner to waste action budget on rotation.
        self.u_bar = cp.clip(
            self.u_bar + u_bar_delta,
            self.action_low,
            self.action_high,
        )
        self.u_bar[:, 6] = 0.0

        # 6. Extract first action, shift horizon (receding horizon).
        action_raw = cp.asnumpy(self.u_bar[0].copy())
        self.u_bar = cp.roll(self.u_bar, -1, axis=0)
        self.u_bar[-1] = 0.0

        # 7. EMA smoothing: blend with previous action to reduce jitter.
        alpha  = self.config.action_alpha
        action = alpha * self._prev_action + (1.0 - alpha) * action_raw
        action = np.clip(action, -2.0, 2.0).astype(np.float32)
        action[6] = 0.0   # wrist_roll frozen — keep it zeroed after EMA blend
        self._prev_action = action.copy()

        return action, timing
