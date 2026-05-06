"""
mppi_mjx.py
JAX/MJX-based MPPI controller.

Replaces the CuPy/CUDA MPPI controller with a pure JAX implementation
that uses MJX for dynamics rollouts.  The entire planning step —
sample → rollout → weight → update → shift — is compiled as a single
jax.jit function for maximum throughput.

Key differences from the old CUDA MPPI (mppi.py):
    - Dynamics: MJX step_one (5 inner mjx.step) instead of analytical RNEA
    - Arrays:   JAX arrays instead of CuPy arrays
    - PRNG:     jax.random.PRNGKey (explicit, functional) instead of cp.random
    - Rollout:  vmap(scan(step_fn)) instead of per-thread CUDA kernel
    - No CuPy dependency

JIT compile times (first call; same-shape calls use cache):
    K=1024, H=20: ~10-20 seconds on CPU, ~5-10 on GPU
    K=1024, H=50: ~15-30 seconds on CPU, ~8-15 on GPU
"""

import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from dynamics_mjx import (
    step_one,
    reset_data,
    cost_step,
    terminal_cost,
    ACTION_DIM,
    ACTION_BOUND,
)


class MPPI_MJX:
    """
    JAX/MJX-based MPPI controller.

    Parameters
    ----------
    mjx_dyn : MJXDynamics
        Provides mjx_model, body IDs, and data helpers.
    config  : Config
        All hyperparameters.
    """

    def __init__(self, mjx_dyn, config):
        self.mjx_dyn = mjx_dyn
        self.config = config
        self.K = config.K
        self.H = config.H
        self.mjx_model = mjx_dyn.mjx_model

        # Body IDs for extracting fork and object positions from mjx.Data
        self._fork_body_id = mjx_dyn.fork_body_id
        self._obj_body_id = mjx_dyn.obj_body_id

        # Nominal control sequence — reset each episode (JAX array)
        self.u_bar: jnp.ndarray = jnp.zeros((self.H, ACTION_DIM), dtype=jnp.float32)

        # Action bounds
        self.action_low = jnp.full(ACTION_DIM, -ACTION_BOUND, dtype=jnp.float32)
        self.action_high = jnp.full(ACTION_DIM, ACTION_BOUND, dtype=jnp.float32)

        # Per-joint sigma weights — effective std = sigma * weight
        self.sigma_vec: jnp.ndarray = jnp.array(
            [w * config.sigma for w in config.sigma_joint_weights],
            dtype=jnp.float32,
        )

        # Target (set once per episode)
        self.target: jnp.ndarray = None

        # EMA smoothing state
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)

        # Active K (for diagnostic output)
        self._K_active = config.K

        # PRNG key — split each step
        self._rng_key = jax.random.PRNGKey(0)

        # Step counter for diagnostics
        self._step = 0

        # JIT-compiled planning function (built lazily on first call)
        self._plan_jit = None
        self._compile_time = None

        # Template MJX data — created once, reused via reset_data inside JIT
        # to avoid expensive C→JAX mjx.put_data transfer every step.
        self._template_data = None

    # ------------------------------------------------------------------ #
    # Episode-level API
    # ------------------------------------------------------------------ #

    def set_target(self, target: np.ndarray):
        """Set 2-D target position for this episode."""
        self.target = jnp.array(target, dtype=jnp.float32)

    def reset(self):
        """Reset nominal sequence and smoothing state for a new episode."""
        self.u_bar = jnp.zeros((self.H, ACTION_DIM), dtype=jnp.float32)
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._step = 0
        self._rng_key = jax.random.PRNGKey(0)

    # ------------------------------------------------------------------ #
    # Core planning step
    # ------------------------------------------------------------------ #

    def compute_action(self, qpos: np.ndarray, qvel: np.ndarray, K: int = None):
        """
        Run one MPPI update and return the next action.

        Unlike the old CUDA version, this takes raw qpos/qvel state
        (not pre-built GPU arrays) since MJX needs full simulation state
        to run physics.

        Parameters
        ----------
        qpos : (nq,) numpy array — generalized positions from env/PF
        qvel : (nv,) numpy array — generalized velocities from env/PF
        K    : int or None — trajectory sample override

        Returns
        -------
        action  : (7,) numpy float32 — action to apply in env
        timing  : dict with 'rollout_ms', 'weight_ms', 'update_ms'
        """
        if K is None:
            K = self.K
        self._K_active = K
        H = self.H

        timing = {"rollout_ms": 0.0, "weight_ms": 0.0, "update_ms": 0.0}

        # Build the JIT-compiled planner on first call (or if not yet built)
        if self._plan_jit is None:
            # Create template MJX data once (used inside JIT via reset_data)
            self._template_data = self.mjx_dyn.make_mjx_data(qpos, qvel)
            self._build_plan_jit()

        # Split PRNG key for this step
        self._rng_key, step_key = jax.random.split(self._rng_key)

        # Pass qpos/qvel as simple JAX arrays — state injection happens
        # inside the JIT boundary via reset_data, avoiding the expensive
        # C-side mj_forward + mjx.put_data transfer every step.
        qpos_jax = jnp.array(qpos, dtype=jnp.float32)
        qvel_jax = jnp.array(qvel, dtype=jnp.float32)

        # Run the JIT-compiled planning function
        t_plan_start = time.perf_counter()
        u_bar_new, costs, weights = self._plan_jit(
            qpos_jax,
            qvel_jax,
            self.u_bar,
            self.target,
            step_key,
        )
        # Block until computation completes (JAX is async by default)
        u_bar_new.block_until_ready()
        t_plan_end = time.perf_counter()
        timing["rollout_ms"] = (t_plan_end - t_plan_start) * 1e3

        # Track compile time (first call is much slower)
        if self._compile_time is None:
            self._compile_time = timing["rollout_ms"]
            print(f"  [MPPI_MJX] First planning call (incl. JIT compile): "
                  f"{self._compile_time:.0f} ms")

        self.u_bar = u_bar_new

        # ---- Diagnostics (every 10 steps) ---------------------------------
        if self._step % 10 == 0:
            c = np.array(costs)
            w = np.array(weights)
            ubar_cpu = np.array(self.u_bar)

            c_mean, c_std = float(c.mean()), float(c.std())
            c_min, c_max = float(c.min()), float(c.max())
            c_cv = c_std / (c_mean + 1e-8)

            w_eff = 1.0 / (float((w ** 2).sum()) + 1e-12)
            w_max = float(w.max())
            ubar_rms = float(np.sqrt((ubar_cpu[:, :6] ** 2).mean()))

            print(
                f"  MPPI step {self._step}: "
                f"cost mean={c_mean:.1f} std={c_std:.2f} "
                f"range={c_max - c_min:.2f} CV={c_cv:.4f} | "
                f"w_eff={w_eff:.0f}/{K} w_max={w_max:.4f} | "
                f"ubar_rms={ubar_rms:.3f}"
            )
        self._step += 1

        # ---- Extract first action, shift horizon --------------------------
        action_raw = np.array(self.u_bar[0])

        # Shift u_bar: drop first, append zero at end (receding horizon)
        self.u_bar = jnp.concatenate([self.u_bar[1:],
                                       jnp.zeros((1, ACTION_DIM), dtype=jnp.float32)],
                                      axis=0)

        # EMA smoothing: blend with previous action
        alpha = self.config.action_alpha
        action = alpha * self._prev_action + (1.0 - alpha) * action_raw
        action = np.clip(action, -ACTION_BOUND, ACTION_BOUND).astype(np.float32)
        action[6] = 0.0  # wrist_roll frozen
        self._prev_action = action.copy()

        return action, timing

    # ------------------------------------------------------------------ #
    # JIT-compiled planning function
    # ------------------------------------------------------------------ #

    def _build_plan_jit(self):
        """
        Build the JIT-compiled planning function.

        This captures K, H, sigma_vec, elite_frac, template_data, etc. as
        constants.  The function takes (qpos, qvel, u_bar, target, rng_key)
        and returns (u_bar_new, costs, weights).

        State injection uses reset_data INSIDE the JIT boundary to avoid
        the expensive C-side mj_forward + mjx.put_data transfer every step.
        Only qpos (11,) and qvel (11,) are transferred as simple arrays.
        """
        K = self.K
        H = self.H
        mjx_model = self.mjx_model
        sigma_vec = self.sigma_vec
        beta = self.config.noise_beta
        lambda_ = self.config.lambda_
        elite_frac = self.config.elite_frac
        fork_body_id = self._fork_body_id
        obj_body_id = self._obj_body_id
        template_data = self._template_data

        def _plan(qpos, qvel, u_bar, target, rng_key):
            """
            Full MPPI planning step (pure function for JIT).

            1. Inject state via reset_data (inside JIT — no C transfer)
            2. Sample AR(1) perturbations
            3. Rollout K trajectories via vmap(scan(step_one))
            4. Compute costs
            5. Importance weights + elite filtering
            6. Weighted u_bar update

            Parameters
            ----------
            qpos     : (nq,) — generalized positions
            qvel     : (nv,) — generalized velocities
            u_bar    : (H, 7) — nominal control sequence
            target   : (2,) — goal position
            rng_key  : PRNGKey

            Returns
            -------
            u_bar_new : (H, 7) — updated nominal sequence
            costs     : (K,)   — per-trajectory costs
            weights   : (K,)   — normalised importance weights
            """
            # Inject state into template data and recompute derived quantities
            # (xpos, contacts, etc.) — all on device, no C-side transfer.
            mjx_data = reset_data(mjx_model, template_data, qpos, qvel)
            # ---- 1. Sample temporally-correlated perturbations ----
            # eps[k, t, a] with AR(1): eps[t] = β·eps[t-1] + √(1-β²)·white[t]
            # white has per-joint sigma scaling applied.
            key_white, _ = jax.random.split(rng_key)
            white = jax.random.normal(key_white, (K, H, ACTION_DIM),
                                      dtype=jnp.float32) * sigma_vec

            if beta > 0:
                scale = jnp.sqrt(1.0 - beta * beta)

                # Build AR(1) using lax.scan over the time axis.
                # scan body: state = previous eps[t-1], input = white[t]
                # output = eps[t]
                def ar1_body(prev, w_t):
                    """prev: (K, 7), w_t: (K, 7) -> (next_state, output)"""
                    next_eps = beta * prev + scale * w_t
                    return next_eps, next_eps

                # Transpose white to (H, K, 7) for scanning over time
                white_t = jnp.transpose(white, (1, 0, 2))  # (H, K, 7)

                # Initial state = first white noise sample
                init = white_t[0]  # (K, 7)
                _, eps_t = lax.scan(ar1_body, init, white_t[1:])
                # eps_t shape: (H-1, K, 7). Prepend init.
                eps_t = jnp.concatenate([init[None, :, :], eps_t], axis=0)
                # Transpose back to (K, H, 7)
                eps = jnp.transpose(eps_t, (1, 0, 2))
            else:
                eps = white

            # ---- 2. Rollout K trajectories ----
            # For each trajectory k, roll out H steps using scan.
            # The rollout function for a single trajectory:

            def single_rollout(mjx_data_init, eps_k):
                """
                Roll out one trajectory of H steps.

                mjx_data_init : mjx.Data — initial state (single)
                eps_k         : (H, 7) — perturbations for this trajectory

                Returns
                -------
                total_cost : scalar
                """
                def scan_body(carry, t_inp):
                    """
                    carry : (mjx_data, running_cost)
                    t_inp : (eps_t, t_index)

                    lax.scan iterates over the leading axis of t_inp.
                    """
                    data, cum_cost = carry
                    eps_t, t_idx = t_inp

                    # Perturbed action: u_bar[t] + eps[t], clipped
                    action = jnp.clip(u_bar[t_idx] + eps_t,
                                      -ACTION_BOUND, ACTION_BOUND)

                    # Running cost (before transition)
                    q = data.qpos[:7]
                    fork_xyz = data.xpos[fork_body_id]
                    obj_xy = data.xpos[obj_body_id, :2]
                    c = cost_step(q, fork_xyz, obj_xy, target)
                    cum_cost = cum_cost + c

                    # Step dynamics
                    data = step_one(mjx_model, data, action)

                    return (data, cum_cost), None

                t_indices = jnp.arange(H, dtype=jnp.int32)
                (final_data, running_cost), _ = lax.scan(
                    scan_body,
                    (mjx_data_init, jnp.float32(0.0)),
                    (eps_k, t_indices),
                )

                # Terminal cost on final state
                q_final = final_data.qpos[:7]
                fork_final = final_data.xpos[fork_body_id]
                obj_final = final_data.xpos[obj_body_id, :2]
                t_cost = terminal_cost(q_final, fork_final, obj_final, target)

                return running_cost + t_cost

            # vmap over K trajectories.
            # mjx_data_init is NOT batched (same for all K) — use in_axes=None.
            # eps has shape (K, H, 7) — vmap over axis 0.
            costs = jax.vmap(single_rollout, in_axes=(None, 0))(mjx_data, eps)
            # costs shape: (K,)

            # ---- 3. Importance weights ----
            # w_k = exp(-(S_k - S_min) / λ)
            min_cost = jnp.min(costs)
            raw_weights = jnp.exp(-(costs - min_cost) / lambda_)

            # ---- 4. Elite filtering ----
            # Zero out weights for the bottom (1-elite_frac)*K trajectories.
            if elite_frac < 1.0:
                elite_k = max(1, int(K * elite_frac))
                # Find the elite_k-th smallest cost as threshold
                # jnp.sort + indexing is XLA-friendly
                sorted_costs = jnp.sort(costs)
                threshold = sorted_costs[elite_k - 1]
                # Mask: 1 for costs <= threshold, 0 otherwise
                mask = jnp.where(costs <= threshold, 1.0, 0.0).astype(jnp.float32)
                raw_weights = raw_weights * mask

            # Normalize
            total_w = jnp.sum(raw_weights)
            weights = jnp.where(total_w > 0, raw_weights / total_w,
                                jnp.ones(K, dtype=jnp.float32) / K)

            # ---- 5. Weighted u_bar update ----
            # u_bar_delta[t, a] = sum_k(w_k * eps[k, t, a])
            # Using einsum: 'k,kta->ta'
            u_bar_delta = jnp.einsum('k,kta->ta', weights, eps)

            u_bar_new = jnp.clip(u_bar + u_bar_delta,
                                  -ACTION_BOUND, ACTION_BOUND)
            # Pin wrist_roll to zero
            u_bar_new = u_bar_new.at[:, 6].set(0.0)

            return u_bar_new, costs, weights

        print(f"  [MPPI_MJX] Compiling planning function "
              f"(K={K}, H={H})... ", end="", flush=True)
        t0 = time.perf_counter()
        self._plan_jit = jax.jit(_plan)
        # The actual compilation happens on first call, but jit() itself
        # does tracing prep work. Report compile time from first call.
        compile_prep = time.perf_counter() - t0
        print(f"jit() prep: {compile_prep:.1f}s")
