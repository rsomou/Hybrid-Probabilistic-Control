"""
particle_filter_mjx.py
JAX/MJX-based bootstrap particle filter.

Replaces the CuPy/CUDA particle filter with a pure JAX implementation
that uses MJX for dynamics propagation.

Key design:
    - All arrays (particles, weights) are JAX arrays
    - Propagation uses vmap(step_one) over N particles
    - Weight update is pure JAX (Gaussian log-likelihood on obj_pos)
    - Systematic resampling via jnp.searchsorted
    - Rao-Blackwellisation: observed q/qdot injected into all particles

State representation
--------------------
Unlike the old PF which used a flat 21-dim state vector, the MJX PF
stores particles as (qpos, qvel) pairs — the native MJX state format.
This avoids converting between flat vectors and MJX Data pytrees.

Particles are stored as:
    particle_qpos : (N, nq) — generalised positions
    particle_qvel : (N, nv) — generalised velocities
"""

import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import mujoco
from mujoco import mjx

from dynamics_mjx import step_one, reset_data


class ParticleFilter_MJX:
    """
    JAX/MJX-based bootstrap particle filter.

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
        self.N = config.N
        self.mjx_model = mjx_dyn.mjx_model

        # Body IDs for extracting positions
        self._fork_body_id = mjx_dyn.fork_body_id
        self._obj_body_id = mjx_dyn.obj_body_id

        # State dimensions
        self.nq = mjx_dyn.nq  # 11
        self.nv = mjx_dyn.nv  # 11

        # Particle state (initialised in initialize())
        self.particle_qpos: jnp.ndarray = None  # (N, nq)
        self.particle_qvel: jnp.ndarray = None  # (N, nv)
        self.weights: jnp.ndarray = None         # (N,)

        # PRNG
        self._rng_key = jax.random.PRNGKey(1)

        # JIT-compiled functions (built lazily)
        self._propagate_jit = None
        self._template_data = None  # MJX Data template for reset_data

    # ------------------------------------------------------------------ #
    # Episode initialisation
    # ------------------------------------------------------------------ #

    def initialize(self, obs: np.ndarray):
        """
        Bootstrap particle set from the first Pusher-v5 observation.

        Observed q/qdot are tiled across all particles.
        Object position is sampled from the Pusher-v5 initial prior
        (uniform over the starting-state distribution in world frame).

        Parameters
        ----------
        obs : (23,) numpy array — first obs from env.reset()
        """
        N = self.N
        nq = self.nq
        nv = self.nv
        obs = np.asarray(obs, dtype=np.float64)

        q = obs[0:7]     # arm joint angles
        qdot = obs[7:14]  # arm joint velocities

        # Object position from observation (for initial placement)
        # obj_xy = obs[17:19]  — but we sample from the prior instead
        # The object body is at MJCF pos (0.45, -0.05, -0.275),
        # with slide joints allowing offset.
        # Pusher-v5 resets: obj_slidey ~ U(-0.3, 0), obj_slidex ~ U(-0.2, 0.2)
        # So in world frame: x = 0.45 + U(-0.2, 0.2), y = -0.05 + U(-0.3, 0)
        self._rng_key, key_obj = jax.random.split(self._rng_key)
        key_ox, key_oy, key_jq, key_jv = jax.random.split(key_obj, 4)

        # Build qpos: [q(7), obj_slidey, obj_slidex, goal_slidey, goal_slidex]
        # qpos[0:7] = arm joints, qpos[7] = obj_slidey, qpos[8] = obj_slidex
        # qpos[9:11] = goal slides (don't change)

        # Get goal position from env (obs[20:22])
        # Goal slide offsets: these don't change during episode
        # We need the INITIAL env qpos to get the goal slides
        # For now, sample object position from prior and set goal to 0
        obj_slidey = jax.random.uniform(key_oy, (N,), minval=-0.3, maxval=0.0)
        obj_slidex = jax.random.uniform(key_ox, (N,), minval=-0.2, maxval=0.2)

        # Build qpos for all particles
        q_tile = jnp.tile(jnp.array(q, dtype=jnp.float32), (N, 1))
        # Add tiny jitter to arm joints
        q_jitter = jax.random.normal(key_jq, (N, 7), dtype=jnp.float32) * 0.01
        q_tile = q_tile + q_jitter

        # qpos layout: [arm(7), obj_slidey, obj_slidex, goal_slidey, goal_slidex]
        self.particle_qpos = jnp.zeros((N, nq), dtype=jnp.float32)
        self.particle_qpos = self.particle_qpos.at[:, :7].set(q_tile)
        self.particle_qpos = self.particle_qpos.at[:, 7].set(obj_slidey)
        self.particle_qpos = self.particle_qpos.at[:, 8].set(obj_slidex)
        # goal slides: 0 (will be overwritten by inject_observation)

        # qvel
        qdot_tile = jnp.tile(jnp.array(qdot, dtype=jnp.float32), (N, 1))
        qdot_jitter = jax.random.normal(key_jv, (N, 7), dtype=jnp.float32) * 0.005
        qdot_tile = qdot_tile + qdot_jitter

        self.particle_qvel = jnp.zeros((N, nv), dtype=jnp.float32)
        self.particle_qvel = self.particle_qvel.at[:, :7].set(qdot_tile)
        # Object and goal velocities are 0

        # Uniform weights
        self.weights = jnp.ones(N, dtype=jnp.float32) / N

        # Create a template MJX data for use in batched propagation
        # Use the first particle's state to create it
        self._template_data = self.mjx_dyn.make_mjx_data(
            np.array(self.particle_qpos[0]),
            np.array(self.particle_qvel[0]),
        )

    # ------------------------------------------------------------------ #
    # Observation injection (Rao-Blackwellisation)
    # ------------------------------------------------------------------ #

    def inject_observation(self, obs: np.ndarray):
        """
        Overwrite every particle's q and qdot with the true observed values
        (plus small jitter to avoid degeneracy).

        Only the hidden state (object position) differs between particles.

        Parameters
        ----------
        obs : (23,) numpy array — raw Pusher-v5 gym observation
        """
        obs = np.asarray(obs, dtype=np.float32)
        q_true = obs[0:7]
        qdot_true = obs[7:14]

        self._rng_key, key_jq, key_jv = jax.random.split(self._rng_key, 3)
        N = self.N

        # Inject q with tiny jitter
        q_obs = jnp.array(q_true, dtype=jnp.float32)
        jitter_q = jax.random.normal(key_jq, (N, 7), dtype=jnp.float32) * self.config.inject_noise_std
        self.particle_qpos = self.particle_qpos.at[:, :7].set(q_obs[None, :] + jitter_q)

        # Inject qdot with tiny jitter
        qdot_obs = jnp.array(qdot_true, dtype=jnp.float32)
        jitter_v = jax.random.normal(key_jv, (N, 7), dtype=jnp.float32) * self.config.inject_noise_std
        self.particle_qvel = self.particle_qvel.at[:, :7].set(qdot_obs[None, :] + jitter_v)

        # Also inject goal slides from obs (they're deterministic)
        # Goal is at obs[20:22] in world frame.
        # Goal slide offsets = goal_world - goal_body_origin
        # goal_body_origin = (0.45, -0.05, -0.323)
        goal_slidey = obs[21] - (-0.05)   # y offset
        goal_slidex = obs[20] - 0.45       # x offset
        self.particle_qpos = self.particle_qpos.at[:, 9].set(goal_slidey)
        self.particle_qpos = self.particle_qpos.at[:, 10].set(goal_slidex)

    # ------------------------------------------------------------------ #
    # Propagation (dynamics step)
    # ------------------------------------------------------------------ #

    def propagate(self, action: np.ndarray):
        """
        Apply MJX dynamics to every particle with process noise on object dims.

        Uses vmap(step_one) to batch-step all N particles.

        Parameters
        ----------
        action : (7,) numpy array — action applied at this step
        """
        ctrl = jnp.array(action, dtype=jnp.float32)
        N = self.N

        # Add process noise to object position dims BEFORE stepping
        # (models uncertainty about object dynamics)
        self._rng_key, key_noise = jax.random.split(self._rng_key)
        obj_noise = jax.random.normal(key_noise, (N, 2), dtype=jnp.float32) * self.config.process_noise_std_obj
        self.particle_qpos = self.particle_qpos.at[:, 7].add(obj_noise[:, 0])  # obj_slidey
        self.particle_qpos = self.particle_qpos.at[:, 8].add(obj_noise[:, 1])  # obj_slidex

        # Build the propagate function if not yet done
        if self._propagate_jit is None:
            self._build_propagate_jit()

        # Run batched propagation
        self.particle_qpos, self.particle_qvel = self._propagate_jit(
            self.particle_qpos, self.particle_qvel, ctrl,
        )

    def _build_propagate_jit(self):
        """Build JIT-compiled batched propagation function."""
        mjx_model = self.mjx_model
        template_data = self._template_data

        def _propagate_one(qpos, qvel, ctrl):
            """Step a single particle through MJX dynamics."""
            # Replace qpos/qvel in template data and recompute derived
            data = reset_data(mjx_model, template_data, qpos, qvel)
            # Step dynamics (5 inner MJX steps)
            data = step_one(mjx_model, data, ctrl)
            return data.qpos, data.qvel

        # vmap over N particles: qpos (N, nq), qvel (N, nv) batched;
        # ctrl (7,) shared across all particles.
        _propagate_batch = jax.vmap(
            _propagate_one,
            in_axes=(0, 0, None),  # batch qpos/qvel, broadcast ctrl
        )

        print(f"  [PF_MJX] Compiling propagation (N={self.N})...")
        self._propagate_jit = jax.jit(_propagate_batch)

    # ------------------------------------------------------------------ #
    # Weight update (likelihood)
    # ------------------------------------------------------------------ #

    def update(self, obs: np.ndarray):
        """
        Likelihood-based weight update using object position.

        Only the HIDDEN dimensions (obj_pos) contribute to the likelihood.
        Joint dims are Rao-Blackwellised (injected), so they're identical
        across particles and carry no discriminative signal.

        Parameters
        ----------
        obs : (23,) numpy array — raw Pusher-v5 gym observation
        """
        obs = np.asarray(obs, dtype=np.float32)

        # True object position from observation (world frame)
        # obs[17:19] = obj_xy in world frame
        obj_obs_x = obs[17]  # world x
        obj_obs_y = obs[18]  # world y

        # Particle object position: convert from slide offsets to world frame
        # obj_world_x = 0.45 + obj_slidex (qpos[8])
        # obj_world_y = -0.05 + obj_slidey (qpos[7])
        particle_obj_x = 0.45 + self.particle_qpos[:, 8]   # (N,)
        particle_obj_y = -0.05 + self.particle_qpos[:, 7]   # (N,)

        # Log-likelihood: Gaussian on object position (2D)
        inv_var = 1.0 / (self.config.obs_noise_std_obj ** 2)
        diff_x = particle_obj_x - obj_obs_x
        diff_y = particle_obj_y - obj_obs_y
        log_lik = -0.5 * inv_var * (diff_x ** 2 + diff_y ** 2)

        # Max-subtraction for numerical stability
        max_ll = jnp.max(log_lik)
        self.weights = self.weights * jnp.exp(log_lik - max_ll)

        # Normalize
        total = jnp.sum(self.weights)
        self.weights = jnp.where(
            total > 0,
            self.weights / total,
            jnp.ones(self.N, dtype=jnp.float32) / self.N,
        )

    # ------------------------------------------------------------------ #
    # Systematic resampling
    # ------------------------------------------------------------------ #

    def resample(self):
        """
        Systematic resampling using the CDF of the weight distribution.
        """
        self._rng_key, key_u = jax.random.split(self._rng_key)

        cdf = jnp.cumsum(self.weights)
        u0 = jax.random.uniform(key_u, (), minval=0.0, maxval=1.0 / self.N)
        u = jnp.arange(self.N, dtype=jnp.float32) / self.N + u0

        indices = jnp.searchsorted(cdf, u, side="left")
        indices = jnp.clip(indices, 0, self.N - 1)

        self.particle_qpos = self.particle_qpos[indices]
        self.particle_qvel = self.particle_qvel[indices]
        self.weights = jnp.ones(self.N, dtype=jnp.float32) / self.N

    # ------------------------------------------------------------------ #
    # State estimation
    # ------------------------------------------------------------------ #

    def estimate_qpos_qvel(self):
        """
        Compute the weighted mean state estimate.

        Returns
        -------
        mean_qpos : (nq,) numpy array
        mean_qvel : (nv,) numpy array
        """
        # Weighted mean over particles
        mean_qpos = jnp.average(self.particle_qpos, axis=0, weights=self.weights)
        mean_qvel = jnp.average(self.particle_qvel, axis=0, weights=self.weights)
        return np.array(mean_qpos), np.array(mean_qvel)

    def get_obj_mean_world(self) -> np.ndarray:
        """
        Get the weighted mean object position in world frame (for diagnostics).

        Returns
        -------
        (2,) numpy array: [obj_x, obj_y] in world frame
        """
        # obj_world = body_origin + slide_offset
        obj_x = 0.45 + jnp.average(self.particle_qpos[:, 8], weights=self.weights)
        obj_y = -0.05 + jnp.average(self.particle_qpos[:, 7], weights=self.weights)
        return np.array([float(obj_x), float(obj_y)], dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def effective_sample_size(self) -> float:
        """ESS = 1 / sum(w_i^2).  Range [1, N]."""
        return float(1.0 / jnp.sum(self.weights ** 2))
