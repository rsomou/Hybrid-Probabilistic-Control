"""
dynamics_mjx.py
MJX-based dynamics for Pusher-v5.

Replaces the analytical RNEA + CUDA dynamics with MuJoCo's JAX/XLA backend
(MJX).  Uses the same MJCF as the Gymnasium environment with identical model
modifications, so rollout-vs-real mismatch is zero by construction.

Key exports
-----------
    apply_model_modifications(model)           — physics tweaks (shared env/planner)
    MJXDynamics                                — model setup, body IDs, data helpers
    step_one(mjx_model, data, ctrl) -> data    — one control step (5 inner mjx.step)
    cost_step(q, fork_xyz, obj_xy, target)     — per-step running cost (pure JAX)
    terminal_cost(...)                         — end-of-horizon cost (pure JAX)
    verify_parity(mjx_dyn, env, ...)           — startup self-check

JIT compile times (first call; subsequent calls with same shapes are cached):
    step_one alone:              ~2-5 seconds
    vmap(step_one, K=1024):      ~5-15 seconds
    Full MPPI plan (scan+vmap):  ~10-30 seconds
    Full PF step (vmap):         ~5-10 seconds
"""

import os

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import mujoco
from mujoco import mjx


# --------------------------------------------------------------------------- #
# Constants — must match envs/pusher.py and runner.py exactly
# --------------------------------------------------------------------------- #
FRAME_SKIP = 5          # Pusher-v5 frame_skip: each control step = 5 mj_step calls
TABLE_Z = -0.275        # z-height of the table surface (MJCF object body z-pos)

# Cost weights (match the CUDA cost_pusher in envs/pusher.py)
GOAL_WEIGHT = 200.0
APPROACH_WEIGHT = 15.0
TERMINAL_WEIGHT = 10.0
JLIMIT_WEIGHT = 0.5
JLIMIT_MARGIN = 10.0
BEHIND_DIST = 0.06

# Joint position limits for the 7 arm joints (qpos indices 0..6).
JOINT_Q_MIN = jnp.array([-2.2854, -0.5236, -1.5, -2.3213, -1.5, -1.094, -1.5],
                         dtype=jnp.float32)
JOINT_Q_MAX = jnp.array([1.7146, 1.3963, 1.7, 0.0, 1.5, 0.0, 1.5],
                         dtype=jnp.float32)

ACTION_DIM = 7
ACTION_BOUND = 2.0


# --------------------------------------------------------------------------- #
# Model modifications (shared between Gymnasium env and MJX planner)
# --------------------------------------------------------------------------- #

def apply_model_modifications(model: mujoco.MjModel) -> None:
    """
    Apply physics modifications so the Pusher-v5 model matches the
    controller's expectations.  Must be applied to BOTH the Gymnasium env
    model AND the MJX planning model so they are identical.

    Modifies ``model`` **in place**.

    Modifications
    -------------
    - Object mass → 0.5 kg  (heavy, needs real pushing)
    - Object slide damping → 2.0  (table friction analog)
    - Newton solver, 100 iterations, tolerance 1e-10
    - Fork + object geoms: condim=3, stiff near-rigid contacts with friction
    """
    # Heavier object
    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    model.body_mass[obj_body_id] = 0.5

    # High viscous damping on object slide joints (like table friction)
    for jname in ["obj_slidey", "obj_slidex"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        dof = model.jnt_dofadr[jid]
        model.dof_damping[dof] = 2.0

    # Stiff contact solver
    model.opt.solver = 2          # Newton (most accurate)
    model.opt.iterations = 100
    model.opt.tolerance = 1e-10

    # Fork capsules (geom 13, 14, 15) + object cylinder (geom 18):
    # enable tangential friction and near-rigid contact.
    for gi in [13, 14, 15, 18]:
        model.geom_condim[gi] = 3                                  # tangential friction
        model.geom_friction[gi] = [1.0, 0.005, 0.0001]
        model.geom_solref[gi] = [0.001, 1.0]                      # 1ms spring → near-rigid
        model.geom_solimp[gi] = [0.99, 0.9999, 0.0001, 0.5, 2.0] # max impedance
        model.geom_margin[gi] = 0.0
        model.geom_gap[gi] = 0.0
        model.geom_priority[gi] = 1                                # these params win the mix


# --------------------------------------------------------------------------- #
# Helper: locate Pusher-v5 MJCF asset
# --------------------------------------------------------------------------- #

def _get_pusher_xml_path() -> str:
    """Return the filesystem path to the Pusher-v5 MJCF XML."""
    import gymnasium.envs.mujoco.pusher_v5 as pv5
    return os.path.join(os.path.dirname(pv5.__file__), "assets", "pusher.xml")


# --------------------------------------------------------------------------- #
# MJX Dynamics class
# --------------------------------------------------------------------------- #

class MJXDynamics:
    """
    MJX-based dynamics for Pusher-v5.

    Loads the MJCF, applies model modifications, and provides helpers for
    creating MJX Data pytrees and running JIT-compiled steps.

    Attributes
    ----------
    mj_model : mujoco.MjModel
        Modified C-side MuJoCo model (for ID lookups and C-reference sim).
    mjx_model : mjx.Model
        MJX model (JAX pytree snapshot of mj_model, for batched rollouts).
    fork_body_id : int
        Body index of r_wrist_roll_link (fork collision geometry, geoms 13-15).
    obj_body_id : int
        Body index of "object" (the puck to push).
    nq, nv, nu : int
        Generalized positions, velocities, and actuators (11, 11, 7).
    """

    def __init__(self, xml_path: str = None):
        if xml_path is None:
            xml_path = _get_pusher_xml_path()

        # Load and modify the MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        apply_model_modifications(self.mj_model)

        # Body IDs (fixed for this MJCF, verified against model)
        self.fork_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "r_wrist_roll_link"
        )  # = 9
        self.obj_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "object"
        )  # = 11
        assert self.fork_body_id >= 0, "r_wrist_roll_link body not found in MJCF"
        assert self.obj_body_id >= 0, "object body not found in MJCF"

        self.nq = self.mj_model.nq  # 11
        self.nv = self.mj_model.nv  # 11
        self.nu = self.mj_model.nu  # 7

        # Convert to MJX model AFTER modifications.
        # mjx.put_model snapshots all model params into a JAX pytree.
        self.mjx_model = mjx.put_model(self.mj_model)

        # A reusable MjData for state → MJX Data conversion (outside JIT).
        self._mj_data = mujoco.MjData(self.mj_model)

    def make_mjx_data(self, qpos: np.ndarray, qvel: np.ndarray) -> mjx.Data:
        """
        Create a single MJX Data pytree from numpy state vectors.

        Sets qpos/qvel on a C-side MjData, runs ``mj_forward`` to populate
        all derived quantities (xpos, xquat, cfrc, …), then converts to MJX.

        This method is for initialization (outside the JIT boundary).
        For batched state injection inside JIT, use ``reset_data``.

        Parameters
        ----------
        qpos : (nq,) array — generalized positions
        qvel : (nv,) array — generalized velocities

        Returns
        -------
        mjx.Data — JAX pytree, all derived quantities populated.
        """
        self._mj_data.qpos[:] = np.asarray(qpos, dtype=np.float64)
        self._mj_data.qvel[:] = np.asarray(qvel, dtype=np.float64)
        mujoco.mj_forward(self.mj_model, self._mj_data)
        return mjx.put_data(self.mj_model, self._mj_data)

    def env_state_to_mjx_data(self, env) -> mjx.Data:
        """
        Snapshot the current Gymnasium env state into an MJX Data pytree.

        Parameters
        ----------
        env : gymnasium.Env — a Pusher-v5 env (possibly wrapped)

        Returns
        -------
        mjx.Data
        """
        qpos = env.unwrapped.data.qpos.copy()
        qvel = env.unwrapped.data.qvel.copy()
        return self.make_mjx_data(qpos, qvel)


# --------------------------------------------------------------------------- #
# Core step function (pure JAX — compatible with jit / vmap / scan)
# --------------------------------------------------------------------------- #

def step_one(mjx_model: mjx.Model, data: mjx.Data,
             ctrl: jnp.ndarray) -> mjx.Data:
    """
    One control step: 5 inner MJX steps with the same ctrl.

    Matches Pusher-v5's frame_skip=5 exactly.  Each inner step advances
    by dt=0.01 s, so one control step = 0.05 s.

    This is a **pure function** — it takes and returns immutable JAX
    pytrees.  Compatible with ``jax.jit``, ``jax.vmap``, ``jax.lax.scan``.

    Parameters
    ----------
    mjx_model : mjx.Model — the MJX model (constant, not batched by vmap)
    data      : mjx.Data  — current simulation state
    ctrl      : (nu,) jax array — joint torques

    Returns
    -------
    mjx.Data — state after 5 inner physics steps
    """
    # data.replace() creates a NEW pytree with ctrl changed (JAX immutability).
    data = data.replace(ctrl=ctrl)

    # jax.lax.fori_loop is JAX's compiled for-loop.
    # body_fn(i, val) -> val;  i is unused because all 5 steps share ctrl.
    def body_fn(_, d):
        return mjx.step(mjx_model, d)

    return lax.fori_loop(0, FRAME_SKIP, body_fn, data)


def reset_data(mjx_model: mjx.Model, data: mjx.Data,
               qpos: jnp.ndarray, qvel: jnp.ndarray) -> mjx.Data:
    """
    Replace qpos/qvel in an MJX Data and recompute derived quantities.

    Use this inside JIT-compiled functions to set a new state without
    going through C MuJoCo.  ``mjx.forward`` recomputes xpos, xquat,
    contact data, etc. from the new qpos/qvel.

    Parameters
    ----------
    mjx_model : mjx.Model
    data      : mjx.Data  — template (only structure is used, values replaced)
    qpos      : (nq,) array
    qvel      : (nv,) array

    Returns
    -------
    mjx.Data — with updated qpos/qvel and recomputed derived fields
    """
    data = data.replace(qpos=qpos, qvel=qvel)
    return mjx.forward(mjx_model, data)


# --------------------------------------------------------------------------- #
# Cost functions (pure JAX — match envs/pusher.py CUDA cost_pusher)
# --------------------------------------------------------------------------- #

def cost_step(q: jnp.ndarray, fork_xyz: jnp.ndarray,
              obj_xy: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """
    Single-step running cost (matches the CUDA cost_pusher exactly).

    Three terms:
        1. GOAL_WEIGHT * d(obj, goal)²  — primary pushing objective
        2. APPROACH_WEIGHT * d(fork, behind_pt) — approach from correct side
        3. JLIMIT_WEIGHT * soft_barrier — avoid joint limits

    Parameters
    ----------
    q        : (7,) joint angles (arm only, qpos[0:7])
    fork_xyz : (3,) fork body position in world frame
    obj_xy   : (2,) object position (x, y)
    target   : (2,) goal position (x, y)

    Returns
    -------
    scalar cost (float32)
    """
    # 1. Object-to-goal distance (Euclidean)
    d_obj_goal = jnp.linalg.norm(obj_xy - target)

    # 2. "Behind" position: BEHIND_DIST behind obj, opposite to goal direction.
    #    behind = obj + unit(obj - target) * BEHIND_DIST
    #    Guides the fork to approach from behind for correct push direction.
    obj_minus_target = obj_xy - target
    gd = jnp.linalg.norm(obj_minus_target) + 1e-6
    behind_pos = obj_xy + (obj_minus_target / gd) * BEHIND_DIST

    # Fork-to-behind 3D distance (includes z → forces arm to table height)
    dx = fork_xyz[0] - behind_pos[0]
    dy = fork_xyz[1] - behind_pos[1]
    dz = fork_xyz[2] - TABLE_Z
    d_fork_behind = jnp.sqrt(dx * dx + dy * dy + dz * dz)

    # 3. Soft joint-limit barrier: sum of exp penalties near each limit
    lo = q - JOINT_Q_MIN    # distance from lower limit (positive = safe)
    hi = JOINT_Q_MAX - q    # distance from upper limit (positive = safe)
    jlim = jnp.sum(jnp.exp(-JLIMIT_MARGIN * lo) + jnp.exp(-JLIMIT_MARGIN * hi))

    return (GOAL_WEIGHT * d_obj_goal ** 2
            + APPROACH_WEIGHT * d_fork_behind
            + JLIMIT_WEIGHT * jlim)


def terminal_cost(q: jnp.ndarray, fork_xyz: jnp.ndarray,
                  obj_xy: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """
    Terminal cost at step H — same structure as running cost, scaled by
    TERMINAL_WEIGHT so the planner cares where trajectories *end*.
    """
    return TERMINAL_WEIGHT * cost_step(q, fork_xyz, obj_xy, target)


# --------------------------------------------------------------------------- #
# Parity verification (importable by runner.py for startup self-check)
# --------------------------------------------------------------------------- #

def verify_parity(mjx_dyn: MJXDynamics, env, n_steps: int = 10,
                  warn_threshold: float = 1e-3, seed: int = 42) -> float:
    """
    Run n_steps with random controls in both the Gymnasium env (C MuJoCo,
    float64) and MJX (float32) from the same initial state.  Print per-step
    qpos divergence and warn if it exceeds ``warn_threshold``.

    Call this at startup (in runner.py) to verify the MJX model matches.

    Parameters
    ----------
    mjx_dyn : MJXDynamics — already constructed
    env     : gymnasium.Env — already reset, with model modifications applied
    n_steps : int
    warn_threshold : float — warn if any step's error exceeds this
    seed    : int — RNG seed for reproducible random controls

    Returns
    -------
    max_err : float — maximum ‖qpos_env - qpos_mjx‖ across all steps
    """
    rng = np.random.RandomState(seed)

    # Snapshot env state → MJX
    qpos0 = env.unwrapped.data.qpos.copy()
    qvel0 = env.unwrapped.data.qvel.copy()
    mjx_data = mjx_dyn.make_mjx_data(qpos0, qvel0)

    # JIT-compile the step function (close over the model)
    mjx_model = mjx_dyn.mjx_model

    @jax.jit
    def _step(data, ctrl):
        return step_one(mjx_model, data, ctrl)

    # Warmup compile
    _step(mjx_data, jnp.zeros(mjx_dyn.nu, dtype=jnp.float32))

    max_err = 0.0
    print(f"  Parity check ({n_steps} steps):")

    for t in range(n_steps):
        action = rng.uniform(-2.0, 2.0, size=7).astype(np.float32)

        # Gymnasium env step (C MuJoCo, float64)
        env.step(action)
        env_qpos = env.unwrapped.data.qpos.copy()

        # MJX step (float32)
        ctrl_jax = jnp.array(action, dtype=jnp.float32)
        mjx_data = _step(mjx_data, ctrl_jax)
        mjx_qpos = np.array(mjx_data.qpos)

        err = float(np.linalg.norm(env_qpos - mjx_qpos))
        max_err = max(max_err, err)

        status = "OK" if err < warn_threshold else "WARN"
        print(f"    step {t:3d}: ‖Δqpos‖ = {err:.8f}  {status}")

    if max_err < warn_threshold:
        print(f"  PASS — max divergence {max_err:.6e} < {warn_threshold}")
    else:
        print(f"  WARNING — max divergence {max_err:.6e} exceeds {warn_threshold}!")
        print(f"  MJX model may not match the env. Investigate before proceeding.")

    return max_err
