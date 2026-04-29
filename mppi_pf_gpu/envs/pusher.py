"""
envs/pusher.py
Analytical dynamics for the Gymnasium Pusher-v5 environment.

State layout (STATE_DIM = 18):
    [0:7]   q        — joint angles (rad)
    [7:14]  qdot     — joint velocities (rad/s)
    [14:16] obj_pos  — 2-D object position (x, y)
    [16:18] obj_vel  — 2-D object velocity (vx, vy)

Observation layout (OBS_DIM = 21, subset of Gym's 23-dim obs):
    [0:7]   cos(q)
    [7:14]  sin(q)
    [14:21] qdot
    (obj_pos and target_pos are handled separately)

The full Gymnasium observation is 23-dim:
    [0:7]   cos(q)      — 7
    [7:14]  sin(q)      — 7
    [14:21] qdot        — 7
    [21:23] unused/dist — 2   (fingertip-to-object x/y in some versions)
    … followed by object xyz (3) and goal xyz (3) in other layouts.

We store target_pos after episode reset and expose it via set_target().

Notes
-----
* The simplified diagonal-mass-matrix arm dynamics are intentionally
  approximate. MuJoCo's coupled mass matrix governs the real simulation;
  this model exists solely for MPPI/PF planning.
* CUDA code is functionally identical to the numpy code — verified by
  `test_dynamics_parity()` in tests/test_pusher.py.
"""

import numpy as np
import sys
import os

# Allow importing from parent package when run standalone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dynamics import AnalyticalDynamics


# --------------------------------------------------------------------------- #
# Physical constants — keep in sync with CUDA #defines in get_cuda_dynamics_code
# --------------------------------------------------------------------------- #
NUM_JOINTS     = 7
LINK_LENGTH    = 0.1        # metres per link
DAMPING        = 0.1        # joint velocity damping coefficient
LINK_MASS      = 1.0        # diagonal mass approximation (kg)
CONTACT_RADIUS = 0.06       # metres — fingertip contact sphere
PUSH_STRENGTH  = 5.0        # N / (m/s) — impulse scaling
FRICTION       = 0.2        # object sliding friction coefficient

STATE_DIM  = 18             # 7q + 7qdot + 2 obj_pos + 2 obj_vel
ACTION_DIM = 7
OBS_DIM    = 21             # cos(q) + sin(q) + qdot

# Pusher-v4 action bounds (verified at runtime against env.action_space)
ACTION_BOUND = 2.0


class PusherDynamics(AnalyticalDynamics):
    """Analytical dynamics model for MuJoCo Pusher-v5."""

    def __init__(self, dt: float = 0.05):
        self._dt = dt
        self._target_pos: np.ndarray = np.zeros(2, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Episode-level API
    # ------------------------------------------------------------------ #

    def set_target(self, target_pos: np.ndarray):
        """Set the 2-D target position for the current episode."""
        self._target_pos = np.asarray(target_pos, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Forward kinematics (CPU)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _forward_kinematics(q: np.ndarray):
        """
        Compute fingertip (x, y) and fingertip velocity direction components.

        Returns
        -------
        tip_x, tip_y : float
            Fingertip position in the planar workspace.
        """
        tip_x = 0.0
        tip_y = 0.0
        cumulative_angle = 0.0
        for j in range(NUM_JOINTS):
            cumulative_angle += q[j]
            tip_x += LINK_LENGTH * np.cos(cumulative_angle)
            tip_y += LINK_LENGTH * np.sin(cumulative_angle)
        return tip_x, tip_y

    @staticmethod
    def _fingertip_velocity(q: np.ndarray, qdot: np.ndarray):
        """
        Approximate fingertip velocity via Jacobian (planar, first-order).
        J_x[j] = -sum_{k>=j} L * sin(sum_{i<=k} q[i])
        J_y[j] =  sum_{k>=j} L * cos(sum_{i<=k} q[i])
        """
        angles = np.cumsum(q)                         # (7,) cumulative angles
        sin_a  = np.sin(angles)
        cos_a  = np.cos(angles)

        # Reverse-cumsum trick: contribution of joint j is sum over k >= j
        J_x = -np.cumsum(LINK_LENGTH * sin_a[::-1])[::-1]  # (7,)
        J_y =  np.cumsum(LINK_LENGTH * cos_a[::-1])[::-1]  # (7,)

        vx = float(J_x @ qdot)
        vy = float(J_y @ qdot)
        return vx, vy

    # ------------------------------------------------------------------ #
    # CPU dynamics
    # ------------------------------------------------------------------ #

    def f_numpy(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        One semi-implicit Euler step on CPU.

        state layout: [q(7), qdot(7), obj_pos(2), obj_vel(2)]
        action: (7,) joint torques
        """
        state   = np.asarray(state, dtype=np.float64)
        action  = np.clip(action, -ACTION_BOUND, ACTION_BOUND).astype(np.float64)
        dt      = self._dt

        q       = state[0:7].copy()
        qdot    = state[7:14].copy()
        obj_pos = state[14:16].copy()
        obj_vel = state[16:18].copy()

        # ---- Arm dynamics (diagonal mass matrix approximation) --------
        tau     = action
        qddot   = (tau - DAMPING * qdot) / LINK_MASS   # shape (7,)

        # Semi-implicit Euler
        qdot_new = qdot + qddot * dt
        q_new    = q    + qdot_new * dt

        # ---- Forward kinematics & contact force on object -------------
        tip_x, tip_y   = self._forward_kinematics(q)
        tip_vx, tip_vy = self._fingertip_velocity(q, qdot)

        diff    = obj_pos - np.array([tip_x, tip_y])
        dist    = np.linalg.norm(diff)

        if dist < CONTACT_RADIUS and dist > 1e-8:
            push_dir    = diff / dist                    # unit vector
            # Component of tip velocity along push direction
            tip_vel     = np.array([tip_vx, tip_vy])
            v_component = float(tip_vel @ push_dir)
            push_force  = PUSH_STRENGTH * v_component
            obj_vel    += push_force * push_dir * dt

        # ---- Object dynamics ------------------------------------------
        obj_vel *= (1.0 - FRICTION * dt)                 # friction damping
        obj_pos  = obj_pos + obj_vel * dt

        # ---- Pack next state ------------------------------------------
        next_state = np.empty(STATE_DIM, dtype=np.float64)
        next_state[0:7]   = q_new
        next_state[7:14]  = qdot_new
        next_state[14:16] = obj_pos
        next_state[16:18] = obj_vel
        return next_state

    # ------------------------------------------------------------------ #
    # CPU cost
    # ------------------------------------------------------------------ #

    def cost_numpy(self, state: np.ndarray, action: np.ndarray) -> float:
        """
        Running cost matching the CUDA cost_pusher device function.

        cost = 1.0 * ||fingertip - obj_pos||
             + 5.0 * ||obj_pos  - target_pos||
             + 0.1 * ||action||^2
        """
        state  = np.asarray(state, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)

        q       = state[0:7]
        obj_pos = state[14:16]

        tip_x, tip_y = self._forward_kinematics(q)
        tip_pos      = np.array([tip_x, tip_y])

        dist_tip_obj    = np.linalg.norm(tip_pos - obj_pos)
        dist_obj_target = np.linalg.norm(obj_pos - self._target_pos)
        action_cost     = float(np.dot(action, action))

        return (1.0 * dist_tip_obj
                + 5.0 * dist_obj_target
                + 0.1 * action_cost)

    # ------------------------------------------------------------------ #
    # Observation model
    # ------------------------------------------------------------------ #

    def obs_model(self, state: np.ndarray) -> np.ndarray:
        """
        Maps internal state → predicted observation.
        Returns 21-dim [cos(q), sin(q), qdot].
        Does NOT include obj_pos / target — those are handled separately.
        """
        state = np.asarray(state, dtype=np.float64)
        q     = state[0:7]
        qdot  = state[7:14]
        return np.concatenate([np.cos(q), np.sin(q), qdot]).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Particle initialisation
    # ------------------------------------------------------------------ #

    def sample_initial_particles(self, obs: np.ndarray, N: int) -> np.ndarray:
        """
        Bootstrap N particles from the first Gym observation.

        Gym Pusher-v5 obs (23-dim):
          [0:7]   cos(q)
          [7:14]  sin(q)
          [14:21] qdot
          [21:23] fingertip-to-obj distances (ignored here)
          *** Note: full obs layout varies slightly by gym version; we use
              the standard 23-dim layout from gymnasium 0.29+ ***

        We recover q via arctan2(sin(q), cos(q)) and use qdot directly.
        Object position and velocity are set to zero + small noise (the true
        obj_pos is embedded elsewhere in the obs in some versions — see
        extract_obj_pos_from_obs in runner.py).
        """
        obs = np.asarray(obs, dtype=np.float64)

        cos_q = obs[0:7]
        sin_q = obs[7:14]
        qdot  = obs[14:21]

        q = np.arctan2(sin_q, cos_q)          # (7,) — exact recovery

        particles = np.zeros((N, STATE_DIM), dtype=np.float32)

        # Tile the deterministic baseline across all N particles explicitly
        particles[:, 0:7]  = np.tile(q.astype(np.float32),    (N, 1))
        particles[:, 7:14] = np.tile(qdot.astype(np.float32), (N, 1))

        # Object pos/vel — small noise (PF will correct quickly)
        particles[:, 14:16] = np.random.normal(0.0, 0.02, (N, 2)).astype(np.float32)
        particles[:, 16:18] = np.random.normal(0.0, 0.01, (N, 2)).astype(np.float32)

        # Add jitter to joint state so the initial cloud is not degenerate
        particles[:, 0:7]  += np.random.normal(0.0, 0.01, (N, 7)).astype(np.float32)
        particles[:, 7:14] += np.random.normal(0.0, 0.01, (N, 7)).astype(np.float32)

        return particles

    # ------------------------------------------------------------------ #
    # Shape properties
    # ------------------------------------------------------------------ #

    @property
    def state_dim(self) -> int:
        return STATE_DIM

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    @property
    def obs_dim(self) -> int:
        return OBS_DIM

    @property
    def action_low(self) -> np.ndarray:
        return np.full(ACTION_DIM, -ACTION_BOUND, dtype=np.float32)

    @property
    def action_high(self) -> np.ndarray:
        return np.full(ACTION_DIM,  ACTION_BOUND, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # CUDA source
    # ------------------------------------------------------------------ #

    def get_cuda_dynamics_code(self) -> str:
        """
        Return CUDA C device code for the Pusher dynamics and cost.

        The returned string contains:
          #define  constants   (STATE_DIM, ACTION_DIM, NUM_JOINTS, …)
          __device__ forward_kinematics(…)
          __device__ f_pusher(…)         — modifies state in place
          __device__ cost_pusher(…)      — returns float running cost

        This string is concatenated with kernel strings and compiled once
        per process via cp.RawKernel.
        """
        return _PUSHER_CUDA_CODE


# --------------------------------------------------------------------------- #
# CUDA device code (kept close to f_numpy for easy side-by-side comparison)
# --------------------------------------------------------------------------- #
_PUSHER_CUDA_CODE = r"""
/* =========================================================
   Pusher-v5 CUDA device code
   State layout: [q(7), qdot(7), obj_pos(2), obj_vel(2)] = 18
   ========================================================= */

#define STATE_DIM       18
#define ACTION_DIM      7
#define NUM_JOINTS      7
#define OBS_DIM         21      /* cos(q)+sin(q)+qdot used by PF weight kernel */
#define LINK_LENGTH     0.1f
#define DAMPING         0.1f
#define LINK_MASS       1.0f
#define CONTACT_RADIUS  0.06f
#define PUSH_STRENGTH   5.0f
#define FRICTION        0.2f
#define ACTION_BOUND    2.0f

/* ------------------------------------------------------------------ */
/* Forward kinematics: cumulative sum of link vectors                   */
/* ------------------------------------------------------------------ */
__device__ void forward_kinematics(const float* q, float* tip_x, float* tip_y)
{
    float cx = 0.0f, cy = 0.0f;
    float angle = 0.0f;
    for (int j = 0; j < NUM_JOINTS; j++) {
        angle += q[j];
        cx    += LINK_LENGTH * cosf(angle);
        cy    += LINK_LENGTH * sinf(angle);
    }
    *tip_x = cx;
    *tip_y = cy;
}

/* ------------------------------------------------------------------ */
/* Fingertip velocity via planar Jacobian                               */
/* J_x[j] = -sum_{k>=j} L*sin(sum_{i<=k} q[i])                       */
/* J_y[j] =  sum_{k>=j} L*cos(sum_{i<=k} q[i])                       */
/* ------------------------------------------------------------------ */
__device__ void fingertip_velocity(const float* q, const float* qdot,
                                   float* vx, float* vy)
{
    /* Precompute cumulative angles */
    float cum[NUM_JOINTS];
    float a = 0.0f;
    for (int j = 0; j < NUM_JOINTS; j++) {
        a     += q[j];
        cum[j] = a;
    }

    /* Reverse cumulative sum for Jacobian rows */
    float Jx[NUM_JOINTS], Jy[NUM_JOINTS];
    float sx = 0.0f, sy = 0.0f;
    for (int j = NUM_JOINTS - 1; j >= 0; j--) {
        sx    += -LINK_LENGTH * sinf(cum[j]);
        sy    +=  LINK_LENGTH * cosf(cum[j]);
        Jx[j] = sx;
        Jy[j] = sy;
    }

    float tvx = 0.0f, tvy = 0.0f;
    for (int j = 0; j < NUM_JOINTS; j++) {
        tvx += Jx[j] * qdot[j];
        tvy += Jy[j] * qdot[j];
    }
    *vx = tvx;
    *vy = tvy;
}

/* ------------------------------------------------------------------ */
/* f_pusher: semi-implicit Euler step, modifies state[] in place       */
/* ------------------------------------------------------------------ */
__device__ void f_pusher(float* state, const float* action, float dt)
{
    float* q       = state;          /* [0..6]   */
    float* qdot    = state + 7;      /* [7..13]  */
    float* obj_pos = state + 14;     /* [14..15] */
    float* obj_vel = state + 16;     /* [16..17] */

    /* ---- Arm dynamics (diagonal mass matrix) ---- */
    float qddot[NUM_JOINTS];
    for (int j = 0; j < NUM_JOINTS; j++) {
        float tau  = fminf(fmaxf(action[j], -ACTION_BOUND), ACTION_BOUND);
        qddot[j]   = (tau - DAMPING * qdot[j]) / LINK_MASS;
    }

    /* Semi-implicit Euler */
    float qdot_new[NUM_JOINTS], q_new[NUM_JOINTS];
    for (int j = 0; j < NUM_JOINTS; j++) {
        qdot_new[j] = qdot[j] + qddot[j] * dt;
        q_new[j]    = q[j]    + qdot_new[j] * dt;
    }

    /* ---- Contact force on object ---- */
    float tip_x, tip_y;
    forward_kinematics(q, &tip_x, &tip_y);

    float tip_vx, tip_vy;
    fingertip_velocity(q, qdot, &tip_vx, &tip_vy);

    float dx   = obj_pos[0] - tip_x;
    float dy   = obj_pos[1] - tip_y;
    float dist = sqrtf(dx * dx + dy * dy);

    if (dist < CONTACT_RADIUS && dist > 1e-8f) {
        float inv_dist   = 1.0f / dist;
        float push_dir_x = dx * inv_dist;
        float push_dir_y = dy * inv_dist;
        float v_comp     = tip_vx * push_dir_x + tip_vy * push_dir_y;
        float push_force = PUSH_STRENGTH * v_comp;
        obj_vel[0]      += push_force * push_dir_x * dt;
        obj_vel[1]      += push_force * push_dir_y * dt;
    }

    /* ---- Object dynamics ---- */
    float one_minus_fric = 1.0f - FRICTION * dt;
    obj_vel[0] *= one_minus_fric;
    obj_vel[1] *= one_minus_fric;
    obj_pos[0] += obj_vel[0] * dt;
    obj_pos[1] += obj_vel[1] * dt;

    /* ---- Write updated joint state ---- */
    for (int j = 0; j < NUM_JOINTS; j++) {
        q[j]    = q_new[j];
        qdot[j] = qdot_new[j];
    }
}

/* ------------------------------------------------------------------ */
/* cost_pusher: running cost for MPPI rollouts                          */
/* target: (2,) — 2-D target position                                  */
/* ------------------------------------------------------------------ */
__device__ float cost_pusher(const float* state, const float* action,
                              const float* target)
{
    const float* q       = state;
    const float* obj_pos = state + 14;

    float tip_x, tip_y;
    forward_kinematics(q, &tip_x, &tip_y);

    /* ||fingertip - obj_pos|| */
    float dx1  = tip_x - obj_pos[0];
    float dy1  = tip_y - obj_pos[1];
    float d1   = sqrtf(dx1 * dx1 + dy1 * dy1);

    /* ||obj_pos - target_pos|| */
    float dx2  = obj_pos[0] - target[0];
    float dy2  = obj_pos[1] - target[1];
    float d2   = sqrtf(dx2 * dx2 + dy2 * dy2);

    /* ||action||^2 */
    float act2 = 0.0f;
    for (int a = 0; a < ACTION_DIM; a++) {
        act2 += action[a] * action[a];
    }

    return 1.0f * d1 + 5.0f * d2 + 0.1f * act2;
}
"""
