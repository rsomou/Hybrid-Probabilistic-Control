"""
envs/pusher.py
Analytical dynamics for the Gymnasium Pusher-v5 environment.

State layout (STATE_DIM = 18):
    [0:7]   q        — joint angles (rad)
    [7:14]  qdot     — joint velocities (rad/s)
    [14:16] obj_pos  — 2-D object position (x, y)
    [16:18] obj_vel  — 2-D object velocity (vx, vy)

True Pusher-v5 Gymnasium observation (23-dim, float64):
    [0:7]   q              — raw joint angles (rad)
    [7:14]  qdot           — joint angular velocities (rad/s)
    [14:17] fingertip_pos  — (x, y, z) in metres
    [17:20] obj_pos        — (x, y, z) in metres
    [20:23] goal_pos       — (x, y, z) FIXED: always (0.45, -0.05, -0.323)

PF observation (OBS_DIM = 14)  — PARTIALLY OBSERVED:
    gym_obs_to_pf_obs() returns ONLY obs[0:14] = [q(7), qdot(7)].
    Object position (obs[17:19]) is deliberately withheld from the controller.
    The particle filter must infer obj_pos from contact cues in the arm
    dynamics and the initial prior.  This makes the PF genuinely necessary.

We store target_pos (x, y only) from obs[20:22] each episode reset.

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
# PARTIAL OBSERVABILITY: PF sees only [q(7), qdot(7)] — object position is hidden.
# gym_obs_to_pf_obs() returns obs[0:14] only. The PF must infer obj_pos from
# contact dynamics (indirect evidence via arm state when fingertip hits object).
OBS_DIM    = 14             # q(7) + qdot(7) only — obj_pos intentionally withheld

# Pusher-v5 action bounds (verified at runtime against env.action_space)
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
            v_component = max(0.0, float(tip_vel @ push_dir))  # contact can only push, not pull
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
        Running cost matching the Pusher-v5 reward (negated for minimisation).

        Pusher-v5 reward:
          reward_near = -0.5 * ||fingertip - obj_pos||
          reward_dist = -1.0 * ||obj_pos   - goal_pos||
          reward_ctrl = -0.1 * ||action||^2

        cost = -reward = 0.5 * ||fingertip - obj_pos||
                       + 1.0 * ||obj_pos   - target_pos||
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

        return (0.5 * dist_tip_obj
                + 1.0 * dist_obj_target
                + 0.1 * action_cost)

    # ------------------------------------------------------------------ #
    # Observation model
    # ------------------------------------------------------------------ #

    def obs_model(self, state: np.ndarray) -> np.ndarray:
        """
        Maps internal particle state → 14-dim PF observation vector:
          [q(7), qdot(7)] = state[0:14]

        PARTIAL OBSERVABILITY (Option 1): only joint angles and velocities
        are observable. Object position (state[14:16]) is hidden — the PF
        must infer it from contact dynamics in the prior.

        This mirrors particle_to_obs() in the CUDA weight kernel, which
        also reads only state[0:OBS_DIM] where OBS_DIM=14.
        """
        return np.asarray(state, dtype=np.float32)[0:OBS_DIM]

    def gym_obs_to_pf_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Extract the 14-dim PF observation from a raw 23-dim Pusher-v5 gym obs.

        PARTIAL OBSERVABILITY: object position (obs[17:19]) is intentionally
        withheld from the controller.  The PF sees only joint state, forcing
        it to track object position via contact dynamics in the prior.

        Pusher-v5 obs layout:
          [0:7]   q              raw joint angles          → included
          [7:14]  qdot           joint velocities          → included
          [14:17] fingertip_pos  (x, y, z)                → skipped
          [17:20] obj_pos        (x, y, z)                → MASKED (hidden)
          [20:23] goal_pos       (x, y, z)                → skipped

        Returns obs[0:14] = [q(7), qdot(7)] only.
        """
        return np.asarray(obs, dtype=np.float32)[0:14]

    # ------------------------------------------------------------------ #
    # Particle initialisation
    # ------------------------------------------------------------------ #

    def sample_initial_particles(self, obs: np.ndarray, N: int) -> np.ndarray:
        """
        Bootstrap N particles from the first Pusher-v5 observation.

        PARTIAL OBSERVABILITY (Option 1):
        The true object position obs[17:19] is NOT used.  Instead, object
        positions are sampled from the Pusher-v5 starting-state prior:
          x ~ Uniform(-0.3, 0.0)
          y ~ Uniform(-0.2, 0.2)
        This ensures the PF never peeks at the hidden object position.

        Joint angles and velocities ARE observable (obs[0:14]).

        Pusher-v5 obs layout (23-dim):
          [0:7]   q              raw joint angles (rad)      → used
          [7:14]  qdot           joint velocities (rad/s)    → used
          [14:17] fingertip_pos  (x, y, z)                   → skipped
          [17:20] obj_pos        (x, y, z)                   → HIDDEN
          [20:23] goal_pos       (x, y, z)                   → skipped
        """
        obs = np.asarray(obs, dtype=np.float64)

        q    = obs[0:7]    # raw joint angles (v5: q at [0:7], NOT cos(q))
        qdot = obs[7:14]   # joint velocities

        particles = np.zeros((N, STATE_DIM), dtype=np.float32)

        # Tile deterministic joint state across all N particles
        particles[:, 0:7]  = np.tile(q.astype(np.float32),    (N, 1))
        particles[:, 7:14] = np.tile(qdot.astype(np.float32), (N, 1))

        # Object position: sample from Pusher-v5 initial prior (NOT from obs)
        #   x ~ Uniform(-0.3, 0.0),  y ~ Uniform(-0.2, 0.2)
        particles[:, 14] = np.random.uniform(-0.3, 0.0, N).astype(np.float32)
        particles[:, 15] = np.random.uniform(-0.2, 0.2, N).astype(np.float32)

        # Object velocity: zero prior (Pusher-v5 starts with zero obj velocity)
        particles[:, 16:18] = 0.0

        # Small jitter on joint dims so initial cloud is not degenerate
        particles[:, 0:7]  += np.random.normal(0.0, 0.01, (N, 7)).astype(np.float32)
        particles[:, 7:14] += np.random.normal(0.0, 0.005, (N, 7)).astype(np.float32)

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
#define OBS_DIM         14      /* q(7)+qdot(7) only — obj_pos withheld (partial obs) */
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
        v_comp           = fmaxf(v_comp, 0.0f);  /* contact can only push, not pull */
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
/* cost_pusher: running cost for MPPI rollouts (negated Pusher-v5 reward) */
/* Pusher-v5 reward:                                                    */
/*   reward_near = -0.5 * ||fingertip - obj_pos||                      */
/*   reward_dist = -1.0 * ||obj_pos   - goal_pos||                     */
/*   reward_ctrl = -0.1 * ||action||^2                                 */
/* cost = -reward = 0.5*d_near + 1.0*d_dist + 0.1*||a||^2             */
/* target: (2,) — 2-D goal position                                    */
/* ------------------------------------------------------------------ */
__device__ float cost_pusher(const float* state, const float* action,
                              const float* target)
{
    const float* q       = state;
    const float* obj_pos = state + 14;

    float tip_x, tip_y;
    forward_kinematics(q, &tip_x, &tip_y);

    /* reward_near: ||fingertip - obj_pos|| */
    float dx1  = tip_x - obj_pos[0];
    float dy1  = tip_y - obj_pos[1];
    float d1   = sqrtf(dx1 * dx1 + dy1 * dy1);

    /* reward_dist: ||obj_pos - target_pos|| */
    float dx2  = obj_pos[0] - target[0];
    float dy2  = obj_pos[1] - target[1];
    float d2   = sqrtf(dx2 * dx2 + dy2 * dy2);

    /* reward_ctrl: ||action||^2 */
    float act2 = 0.0f;
    for (int a = 0; a < ACTION_DIM; a++) {
        act2 += action[a] * action[a];
    }

    return 0.5f * d1 + 1.0f * d2 + 0.1f * act2;
}
"""
