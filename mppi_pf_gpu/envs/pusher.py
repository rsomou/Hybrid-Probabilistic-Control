"""
envs/pusher.py
Analytical dynamics for the Gymnasium Pusher-v5 environment.

State layout (STATE_DIM = 20):
    [0:7]   q        — joint angles (rad)
    [7:14]  qdot     — joint velocities (rad/s)
    [14:16] obj_pos  — 2-D object position (x, y)       — HIDDEN from PF
    [16:18] obj_vel  — 2-D object velocity (vx, vy)     — HIDDEN from PF
    [18:20] tip_pos  — 2-D fingertip position (x, y)    — INJECTED from obs

    tip_pos is the TRUE fingertip position from MuJoCo, injected each step
    via inject_observation(). It is used for contact detection instead of
    the approximate analytical FK.  During MPPI multi-step rollouts,
    tip_pos is updated by the (approximate) analytical FK after each step.

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
* Contact detection uses the INJECTED true fingertip position (state[18:20])
  rather than the analytical FK.  This is critical because the analytical
  FK (a 2-link planar approximation of a 3D arm) has 0.1–0.7m error.
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
DAMPING        = 0.1        # joint velocity damping coefficient
LINK_MASS      = 1.0        # diagonal mass approximation (kg)
CONTACT_RADIUS = 0.17       # metres — enlarged contact zone so PF can detect proximity
PUSH_STRENGTH  = 20.0       # N / (m/s) — strong enough for reaction signal to exceed obs_noise
FRICTION       = 0.2        # object sliding friction coefficient

# --- Effective planar FK geometry (from MuJoCo Pusher-v5 MJCF) ---
# The real arm is 3D, but operates mostly in the XY plane (gravity=0).
# The Gymnasium obs "fingertip" (get_body_com("tips_arm")) is at the
# WRIST body frame, NOT 0.1m past it.  Verified: at q=0 real tip =
# (0.821, -0.6) = base + 0.5 + 0.321, confirming 2-link model.
#
# Effective 2-link chain in XY (used only for FK during MPPI rollouts):
#   link 0: shoulder → elbow  = 0.1 + 0.4 = 0.5m  (rotated by q[0])
#   link 1: elbow → wrist     = 0.321m             (rotated by q[0]+q[3])
BASE_X         = 0.0
BASE_Y         = -0.6
EFF_LINK_LENGTHS = [0.5, 0.321]          # metres
EFF_JOINT_INDICES = [0, 3]               # which q[] indices affect XY FK
N_EFF_LINKS    = 2

STATE_DIM  = 20             # 7q + 7qdot + 2 obj_pos + 2 obj_vel + 2 tip_pos
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
        Compute fingertip (x, y) using the effective 3-link planar model.

        Only joints 0, 3, 5 affect the XY plane position (the others are
        lift/roll joints that mostly change Z).

        Returns
        -------
        tip_x, tip_y : float
            Fingertip position in the planar workspace.
        """
        tip_x = BASE_X
        tip_y = BASE_Y
        cumulative_angle = 0.0
        for k in range(N_EFF_LINKS):
            cumulative_angle += q[EFF_JOINT_INDICES[k]]
            tip_x += EFF_LINK_LENGTHS[k] * np.cos(cumulative_angle)
            tip_y += EFF_LINK_LENGTHS[k] * np.sin(cumulative_angle)
        return tip_x, tip_y

    @staticmethod
    def _planar_jacobian(q: np.ndarray):
        """
        Compute the 2×7 Jacobian mapping qdot → fingertip velocity in XY.

        Only the effective joints have non-zero columns.
        Returns J_x, J_y as (7,) arrays (zero for non-effective joints).
        """
        J_x = np.zeros(NUM_JOINTS, dtype=np.float64)
        J_y = np.zeros(NUM_JOINTS, dtype=np.float64)

        # Cumulative angle at each effective link end
        cum_angles = np.zeros(N_EFF_LINKS)
        a = 0.0
        for k in range(N_EFF_LINKS):
            a += q[EFF_JOINT_INDICES[k]]
            cum_angles[k] = a

        # For effective joint k, its column in J is the sum of
        # -L_m * sin(cum_m) / +L_m * cos(cum_m) for all m >= k
        for k in range(N_EFF_LINKS):
            jx = 0.0
            jy = 0.0
            for m in range(k, N_EFF_LINKS):
                jx += -EFF_LINK_LENGTHS[m] * np.sin(cum_angles[m])
                jy +=  EFF_LINK_LENGTHS[m] * np.cos(cum_angles[m])
            j_idx = EFF_JOINT_INDICES[k]
            J_x[j_idx] = jx
            J_y[j_idx] = jy

        return J_x, J_y

    # ------------------------------------------------------------------ #
    # CPU dynamics
    # ------------------------------------------------------------------ #

    def f_numpy(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        One semi-implicit Euler step on CPU.

        state layout: [q(7), qdot(7), obj_pos(2), obj_vel(2), tip_pos(2)]
        action: (7,) joint torques

        Contact uses state[18:20] (injected true fingertip position) rather
        than the analytical FK.  This is critical because the FK is a 2-link
        planar approximation of a 3D arm and can be off by 0.1–0.7m.
        After integration, tip_pos is updated via FK for MPPI rollouts.
        """
        state   = np.asarray(state, dtype=np.float64)
        action  = np.clip(action, -ACTION_BOUND, ACTION_BOUND).astype(np.float64)
        dt      = self._dt

        q       = state[0:7].copy()
        qdot    = state[7:14].copy()
        obj_pos = state[14:16].copy()
        obj_vel = state[16:18].copy()
        tip_pos = state[18:20].copy()     # injected real fingertip xy

        # ---- Arm acceleration from applied torques --------------------
        tau     = action
        qddot   = (tau - DAMPING * qdot) / LINK_MASS   # shape (7,)

        # ---- Contact detection using REAL tip position ----------------
        # tip_pos comes from obs[14:16] (injected each step), so contact
        # is detected at the TRUE fingertip location.  The Jacobian is
        # still approximate but only used for velocity and reaction torque.
        J_x, J_y = self._planar_jacobian(q)
        tip_vx = float(J_x @ qdot)
        tip_vy = float(J_y @ qdot)

        diff    = obj_pos - tip_pos
        dist    = np.linalg.norm(diff)

        if dist < CONTACT_RADIUS and dist > 1e-8:
            push_dir    = diff / dist
            tip_vel     = np.array([tip_vx, tip_vy])
            v_component = max(0.0, float(tip_vel @ push_dir))
            push_force  = PUSH_STRENGTH * v_component

            obj_vel += push_force * push_dir * dt

            # Newton's 3rd law: reaction on arm via J^T
            rx = -push_force * push_dir[0]
            ry = -push_force * push_dir[1]
            reaction_torque = J_x * rx + J_y * ry
            qddot += reaction_torque / LINK_MASS

        # ---- Semi-implicit Euler (with reaction force in qddot) -------
        qdot_new = qdot + qddot * dt
        q_new    = q    + qdot_new * dt

        # ---- Object dynamics ------------------------------------------
        obj_vel *= (1.0 - FRICTION * dt)
        obj_pos  = obj_pos + obj_vel * dt

        # ---- Update tip_pos via FK (approximate, for MPPI rollouts) ---
        tip_new_x, tip_new_y = self._forward_kinematics(q_new)

        # ---- Pack next state ------------------------------------------
        next_state = np.empty(STATE_DIM, dtype=np.float64)
        next_state[0:7]   = q_new
        next_state[7:14]  = qdot_new
        next_state[14:16] = obj_pos
        next_state[16:18] = obj_vel
        next_state[18]    = tip_new_x
        next_state[19]    = tip_new_y
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

        obj_pos = state[14:16]
        tip_pos = state[18:20]           # use stored tip position

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
        positions are sampled from the Pusher-v5 starting-state prior
        in WORLD FRAME coordinates.  The MJCF body origin is (0.45, -0.05)
        and slide joints are offset by U[-0.3, 0] x U[-0.2, 0.2], so:
          x ~ Uniform(0.15, 0.45)
          y ~ Uniform(-0.25, 0.15)
        This ensures the PF never peeks at the hidden object position.

        Joint angles and velocities ARE observable (obs[0:14]).

        Pusher-v5 obs layout (23-dim):
          [0:7]   q              raw joint angles (rad)      → used
          [7:14]  qdot           joint velocities (rad/s)    → used
          [14:17] fingertip_pos  (x, y, z)                   → tip_pos
          [17:20] obj_pos        (x, y, z)                   → HIDDEN
          [20:23] goal_pos       (x, y, z)                   → skipped
        """
        obs = np.asarray(obs, dtype=np.float64)

        q      = obs[0:7]
        qdot   = obs[7:14]
        tip_xy = obs[14:16]   # real fingertip x, y

        particles = np.zeros((N, STATE_DIM), dtype=np.float32)

        # Tile deterministic joint state across all N particles
        particles[:, 0:7]  = np.tile(q.astype(np.float32),    (N, 1))
        particles[:, 7:14] = np.tile(qdot.astype(np.float32), (N, 1))

        # Object position: sample from Pusher-v5 initial prior in WORLD FRAME.
        # MJCF body origin (0.45, -0.05) + slide offset U[-0.3,0] x U[-0.2,0.2]
        #   x ~ Uniform(0.15, 0.45),  y ~ Uniform(-0.25, 0.15)
        particles[:, 14] = np.random.uniform(0.15, 0.45, N).astype(np.float32)
        particles[:, 15] = np.random.uniform(-0.25, 0.15, N).astype(np.float32)

        # Object velocity: zero prior
        particles[:, 16:18] = 0.0

        # Fingertip position: set from real observation
        particles[:, 18] = tip_xy[0]
        particles[:, 19] = tip_xy[1]

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
   State layout: [q(7), qdot(7), obj_pos(2), obj_vel(2), tip_pos(2)] = 20
   tip_pos = injected real fingertip position; used for contact detection
   ========================================================= */

#define STATE_DIM       20
#define ACTION_DIM      7
#define NUM_JOINTS      7
#define OBS_DIM         14      /* q(7)+qdot(7) only — obj_pos withheld (partial obs) */
#define DAMPING         0.1f
#define LINK_MASS       1.0f
#define CONTACT_RADIUS  0.17f
#define PUSH_STRENGTH   20.0f
#define FRICTION        0.2f
#define ACTION_BOUND    2.0f

/* --- Effective 2-link planar FK (from MuJoCo Pusher-v5 MJCF) ---
   "fingertip" obs is at the wrist body frame, not 0.1m past it.
   Verified: at q=0, real_tip=(0.821,-0.6) = base + 0.5 + 0.321.
   Only used for updating tip_pos during MPPI multi-step rollouts;
   contact detection uses the injected real tip_pos (state[18:20]). */
#define BASE_X          0.0f
#define BASE_Y         -0.6f
#define N_EFF_LINKS     2
__device__ const int   EFF_JOINT_IDX[2] = {0, 3};
__device__ const float EFF_LINK_LEN[2]  = {0.5f, 0.321f};

/* ------------------------------------------------------------------ */
/* Forward kinematics: 3-link effective planar chain                    */
/* ------------------------------------------------------------------ */
__device__ void forward_kinematics(const float* q, float* tip_x, float* tip_y)
{
    float cx = BASE_X, cy = BASE_Y;
    float angle = 0.0f;
    for (int k = 0; k < N_EFF_LINKS; k++) {
        angle += q[EFF_JOINT_IDX[k]];
        cx    += EFF_LINK_LEN[k] * cosf(angle);
        cy    += EFF_LINK_LEN[k] * sinf(angle);
    }
    *tip_x = cx;
    *tip_y = cy;
}

/* ------------------------------------------------------------------ */
/* Planar Jacobian: computes Jx[7], Jy[7] and tip velocity             */
/* Only columns EFF_JOINT_IDX[k] are non-zero.                         */
/* ------------------------------------------------------------------ */
__device__ void planar_jacobian(const float* q, const float* qdot,
                                float* Jx, float* Jy,
                                float* vx, float* vy)
{
    /* Zero all 7 columns */
    for (int j = 0; j < NUM_JOINTS; j++) { Jx[j] = 0.0f; Jy[j] = 0.0f; }

    /* Cumulative angles at each effective link end */
    float cum[N_EFF_LINKS];
    float a = 0.0f;
    for (int k = 0; k < N_EFF_LINKS; k++) {
        a      += q[EFF_JOINT_IDX[k]];
        cum[k]  = a;
    }

    /* For effective joint k, J column = sum_{m>=k} of link contributions */
    for (int k = 0; k < N_EFF_LINKS; k++) {
        float jx = 0.0f, jy = 0.0f;
        for (int m = k; m < N_EFF_LINKS; m++) {
            jx += -EFF_LINK_LEN[m] * sinf(cum[m]);
            jy +=  EFF_LINK_LEN[m] * cosf(cum[m]);
        }
        Jx[EFF_JOINT_IDX[k]] = jx;
        Jy[EFF_JOINT_IDX[k]] = jy;
    }

    /* tip velocity = J @ qdot */
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
/*                                                                      */
/* Contact is computed BEFORE Euler integration so the reaction force   */
/* modifies qddot.  This makes q_new/qdot_new depend on obj_pos —     */
/* the signal the particle filter uses to discriminate hypotheses.      */
/* ------------------------------------------------------------------ */
__device__ void f_pusher(float* state, const float* action, float dt)
{
    float* q       = state;          /* [0..6]   */
    float* qdot    = state + 7;      /* [7..13]  */
    float* obj_pos = state + 14;     /* [14..15] */
    float* obj_vel = state + 16;     /* [16..17] */
    float* tip_pos = state + 18;     /* [18..19] injected real tip xy */

    /* ---- Arm acceleration from applied torques ---- */
    float qddot[NUM_JOINTS];
    for (int j = 0; j < NUM_JOINTS; j++) {
        float tau = fminf(fmaxf(action[j], -ACTION_BOUND), ACTION_BOUND);
        qddot[j]  = (tau - DAMPING * qdot[j]) / LINK_MASS;
    }

    /* ---- Contact using REAL tip position (state[18:20]) ---- */
    /* Jacobian is still approximate — only used for velocity + reaction */
    float Jx[NUM_JOINTS], Jy[NUM_JOINTS];
    float tip_vx, tip_vy;
    planar_jacobian(q, qdot, Jx, Jy, &tip_vx, &tip_vy);

    float dx   = obj_pos[0] - tip_pos[0];
    float dy   = obj_pos[1] - tip_pos[1];
    float dist = sqrtf(dx * dx + dy * dy);

    if (dist < CONTACT_RADIUS && dist > 1e-8f) {
        float inv_dist   = 1.0f / dist;
        float push_dir_x = dx * inv_dist;
        float push_dir_y = dy * inv_dist;
        float v_comp     = tip_vx * push_dir_x + tip_vy * push_dir_y;
        v_comp           = fmaxf(v_comp, 0.0f);
        float push_force = PUSH_STRENGTH * v_comp;

        /* Force on object */
        obj_vel[0] += push_force * push_dir_x * dt;
        obj_vel[1] += push_force * push_dir_y * dt;

        /* Newton's 3rd law: reaction on arm via J^T */
        float rx = -push_force * push_dir_x;
        float ry = -push_force * push_dir_y;
        for (int j = 0; j < NUM_JOINTS; j++) {
            qddot[j] += (Jx[j] * rx + Jy[j] * ry) / LINK_MASS;
        }
    }

    /* ---- Semi-implicit Euler (reaction force included in qddot) ---- */
    for (int j = 0; j < NUM_JOINTS; j++) {
        qdot[j] = qdot[j] + qddot[j] * dt;
        q[j]    = q[j]    + qdot[j] * dt;
    }

    /* ---- Object dynamics ---- */
    float one_minus_fric = 1.0f - FRICTION * dt;
    obj_vel[0] *= one_minus_fric;
    obj_vel[1] *= one_minus_fric;
    obj_pos[0] += obj_vel[0] * dt;
    obj_pos[1] += obj_vel[1] * dt;

    /* ---- Update tip_pos via FK (approximate, for MPPI rollouts) ---- */
    forward_kinematics(q, &tip_pos[0], &tip_pos[1]);
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
    const float* obj_pos = state + 14;
    const float* tip_pos = state + 18;   /* use stored tip position */

    /* reward_near: ||fingertip - obj_pos|| */
    float dx1  = tip_pos[0] - obj_pos[0];
    float dy1  = tip_pos[1] - obj_pos[1];
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
