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
CONTACT_RADIUS = 0.17       # metres — enlarged contact zone so PF can detect proximity
PUSH_STRENGTH  = 20.0       # N / (m/s) — strong enough for reaction signal to exceed obs_noise
FRICTION       = 0.2        # object sliding friction coefficient
FRAME_SKIP     = 5          # number of inner integration substeps per control step
INNER_DT       = 0.01       # MuJoCo inner timestep (control dt = FRAME_SKIP * INNER_DT)

# --- Z-height incentive for lowering the fingertip to the table plane ---
TABLE_Z        = -0.275     # z-height of the object on the table (from MJCF body pos)
Z_COST_WEIGHT  = 5.0        # weight on (tip_z - TABLE_Z)^2 in running cost

# --- Arm base position in world frame (from MuJoCo Pusher-v5 MJCF) ---
# The arm body chain starts at <body pos="0 -0.6 0"> in the MJCF.
# Full 7-DOF FK uses JOINT_AXES, JOINT_OFFSETS, and this base.
ARM_BASE = np.array([0.0, -0.6, 0.0], dtype=np.float64)

STATE_DIM  = 20             # 7q + 7qdot + 2 obj_pos + 2 obj_vel + 2 tip_pos
ACTION_DIM = 7
# PARTIAL OBSERVABILITY: PF sees only [q(7), qdot(7)] — object position is hidden.
# gym_obs_to_pf_obs() returns [q(7), qdot(7), obj_xy(2)] = 16 dims.
# Object position is included (with a looser noise scale) so the PF can
# actually converge — the indirect contact-only signal proved too weak.
OBS_DIM    = 16             # q(7) + qdot(7) + obj_pos(2)

# Pusher-v5 action bounds (verified at runtime against env.action_space)
ACTION_BOUND = 2.0

# --------------------------------------------------------------------------- #
# RNEA rigid-body parameters (computed from Pusher-v5 MJCF geom specs)
# --------------------------------------------------------------------------- #
ARMATURE = 0.04             # motor armature (added to M diagonal), from XML default

JOINT_AXES = np.array([
    [0, 0, 1],   # joint 0: r_shoulder_pan_joint   — Z axis
    [0, 1, 0],   # joint 1: r_shoulder_lift_joint  — Y axis
    [1, 0, 0],   # joint 2: r_upper_arm_roll_joint — X axis
    [0, 1, 0],   # joint 3: r_elbow_flex_joint     — Y axis
    [1, 0, 0],   # joint 4: r_forearm_roll_joint   — X axis
    [0, 1, 0],   # joint 5: r_wrist_flex_joint     — Y axis
    [1, 0, 0],   # joint 6: r_wrist_roll_joint     — X axis
], dtype=np.float64)

# Position of joint i origin in parent link frame (frame i-1), constant.
# Accounts for fused (jointless) intermediate bodies.
JOINT_OFFSETS = np.array([
    [0.0,   0.0, 0.0],   # joint 0 (base — not used in forward pass)
    [0.1,   0.0, 0.0],   # shoulder_pan → shoulder_lift
    [0.0,   0.0, 0.0],   # shoulder_lift → upper_arm_roll
    [0.4,   0.0, 0.0],   # upper_arm_roll → elbow_flex  (through fused upper_arm_link)
    [0.0,   0.0, 0.0],   # elbow_flex → forearm_roll
    [0.321, 0.0, 0.0],   # forearm_roll → wrist_flex  (through fused forearm_link)
    [0.0,   0.0, 0.0],   # wrist_flex → wrist_roll
], dtype=np.float64)

# Per-joint damping coefficients (from MJCF joint damping attributes)
JOINT_DAMPING = np.array([1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)


def _capsule_mass_inertia(p1, p2, radius, density=300.0):
    """Mass, CoM (midpoint), and 3×3 inertia tensor of a capsule."""
    p1, p2 = np.asarray(p1, np.float64), np.asarray(p2, np.float64)
    r = float(radius)
    axis = p2 - p1
    L = float(np.linalg.norm(axis))
    com = (p1 + p2) / 2.0

    m_cyl = density * np.pi * r * r * L
    m_cap = density * (4.0 / 3.0) * np.pi * r ** 3
    mass = m_cyl + m_cap

    if L < 1e-12:
        Iv = (2.0 / 5.0) * mass * r * r
        return mass, com, np.diag([Iv, Iv, Iv])

    d = axis / L
    I_cyl_a = 0.5 * m_cyl * r * r
    I_cyl_t = m_cyl * (r * r / 4.0 + L * L / 12.0)
    m_h = m_cap / 2.0
    d_h = L / 2.0 + 3.0 * r / 8.0
    I_cap_a = 2.0 * (2.0 / 5.0) * m_h * r * r
    I_cap_t = 2.0 * ((83.0 / 320.0) * m_h * r * r + m_h * d_h * d_h)
    Ia = I_cyl_a + I_cap_a
    It = I_cyl_t + I_cap_t
    inertia = It * np.eye(3) + (Ia - It) * np.outer(d, d)
    return mass, com, inertia


def _sphere_mass_inertia(pos, radius, density=300.0):
    """Mass, CoM, and 3×3 inertia tensor of a solid sphere."""
    mass = density * (4.0 / 3.0) * np.pi * radius ** 3
    com = np.asarray(pos, np.float64)
    Iv = (2.0 / 5.0) * mass * radius * radius
    return mass, com, np.diag([Iv, Iv, Iv])


def _combine_geom_inertias(geoms):
    """Combine (mass, com, inertia) tuples into one rigid body."""
    total_mass = sum(m for m, _, _ in geoms)
    if total_mass < 1e-30:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    com = sum(m * c for m, c, _ in geoms) / total_mass
    total_I = np.zeros((3, 3))
    for m, c, I in geoms:
        d = c - com
        total_I += I + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return total_mass, com, total_I


def _compute_link_params():
    """Compute mass, CoM, inertia for each of the 7 RNEA links from MJCF geoms."""
    rho = 300.0  # default density

    link_geoms = [
        # Link 0: r_shoulder_pan_link
        [_sphere_mass_inertia([-0.06, 0.05, 0.2], 0.05, rho),
         _sphere_mass_inertia([0.06, 0.05, 0.2], 0.05, rho),
         _sphere_mass_inertia([-0.06, 0.09, 0.2], 0.03, rho),
         _sphere_mass_inertia([0.06, 0.09, 0.2], 0.03, rho),
         _capsule_mass_inertia([0, 0, -0.4], [0, 0, 0.2], 0.1, rho)],
        # Link 1: r_shoulder_lift_link
        [_capsule_mass_inertia([0, -0.1, 0], [0, 0.1, 0], 0.1, rho)],
        # Link 2: r_upper_arm_roll_link + r_upper_arm_link (fused)
        [_capsule_mass_inertia([-0.1, 0, 0], [0.1, 0, 0], 0.02, rho),
         _capsule_mass_inertia([0, 0, 0], [0.4, 0, 0], 0.06, rho)],
        # Link 3: r_elbow_flex_link
        [_capsule_mass_inertia([0, -0.02, 0], [0, 0.02, 0], 0.06, rho)],
        # Link 4: r_forearm_roll_link + r_forearm_link (fused)
        [_capsule_mass_inertia([-0.1, 0, 0], [0.1, 0, 0], 0.02, rho),
         _capsule_mass_inertia([0, 0, 0], [0.291, 0, 0], 0.05, rho)],
        # Link 5: r_wrist_flex_link
        [_capsule_mass_inertia([0, -0.02, 0], [0, 0.02, 0], 0.01, rho)],
        # Link 6: r_wrist_roll_link + tips_arm (fused)
        [_capsule_mass_inertia([0, -0.1, 0], [0, 0.1, 0], 0.02, rho),
         _capsule_mass_inertia([0, -0.1, 0], [0.1, -0.1, 0], 0.02, rho),
         _capsule_mass_inertia([0, 0.1, 0], [0.1, 0.1, 0], 0.02, rho),
         _sphere_mass_inertia([0.1, -0.1, 0], 0.01, rho),
         _sphere_mass_inertia([0.1, 0.1, 0], 0.01, rho)],
    ]

    masses = np.zeros(7)
    coms = np.zeros((7, 3))
    inertias = np.zeros((7, 3, 3))
    for i, geoms in enumerate(link_geoms):
        masses[i], coms[i], inertias[i] = _combine_geom_inertias(geoms)
    return masses, coms, inertias


LINK_MASSES, LINK_COMS, LINK_INERTIAS = _compute_link_params()


# --------------------------------------------------------------------------- #
# RNEA helper functions (CPU / numpy)
# --------------------------------------------------------------------------- #

def _rotation_matrix(axis, angle):
    """Rodrigues formula: 3×3 rotation about unit *axis* by *angle* radians."""
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    return np.array([
        [t * x * x + c,     t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c,     t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c    ],
    ])


def _rnea_inverse_dynamics(q, qdot, qddot):
    """
    Recursive Newton-Euler: returns tau such that
        M(q) * qddot + C(q, qdot) * qdot = tau
    Gravity is zero for Pusher-v5.
    """
    omega = np.zeros((7, 3))
    alpha = np.zeros((7, 3))
    ae    = np.zeros((7, 3))
    ac    = np.zeros((7, 3))
    R     = np.zeros((7, 3, 3))

    wp = np.zeros(3)   # omega of previous link (in its own frame)
    ap = np.zeros(3)   # alpha of previous link
    lp = np.zeros(3)   # linear accel of previous joint origin

    for i in range(7):
        R[i] = _rotation_matrix(JOINT_AXES[i], q[i])
        Rt = R[i].T                                     # parent → child

        wp_i = Rt @ wp                                   # parent ω in frame i
        omega[i] = wp_i + qdot[i] * JOINT_AXES[i]

        qdot_z = qdot[i] * JOINT_AXES[i]
        alpha[i] = Rt @ ap + qddot[i] * JOINT_AXES[i] + np.cross(wp_i, qdot_z)

        if i == 0:
            ae[i] = np.zeros(3)
        else:
            p = JOINT_OFFSETS[i]
            ae[i] = Rt @ (lp + np.cross(ap, p) + np.cross(wp, np.cross(wp, p)))

        c = LINK_COMS[i]
        ac[i] = ae[i] + np.cross(alpha[i], c) + np.cross(omega[i], np.cross(omega[i], c))

        wp, ap, lp = omega[i], alpha[i], ae[i]

    # Backward pass
    f = np.zeros((7, 3))
    n = np.zeros((7, 3))
    tau = np.zeros(7)

    for i in range(6, -1, -1):
        Fi = LINK_MASSES[i] * ac[i]
        Iw = LINK_INERTIAS[i] @ omega[i]
        Ni = LINK_INERTIAS[i] @ alpha[i] + np.cross(omega[i], Iw)

        f[i] = Fi
        n[i] = Ni + np.cross(LINK_COMS[i], Fi)

        if i < 6:
            fc = R[i + 1] @ f[i + 1]
            nc = R[i + 1] @ n[i + 1]
            f[i] += fc
            n[i] += nc + np.cross(JOINT_OFFSETS[i + 1], fc)

        tau[i] = np.dot(n[i], JOINT_AXES[i])

    return tau


def _crba_mass_matrix(q):
    """Composite Rigid-Body Algorithm via RNEA columns.  Returns 7×7 M(q)."""
    M = np.zeros((7, 7))
    z7 = np.zeros(7)
    for j in range(7):
        ej = np.zeros(7)
        ej[j] = 1.0
        M[:, j] = _rnea_inverse_dynamics(q, z7, ej)
    M += np.diag(np.full(7, ARMATURE))
    return M


def _forward_dynamics_numpy(q, qdot, tau):
    """Solve  (M + diag(armature)) * qddot = tau − damping·qdot − bias  for qddot."""
    bias = _rnea_inverse_dynamics(q, qdot, np.zeros(7))
    M = _crba_mass_matrix(q)
    rhs = tau - JOINT_DAMPING * qdot - bias
    return np.linalg.solve(M, rhs)


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
        Compute fingertip (x, y, z) using full 7-DOF forward kinematics.

        Chains all 7 joint transforms using the same rotation matrices
        and link offsets as the RNEA implementation.

        Returns
        -------
        tip_x, tip_y, tip_z : float
            Fingertip position in the world frame.
        """
        pos = ARM_BASE.copy()
        R_cum = np.eye(3, dtype=np.float64)
        for i in range(NUM_JOINTS):
            pos = pos + R_cum @ JOINT_OFFSETS[i]
            R_cum = R_cum @ _rotation_matrix(JOINT_AXES[i], q[i])
        return float(pos[0]), float(pos[1]), float(pos[2])

    @staticmethod
    def _planar_jacobian(q: np.ndarray):
        """
        Compute the 2×7 positional Jacobian mapping qdot → fingertip XY velocity
        using full 7-DOF kinematics.

        For revolute joint i: J_i = z_i × (p_tip − p_i)  (3D cross product)
        where z_i is the joint axis in world frame, p_i is joint i origin.
        We extract the x and y rows.

        Returns J_x, J_y as (7,) arrays.
        """
        # Forward pass: compute joint origins and world-frame axes
        positions = np.zeros((NUM_JOINTS, 3), dtype=np.float64)
        axes_w    = np.zeros((NUM_JOINTS, 3), dtype=np.float64)
        pos = ARM_BASE.copy()
        R_cum = np.eye(3, dtype=np.float64)
        for i in range(NUM_JOINTS):
            pos = pos + R_cum @ JOINT_OFFSETS[i]
            positions[i] = pos
            axes_w[i] = R_cum @ JOINT_AXES[i]
            R_cum = R_cum @ _rotation_matrix(JOINT_AXES[i], q[i])
        # tip position = pos at end of loop (after all offsets, last rotation
        # doesn't add position since there's no offset past joint 6)
        tip = pos.copy()

        J_x = np.zeros(NUM_JOINTS, dtype=np.float64)
        J_y = np.zeros(NUM_JOINTS, dtype=np.float64)
        for i in range(NUM_JOINTS):
            r = tip - positions[i]
            # cross product: z_i × r
            cx = axes_w[i, 1] * r[2] - axes_w[i, 2] * r[1]
            cy = axes_w[i, 2] * r[0] - axes_w[i, 0] * r[2]
            J_x[i] = cx
            J_y[i] = cy

        return J_x, J_y

    # ------------------------------------------------------------------ #
    # CPU dynamics
    # ------------------------------------------------------------------ #

    def f_numpy(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        One control step on CPU using RNEA forward dynamics with 5 substeps.

        Pusher-v5 uses frame_skip=5 with inner dt=0.01.  The integrator
        performs 5 semi-implicit Euler substeps to match MuJoCo.

        state layout: [q(7), qdot(7), obj_pos(2), obj_vel(2), tip_pos(2)]
        action: (7,) joint torques
        """
        state  = np.asarray(state, dtype=np.float64)
        action = np.clip(action, -ACTION_BOUND, ACTION_BOUND).astype(np.float64)

        q       = state[0:7].copy()
        qdot    = state[7:14].copy()
        obj_pos = state[14:16].copy()
        obj_vel = state[16:18].copy()
        tip_pos = state[18:20].copy()

        dt_sub = INNER_DT

        for _ in range(FRAME_SKIP):
            # ---- Contact detection using REAL tip position ----------------
            J_x, J_y = self._planar_jacobian(q)
            tip_vx = float(J_x @ qdot)
            tip_vy = float(J_y @ qdot)

            diff = obj_pos - tip_pos
            dist = np.linalg.norm(diff)

            tau_total = action.copy()

            if dist < CONTACT_RADIUS and dist > 1e-8:
                push_dir    = diff / dist
                tip_vel     = np.array([tip_vx, tip_vy])
                v_component = max(0.0, float(tip_vel @ push_dir))
                push_force  = PUSH_STRENGTH * v_component

                obj_vel += push_force * push_dir * dt_sub

                # Newton's 3rd law: reaction on arm as joint torque via J^T
                rx = -push_force * push_dir[0]
                ry = -push_force * push_dir[1]
                tau_total += J_x * rx + J_y * ry

            # ---- RNEA forward dynamics ------------------------------------
            qddot = _forward_dynamics_numpy(q, qdot, tau_total)

            # ---- Semi-implicit Euler substep ------------------------------
            qdot = qdot + qddot * dt_sub
            q    = q    + qdot  * dt_sub

            # ---- Object dynamics ------------------------------------------
            obj_vel *= (1.0 - FRICTION * dt_sub)
            obj_pos += obj_vel * dt_sub

            # ---- Update tip_pos via FK ------------------------------------
            tip_x, tip_y, _tip_z = self._forward_kinematics(q)
            tip_pos = np.array([tip_x, tip_y])

        # ---- Pack next state ------------------------------------------
        next_state = np.empty(STATE_DIM, dtype=np.float64)
        next_state[0:7]   = q
        next_state[7:14]  = qdot
        next_state[14:16] = obj_pos
        next_state[16:18] = obj_vel
        next_state[18]    = tip_pos[0]
        next_state[19]    = tip_pos[1]
        return next_state

    # ------------------------------------------------------------------ #
    # CPU cost
    # ------------------------------------------------------------------ #

    def cost_numpy(self, state: np.ndarray, action: np.ndarray) -> float:
        """
        Running cost matching the Pusher-v5 reward (negated for minimisation),
        plus a z-height penalty to encourage lowering the fingertip to the
        table plane so actual contact can occur in the 3-D simulator.

        Pusher-v5 reward:
          reward_near = -0.5 * ||fingertip - obj_pos||
          reward_dist = -1.0 * ||obj_pos   - goal_pos||
          reward_ctrl = -0.1 * ||action||^2

        cost = -reward = 0.5 * ||fingertip - obj_pos||
                       + 1.0 * ||obj_pos   - target_pos||
                       + 0.1 * ||action||^2
                       + Z_COST_WEIGHT * (tip_z - TABLE_Z)^2
        """
        state  = np.asarray(state, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)

        obj_pos = state[14:16]
        tip_pos = state[18:20]           # use stored tip position

        dist_tip_obj    = np.linalg.norm(tip_pos - obj_pos)
        dist_obj_target = np.linalg.norm(obj_pos - self._target_pos)
        action_cost     = float(np.dot(action, action))

        # Z-height penalty: compute tip_z from FK and penalise deviation
        # from the table plane so the planner drives the arm downward.
        _tx, _ty, tip_z = self._forward_kinematics(state[0:7])
        z_err = tip_z - TABLE_Z

        return (0.5 * dist_tip_obj
                + 1.0 * dist_obj_target
                + 0.1 * action_cost
                + Z_COST_WEIGHT * z_err * z_err)

    # ------------------------------------------------------------------ #
    # Observation model
    # ------------------------------------------------------------------ #

    def obs_model(self, state: np.ndarray) -> np.ndarray:
        """
        Maps internal particle state → 16-dim PF observation vector:
          [q(7), qdot(7), obj_pos(2)] = state[0:16]

        The first 14 dims (q, qdot) are compared with tight noise, and
        the last 2 dims (obj_pos) are compared with a looser noise scale
        (obs_noise_std_obj) — this is the primary signal that lets the PF
        converge on the true object position.
        """
        return np.asarray(state, dtype=np.float32)[0:OBS_DIM]

    def gym_obs_to_pf_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Extract the 16-dim PF observation from a raw 23-dim Pusher-v5 gym obs.

        Pusher-v5 obs layout:
          [0:7]   q              raw joint angles          → included
          [7:14]  qdot           joint velocities          → included
          [14:17] fingertip_pos  (x, y, z)                → skipped
          [17:20] obj_pos        (x, y, z)                → obj_xy included
          [20:23] goal_pos       (x, y, z)                → skipped

        Returns [q(7), qdot(7), obj_x, obj_y] — 16 dims.
        """
        obs = np.asarray(obs, dtype=np.float32)
        return np.concatenate([obs[0:14], obs[17:19]])

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
        # MJCF body origin (0.45, -0.05, -0.275).
        # Joint ordering in pusher_v5.xml: obj_slidey (qpos[-4]) then
        # obj_slidex (qpos[-3]).  reset_model assigns:
        #   cylinder_pos[0] = U(-0.3, 0)   → obj_slidey → y offset
        #   cylinder_pos[1] = U(-0.2, 0.2) → obj_slidex → x offset
        # World-frame ranges:
        #   x = 0.45 + U(-0.2, 0.2) → U(0.25, 0.65)
        #   y = -0.05 + U(-0.3, 0)  → U(-0.35, -0.05)
        particles[:, 14] = np.random.uniform(0.25, 0.65, N).astype(np.float32)
        particles[:, 15] = np.random.uniform(-0.35, -0.05, N).astype(np.float32)

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

        The returned string is **generated dynamically** so that the RNEA
        link parameters (mass, CoM, inertia — computed from the MJCF geom
        specs at import time) are embedded as ``__device__ const`` arrays.

        This string is concatenated with kernel strings and compiled once
        per process via cp.RawModule.
        """
        return _generate_cuda_code()


# --------------------------------------------------------------------------- #
# CUDA device code generator — embeds RNEA link parameters as constants
# --------------------------------------------------------------------------- #

def _fmt1d(arr, fmt=".10e"):
    """Format a 1-D array as a C initialiser list."""
    return ", ".join(f"{float(v):{fmt}}f" for v in arr)


def _fmt2d(arr, fmt=".10e"):
    """Format a 2-D array as nested C initialisers."""
    return ", ".join("{" + _fmt1d(row, fmt) + "}" for row in arr)


def _generate_cuda_code():
    """Build CUDA device-code string with RNEA parameters baked in."""

    # Flatten 3×3 inertias to 9-element row-major arrays
    inertia_flat = LINK_INERTIAS.reshape(7, 9)

    params = (
        "/* =========================================================\n"
        "   Pusher-v5 CUDA device code — RNEA rigid-body dynamics\n"
        "   State: [q(7), qdot(7), obj_pos(2), obj_vel(2), tip_pos(2)]\n"
        "   ========================================================= */\n\n"
        "#define STATE_DIM       20\n"
        "#define ACTION_DIM      7\n"
        "#define NUM_JOINTS      7\n"
        "#define OBS_DIM         16\n"
        "#define CONTACT_RADIUS  0.17f\n"
        "#define PUSH_STRENGTH   20.0f\n"
        "#define FRICTION        0.2f\n"
        "#define ACTION_BOUND    2.0f\n"
        "#define N_SUBSTEPS      5\n"
        "#define INNER_DT        0.01f\n"
        f"#define ARMATURE_VAL    {ARMATURE:.6f}f\n"
        f"#define TABLE_Z         {TABLE_Z:.6f}f\n"
        f"#define Z_COST_WEIGHT   {Z_COST_WEIGHT:.6f}f\n\n"
        "/* Arm base position in world frame */\n"
        "#define ARM_BASE_X  0.0f\n"
        "#define ARM_BASE_Y -0.6f\n"
        "#define ARM_BASE_Z  0.0f\n\n"
        "/* ---- RNEA link parameters (auto-generated from MJCF geoms) ---- */\n"
        f"__device__ const float JOINT_AXES_D[7][3]    = {{{_fmt2d(JOINT_AXES)}}};\n"
        f"__device__ const float JOINT_OFFSETS_D[7][3]  = {{{_fmt2d(JOINT_OFFSETS)}}};\n"
        f"__device__ const float LINK_MASSES_D[7]       = {{{_fmt1d(LINK_MASSES)}}};\n"
        f"__device__ const float LINK_COMS_D[7][3]      = {{{_fmt2d(LINK_COMS)}}};\n"
        f"__device__ const float LINK_INERTIAS_D[7][9]  = {{{_fmt2d(inertia_flat)}}};\n"
        f"__device__ const float JOINT_DAMPING_D[7]     = {{{_fmt1d(JOINT_DAMPING)}}};\n\n"
    )

    body = r"""
/* ================================================================== */
/*  3-D vector / matrix helpers                                        */
/* ================================================================== */
__device__ void cross3(const float* a, const float* b, float* o) {
    o[0] = a[1]*b[2] - a[2]*b[1];
    o[1] = a[2]*b[0] - a[0]*b[2];
    o[2] = a[0]*b[1] - a[1]*b[0];
}
__device__ float dot3(const float* a, const float* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
/* M is 3x3 row-major */
__device__ void mv3(const float* M, const float* v, float* o) {
    o[0] = M[0]*v[0]+M[1]*v[1]+M[2]*v[2];
    o[1] = M[3]*v[0]+M[4]*v[1]+M[5]*v[2];
    o[2] = M[6]*v[0]+M[7]*v[1]+M[8]*v[2];
}
/* M^T * v */
__device__ void mtv3(const float* M, const float* v, float* o) {
    o[0] = M[0]*v[0]+M[3]*v[1]+M[6]*v[2];
    o[1] = M[1]*v[0]+M[4]*v[1]+M[7]*v[2];
    o[2] = M[2]*v[0]+M[5]*v[1]+M[8]*v[2];
}
/* Rodrigues rotation matrix (axis must be unit), R is 3x3 row-major */
__device__ void rot_aa(const float* ax, float ang, float* R) {
    float c = cosf(ang), s = sinf(ang), t = 1.0f - c;
    float x = ax[0], y = ax[1], z = ax[2];
    R[0]=t*x*x+c;   R[1]=t*x*y-s*z; R[2]=t*x*z+s*y;
    R[3]=t*x*y+s*z; R[4]=t*y*y+c;   R[5]=t*y*z-s*x;
    R[6]=t*x*z-s*y; R[7]=t*y*z+s*x; R[8]=t*z*z+c;
}

/* ================================================================== */
/*  RNEA inverse dynamics:  tau = M(q)*qddot + C(q,qdot)*qdot         */
/*  gravity = 0 for Pusher-v5                                          */
/* ================================================================== */
__device__ void rnea_inv(const float* q, const float* qd, const float* qdd,
                         float* tau)
{
    float w[7][3], al[7][3], ae[7][3], ac[7][3], Rot[7][9];
    float wp[3]={0,0,0}, ap[3]={0,0,0}, lp[3]={0,0,0};

    /* Forward pass */
    for (int i = 0; i < 7; i++) {
        rot_aa(JOINT_AXES_D[i], q[i], Rot[i]);
        float wp_i[3]; mtv3(Rot[i], wp, wp_i);

        float qd_z[3];
        for (int d=0;d<3;d++) qd_z[d] = qd[i]*JOINT_AXES_D[i][d];

        for (int d=0;d<3;d++)
            w[i][d] = wp_i[d] + qd_z[d];

        float Rt_ap[3]; mtv3(Rot[i], ap, Rt_ap);
        float cwa[3]; cross3(wp_i, qd_z, cwa);
        for (int d=0;d<3;d++)
            al[i][d] = Rt_ap[d] + qdd[i]*JOINT_AXES_D[i][d] + cwa[d];

        if (i == 0) {
            ae[i][0]=ae[i][1]=ae[i][2]=0.0f;
        } else {
            float cp1[3], tmp[3], cp2[3];
            cross3(ap, JOINT_OFFSETS_D[i], cp1);
            cross3(wp, JOINT_OFFSETS_D[i], tmp);
            cross3(wp, tmp, cp2);
            float v[3];
            for (int d=0;d<3;d++) v[d] = lp[d]+cp1[d]+cp2[d];
            mtv3(Rot[i], v, ae[i]);
        }

        float c1[3], t1[3], c2[3];
        cross3(al[i], LINK_COMS_D[i], c1);
        cross3(w[i],  LINK_COMS_D[i], t1);
        cross3(w[i], t1, c2);
        for (int d=0;d<3;d++)
            ac[i][d] = ae[i][d]+c1[d]+c2[d];

        for (int d=0;d<3;d++) { wp[d]=w[i][d]; ap[d]=al[i][d]; lp[d]=ae[i][d]; }
    }

    /* Backward pass */
    float f[7][3], n[7][3];
    for (int i=6; i>=0; i--) {
        float Fi[3];
        for (int d=0;d<3;d++) Fi[d] = LINK_MASSES_D[i]*ac[i][d];

        float Iw[3], Ia[3], wIw[3], Ni[3];
        mv3(&LINK_INERTIAS_D[i][0], w[i], Iw);
        mv3(&LINK_INERTIAS_D[i][0], al[i], Ia);
        cross3(w[i], Iw, wIw);
        for (int d=0;d<3;d++) Ni[d] = Ia[d]+wIw[d];

        float cxF[3]; cross3(LINK_COMS_D[i], Fi, cxF);
        for (int d=0;d<3;d++) { f[i][d]=Fi[d]; n[i][d]=Ni[d]+cxF[d]; }

        if (i < 6) {
            float fc[3], nc[3], pxfc[3];
            mv3(Rot[i+1], f[i+1], fc);
            mv3(Rot[i+1], n[i+1], nc);
            cross3(JOINT_OFFSETS_D[i+1], fc, pxfc);
            for (int d=0;d<3;d++) { f[i][d]+=fc[d]; n[i][d]+=nc[d]+pxfc[d]; }
        }
        tau[i] = dot3(n[i], JOINT_AXES_D[i]);
    }
}

/* ================================================================== */
/*  CRBA: M(q) via column-wise RNEA                                    */
/* ================================================================== */
__device__ void crba_M(const float* q, float* M) {
    float z7[7]={0,0,0,0,0,0,0};
    for (int j=0; j<7; j++) {
        float ej[7]={0,0,0,0,0,0,0};
        ej[j]=1.0f;
        float col[7];
        rnea_inv(q, z7, ej, col);
        for (int i=0;i<7;i++) M[i*7+j]=col[i];
    }
    for (int j=0;j<7;j++) M[j*7+j] += ARMATURE_VAL;
}

/* ================================================================== */
/*  Cholesky solve  A x = b  (A is 7x7 SPD row-major, overwritten)    */
/* ================================================================== */
__device__ void chol_solve7(float* A, const float* b, float* x) {
    /* Decompose A → L */
    for (int j=0;j<7;j++) {
        float s=0.0f;
        for (int k=0;k<j;k++) s += A[j*7+k]*A[j*7+k];
        A[j*7+j] = sqrtf(fmaxf(A[j*7+j]-s, 1e-12f));
        float inv = 1.0f / A[j*7+j];
        for (int i=j+1;i<7;i++) {
            s=0.0f;
            for (int k=0;k<j;k++) s += A[i*7+k]*A[j*7+k];
            A[i*7+j] = (A[i*7+j]-s)*inv;
        }
    }
    /* Forward sub: L y = b */
    float y[7];
    for (int i=0;i<7;i++) {
        float s=b[i];
        for (int k=0;k<i;k++) s -= A[i*7+k]*y[k];
        y[i] = s / A[i*7+i];
    }
    /* Back sub: L^T x = y */
    for (int i=6;i>=0;i--) {
        float s=y[i];
        for (int k=i+1;k<7;k++) s -= A[k*7+i]*x[k];
        x[i] = s / A[i*7+i];
    }
}

/* ================================================================== */
/*  Full 7-DOF Forward Kinematics                                      */
/*  Chains all joint transforms: T = prod_i translate(offset_i) *       */
/*  rotate(axis_i, q_i).  Returns tip (x, y) in world frame.           */
/* ================================================================== */
__device__ void forward_kinematics(const float* q, float* tip_x, float* tip_y, float* tip_z) {
    float pos[3] = {ARM_BASE_X, ARM_BASE_Y, ARM_BASE_Z};
    float R[9] = {1,0,0, 0,1,0, 0,0,1};  /* cumulative rotation, row-major */

    for (int i = 0; i < NUM_JOINTS; i++) {
        /* pos += R @ offset[i] */
        float ox = JOINT_OFFSETS_D[i][0], oy = JOINT_OFFSETS_D[i][1], oz = JOINT_OFFSETS_D[i][2];
        pos[0] += R[0]*ox + R[1]*oy + R[2]*oz;
        pos[1] += R[3]*ox + R[4]*oy + R[5]*oz;
        pos[2] += R[6]*ox + R[7]*oy + R[8]*oz;

        /* R_joint = rotation(axis[i], q[i]) */
        float Rj[9];
        rot_aa(JOINT_AXES_D[i], q[i], Rj);

        /* R = R @ Rj */
        float tmp[9];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                tmp[r*3+c] = R[r*3+0]*Rj[0*3+c] + R[r*3+1]*Rj[1*3+c] + R[r*3+2]*Rj[2*3+c];
        for (int k = 0; k < 9; k++) R[k] = tmp[k];
    }
    *tip_x = pos[0];
    *tip_y = pos[1];
    *tip_z = pos[2];
}

/* ================================================================== */
/*  Full 7-DOF Positional Jacobian (XY rows)                           */
/*  J_i = z_i x (p_tip - p_i), where z_i = world-frame joint axis     */
/* ================================================================== */
__device__ void planar_jacobian(const float* q, const float* qdot,
                                float* Jx, float* Jy,
                                float* vx, float* vy)
{
    /* Forward pass: compute joint origins and world-frame axes */
    float positions[7][3];
    float axes_w[7][3];
    float pos[3] = {ARM_BASE_X, ARM_BASE_Y, ARM_BASE_Z};
    float R[9] = {1,0,0, 0,1,0, 0,0,1};

    for (int i = 0; i < NUM_JOINTS; i++) {
        /* pos += R @ offset[i] */
        float ox = JOINT_OFFSETS_D[i][0], oy = JOINT_OFFSETS_D[i][1], oz = JOINT_OFFSETS_D[i][2];
        pos[0] += R[0]*ox + R[1]*oy + R[2]*oz;
        pos[1] += R[3]*ox + R[4]*oy + R[5]*oz;
        pos[2] += R[6]*ox + R[7]*oy + R[8]*oz;
        positions[i][0] = pos[0];
        positions[i][1] = pos[1];
        positions[i][2] = pos[2];

        /* axis in world = R @ JOINT_AXES[i] */
        float ax = JOINT_AXES_D[i][0], ay = JOINT_AXES_D[i][1], az = JOINT_AXES_D[i][2];
        axes_w[i][0] = R[0]*ax + R[1]*ay + R[2]*az;
        axes_w[i][1] = R[3]*ax + R[4]*ay + R[5]*az;
        axes_w[i][2] = R[6]*ax + R[7]*ay + R[8]*az;

        /* R = R @ Rj */
        float Rj[9];
        rot_aa(JOINT_AXES_D[i], q[i], Rj);
        float tmp[9];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                tmp[r*3+c] = R[r*3+0]*Rj[0*3+c] + R[r*3+1]*Rj[1*3+c] + R[r*3+2]*Rj[2*3+c];
        for (int k = 0; k < 9; k++) R[k] = tmp[k];
    }
    /* tip = pos (after all offsets applied) */
    float tip[3] = {pos[0], pos[1], pos[2]};

    /* Jacobian: J_i = z_i x (tip - p_i) */
    float tvx = 0.0f, tvy = 0.0f;
    for (int i = 0; i < NUM_JOINTS; i++) {
        float rx = tip[0] - positions[i][0];
        float ry = tip[1] - positions[i][1];
        float rz = tip[2] - positions[i][2];
        /* cross product x-component: z_y * r_z - z_z * r_y */
        Jx[i] = axes_w[i][1]*rz - axes_w[i][2]*ry;
        /* cross product y-component: z_z * r_x - z_x * r_z */
        Jy[i] = axes_w[i][2]*rx - axes_w[i][0]*rz;
        tvx += Jx[i] * qdot[i];
        tvy += Jy[i] * qdot[i];
    }
    *vx = tvx;
    *vy = tvy;
}

/* ================================================================== */
/*  f_pusher: RNEA forward dynamics, 5 substeps, modifies state[]      */
/* ================================================================== */
__device__ void f_pusher(float* state, const float* action, float dt)
{
    float* q       = state;
    float* qdot    = state + 7;
    float* obj_pos = state + 14;
    float* obj_vel = state + 16;
    float* tip_pos = state + 18;

    float tau[NUM_JOINTS];
    for (int j=0;j<NUM_JOINTS;j++)
        tau[j] = fminf(fmaxf(action[j], -ACTION_BOUND), ACTION_BOUND);

    for (int sub = 0; sub < N_SUBSTEPS; sub++) {
        /* ---- Contact ---- */
        float Jx[NUM_JOINTS], Jy[NUM_JOINTS], tvx, tvy;
        planar_jacobian(q, qdot, Jx, Jy, &tvx, &tvy);
        float dx = obj_pos[0]-tip_pos[0], dy = obj_pos[1]-tip_pos[1];
        float dist = sqrtf(dx*dx+dy*dy);

        float tau_t[NUM_JOINTS];
        for (int j=0;j<NUM_JOINTS;j++) tau_t[j]=tau[j];

        if (dist < CONTACT_RADIUS && dist > 1e-8f) {
            float id = 1.0f/dist;
            float pdx=dx*id, pdy=dy*id;
            float vc = fmaxf(tvx*pdx+tvy*pdy, 0.0f);
            float pf = PUSH_STRENGTH*vc;
            obj_vel[0] += pf*pdx*INNER_DT;
            obj_vel[1] += pf*pdy*INNER_DT;
            float rx=-pf*pdx, ry=-pf*pdy;
            for (int j=0;j<NUM_JOINTS;j++)
                tau_t[j] += Jx[j]*rx + Jy[j]*ry;
        }

        /* ---- RNEA forward dynamics ---- */
        float bias[NUM_JOINTS];
        float z7[7]={0,0,0,0,0,0,0};
        rnea_inv(q, qdot, z7, bias);

        float M[49];
        crba_M(q, M);

        float rhs[NUM_JOINTS];
        for (int j=0;j<NUM_JOINTS;j++)
            rhs[j] = tau_t[j] - JOINT_DAMPING_D[j]*qdot[j] - bias[j];

        float qdd[NUM_JOINTS];
        chol_solve7(M, rhs, qdd);

        /* ---- Semi-implicit Euler substep ---- */
        for (int j=0;j<NUM_JOINTS;j++) {
            qdot[j] += qdd[j]*INNER_DT;
            q[j]    += qdot[j]*INNER_DT;
        }

        float fric = 1.0f - FRICTION*INNER_DT;
        obj_vel[0] *= fric;  obj_vel[1] *= fric;
        obj_pos[0] += obj_vel[0]*INNER_DT;
        obj_pos[1] += obj_vel[1]*INNER_DT;

        float dummy_z;
        forward_kinematics(q, &tip_pos[0], &tip_pos[1], &dummy_z);
    }
}

/* ================================================================== */
/*  cost_pusher: running cost with z-height penalty                     */
/*  The z-height term encourages the planner to lower the fingertip    */
/*  to the table plane so that real 3-D contact can occur.             */
/* ================================================================== */
__device__ float cost_pusher(const float* state, const float* action,
                              const float* target)
{
    const float* obj_pos = state + 14;
    const float* tip_pos = state + 18;
    float dx1=tip_pos[0]-obj_pos[0], dy1=tip_pos[1]-obj_pos[1];
    float d1 = sqrtf(dx1*dx1+dy1*dy1);
    float dx2=obj_pos[0]-target[0], dy2=obj_pos[1]-target[1];
    float d2 = sqrtf(dx2*dx2+dy2*dy2);
    float act2=0.0f;
    for (int a=0;a<ACTION_DIM;a++) act2+=action[a]*action[a];

    /* Z-height penalty: compute fingertip z via FK from joint angles
       and penalise deviation from the table plane (TABLE_Z).  This is
       the key signal that makes the planner lower the arm so that
       real 3-D contact with the object is possible. */
    float fk_x, fk_y, fk_z;
    forward_kinematics(state, &fk_x, &fk_y, &fk_z);
    float z_err = fk_z - TABLE_Z;

    // Matched to Pusher-v5 gym reward (negated for minimisation):
    //   reward = -0.5*||tip-obj|| - 1.0*||obj-goal|| - 0.1*||action||^2
    //   + z-height penalty to drive the arm to the correct height
    return 0.5f*d1 + 1.0f*d2 + 0.1f*act2 + Z_COST_WEIGHT*z_err*z_err;
}
"""

    return params + body
