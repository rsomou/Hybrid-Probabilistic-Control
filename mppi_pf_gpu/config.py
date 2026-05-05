"""
config.py
All hyperparameters and GPU settings for the MPPI + Particle Filter system.
Scheduler placeholder fields are included for future deadline-aware scheduling.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # Environment
    # ------------------------------------------------------------------ #
    env_name: str = "Pusher-v5"

    # ------------------------------------------------------------------ #
    # Particle Filter
    # ------------------------------------------------------------------ #
    N: int = 1000                        # number of particles
    process_noise_std: float = 0.0       # NO noise on joint dims during propagation — we inject
                                         # true q/qdot every step, so joint noise is pure contamination
                                         # that washes out the contact reaction signal
    process_noise_std_obj: float = 0.01  # std of process noise for object-state dims — allows particles to
                                         # explore ≈ 0.17m over 300 steps (0.01 * sqrt(300) ≈ 0.17m)
                                         # while staying tight enough that resampled particles don't drift far
    obs_noise_std: float = 0.01          # likelihood std for joint dims — tight so one-step contact signal discriminates
    obs_noise_std_obj: float = 0.05      # likelihood std for object position dims — looser than joint dims
                                         # to account for the PF's approximate dynamics model
    inject_noise_std: float = 0.001      # jitter added when injecting true q/qdot into particles
    resample_threshold: float = 0.5      # resample only when ESS < threshold * N

    # ------------------------------------------------------------------ #
    # MPPI
    # ------------------------------------------------------------------ #
    K: int = 1024                    # number of trajectory samples
    H: int = 20                      # planning horizon — 1s look-ahead.
                                     # Shorter = less model-reality divergence during contact.
                                     # The analytical contact model disagrees with MuJoCo's
                                     # constraint solver, so long rollouts accumulate error.
    lambda_: float = 200.0           # temperature — scaled to d² cost magnitude.
                                     # With GOAL_WEIGHT=100 and d²: total costs ~200-500.
                                     # λ=200 gives exp(-300/200)=0.22, w_eff ≈ 300-800.
    sigma: float = 0.8               # global perturbation scale — wider exploration to find
                                     # contact configurations after bouncing off the object.
    noise_beta: float = 0.5          # AR(1) temporal correlation — lower = faster recovery.
                                     # β=0.8 was too sticky: after losing contact the planner
                                     # kept replaying the old approach shape.  β=0.5 lets it
                                     # find new trajectories within 2-3 steps.
    action_alpha: float = 0.3        # EMA smoothing on output action.
                                     # Lower = faster reaction to plan changes after contact.
    N_CP: int = 6                    # number of spline control points per rollout trajectory.
                                     # Noise is sampled as (K, N_CP, 7) then linearly interpolated
                                     # to (K, H, 7) before rollout — reduces effective sample
                                     # dimension from H*7=350 to N_CP*7=42 and guarantees smooth
                                     # action sequences without relying on AR(1) alone.
    # Per-joint sigma multipliers for Pusher-v5 (7 joints):
    #   0=shoulder_pan, 1=shoulder_lift, 2=upper_arm_roll,
    #   3=elbow_flex,   4=forearm_roll,  5=wrist_flex, 6=wrist_roll
    # wrist_roll (6) is frozen: sigma=0 so no perturbation is ever sampled,
    # and u_bar[:,6] is pinned to 0 after every update in mppi.py.
    # Rationale: J_6=0 always (zero-offset joint), so action[6] cannot
    # transfer force to the object and is a pure waste of action budget.
    sigma_joint_weights: tuple = (1.5, 1.2, 1.0, 2.0, 0.8, 0.8, 0.0)

    elite_frac: float = 0.3          # fraction of lowest-cost rollouts used for u_bar update.
                                     # Weights for the other (1-elite_frac)*K trajectories are
                                     # zeroed before normalisation.  0.3 = top 30% only.
                                     # 1.0 = standard MPPI (all K used).

    # ------------------------------------------------------------------ #
    # Observation Delay
    # ------------------------------------------------------------------ #
    obs_delay: int = 3               # PF receives obs from d steps ago
    sensor_noise_std: float = 0.02   # std of additive sensor noise on observations

    # ------------------------------------------------------------------ #
    # Simulation
    # ------------------------------------------------------------------ #
    max_steps: int = 300
    dt: float = 0.05                 # integration timestep matching MuJoCo

    # ------------------------------------------------------------------ #
    # GPU
    # ------------------------------------------------------------------ #
    threads_per_block: int = 256
    device_id: int = 0

    # ------------------------------------------------------------------ #
    # Timing (for future deadline-aware scheduler)
    # ------------------------------------------------------------------ #
    deadline_ms: float = 50.0        # per-step deadline in milliseconds
    enable_timing: bool = True       # toggle CUDA event timing

    # ------------------------------------------------------------------ #
    # Future scheduler placeholders — do NOT implement logic here
    # These are stored for the adaptive scheduler to read and modify.
    # ------------------------------------------------------------------ #
    K_min: int = 64
    K_max: int = 4096
    safety_margin_ms: float = 2.0

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    def __post_init__(self):
        assert self.K_min <= self.K <= self.K_max, (
            f"K={self.K} must be in [{self.K_min}, {self.K_max}]"
        )
        assert self.H > 0, "Planning horizon H must be positive"
        assert self.N > 0, "Number of particles N must be positive"
        assert self.lambda_ > 0, "Temperature lambda_ must be positive"
        assert self.sigma > 0, "Perturbation std sigma must be positive"
