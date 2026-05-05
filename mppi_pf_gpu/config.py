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
    H: int = 50                      # planning horizon — longer = sees full approach+push sequence
                                     # (dt=0.05 * H=50 = 2.5s look-ahead)
    lambda_: float = 200.0           # temperature — must be proportional to cost magnitude.
                                     # With H=50 steps and per-step cost ~10-15, total costs are
                                     # ~500-900 (std ~200-400).  λ=0.5 made exp(-range/λ) = 0 for
                                     # every trajectory except the single best, giving w_eff=1.
                                     # λ=200 gives exp(-400/200)=0.13, so ~100-500 trajectories
                                     # contribute to the weighted update.
    sigma: float = 0.5               # global perturbation scale — reduced from 0.8 to keep
                                     # rollout noise tighter around the nominal trajectory
    noise_beta: float = 0.8          # temporal correlation for AR(1) applied in control-point space.
                                     # eps_cp[t] = beta*eps_cp[t-1] + sqrt(1-beta^2)*white[t]
                                     # The spline interpolation then smooths CPs → H steps automatically.
    action_alpha: float = 0.5        # EMA smoothing on output action.
                                     # a_out = alpha*a_prev + (1-alpha)*a_mppi
                                     # 0 = no smoothing, 1 = frozen
                                     # 0.5 = equal blend, halves step-to-step variation
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
