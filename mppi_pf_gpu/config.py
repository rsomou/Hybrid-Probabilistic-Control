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
    H: int = 30                      # planning horizon
    lambda_: float = 1.0             # temperature parameter
    sigma: float = 0.5               # perturbation std

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
