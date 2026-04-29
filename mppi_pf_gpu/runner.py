"""
runner.py
CPU-side orchestration loop for MPPI + Particle Filter on Pusher-v5.

Responsibilities
----------------
1. Construct and connect all components (env, dynamics, GPU, PF, MPPI).
2. Run the per-step control loop.
3. Record fine-grained timing data (T_gpu_ms, T_env_ms, ESS, K_used).
4. Save the timing log for offline analysis / future scheduler design.

Timing design
-------------
  t0  ──── GPU work begins ─────────────────────── t1
  t1  ──── env.step() ──────────────────────── t2
  t2  ──── pf.propagate() (prior update) ──── t3

  T_gpu_ms = (t1 - t0) * 1e3   (includes explicit Device.synchronize())
  T_env_ms = (t2 - t1) * 1e3
  T_total  = (t3 - t0) * 1e3

This matches the data format the future deadline-aware scheduler expects.

CPU-GPU bus crossings per step
------------------------------
  CPU → GPU : pf_obs (16 floats: q, qdot, obj_xy)  — in pf.update()
  CPU → GPU : action         (7 floats)    — in pf.propagate()
  GPU → CPU : u_bar[0]       (7 floats)    — from mppi.compute_action()
  GPU → CPU : ESS            (1 float)     — from pf.effective_sample_size()
"""

import argparse
import time

import gymnasium as gym
import numpy as np
import cupy as cp

from config import Config
from envs.pusher import PusherDynamics
from gpu_utils import GPUUtils
from particle_filter import ParticleFilter
from mppi import MPPI


# --------------------------------------------------------------------------- #
# Observation parsing helpers
# --------------------------------------------------------------------------- #

def _get_target(obs: np.ndarray) -> np.ndarray:
    """
    Extract the 2-D goal position (x, y) from a Pusher-v5 observation.

    Pusher-v5 obs[20:23] is always the goal position (x, y, z).
    The goal is fixed at [0.45, -0.05, -0.323] for all episodes.
    We take obs[20:22] (x, y only; z is not needed by the planner).
    """
    return obs[20:22].astype(np.float32)


# --------------------------------------------------------------------------- #
# Main control loop
# --------------------------------------------------------------------------- #

def run(config: Config, render: bool = False):
    """
    Execute one episode of Pusher-v4 with MPPI + Particle Filter.

    Parameters
    ----------
    config : Config
    render : bool — if True opens the MuJoCo viewer

    Returns
    -------
    total_reward : float
    timing_log   : list[dict]  — one entry per step
    """
    # ---- Environment -------------------------------------------------------
    render_mode = "human" if render else None
    # Pass max_episode_steps to override the default 100-step truncation.
    env = gym.make(config.env_name, render_mode=render_mode,
                   max_episode_steps=config.max_steps)

    # ---- Components --------------------------------------------------------
    dynamics = PusherDynamics(dt=config.dt)
    gpu      = GPUUtils(config)
    pf       = ParticleFilter(dynamics, config, gpu)
    mppi     = MPPI(dynamics, config, gpu)

    # ---- Episode reset -----------------------------------------------------
    obs, _info = env.reset()
    obs        = obs.astype(np.float32)

    # Goal is fixed — read directly from obs[20:22] (always correct in v5)
    target = _get_target(obs)
    dynamics.set_target(target)
    mppi.set_target(target)
    pf.initialize(obs)

    total_reward = 0.0
    timing_log   = []

    # ---- Control loop ------------------------------------------------------
    for t in range(config.max_steps):
        step_start = time.perf_counter()

        # ========================= GPU WORK =================================
        # 1. Weight update: multiply particle weights by obs likelihood
        pf.update(obs)

        # 2. ESS must be computed BEFORE resample() — resample resets all
        #    weights to 1/N, which would always give ESS == N and make the
        #    metric useless for the future scheduler.
        ess = pf.effective_sample_size()

        # 3. Adaptive resampling: only resample when diversity falls below
        #    the threshold.  Always resampling every step destroys diversity
        #    once the cloud collapses — adaptive resampling lets the PF
        #    recover naturally via the process noise.
        if ess < config.resample_threshold * config.N:
            pf.resample()

        # 4. Sample K initial states from belief for MPPI rollouts
        initial_states = pf.sample(mppi.K)          # (K, state_dim) on GPU

        # 5. MPPI planning step — returns next action on CPU
        action, mppi_timing = mppi.compute_action(initial_states)

        # Flush all pending GPU kernels before timestamping
        cp.cuda.Device(config.device_id).synchronize()
        # ========================= END GPU WORK =============================
        t_gpu_end = time.perf_counter()

        # ========================= CPU / ENV WORK ===========================
        obs, reward, terminated, truncated, _info = env.step(action)
        obs = obs.astype(np.float32)
        t_env_end = time.perf_counter()

        # 6. Propagate particles forward with the action just applied
        #    (prior update — runs asynchronously before next GPU sync)
        pf.propagate(action)
        # ========================= END CPU / ENV WORK =======================

        total_reward += reward

        # ---- Timing record -------------------------------------------------
        T_gpu_ms   = (t_gpu_end - step_start) * 1e3
        T_env_ms   = (t_env_end - t_gpu_end)  * 1e3
        T_total_ms = (t_env_end - step_start) * 1e3

        record = {
            "step":       t,
            "T_total_ms": T_total_ms,
            "T_gpu_ms":   T_gpu_ms,
            "T_env_ms":   T_env_ms,
            "ESS":        ess,
            "K_used":     mppi._K_active,  # reflects any per-step K override
            "reward":     float(reward),
            # Scheduler placeholders — future scheduler fills these:
            "deadline_ms":      config.deadline_ms,
            "safety_margin_ms": config.safety_margin_ms,
        }
        timing_log.append(record)

        if config.enable_timing:
            print(
                f"Step {t:4d} | "
                f"R={reward:7.3f} | "
                f"T={T_total_ms:6.2f}ms (GPU={T_gpu_ms:5.2f} ENV={T_env_ms:5.2f}) | "
                f"ESS={ess:6.0f}/{config.N} | "
                f"K={mppi.K}"
            )

        if terminated or truncated:
            break

    env.close()

    # ---- Summary -----------------------------------------------------------
    n_steps       = len(timing_log)
    avg_total_ms  = float(np.mean([r["T_total_ms"] for r in timing_log]))
    avg_gpu_ms    = float(np.mean([r["T_gpu_ms"]   for r in timing_log]))
    avg_env_ms    = float(np.mean([r["T_env_ms"]   for r in timing_log]))
    deadline_hits = sum(
        1 for r in timing_log if r["T_total_ms"] <= config.deadline_ms
    )

    print(f"\n{'='*60}")
    print(f"  Total reward  : {total_reward:.3f}")
    print(f"  Steps         : {n_steps}")
    print(f"  Avg step time : {avg_total_ms:.2f} ms  "
          f"(GPU={avg_gpu_ms:.2f}  ENV={avg_env_ms:.2f})")
    print(f"  Deadline hits : {deadline_hits}/{n_steps} "
          f"({100*deadline_hits/max(n_steps,1):.1f}%  <= {config.deadline_ms:.0f} ms)")
    print(f"{'='*60}")

    # ---- Save timing log ---------------------------------------------------
    np.save("timing_log.npy", timing_log)
    print("Timing log saved → timing_log.npy")

    return total_reward, timing_log


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MPPI + Particle Filter on Pusher-v5"
    )
    parser.add_argument("--K",        type=int,   default=1024,
                        help="MPPI trajectory samples")
    parser.add_argument("--N",        type=int,   default=1000,
                        help="Particle filter particles")
    parser.add_argument("--H",        type=int,   default=30,
                        help="MPPI planning horizon")
    parser.add_argument("--deadline", type=float, default=50.0,
                        help="Per-step deadline in ms")
    parser.add_argument("--sigma",    type=float, default=0.5,
                        help="MPPI perturbation std")
    parser.add_argument("--lambda_",  type=float, default=1.0,
                        help="MPPI temperature")
    parser.add_argument("--steps",    type=int,   default=300,
                        help="Max episode steps")
    parser.add_argument("--device",   type=int,   default=0,
                        help="CUDA device ID")
    parser.add_argument("--render",   action="store_true",
                        help="Open MuJoCo viewer")
    parser.add_argument("--no-timing", action="store_true",
                        help="Suppress per-step timing output")
    args = parser.parse_args()

    cfg = Config(
        K              = args.K,
        N              = args.N,
        H              = args.H,
        deadline_ms    = args.deadline,
        sigma          = args.sigma,
        lambda_        = args.lambda_,
        max_steps      = args.steps,
        device_id      = args.device,
        enable_timing  = not args.no_timing,
    )

    run(cfg, render=args.render)
