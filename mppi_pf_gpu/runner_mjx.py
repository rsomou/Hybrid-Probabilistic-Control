"""
runner_mjx.py
CPU-side orchestration loop for MJX-based MPPI + Particle Filter on Pusher-v5.

This replaces runner.py with JAX/MJX dynamics instead of CuPy/CUDA.
The control loop structure is identical — only the dynamics backend changes.

Until Step 3, the PF is not available (requires --no-pf).  After Step 3,
the PF will use JAX/MJX dynamics as well.
"""

import argparse
import time

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import jax
import mujoco

from config import Config
from dynamics_mjx import (
    MJXDynamics,
    apply_model_modifications,
    verify_parity,
    TABLE_Z,
    BEHIND_DIST,
)
from mppi_mjx import MPPI_MJX


# --------------------------------------------------------------------------- #
# Observation parsing helpers (same as runner.py)
# --------------------------------------------------------------------------- #

def _get_target(obs: np.ndarray) -> np.ndarray:
    """Extract 2-D goal position (x, y) from Pusher-v5 obs[20:22]."""
    return obs[20:22].astype(np.float32)


# --------------------------------------------------------------------------- #
# Main control loop
# --------------------------------------------------------------------------- #

def run(config: Config, render: bool = False, record: bool = False,
        no_pf: bool = False, skip_parity: bool = False):
    """
    Execute one episode of Pusher-v5 with MJX-based MPPI.

    Parameters
    ----------
    config : Config
    render : bool
    record : bool
    no_pf  : bool — if True, bypass PF and use perfect state (required until Step 3)
    skip_parity : bool — if True, skip the startup parity check

    Returns
    -------
    total_reward : float
    timing_log   : list[dict]
    """
    if not no_pf:
        print("ERROR: PF not yet available with MJX backend. Use --no-pf.")
        print("  (PF migration is Step 3; this is Step 2 — MPPI only.)")
        return 0.0, []

    # ---- Environment -------------------------------------------------------
    if record:
        render_mode = "rgb_array"
    elif render:
        render_mode = "human"
    else:
        render_mode = None

    env = gym.make(config.env_name, render_mode=render_mode,
                   max_episode_steps=config.max_steps)
    if record:
        env = RecordVideo(env, video_folder="./videos",
                          name_prefix="pusher_mppi_mjx",
                          episode_trigger=lambda _: True)

    # ---- Episode reset -----------------------------------------------------
    obs, _info = env.reset()
    obs = obs.astype(np.float32)

    # ---- Apply model modifications to BOTH env and planner ----------------
    model = env.unwrapped.model
    data = env.unwrapped.data
    apply_model_modifications(model)
    mujoco.mj_forward(model, data)

    # ---- MJX dynamics + MPPI -----------------------------------------------
    print("Initializing MJX dynamics...")
    mjx_dyn = MJXDynamics()  # loads & modifies its own copy of the MJCF
    mppi = MPPI_MJX(mjx_dyn, config)

    # ---- Startup parity check (10 steps) -----------------------------------
    if not skip_parity:
        print("\nRunning startup parity check...")
        max_err = verify_parity(mjx_dyn, env, n_steps=10)
        if max_err > 1e-2:
            print("FATAL: MJX parity check failed. Aborting.")
            env.close()
            return 0.0, []
        # Re-reset env since parity check modified it
        obs, _info = env.reset()
        obs = obs.astype(np.float32)
        apply_model_modifications(model)
        mujoco.mj_forward(model, data)

    # ---- Target + reset MPPI -----------------------------------------------
    target = _get_target(obs)
    mppi.set_target(target)
    mppi.reset()

    total_reward = 0.0
    timing_log = []

    # ---- Initial diagnostics -----------------------------------------------
    q0 = obs[0:7]
    obj0 = obs[17:20]
    fork0 = data.xpos[mjx_dyn.fork_body_id]
    print(f"\n{'='*60}")
    print(f"  INIT DIAG")
    print(f"  fork0  = ({fork0[0]:+.3f}, {fork0[1]:+.3f}, {fork0[2]:+.3f})")
    print(f"  obj0   = ({obj0[0]:+.3f}, {obj0[1]:+.3f}, {obj0[2]:+.3f})")
    print(f"  goal   = ({target[0]:+.3f}, {target[1]:+.3f})")
    print(f"{'='*60}\n")

    # ---- Control loop -------------------------------------------------------
    for t in range(config.max_steps):
        step_start = time.perf_counter()

        # ========================= PLANNING ================================
        # --no-pf mode: give MPPI the exact env state
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        action, mppi_timing = mppi.compute_action(qpos, qvel)

        t_plan_end = time.perf_counter()

        # ========================= ENV STEP ================================
        obs, reward, terminated, truncated, _info = env.step(action)
        obs = obs.astype(np.float32)

        t_env_end = time.perf_counter()
        # ===================================================================

        total_reward += reward

        # ---- Timing record ------------------------------------------------
        T_plan_ms = (t_plan_end - step_start) * 1e3
        T_env_ms = (t_env_end - t_plan_end) * 1e3
        T_total_ms = (t_env_end - step_start) * 1e3

        timing_entry = {
            "step": t,
            "T_total_ms": T_total_ms,
            "T_gpu_ms": T_plan_ms,  # "gpu" = planning time (JAX)
            "T_env_ms": T_env_ms,
            "ESS": float(config.N),  # no PF → sentinel
            "K_used": mppi._K_active,
            "reward": float(reward),
            "deadline_ms": config.deadline_ms,
            "safety_margin_ms": config.safety_margin_ms,
        }
        timing_log.append(timing_entry)

        if config.enable_timing:
            print(
                f"Step {t:4d} | "
                f"R={reward:7.3f} | "
                f"T={T_total_ms:6.2f}ms (plan={T_plan_ms:5.2f} env={T_env_ms:5.2f}) | "
                f"K={mppi.K}"
            )

        # ---- Diagnostic every 10 steps ------------------------------------
        if t % 10 == 0:
            q_now = obs[0:7]
            real_obj = obs[17:20]
            fork_xyz = data.xpos[mjx_dyn.fork_body_id]
            fork_obj_3d = float(np.linalg.norm(
                fork_xyz - np.array([real_obj[0], real_obj[1], TABLE_Z])))
            print(
                f"  DIAG step {t}: "
                f"fork=({fork_xyz[0]:+.3f},{fork_xyz[1]:+.3f},{fork_xyz[2]:+.3f}) "
                f"obj=({real_obj[0]:+.3f},{real_obj[1]:+.3f}) "
                f"fork→obj_3d={fork_obj_3d:.3f}m"
            )

        if terminated or truncated:
            break

    env.close()

    # ---- Summary -----------------------------------------------------------
    n_steps = len(timing_log)
    avg_total_ms = float(np.mean([r["T_total_ms"] for r in timing_log]))
    avg_plan_ms = float(np.mean([r["T_gpu_ms"] for r in timing_log]))
    avg_env_ms = float(np.mean([r["T_env_ms"] for r in timing_log]))
    deadline_hits = sum(
        1 for r in timing_log if r["T_total_ms"] <= config.deadline_ms
    )

    print(f"\n{'='*60}")
    print(f"  Total reward  : {total_reward:.3f}")
    print(f"  Steps         : {n_steps}")
    print(f"  Avg step time : {avg_total_ms:.2f} ms  "
          f"(plan={avg_plan_ms:.2f}  env={avg_env_ms:.2f})")
    print(f"  Deadline hits : {deadline_hits}/{n_steps} "
          f"({100*deadline_hits/max(n_steps,1):.1f}%  <= {config.deadline_ms:.0f} ms)")
    print(f"{'='*60}")

    np.save("timing_log.npy", timing_log)
    print("Timing log saved → timing_log.npy")

    return total_reward, timing_log


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MPPI (MJX backend) on Pusher-v5"
    )
    parser.add_argument("--K", type=int, default=1024,
                        help="MPPI trajectory samples")
    parser.add_argument("--N", type=int, default=1000,
                        help="Particle filter particles (unused with --no-pf)")
    parser.add_argument("--H", type=int, default=20,
                        help="MPPI planning horizon")
    parser.add_argument("--deadline", type=float, default=50.0,
                        help="Per-step deadline in ms")
    parser.add_argument("--sigma", type=float, default=0.8,
                        help="MPPI perturbation scale")
    parser.add_argument("--lambda_", type=float, default=200.0,
                        help="MPPI temperature")
    parser.add_argument("--steps", type=int, default=300,
                        help="Max episode steps")
    parser.add_argument("--render", action="store_true",
                        help="Open MuJoCo viewer")
    parser.add_argument("--record", action="store_true",
                        help="Record video")
    parser.add_argument("--no-timing", action="store_true",
                        help="Suppress per-step timing")
    parser.add_argument("--no-pf", action="store_true",
                        help="Bypass PF; perfect state (required until PF migrated)")
    parser.add_argument("--skip-parity", action="store_true",
                        help="Skip startup parity check")
    args = parser.parse_args()

    cfg = Config(
        K=args.K,
        N=args.N,
        H=args.H,
        deadline_ms=args.deadline,
        sigma=args.sigma,
        lambda_=args.lambda_,
        max_steps=args.steps,
        enable_timing=not args.no_timing,
    )

    run(cfg, render=args.render, record=args.record,
        no_pf=args.no_pf, skip_parity=args.skip_parity)
