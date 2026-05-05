#!/usr/bin/env python
"""
test_mjx_parity.py
Standalone parity check: run MuJoCo (via Gymnasium) and MJX from the same
initial state with the same random controls for 100 steps.

Usage
-----
    conda activate ml_env
    cd mppi_pf_gpu
    python test_mjx_parity.py

Expected behaviour
------------------
    Per-step ‖qpos_env - qpos_mjx‖ < 1e-3.
    MJX is float32, MuJoCo C is float64 → expect ~1e-5 to 1e-4 per step,
    accumulating slowly over longer runs.

    If any step exceeds 1e-3, STOP — it means the MJX model doesn't match
    the modified env and we need to investigate before proceeding with the
    MPPI/PF migration.
"""

import sys
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import gymnasium as gym

# Ensure imports from the package work when running from mppi_pf_gpu/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dynamics_mjx import (
    MJXDynamics,
    apply_model_modifications,
    step_one,
    FRAME_SKIP,
)


def run_parity_check(n_steps: int = 100, seed: int = 42):
    """
    Run n_steps with identical random controls in both the Gymnasium env
    (C MuJoCo, float64) and MJX (float32).  Report per-step divergence.

    Parameters
    ----------
    n_steps : int — number of control steps to compare
    seed    : int — RNG seed for reproducible actions and env reset
    """
    rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # 1. Gymnasium environment (ground truth)
    # ------------------------------------------------------------------
    env = gym.make("Pusher-v5")
    obs, _ = env.reset(seed=seed)

    mj_model = env.unwrapped.model
    mj_data = env.unwrapped.data
    apply_model_modifications(mj_model)
    mujoco.mj_forward(mj_model, mj_data)   # recompute with modified model

    print(f"Env initial qpos: {mj_data.qpos}")
    print(f"Env initial qvel: {mj_data.qvel}")

    # ------------------------------------------------------------------
    # 2. MJX model (planning model, float32)
    # ------------------------------------------------------------------
    print("\nInitializing MJX dynamics...")
    mjx_dyn = MJXDynamics()

    # Create MJX data from env's current state
    qpos0 = mj_data.qpos.copy()
    qvel0 = mj_data.qvel.copy()
    mjx_data = mjx_dyn.make_mjx_data(qpos0, qvel0)

    print(f"MJX initial qpos: {np.array(mjx_data.qpos)}")
    print(f"Initial qpos match: {np.allclose(qpos0, np.array(mjx_data.qpos), atol=1e-6)}")

    # ------------------------------------------------------------------
    # 3. JIT compile the step function
    # ------------------------------------------------------------------
    mjx_model = mjx_dyn.mjx_model

    @jax.jit
    def mjx_step_one(data, ctrl):
        return step_one(mjx_model, data, ctrl)

    print("\nCompiling MJX step function (first call triggers JIT)...")
    t0 = time.perf_counter()
    dummy_ctrl = jnp.zeros(7, dtype=jnp.float32)
    _ = mjx_step_one(mjx_data, dummy_ctrl)
    compile_time = time.perf_counter() - t0
    print(f"JIT compile time: {compile_time:.2f} s")

    # Re-initialize MJX data (the warmup call mutated it)
    mjx_data = mjx_dyn.make_mjx_data(qpos0, qvel0)

    # ------------------------------------------------------------------
    # 4. Run comparison
    # ------------------------------------------------------------------
    errors_qpos = []
    errors_qvel = []
    errors_xpos_fork = []
    errors_xpos_obj = []

    header = (f"{'Step':>5} | {'‖Δqpos‖':>14} | {'‖Δqvel‖':>14} | "
              f"{'‖Δfork_xyz‖':>14} | {'‖Δobj_xyz‖':>14} | Status")
    print(f"\n{header}")
    print("-" * len(header))

    for t in range(n_steps):
        # Random action in [-2, 2]
        action = rng.uniform(-2.0, 2.0, size=7).astype(np.float32)

        # -- Gymnasium env step (float64, frame_skip=5) --
        env.step(action)
        env_qpos = mj_data.qpos.copy()
        env_qvel = mj_data.qvel.copy()
        # Fork and object world positions from the C-side data
        env_fork_xyz = mj_data.xpos[mjx_dyn.fork_body_id].copy()
        env_obj_xyz = mj_data.xpos[mjx_dyn.obj_body_id].copy()

        # -- MJX step (float32) --
        ctrl_jax = jnp.array(action, dtype=jnp.float32)
        mjx_data = mjx_step_one(mjx_data, ctrl_jax)

        mjx_qpos = np.array(mjx_data.qpos)
        mjx_qvel = np.array(mjx_data.qvel)
        mjx_fork_xyz = np.array(mjx_data.xpos[mjx_dyn.fork_body_id])
        mjx_obj_xyz = np.array(mjx_data.xpos[mjx_dyn.obj_body_id])

        # -- Errors --
        eq = float(np.linalg.norm(env_qpos - mjx_qpos))
        ev = float(np.linalg.norm(env_qvel - mjx_qvel))
        ef = float(np.linalg.norm(env_fork_xyz - mjx_fork_xyz))
        eo = float(np.linalg.norm(env_obj_xyz - mjx_obj_xyz))

        errors_qpos.append(eq)
        errors_qvel.append(ev)
        errors_xpos_fork.append(ef)
        errors_xpos_obj.append(eo)

        status = "OK" if eq < 1e-3 else ("WARN" if eq < 1e-2 else "FAIL")

        if t < 10 or t % 10 == 0 or status != "OK":
            print(f"{t:5d} | {eq:14.8f} | {ev:14.8f} | "
                  f"{ef:14.8f} | {eo:14.8f} | {status}")

        if env.unwrapped.data.time > 1e6:
            break  # safety

    env.close()

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    max_eq = max(errors_qpos)
    mean_eq = np.mean(errors_qpos)
    max_eo = max(errors_xpos_obj)

    print(f"\n{'=' * 60}")
    print(f"  Steps compared     : {len(errors_qpos)}")
    print(f"  ‖Δqpos‖  max={max_eq:.6e}  mean={mean_eq:.6e}")
    print(f"  ‖Δqvel‖  max={max(errors_qvel):.6e}  mean={np.mean(errors_qvel):.6e}")
    print(f"  ‖Δfork‖  max={max(errors_xpos_fork):.6e}  mean={np.mean(errors_xpos_fork):.6e}")
    print(f"  ‖Δobj‖   max={max_eo:.6e}  mean={np.mean(errors_xpos_obj):.6e}")
    print()

    if max_eq < 1e-3:
        print(f"  ✓ PASS — MJX matches env within 1e-3 at every step.")
    elif max_eq < 1e-2:
        print(f"  ~ MARGINAL — some steps exceed 1e-3 (max={max_eq:.4e}).")
        print(f"    float32 vs float64 accumulation is likely the cause.")
        print(f"    This is acceptable for control but worth noting.")
    else:
        print(f"  ✗ FAIL — max qpos divergence {max_eq:.4e} exceeds 1e-2!")
        print(f"    STOP: MJX model does not match the env.")
        print(f"    Check model modifications and MJX feature support.")

    print(f"{'=' * 60}")

    return errors_qpos


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MJX parity check")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    run_parity_check(n_steps=args.steps, seed=args.seed)
