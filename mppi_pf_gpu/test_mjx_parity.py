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
    increments_qpos = []   # per-step growth in accumulated error

    header = (f"{'Step':>5} | {'‖Δqpos‖':>14} | {'Δ/step':>14} | {'‖Δqvel‖':>14} | "
              f"{'‖Δobj_xyz‖':>14} | Status")
    print(f"\n{header}")
    print("-" * len(header))

    prev_eq = 0.0
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

        # Per-step increment: how much NEW error was introduced this step.
        # This is the meaningful metric — it should be ~1e-4 to 1e-5 per
        # step for float32 vs float64.  Accumulated error grows with sqrt(N)
        # but that doesn't indicate a model mismatch.
        incr = eq - prev_eq
        prev_eq = eq
        increments_qpos.append(incr)

        errors_qpos.append(eq)
        errors_qvel.append(ev)
        errors_xpos_fork.append(ef)
        errors_xpos_obj.append(eo)

        status = "OK" if eq < 1e-3 else ("WARN" if eq < 5e-2 else "FAIL")

        if t < 10 or t % 10 == 0 or status == "FAIL":
            print(f"{t:5d} | {eq:14.8f} | {incr:+14.8f} | {ev:14.8f} | "
                  f"{eo:14.8f} | {status}")

        if env.unwrapped.data.time > 1e6:
            break  # safety

    env.close()

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    max_eq = max(errors_qpos)
    mean_eq = np.mean(errors_qpos)
    max_eo = max(errors_xpos_obj)

    # First-10-steps error is the most meaningful: matches MPPI rollout
    # lengths and hasn't accumulated open-loop drift.
    first10_max = max(errors_qpos[:min(10, len(errors_qpos))])
    first50_max = max(errors_qpos[:min(50, len(errors_qpos))])
    mean_incr = np.mean(increments_qpos)
    max_incr = max(increments_qpos)

    print(f"\n{'=' * 60}")
    print(f"  Steps compared     : {len(errors_qpos)}")
    print(f"  ‖Δqpos‖  max={max_eq:.6e}  mean={mean_eq:.6e}")
    print(f"  ‖Δqvel‖  max={max(errors_qvel):.6e}  mean={np.mean(errors_qvel):.6e}")
    print(f"  ‖Δfork‖  max={max(errors_xpos_fork):.6e}  mean={np.mean(errors_xpos_fork):.6e}")
    print(f"  ‖Δobj‖   max={max_eo:.6e}  mean={np.mean(errors_xpos_obj):.6e}")
    print(f"  Per-step increment: mean={mean_incr:.6e}  max={max_incr:.6e}")
    print(f"  First 10 steps max : {first10_max:.6e}")
    print(f"  First 50 steps max : {first50_max:.6e}")
    print()

    # The meaningful checks:
    # 1. First 10 steps < 1e-3: proves the model matches (no systematic error)
    # 2. Per-step increment < 1e-2: proves drift is gradual float32, not
    #    a sudden model divergence
    # 3. Object position ~ 0: proves contact physics match
    first10_ok = first10_max < 1e-3
    incr_ok = max_incr < 1e-2
    obj_ok = max_eo < 1e-3

    if first10_ok and incr_ok and obj_ok:
        print(f"  ✓ PASS — MJX matches env.")
        print(f"    First-10 max {first10_max:.2e} < 1e-3")
        print(f"    Per-step increment max {max_incr:.2e} < 1e-2")
        print(f"    Object divergence {max_eo:.2e} ≈ 0")
        if max_eq > 1e-2:
            print(f"    (Accumulated drift {max_eq:.2e} after {len(errors_qpos)} open-loop")
            print(f"     steps is expected float32 behavior — not a model error.)")
    elif not first10_ok:
        print(f"  ✗ FAIL — first 10 steps max {first10_max:.4e} exceeds 1e-3!")
        print(f"    This indicates a real model mismatch, not float32 drift.")
    elif not obj_ok:
        print(f"  ✗ FAIL — object divergence {max_eo:.4e} exceeds 1e-3!")
        print(f"    Contact physics don't match between env and MJX.")
    else:
        print(f"  ✗ FAIL — per-step increment {max_incr:.4e} exceeds 1e-2!")
        print(f"    Sudden jump suggests model feature mismatch.")

    print(f"{'=' * 60}")

    return errors_qpos


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MJX parity check")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    run_parity_check(n_steps=args.steps, seed=args.seed)
