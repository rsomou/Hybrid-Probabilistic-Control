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
  t0  ──── GPU work begins (inject + propagate + update + MPPI) ──── t1
  t1  ──── env.step() ──────────────────────────────────────── t2

  T_gpu_ms = (t1 - t0) * 1e3   (includes explicit Device.synchronize())
  T_env_ms = (t2 - t1) * 1e3
  T_total  = (t2 - t0) * 1e3

This matches the data format the future deadline-aware scheduler expects.

CPU-GPU bus crossings per step
------------------------------
  CPU → GPU : pf_obs (14 floats: q, qdot)  — in pf.update()  (obj_pos hidden)
  CPU → GPU : action         (7 floats)    — in pf.propagate()
  GPU → CPU : u_bar[0]       (7 floats)    — from mppi.compute_action()
  GPU → CPU : ESS            (1 float)     — from pf.effective_sample_size()
"""

import argparse
import time
from collections import deque

import gymnasium as gym
import numpy as np
import cupy as cp

from config import Config
from envs.pusher import PusherDynamics, CONTACT_RADIUS
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
    Execute one episode of Pusher-v5 with MPPI + Particle Filter.

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

    # ---- Observation-delay buffers -----------------------------------------
    # obs_buffer:    maxlen = d+1  (stores delayed obs window)
    # action_buffer: maxlen = d+1  (oldest entry = action for PF propagation,
    #                               remaining d entries = actions for sample_current)
    obs_buffer    = deque(maxlen=config.obs_delay + 1)
    action_buffer = deque(maxlen=config.obs_delay + 1)
    # Seed with the initial observation (no noise) so delay doesn't starve
    obs_buffer.append(obs.copy())
    prev_delayed_obs = obs.copy()   # safe initialisation for first inject

    # ---- Initial diagnostics -----------------------------------------------
    print(f"\n{'='*60}")
    print(f"  INIT DIAG")
    q0       = obs[0:7]
    tip0     = obs[14:17]
    obj0     = obs[17:20]
    goal0    = obs[20:23]
    anal0    = dynamics._forward_kinematics(q0)
    print(f"  q0         = {np.array2string(q0, precision=3, separator=',')}")
    print(f"  real_tip0  = ({tip0[0]:+.3f}, {tip0[1]:+.3f}, {tip0[2]:+.3f})")
    print(f"  anal_tip0  = ({anal0[0]:+.3f}, {anal0[1]:+.3f})")
    print(f"  real_obj0  = ({obj0[0]:+.3f}, {obj0[1]:+.3f}, {obj0[2]:+.3f})")
    print(f"  goal       = ({goal0[0]:+.3f}, {goal0[1]:+.3f}, {goal0[2]:+.3f})")
    print(f"  target(2d) = ({target[0]:+.3f}, {target[1]:+.3f})")
    print(f"{'='*60}\n")

    # ---- Control loop ------------------------------------------------------
    for t in range(config.max_steps):
        step_start = time.perf_counter()

        # ========================= GPU WORK =================================
        #
        # Observation-delay protocol:
        #   PF particles track belief at time (t − d).  Each step we feed
        #   the DELAYED observation (from d steps ago) to pf.update().
        #   For MPPI we need current-time estimates, so sample_current()
        #   propagates temporary copies through the d recent actions.

        # -- Delayed observation ------------------------------------------
        delayed_obs = obs_buffer[0]          # oldest buffered obs (d steps old)

        # PF inject + propagate: only after the action buffer has d+1 entries
        # so that action_buffer[0] is the action from the DELAYED transition
        # (the action applied at step t-d-1 that moved state from
        #  prev_delayed_obs to delayed_obs).
        if len(action_buffer) > config.obs_delay:
            delayed_action = action_buffer[0]
            pf.inject_observation(prev_delayed_obs)
            pf.propagate(delayed_action)

        # Weight update against delayed observation
        pf.update(delayed_obs)

        ess = pf.effective_sample_size()

        if ess < config.resample_threshold * config.N:
            pf.resample()

        # -- Delay-aware state estimate for MPPI --------------------------
        pf.inject_observation(delayed_obs)

        t_delay_start = time.perf_counter()
        # Use PF weighted mean as a SINGLE initial state for all K MPPI
        # rollouts.  Sampling K different particle states introduces
        # initial-condition noise (varying obj_pos -> varying obj-target
        # cost) that DOMINATES the action-quality signal, preventing
        # MPPI from selecting approach-improving trajectories.
        mean_state = pf.estimate()            # (state_dim,) numpy float32

        if len(action_buffer) > config.obs_delay:
            recent_actions = list(action_buffer)[1:]   # last d actions
        else:
            recent_actions = list(action_buffer)        # warmup: all we have

        # Propagate mean state through delay actions on CPU
        for act in recent_actions:
            mean_state = dynamics.f_numpy(mean_state, act).astype(np.float32)

        # Tile the single state across all K MPPI rollout starts
        mean_gpu = cp.asarray(mean_state[None, :], dtype=cp.float32)
        initial_states = cp.repeat(mean_gpu, mppi.K, axis=0)
        cp.cuda.Device(config.device_id).synchronize()
        t_delay_end = time.perf_counter()
        T_pf_delay_ms = (t_delay_end - t_delay_start) * 1e3

        # MPPI planning
        action, mppi_timing = mppi.compute_action(initial_states)

        cp.cuda.Device(config.device_id).synchronize()
        # ========================= END GPU WORK =============================
        t_gpu_end = time.perf_counter()

        # ========================= CPU / ENV WORK ===========================
        prev_delayed_obs = delayed_obs

        obs, reward, terminated, truncated, _info = env.step(action)
        obs = obs.astype(np.float32)

        # -- Sensor noise + delay buffers ---------------------------------
        noisy_obs = obs.copy()
        noisy_obs += np.random.normal(
            0.0, config.sensor_noise_std, obs.shape,
        ).astype(np.float32)
        obs_buffer.append(noisy_obs)
        action_buffer.append(action.copy())

        t_env_end = time.perf_counter()
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
            "T_pf_delay_propagate_ms": T_pf_delay_ms,
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
                f"T={T_total_ms:6.2f}ms (GPU={T_gpu_ms:5.2f} ENV={T_env_ms:5.2f} "
                f"delay={T_pf_delay_ms:5.2f}) | "
                f"ESS={ess:6.0f}/{config.N} | "
                f"K={mppi.K}"
            )

        # ---- Diagnostic output every 10 steps ----------------------------
        if t % 10 == 0:
            q_now        = obs[0:7]
            real_tip     = obs[14:17]                        # MuJoCo fingertip xyz
            real_obj     = obs[17:20]                        # MuJoCo object xyz
            anal_tip     = dynamics._forward_kinematics(q_now)  # our FK (x, y)
            tip_err      = np.sqrt((anal_tip[0] - real_tip[0])**2
                                   + (anal_tip[1] - real_tip[1])**2)
            tip_obj_dist = np.sqrt((real_tip[0] - real_obj[0])**2
                                   + (real_tip[1] - real_obj[1])**2)
            anal_obj_dist = np.sqrt((anal_tip[0] - real_obj[0])**2
                                    + (anal_tip[1] - real_obj[1])**2)

            # Check how many particles have obj_pos within contact radius
            # of the INJECTED tip position (state[18:20])
            particles_cpu = cp.asnumpy(pf.particles)
            p_obj = particles_cpu[:, 14:16]                  # (N, 2)
            p_tip = particles_cpu[:, 18:20]                  # (N, 2) injected real tip
            p_dists = np.linalg.norm(p_obj - p_tip, axis=1)
            n_contact = int(np.sum(p_dists < CONTACT_RADIUS))

            # Particle obj_pos spread
            obj_mean = p_obj.mean(axis=0)
            obj_std  = p_obj.std(axis=0)

            print(
                f"  DIAG step {t}: "
                f"real_tip=({real_tip[0]:+.3f},{real_tip[1]:+.3f},{real_tip[2]:+.3f}) "
                f"anal_tip=({anal_tip[0]:+.3f},{anal_tip[1]:+.3f}) "
                f"FK_err={tip_err:.4f}m"
            )
            # Use the first particle's tip_pos as representative (all same after inject)
            injected_tip = particles_cpu[0, 18:20] if particles_cpu.shape[1] > 18 else np.array(anal_tip)
            inj_obj_dist = np.sqrt((injected_tip[0] - real_obj[0])**2
                                   + (injected_tip[1] - real_obj[1])**2)
            print(
                f"         real_obj=({real_obj[0]:+.3f},{real_obj[1]:+.3f}) "
                f"tip→obj(real)={tip_obj_dist:.3f}m "
                f"tip→obj(injected)={inj_obj_dist:.3f}m "
                f"contact_r={CONTACT_RADIUS}"
            )
            # Weight diagnostics
            w_cpu = cp.asnumpy(pf.weights)
            w_max = w_cpu.max()
            w_min = w_cpu[w_cpu > 0].min() if (w_cpu > 0).any() else 0.0
            w_ratio = w_max / w_min if w_min > 0 else float('inf')

            print(
                f"         particles: obj_mean=({obj_mean[0]:+.3f},{obj_mean[1]:+.3f}) "
                f"obj_std=({obj_std[0]:.3f},{obj_std[1]:.3f}) "
                f"n_in_contact={n_contact}/{config.N}"
            )
            print(
                f"         weights: max={w_max:.6f} min={w_min:.6f} "
                f"ratio={w_ratio:.1f} std={w_cpu.std():.6f}"
            )
            print(
                f"         action={np.array2string(action, precision=2, separator=',')}"
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
