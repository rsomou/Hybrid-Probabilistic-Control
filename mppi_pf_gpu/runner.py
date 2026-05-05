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
from gymnasium.wrappers import RecordVideo
import numpy as np
import cupy as cp

from config import Config
from envs.pusher import PusherDynamics, CONTACT_RADIUS, TABLE_Z
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


def _obs_to_state(obs: np.ndarray, dynamics) -> np.ndarray:
    """
    Build a full 21-dim state vector directly from a Pusher-v5 observation.
    Used in --no-pf mode to give MPPI perfect state information.

    State layout: [q(7), qdot(7), obj_pos(2), obj_vel(2), fork_xy(2), fork_z(1)]
    """
    fk_x, fk_y, fk_z = dynamics._forward_kinematics(obs[0:7])
    state = np.zeros(21, dtype=np.float32)
    state[0:7]   = obs[0:7]    # joint angles
    state[7:14]  = obs[7:14]   # joint velocities
    state[14:16] = obs[17:19]  # object xy
    state[16:18] = 0.0         # object velocity — not in obs, assume zero
    state[18]    = fk_x        # fork xy from analytical FK
    state[19]    = fk_y
    state[20]    = fk_z        # fork z from analytical FK
    return state


# --------------------------------------------------------------------------- #
# Main control loop
# --------------------------------------------------------------------------- #

def run(config: Config, render: bool = False, record: bool = False,
        no_pf: bool = False):
    """
    Execute one episode of Pusher-v5 with MPPI (+ Particle Filter unless no_pf).

    Parameters
    ----------
    config : Config
    render : bool — if True opens the MuJoCo viewer
    record : bool — if True saves an MP4 video to ./videos/
    no_pf  : bool — if True bypass the PF; feed MPPI perfect state from obs

    Returns
    -------
    total_reward : float
    timing_log   : list[dict]  — one entry per step
    """
    # ---- Environment -------------------------------------------------------
    if record:
        render_mode = "rgb_array"
    elif render:
        render_mode = "human"
    else:
        render_mode = None
    # Pass max_episode_steps to override the default 100-step truncation.
    env = gym.make(config.env_name, render_mode=render_mode,
                   max_episode_steps=config.max_steps)
    if record:
        env = RecordVideo(env, video_folder="./videos",
                          name_prefix="pusher_mppi",
                          episode_trigger=lambda _: True)

    # ---- Components --------------------------------------------------------
    dynamics = PusherDynamics(dt=config.dt)
    gpu      = GPUUtils(config)
    pf       = ParticleFilter(dynamics, config, gpu)
    mppi     = MPPI(dynamics, config, gpu)

    # ---- Episode reset -----------------------------------------------------
    obs, _info = env.reset()
    obs        = obs.astype(np.float32)

    # ---- Tune MuJoCo contact so collisions impart more force on object ---
    import mujoco
    model = env.unwrapped.model
    data  = env.unwrapped.data

    # Heavier object = needs sustained force to move, doesn't fly away on contact.
    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    model.body_mass[obj_body_id] = 0.5   # 500g — substantial, needs real pushing

    # Higher damping = object decelerates quickly after contact ends.
    # Acts like table friction — the puck stops near where you pushed it.
    for jname in ["obj_slidey", "obj_slidex"]:
        jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        dof  = model.jnt_dofadr[jid]
        model.dof_damping[dof] = 2.0    # high viscous friction

    # Stiffen contacts to near-rigid so the fork cannot penetrate the object.
    # solref[0]=0.001 → 1 ms time-constant (≈ rigid spring).
    # solimp approaching 1.0 → maximum constraint impedance (near-zero penetration).
    # margin=0 / gap=0 → contact force activates only on actual overlap, no soft zone.
    # Newton solver + more iterations → constraint solver converges to near-zero penetration.
    model.opt.solver     = 2    # 0=PGS, 1=CG, 2=Newton — Newton is most accurate
    model.opt.iterations = 100  # was default 100; keep high for hard contacts
    model.opt.tolerance  = 1e-10  # tighter convergence

    for gi in [13, 14, 15, 18]:   # wrist fork capsules + object cylinder
        model.geom_condim[gi]  = 3                           # tangential friction
        model.geom_friction[gi] = [1.0, 0.005, 0.0001]
        model.geom_solref[gi]  = [0.001, 1.0]               # 1 ms — near-rigid
        model.geom_solimp[gi]  = [0.99, 0.9999, 0.0001, 0.5, 2.0]  # max impedance
        model.geom_margin[gi]  = 0.0                         # no soft detection zone
        model.geom_gap[gi]     = 0.0                         # no free zone inside margin
        model.geom_priority[gi] = 1                          # these params win the mix

    mujoco.mj_forward(model, data)

    # Goal is fixed — read directly from obs[20:22] (always correct in v5)
    target = _get_target(obs)
    dynamics.set_target(target)
    mppi.set_target(target)
    mppi.reset()
    pf.initialize(obs)

    total_reward   = 0.0
    timing_log     = []
    resample_count = 0    # number of times PF resampled this episode

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
    q0    = obs[0:7]
    obj0  = obs[17:20]
    fork0 = dynamics._forward_kinematics(q0)
    print(f"  fork0  = ({fork0[0]:+.3f}, {fork0[1]:+.3f}, {fork0[2]:+.3f})  (FK = wrist fork)")
    print(f"  obj0   = ({obj0[0]:+.3f}, {obj0[1]:+.3f}, {obj0[2]:+.3f})")
    print(f"  goal   = ({target[0]:+.3f}, {target[1]:+.3f})")
    print(f"  fork_z = {fork0[2]:+.4f}  TABLE_Z = {TABLE_Z:.4f}")
    print(f"{'='*60}\n")

    # ---- Control loop ------------------------------------------------------
    for t in range(config.max_steps):
        step_start = time.perf_counter()

        # ========================= GPU WORK =================================

        if no_pf:
            # ---- Perfect-information mode: bypass PF entirely --------------
            # Build state directly from the current observation.
            state_vec   = _obs_to_state(obs, dynamics)
            state_gpu   = cp.asarray(state_vec, dtype=cp.float32).reshape(1, -1)
            initial_states = cp.repeat(state_gpu, mppi.K, axis=0)
            ess = float(config.N)   # sentinel — PF not running
            cp.cuda.Device(config.device_id).synchronize()
            t_delay_start = t_delay_end = time.perf_counter()
            T_pf_delay_ms = 0.0
        else:
            # ---- Normal PF path --------------------------------------------
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
                resample_count += 1

            # -- Delay-aware state estimate for MPPI --------------------------
            pf.inject_observation(delayed_obs)

            t_delay_start = time.perf_counter()
            # Use PF weighted mean as a SINGLE initial state for all K MPPI
            # rollouts.  Sampling K different particle states introduces
            # initial-condition noise (varying obj_pos -> varying obj-target
            # cost) that DOMINATES the action-quality signal, preventing
            # MPPI from selecting approach-improving trajectories.
            mean_gpu = pf.estimate_gpu()          # (1, state_dim) on GPU

            if len(action_buffer) > config.obs_delay:
                recent_actions = list(action_buffer)[1:]   # last d actions
            else:
                recent_actions = list(action_buffer)        # warmup: all we have

            # Propagate mean state through delay actions ON GPU (single particle,
            # zero noise) — avoids the ~85ms CPU RNEA bottleneck.
            state_dim = dynamics.state_dim
            zero_noise = cp.zeros((1, state_dim), dtype=cp.float32)
            for act in recent_actions:
                act_gpu = cp.asarray(act, dtype=cp.float32)
                pf._propagate_kernel(
                    (1,), (1,),
                    (
                        mean_gpu,
                        act_gpu,
                        zero_noise,
                        cp.float32(0.0),   # no joint noise
                        cp.float32(0.0),   # no obj noise
                        cp.float32(config.dt),
                        np.int32(1),
                    ),
                )

            # Compute fork position from FK using the latest observed q.
            # This is the wrist fork (collision geometry), NOT obs[14:17]
            # (fingertip marker which is ~8cm offset from the fork).
            latest_obs = obs_buffer[-1] if len(obs_buffer) > 0 else obs
            fk_x, fk_y, fk_z = dynamics._forward_kinematics(latest_obs[0:7])
            mean_gpu[0, 18] = cp.float32(fk_x)
            mean_gpu[0, 19] = cp.float32(fk_y)
            mean_gpu[0, 20] = cp.float32(fk_z)

            # Tile the single state across all K MPPI rollout starts
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
        if not no_pf:
            prev_delayed_obs = delayed_obs

        obs, reward, terminated, truncated, _info = env.step(action)
        obs = obs.astype(np.float32)

        # -- Sensor noise + delay buffers ---------------------------------
        noisy_obs = obs.copy()
        if not no_pf:
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

        timing_entry = {
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
        timing_log.append(timing_entry)

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
            q_now    = obs[0:7]
            real_obj = obs[17:20]
            # FK = fork position (r_wrist_roll_link body = collision geometry)
            fork     = dynamics._forward_kinematics(q_now)
            fork_xyz = np.array(fork)
            fork_obj_3d = float(np.linalg.norm(
                fork_xyz - np.array([real_obj[0], real_obj[1], TABLE_Z])))

            if not no_pf:
                particles_cpu = cp.asnumpy(pf.particles)
                p_obj   = particles_cpu[:, 14:16]
                p_fork  = particles_cpu[:, 18:20]
                p_fz    = particles_cpu[:, 20]
                dxy = p_obj - p_fork
                dz  = p_fz - TABLE_Z
                p_d3d = np.sqrt(dxy[:, 0]**2 + dxy[:, 1]**2 + dz**2)
                n_contact = int(np.sum(p_d3d < CONTACT_RADIUS))

            print(
                f"  DIAG step {t}: "
                f"fork=({fork[0]:+.3f},{fork[1]:+.3f},{fork[2]:+.3f}) "
                f"obj=({real_obj[0]:+.3f},{real_obj[1]:+.3f}) "
                f"fork→obj_3d={fork_obj_3d:.3f}m"
            )
            if not no_pf:
                w_cpu = cp.asnumpy(pf.weights)
                w_max = w_cpu.max()
                w_min = w_cpu[w_cpu > 0].min() if (w_cpu > 0).any() else 0.0

                # Spread (std) of object position belief
                obj_std_x = float(p_obj[:, 0].std())
                obj_std_y = float(p_obj[:, 1].std())

                # PF tracking error vs ground truth
                pf_mean_obj = p_obj.mean(axis=0)
                pf_err = float(np.linalg.norm(
                    pf_mean_obj - np.array([real_obj[0], real_obj[1]])))

                print(
                    f"         PF: obj_mean=({pf_mean_obj[0]:+.3f},{pf_mean_obj[1]:+.3f}) "
                    f"err={pf_err:.3f}m  "
                    f"std=({obj_std_x:.3f},{obj_std_y:.3f})  "
                    f"n_contact={n_contact}/{config.N}  "
                    f"ESS={ess:.0f}  "
                    f"w_ratio={w_max/w_min if w_min>0 else float('inf'):.0f}"
                )
            print(f"         fork_z={fork[2]:+.4f}  TABLE_Z={TABLE_Z:.4f}")

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
    print(f"  PF resamples  : {resample_count} / {n_steps} steps "
          f"({'N/A — --no-pf' if no_pf else f'{100*resample_count/max(n_steps,1):.1f}%'})")
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
    parser.add_argument("--H",        type=int,   default=20,
                        help="MPPI planning horizon")
    parser.add_argument("--deadline", type=float, default=50.0,
                        help="Per-step deadline in ms")
    parser.add_argument("--sigma",    type=float, default=0.5,
                        help="MPPI perturbation global scale (multiplied by per-joint weights)")
    parser.add_argument("--lambda_",  type=float, default=200.0,
                        help="MPPI temperature (scale with total cost magnitude)")
    parser.add_argument("--steps",    type=int,   default=300,
                        help="Max episode steps")
    parser.add_argument("--device",   type=int,   default=0,
                        help="CUDA device ID")
    parser.add_argument("--render",   action="store_true",
                        help="Open MuJoCo viewer")
    parser.add_argument("--record",   action="store_true",
                        help="Record MP4 video to ./videos/")
    parser.add_argument("--no-timing", action="store_true",
                        help="Suppress per-step timing output")
    parser.add_argument("--no-pf",    action="store_true",
                        help="Bypass particle filter; give MPPI perfect state from obs")
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

    run(cfg, render=args.render, record=args.record, no_pf=args.no_pf)
