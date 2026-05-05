#!/usr/bin/env python3
"""
test_analytical_push.py — Analytically place the wrist fork behind the puck
(relative to the goal), then push straight through it toward the goal.

The fork crossbar (geom 13) sits at the r_wrist_roll_link body origin.
Our FK computes exactly this point (no offset).

Phase 1: PD-servo the arm to place the fork behind the object.
Phase 2: Track a Cartesian velocity along push_dir using the Jacobian,
         re-targeting the object's live position so the fork chases it.
"""

import gymnasium as gym
import numpy as np
import mujoco
import time


TABLE_Z = -0.275
STANDOFF = 0.10   # start 10cm behind the object
PUSH_SPEED = 0.10  # desired end-effector speed in m/s


def compute_push_plan(obj_xy, goal_xy):
    push_dir = goal_xy - obj_xy
    push_dist = np.linalg.norm(push_dir)
    push_dir = push_dir / push_dist
    # Fork starts behind the object (opposite to push direction)
    fork_start_xy = obj_xy - push_dir * STANDOFF
    return push_dir, fork_start_xy, push_dist


def find_best_config(model, data, target_xyz):
    """Scan joint space to place fork body (body 9) at target_xyz."""
    best_dist = 1e9
    best_q = None
    for q0 in np.linspace(-2.2, 1.7, 50):
        for q1 in np.linspace(-0.5, 1.3, 30):
            for q3 in np.linspace(-2.3, 0.0, 30):
                for q5 in np.linspace(-1.0, 0.0, 8):
                    data.qpos[:7] = [q0, q1, 0.0, q3, 0.0, q5, 0.0]
                    data.qvel[:] = 0.0
                    mujoco.mj_forward(model, data)
                    fork_pos = data.xpos[9]
                    d = np.linalg.norm(fork_pos - target_xyz)
                    if d < best_dist:
                        best_dist = d
                        best_q = np.array([q0, q1, 0.0, q3, 0.0, q5, 0.0])
    return best_q, best_dist


def compute_jacobian(model, data):
    """3x7 positional Jacobian for wrist_roll_link body (body 9)."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, 9)
    return jacp[:, :7]


def main():
    env = gym.make("Pusher-v5", max_episode_steps=800, render_mode="human")
    obs, _ = env.reset(seed=42)
    model = env.unwrapped.model
    data  = env.unwrapped.data

    obj_xy = obs[17:19].copy()
    goal_xy = obs[20:22].copy()
    obj_start = obj_xy.copy()

    print("=" * 70)
    print("  ANALYTICAL PUSH PLAN")
    print("=" * 70)

    push_dir, fork_start_xy, push_dist = compute_push_plan(obj_xy, goal_xy)
    target_xyz = np.array([fork_start_xy[0], fork_start_xy[1], TABLE_Z])

    print(f"  Object:     ({obj_xy[0]:+.4f}, {obj_xy[1]:+.4f})")
    print(f"  Goal:       ({goal_xy[0]:+.4f}, {goal_xy[1]:+.4f})")
    print(f"  Push dir:   ({push_dir[0]:+.4f}, {push_dir[1]:+.4f})")
    print(f"  Fork start: ({fork_start_xy[0]:+.4f}, {fork_start_xy[1]:+.4f}, {TABLE_Z})")

    print(f"\n  Scanning joint space...")
    best_q, best_dist = find_best_config(model, data, target_xyz)
    # Set and verify
    data.qpos[:7] = best_q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    fork_pos = data.xpos[9].copy()
    print(f"  Best config: [{', '.join(f'{x:+.3f}' for x in best_q)}]")
    print(f"  Fork at:     ({fork_pos[0]:+.4f}, {fork_pos[1]:+.4f}, {fork_pos[2]:+.4f})")
    print(f"  Error:       {best_dist:.4f}m")

    # ---- Phase 1: Servo to approach config ----
    print(f"\n{'='*70}")
    print(f"  PHASE 1: Servo to approach position")
    print(f"{'='*70}")

    Kp = np.array([50.0, 50.0, 15.0, 40.0, 15.0, 25.0, 15.0])
    Kd = np.array([8.0, 8.0, 2.0, 5.0, 2.0, 3.0, 2.0])

    for t in range(250):
        q_now = obs[0:7]
        qdot_now = obs[7:14]
        q_err = best_q - q_now
        action = np.clip(Kp * q_err - Kd * qdot_now, -2.0, 2.0)
        obs, reward, term, trunc, _ = env.step(action)

        if t % 30 == 0:
            fp = data.xpos[9]
            print(f"  t={t:3d}  fork=({fp[0]:+.3f},{fp[1]:+.3f},{fp[2]:+.3f})  "
                  f"q_err={np.linalg.norm(q_err):.4f}")

        if np.linalg.norm(best_q - obs[0:7]) < 0.03 and np.linalg.norm(obs[7:14]) < 0.1:
            print(f"  Converged at t={t}")
            break

    # ---- Phase 2: Push toward goal ----
    print(f"\n{'='*70}")
    print(f"  PHASE 2: Push (Jacobian velocity tracking)")
    print(f"{'='*70}")

    Kv = 12.0       # velocity gain
    Kz = 3.0        # z height correction gain

    for t in range(400):
        q_now = obs[0:7]
        qdot_now = obs[7:14]

        # Update Jacobian at current config
        data.qpos[:7] = q_now
        data.qvel[:7] = qdot_now
        mujoco.mj_forward(model, data)
        J = compute_jacobian(model, data)
        fork_pos = data.xpos[9].copy()

        # Desired velocity: push toward the object's CURRENT position,
        # then through it along push_dir. This chases the object.
        obj_now = obs[17:19]
        fork_to_obj = obj_now - fork_pos[:2]
        d_to_obj = np.linalg.norm(fork_to_obj)

        if d_to_obj > 0.02:
            # Steer toward the object (approach/chase)
            approach_dir = fork_to_obj / d_to_obj
            # Blend: mostly toward object, some along push_dir
            blend = min(d_to_obj / 0.10, 1.0)  # 1.0 = far, approach. 0.0 = close, push
            cmd_dir = blend * approach_dir + (1.0 - blend) * push_dir
            cmd_dir = cmd_dir / np.linalg.norm(cmd_dir)
        else:
            # Close enough — push straight through
            cmd_dir = push_dir

        z_err = TABLE_Z - fork_pos[2]
        v_cmd = np.array([cmd_dir[0] * PUSH_SPEED,
                          cmd_dir[1] * PUSH_SPEED,
                          Kz * z_err])

        # Damped pseudoinverse for stability
        JJT = J @ J.T + 0.01 * np.eye(3)
        qdot_cmd = J.T @ np.linalg.solve(JJT, v_cmd)

        action = np.clip(Kv * (qdot_cmd - qdot_now), -2.0, 2.0)
        obs, reward, term, trunc, _ = env.step(action)

        if t % 20 == 0:
            obj_disp = np.linalg.norm(obj_now - obj_start)
            d_obj_goal = np.linalg.norm(obj_now - goal_xy)
            print(f"  t={t:3d}  fork=({fork_pos[0]:+.3f},{fork_pos[1]:+.3f},{fork_pos[2]:+.3f})  "
                  f"obj=({obj_now[0]:+.3f},{obj_now[1]:+.3f})  "
                  f"disp={obj_disp:.4f}m  obj->goal={d_obj_goal:.3f}m  "
                  f"fork->obj={d_to_obj:.3f}m")

        if term or trunc:
            break

    # ---- Summary ----
    final_obj = obs[17:19]
    total_disp = np.linalg.norm(final_obj - obj_start)
    dist_to_goal = np.linalg.norm(final_obj - goal_xy)
    initial_to_goal = np.linalg.norm(obj_start - goal_xy)
    print(f"\n{'='*70}")
    print(f"  RESULT")
    print(f"  Object start:       ({obj_start[0]:+.4f}, {obj_start[1]:+.4f})")
    print(f"  Object final:       ({final_obj[0]:+.4f}, {final_obj[1]:+.4f})")
    print(f"  Goal:               ({goal_xy[0]:+.4f}, {goal_xy[1]:+.4f})")
    print(f"  Total displacement: {total_disp:.4f}m")
    print(f"  Initial to goal:    {initial_to_goal:.4f}m")
    print(f"  Final to goal:      {dist_to_goal:.4f}m")
    print(f"  Progress:           {initial_to_goal - dist_to_goal:.4f}m closer")
    print(f"{'='*70}")

    time.sleep(3.0)
    env.close()


if __name__ == "__main__":
    main()
