#!/usr/bin/env python3
"""
test_push.py — Minimal diagnostic: Is the puck even pushable?

Bypasses MPPI entirely. Runs two tests:
  1. Dumps MuJoCo model info (masses, contact geoms, contype/conaffinity)
  2. Scripted action sequence: descend to table height, then ram the puck

Run:  python test_push.py
"""

import gymnasium as gym
import numpy as np


def dump_model_info(env):
    """Print everything relevant about contact geometry."""
    model = env.unwrapped.model
    data  = env.unwrapped.data

    print("=" * 70)
    print("  MuJoCo MODEL DIAGNOSTICS")
    print("=" * 70)

    # Body masses
    print("\n--- Body masses ---")
    for i in range(model.nbody):
        name = model.body(i).name
        mass = model.body_mass[i]
        print(f"  body {i:2d} '{name:30s}' mass={mass:.6f}")

    # Geom properties (focus on contype/conaffinity)
    print("\n--- Geom contact properties ---")
    print(f"  {'idx':>3s}  {'name':30s}  {'type':>4s}  {'size':20s}  "
          f"{'contype':>7s}  {'conaffinity':>11s}  {'body':20s}")
    for i in range(model.ngeom):
        gname = model.geom(i).name
        gtype = model.geom_type[i]
        gsize = model.geom_size[i]
        contype = model.geom_contype[i]
        conaffinity = model.geom_conaffinity[i]
        body_id = model.geom_bodyid[i]
        bname = model.body(body_id).name
        type_names = {0: 'plane', 1: 'hfield', 2: 'sphere', 3: 'capsule',
                      4: 'ellip', 5: 'cyl', 6: 'box', 7: 'mesh'}
        tname = type_names.get(gtype, str(gtype))
        if contype > 0 or conaffinity > 0:
            marker = " <<<< COLLISION-ENABLED"
        else:
            marker = ""
        print(f"  {i:3d}  {gname:30s}  {tname:>4s}  {str(gsize):20s}  "
              f"{contype:7d}  {conaffinity:11d}  {bname:20s}{marker}")

    # Joint info for object
    print("\n--- Joints ---")
    for i in range(model.njnt):
        jname = model.joint(i).name
        jtype = model.jnt_type[i]
        jbody = model.jnt_bodyid[i]
        bname = model.body(jbody).name
        type_names = {0: 'free', 1: 'ball', 2: 'slide', 3: 'hinge'}
        tname = type_names.get(jtype, str(jtype))
        jrange = model.jnt_range[i]
        jdamp = model.dmp_jnt_dmpprm if hasattr(model, 'dmp_jnt_dmpprm') else "N/A"
        print(f"  joint {i:2d} '{jname:25s}' type={tname:5s} body='{bname}' "
              f"range=[{jrange[0]:.3f}, {jrange[1]:.3f}]")

    # Check equality constraints
    print(f"\n--- Equality constraints: {model.neq} ---")
    for i in range(model.neq):
        print(f"  eq {i}: type={model.eq_type[i]} obj1={model.eq_obj1id[i]} obj2={model.eq_obj2id[i]}")

    # Gravity
    print(f"\n--- Gravity: {model.opt.gravity} ---")

    print("=" * 70)


def check_contacts(env):
    """Print active contact pairs after a step."""
    data = env.unwrapped.data
    model = env.unwrapped.model
    print(f"  Active contacts (ncon={data.ncon}):")
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = model.geom(c.geom1).name
        g2 = model.geom(c.geom2).name
        dist = c.dist
        print(f"    contact {i}: {g1} <-> {g2}  dist={dist:.6f}  "
              f"pos=({c.pos[0]:.3f},{c.pos[1]:.3f},{c.pos[2]:.3f})")
    if data.ncon == 0:
        print("    (none)")


def test_scripted_push():
    """
    Test 1: Scripted action sequence.
    Strategy: use joint torques to drive the tip toward the puck.
    """
    print("\n" + "=" * 70)
    print("  TEST: Scripted push sequence")
    print("=" * 70)

    env = gym.make("Pusher-v5", max_episode_steps=200, render_mode="human")
    obs, _ = env.reset(seed=42)

    tip0 = obs[14:17]
    obj0 = obs[17:20]
    goal0 = obs[20:23]
    print(f"  Initial tip : ({tip0[0]:+.4f}, {tip0[1]:+.4f}, {tip0[2]:+.4f})")
    print(f"  Initial obj : ({obj0[0]:+.4f}, {obj0[1]:+.4f}, {obj0[2]:+.4f})")
    print(f"  Goal        : ({goal0[0]:+.4f}, {goal0[1]:+.4f}, {goal0[2]:+.4f})")
    print(f"  tip→obj dist: {np.linalg.norm(tip0 - obj0):.4f}")
    print()

    obj_start = obj0[:2].copy()

    for t in range(150):
        tip = obs[14:17]
        obj = obs[17:20]

        # Simple proportional controller:
        # - shoulder_pan (joint 0, Z axis): steer tip_x toward obj_x
        # - shoulder_lift (joint 1, Y axis): steer tip_z down to table height
        # - elbow_flex (joint 3, Y axis): help with reach and descent
        dx = obj[0] - tip[0]
        dy = obj[1] - tip[1]
        dz = -0.275 - tip[2]  # target table height

        # Rough proportional gains
        action = np.zeros(7, dtype=np.float32)
        action[0] = np.clip(3.0 * dy, -2, 2)        # shoulder pan → y control
        action[1] = np.clip(-3.0 * dz, -2, 2)       # shoulder lift → z descent
        action[3] = np.clip(-2.0 * dz - 1.0, -2, 2) # elbow flex → help descend
        action[5] = np.clip(-2.0 * dz, -2, 2)       # wrist flex → angle tip down

        # Once close in z, push toward object in x
        if abs(tip[2] - (-0.275)) < 0.05:
            action[0] = np.clip(5.0 * dy + 2.0 * dx, -2, 2)
            action[1] = np.clip(-5.0 * dz + 1.0 * dx, -2, 2)

        obs, reward, term, trunc, info = env.step(action)

        if t % 5 == 0:
            tip_now = obs[14:17]
            obj_now = obs[17:20]
            obj_disp = np.linalg.norm(obj_now[:2] - obj_start)
            d_xy = np.linalg.norm(tip_now[:2] - obj_now[:2])
            d_3d = np.linalg.norm(tip_now - obj_now)
            print(f"  t={t:3d}  tip=({tip_now[0]:+.3f},{tip_now[1]:+.3f},{tip_now[2]:+.3f})  "
                  f"obj=({obj_now[0]:+.3f},{obj_now[1]:+.3f})  "
                  f"tip→obj_xy={d_xy:.3f}  tip→obj_3d={d_3d:.3f}  "
                  f"obj_moved={obj_disp:.4f}m  R={reward:.3f}")
            check_contacts(env)

        if term or trunc:
            break

    final_obj = obs[17:20]
    total_disp = np.linalg.norm(final_obj[:2] - obj_start)
    print(f"\n  RESULT: Object displaced {total_disp:.4f}m from start")
    if total_disp < 0.01:
        print("  >>> OBJECT DID NOT MOVE — problem is in env/contact geometry <<<")
    else:
        print(f"  >>> Object moved {total_disp:.3f}m — contact works! <<<")

    env.close()


def test_brute_force_push():
    """
    Test 2: Just slam max torque on every joint and see if anything happens.
    If the puck moves even with random max-torque, we know contact CAN work.
    """
    print("\n" + "=" * 70)
    print("  TEST: Brute-force max torque on all joints")
    print("=" * 70)

    env = gym.make("Pusher-v5", max_episode_steps=100, render_mode="human")
    obs, _ = env.reset(seed=42)
    obj_start = obs[17:19].copy()

    for t in range(100):
        # Alternate between joint configurations that swing the arm around
        if t % 20 < 10:
            action = np.array([2.0, 2.0, 0.0, -2.0, 0.0, -2.0, 0.0])
        else:
            action = np.array([-2.0, -2.0, 0.0, 2.0, 0.0, 2.0, 0.0])
        obs, *_ = env.step(action)

        if t % 10 == 0:
            tip = obs[14:17]
            obj = obs[17:20]
            disp = np.linalg.norm(obj[:2] - obj_start)
            print(f"  t={t:3d}  tip=({tip[0]:+.3f},{tip[1]:+.3f},{tip[2]:+.3f})  "
                  f"obj=({obj[0]:+.3f},{obj[1]:+.3f})  disp={disp:.4f}m")
            check_contacts(env)

    final_disp = np.linalg.norm(obs[17:19] - obj_start)
    print(f"\n  RESULT: Object displaced {final_disp:.4f}m")
    env.close()


def test_position_control_push():
    """
    Test 3: Use MuJoCo's own qpos to drive the arm directly to a
    configuration that places the wrist fork at the puck location,
    then push.  This bypasses the controller entirely and answers:
    "CAN the arm physically reach the object and push it?"
    """
    print("\n" + "=" * 70)
    print("  TEST: Position-control push (set qpos directly)")
    print("=" * 70)

    import mujoco

    env = gym.make("Pusher-v5", max_episode_steps=300, render_mode="human")
    obs, _ = env.reset(seed=42)
    model = env.unwrapped.model
    data  = env.unwrapped.data

    obj_pos = obs[17:20].copy()
    print(f"  Object at: ({obj_pos[0]:+.4f}, {obj_pos[1]:+.4f}, {obj_pos[2]:+.4f})")

    # Print collision geom positions (wrist fork, geoms 13-15)
    for gi in [13, 14, 15]:
        gpos = data.geom_xpos[gi]
        print(f"  Collision geom {gi} at: ({gpos[0]:+.4f}, {gpos[1]:+.4f}, {gpos[2]:+.4f})")

    # Try a range of joint configurations to find one that reaches the object
    print("\n  Scanning q3 (elbow_flex) to find reachable configs...")
    best_dist = 1e9
    best_q = None
    for q1 in np.linspace(-0.5, 1.3, 10):
        for q3 in np.linspace(-2.3, 0.0, 10):
            for q5 in np.linspace(-1.0, 0.0, 5):
                # Set joint angles
                data.qpos[0] = 0.0   # shoulder pan
                data.qpos[1] = q1    # shoulder lift
                data.qpos[2] = 0.0   # upper arm roll
                data.qpos[3] = q3    # elbow flex
                data.qpos[4] = 0.0   # forearm roll
                data.qpos[5] = q5    # wrist flex
                data.qpos[6] = 0.0   # wrist roll
                data.qvel[:7] = 0.0
                mujoco.mj_forward(model, data)

                # Check wrist fork collision geom positions
                for gi in [13, 14, 15]:
                    gpos = data.geom_xpos[gi]
                    d = np.linalg.norm(gpos - obj_pos)
                    if d < best_dist:
                        best_dist = d
                        best_q = [0.0, q1, 0.0, q3, 0.0, q5, 0.0]
                        best_gi = gi

    print(f"  Best config: q={[f'{x:.2f}' for x in best_q]}")
    print(f"  Best wrist-geom dist to obj: {best_dist:.4f}m (geom {best_gi})")

    # Also scan shoulder_pan
    print("\n  Scanning q0 (shoulder_pan) + q1 + q3...")
    for q0 in np.linspace(-2.2, 1.7, 20):
        for q1 in np.linspace(-0.5, 1.3, 10):
            for q3 in np.linspace(-2.3, 0.0, 10):
                data.qpos[0] = q0
                data.qpos[1] = q1
                data.qpos[2] = 0.0
                data.qpos[3] = q3
                data.qpos[4] = 0.0
                data.qpos[5] = -0.5
                data.qpos[6] = 0.0
                data.qvel[:7] = 0.0
                mujoco.mj_forward(model, data)

                for gi in [13, 14, 15]:
                    gpos = data.geom_xpos[gi]
                    d = np.linalg.norm(gpos - obj_pos)
                    if d < best_dist:
                        best_dist = d
                        best_q = [q0, q1, 0.0, q3, 0.0, -0.5, 0.0]
                        best_gi = gi

    print(f"  Best config: q={[f'{x:.2f}' for x in best_q]}")
    print(f"  Best wrist-geom dist to obj: {best_dist:.4f}m (geom {best_gi})")

    # Set to best config and check
    for i in range(7):
        data.qpos[i] = best_q[i]
    data.qvel[:7] = 0.0
    mujoco.mj_forward(model, data)

    tip = data.site_xpos[0] if model.nsite > 0 else data.geom_xpos[16]  # tip_arml
    print(f"\n  At best config:")
    print(f"    tip_arml:  ({data.geom_xpos[16][0]:+.4f}, {data.geom_xpos[16][1]:+.4f}, {data.geom_xpos[16][2]:+.4f})")
    for gi in [13, 14, 15]:
        gpos = data.geom_xpos[gi]
        print(f"    geom {gi}:   ({gpos[0]:+.4f}, {gpos[1]:+.4f}, {gpos[2]:+.4f})")

    # Now simulate from this config with a push action
    print("\n  Simulating with push torque from best config...")
    obj_start = data.qpos[8:10].copy()  # obj slide joints (y, x offsets)
    for t in range(100):
        # Apply a torque that tries to push arm toward object
        data.ctrl[:] = np.zeros(7)
        data.ctrl[0] = 2.0  # shoulder pan to sweep
        mujoco.mj_step(model, data)

        if t % 10 == 0:
            obj_xy_now = np.array([0.45 + data.qpos[9], -0.05 + data.qpos[8]])
            disp = np.linalg.norm(obj_xy_now - obj_pos[:2])
            ncon = data.ncon
            contact_names = []
            for ci in range(ncon):
                c = data.contact[ci]
                g1 = model.geom(c.geom1).name
                g2 = model.geom(c.geom2).name
                contact_names.append(f"{g1}<->{g2}")
            wrist_geom_pos = data.geom_xpos[13]
            print(f"    t={t:3d}  wrist_fork=({wrist_geom_pos[0]:+.3f},{wrist_geom_pos[1]:+.3f},{wrist_geom_pos[2]:+.3f})  "
                  f"obj_disp={disp:.4f}m  ncon={ncon}  contacts={contact_names}")

    env.close()


if __name__ == "__main__":
    env = gym.make("Pusher-v5")
    obs, _ = env.reset(seed=42)
    dump_model_info(env)
    env.close()

    test_scripted_push()
    test_brute_force_push()
    test_position_control_push()
