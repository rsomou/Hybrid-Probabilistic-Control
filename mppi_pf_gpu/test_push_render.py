#!/usr/bin/env python3
"""
test_push_render.py — Visual demo: pre-position arm near puck, then sweep to push it.
Uses the configuration that was proven to move the puck (4.2cm displacement).
"""

import gymnasium as gym
import numpy as np
import mujoco
import time


env = gym.make("Pusher-v5", max_episode_steps=300, render_mode="human")
obs, _ = env.reset(seed=42)
model = env.unwrapped.model
data  = env.unwrapped.data

obj_start = obs[17:19].copy()
print(f"Object start: ({obj_start[0]:+.4f}, {obj_start[1]:+.4f})")

# Pre-position the arm so the wrist fork is right next to the puck.
# This config was found by scanning joint space for closest wrist-geom-to-object distance.
best_q = [0.221, 0.914, 0.0, -0.986, 0.0, -0.75, 0.0]
for i in range(7):
    data.qpos[i] = best_q[i]
data.qvel[:7] = 0.0
mujoco.mj_forward(model, data)

# Let the viewer show the starting pose
env.render()
time.sleep(1.0)

# Sweep shoulder pan to push the wrist fork through the object
print("\nSweeping arm through object...")
for t in range(200):
    action = np.array([2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    obs, reward, term, trunc, _ = env.step(action)

    if t % 10 == 0:
        obj_now = obs[17:19]
        disp = np.linalg.norm(obj_now - obj_start)
        tip = obs[14:17]
        print(f"  t={t:3d}  obj=({obj_now[0]:+.3f},{obj_now[1]:+.3f})  "
              f"disp={disp:.4f}m  tip_z={tip[2]:+.3f}")

    if term or trunc:
        break

final_disp = np.linalg.norm(obs[17:19] - obj_start)
print(f"\nFinal displacement: {final_disp:.4f}m")

# Hold the viewer open for a moment
time.sleep(2.0)
env.close()
