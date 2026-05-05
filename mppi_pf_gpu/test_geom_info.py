import gymnasium as gym
import numpy as np
import mujoco

env = gym.make("Pusher-v5", max_episode_steps=300)
obs, _ = env.reset(seed=42)
model = env.unwrapped.model
data  = env.unwrapped.data

for gi in [13, 14, 15, 16, 17]:
    bid = model.geom_bodyid[gi]
    bname = model.body(bid).name
    gname = model.geom(gi).name
    gpos = model.geom_pos[gi]
    gsize = model.geom_size[gi]
    print(f"geom {gi}: name={gname!r:10s} body={bname:25s} local_pos={gpos} size={gsize}")

mujoco.mj_forward(model, data)
print()
print(f"r_wrist_roll_link body xpos at q=0: {data.xpos[9]}")
print(f"tips_arm body xpos at q=0:          {data.xpos[10]}")
for gi in [13, 14, 15, 16, 17]:
    print(f"geom {gi} world xpos at q=0: {data.geom_xpos[gi]}")

print()
body_pos = data.xpos[9]
for gi in [13, 14, 15]:
    offset = data.geom_xpos[gi] - body_pos
    print(f"geom {gi} offset from wrist_roll body: {offset}  dist={np.linalg.norm(offset):.4f}")

env.close()
