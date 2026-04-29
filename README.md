# Hybrid-Probabilistic-Control

GPU-accelerated hybrid controller combining a **Particle Filter** (state estimation) with **MPPI** (stochastic optimal control) under real-time deadline constraints. Implemented in Python/CuPy with raw CUDA kernels. Tested on MuJoCo **Pusher-v5** via Gymnasium.

The environment is made **partially observable** (Option 1): the object position (`obs[17:19]`) is masked from the controller, so the particle filter must infer where the object is from indirect cues — contact forces that perturb the arm's joint state. This makes the particle filter genuinely necessary rather than decorative.

---

## Repository Structure

```
Hybrid-Probabilistic-Control/
├── README.md
└── mppi_pf_gpu/
    ├── config.py               # All hyperparameters and scheduler placeholders
    ├── dynamics.py             # Abstract interface every environment must implement
    ├── gpu_utils.py            # CuPy utility layer (reductions, RNG, scan, grid dims)
    ├── particle_filter.py      # GPU-resident bootstrap particle filter
    ├── mppi.py                 # GPU-resident MPPI controller
    ├── runner.py               # CPU orchestration loop with timing instrumentation
    ├── envs/
    │   └── pusher.py           # PusherDynamics: numpy + CUDA implementations
    └── kernels/
        └── pusher_kernels.py   # Raw CUDA C kernel source strings
```

---

## Algorithm

The system solves a discrete-time partially observable control problem:

$$s_{t+1} = f(s_t, a_t) + \mathcal{N}(0, \Sigma_p), \qquad o_t = h(s_t) + \mathcal{N}(0, \Sigma_o)$$

The true state $s_t$ is never observed directly. A particle filter estimates the posterior over $s_t$; MPPI plans optimal actions over that posterior.

### Particle Filter (State Estimation)

The posterior $p(s_t \mid o_{0:t}, a_{0:t-1})$ is approximated by $N$ weighted particles $\{(s_i, w_i)\}$.

Each step has three stages:

**(1) Propagation** -- draw each particle forward through the dynamics (parallel, one thread per particle):

$$s_t^{(i)} \sim p(s_t \mid s_{t-1}^{(i)}, a_{t-1}) = f(s_{t-1}^{(i)}, a_{t-1}) + \mathcal{N}(0, \sigma_p^2 I)$$

**(2) Weighting** -- multiply each particle's weight by its observation likelihood (parallel, one thread per particle):

$$w_i \propto w_i \cdot p(o_t \mid s_t^{(i)}) = w_i \cdot \exp\left(-\frac{\|h(s_t^{(i)}) - o_t\|^2}{2\sigma_o^2}\right)$$

Weights are renormalised to sum to one.

**(3) Resampling** -- when ESS $= 1/\sum w_i^2$ drops below a threshold, draw $N$ replacement particles using systematic resampling. The weight CDF is computed via `cp.cumsum`; stratum boundaries are resolved via `cp.searchsorted`. This is the only collective step and runs entirely on-device.

### MPPI (Stochastic Optimal Control)

MPPI frames finite-horizon control as inference. Given a nominal sequence $\bar{u} = \{u_0, \ldots, u_{H-1}\}$ (initially zero, warm-started on subsequent steps), each call to `compute_action` does the following:

**(1)** Sample $K$ perturbation sequences, each of length $H$:

$$\varepsilon^{(k)} \sim \mathcal{N}(0, \sigma^2 I)^H, \quad k = 1, \ldots, K$$

**(2)** Form perturbed candidates: $u^{(k)} = \bar{u} + \varepsilon^{(k)}$

**(3)** Draw $K$ initial states $x_0^{(k)}$ from the particle filter belief.

**(4)** Roll each candidate out through the dynamics and accumulate cost (parallel, one GPU thread per trajectory):

$$S_k = \sum_{t=0}^{H-1} c(s_t^{(k)}, u_t^{(k)}), \qquad s_{t+1}^{(k)} = f(s_t^{(k)}, u_t^{(k)})$$

**(5)** Compute importance weights. Subtracting $S_\min$ before exponentiation prevents overflow and is mathematically neutral (it cancels in the normalisation):

$$h_k = \frac{\exp\left(-(S_k - S_\min)/\lambda\right)}{\sum_j \exp\left(-(S_j - S_\min)/\lambda\right)}$$

Temperature $\lambda$ controls selectivity: $\lambda \to 0$ concentrates weight on the minimum-cost trajectory; $\lambda \to \infty$ averages all trajectories equally.

**(6)** Update the nominal sequence:

$$\bar{u} \leftarrow \bar{u} + \sum_{k=1}^{K} h_k \cdot \varepsilon^{(k)}$$

**(7)** Execute only $u_0$, then shift the horizon:

$$\{u_0, \ldots, u_{H-1}\} \to \{u_1, \ldots, u_{H-1}, 0\}$$

This warm-start means each optimisation step begins from the previous solution.

### Per-Step Execution Order

```
CPU (runner.py)                         GPU (CUDA kernels)
────────────────                        ──────────────────────────────────
gym obs (23-dim)                        
  → gym_obs_to_pf_obs() strips to 14-dim (q, qdot only; obj_pos HIDDEN)
pf_obs (14 floats) ──────────────────► pf_weight_update   (N threads)
                                         pf resampling      (cumsum + searchsorted)
                                         pf.sample(K)       (gather K particles)
                      K initial states ◄─
mppi.compute_action(states) ──────────► mppi_rollout        (K threads × H steps)
                                         compute_importance_weights  (K threads)
                                         weighted_eps_update  (H × 7 threads)
                      action u_bar[0] ◄─
env.step(action) ──► obs, reward
pf.propagate(action) ───────────────►   pf_propagate        (N threads)
```

Only ~29 floats cross the CPU-GPU bus per step (14 pf_obs in, 7 action in, 7 action out, 1 ESS scalar) regardless of $K$ or $N$. All particles, weights, perturbations, and rollout costs stay resident on-device for the full episode.

---

## Pusher-v5 Environment

Pusher-v5 is a MuJoCo environment from Gymnasium in which a 7-DOF planar robotic arm must push a small cylinder to a fixed goal position on a table. The arm is controlled by joint torques; the object moves only through contact with the fingertip. There is no gripper — the task requires the controller to plan an approach trajectory, make contact, and push the object to the goal without being able to grasp it.

**Gymnasium observation (23-dim):** `[q(7), qdot(7), fingertip_xyz(3), obj_xyz(3), goal_xyz(3)]`

The raw gym observation is a flattened vector of joint angles, joint velocities, and 3-D Cartesian positions of the fingertip, object, and goal.

### Partial Observability (Option 1)

The controller **never sees the object position**. Before any observation reaches the particle filter or MPPI, `gym_obs_to_pf_obs()` strips it down to 14 dimensions:

| Gym obs indices | Content | Given to PF? |
|---|---|---|
| `[0:7]` | Joint angles q | ✅ Yes |
| `[7:14]` | Joint velocities qdot | ✅ Yes |
| `[14:17]` | Fingertip position (x,y,z) | ❌ Skipped |
| `[17:20]` | Object position (x,y,z) | ❌ **Hidden** |
| `[20:23]` | Goal position (x,y,z) | ❌ Skipped (read once at reset) |

The particle filter must **infer** the object position from indirect evidence: when the fingertip contacts the object, the resulting forces perturb the arm's joint state differently depending on where the object is. Particles with incorrect object position predictions will produce joint-state predictions that diverge from the measured joint state, causing their weights to decrease. This makes the particle filter genuinely necessary.

At episode start, particles sample object positions from the Pusher-v5 initial prior ($x \sim U[-0.3, 0.0]$, $y \sim U[-0.2, 0.2]$) rather than peeking at the true position.

**Internal state tracked by the particle filter (18-dim):** `[q(7), qdot(7), obj_pos_2d(2), obj_vel_2d(2)]`

The object is constrained to the table plane, so its 3-D position reduces to 2-D. Object velocity is not directly observable but is needed to integrate the contact dynamics, so it is maintained as part of the latent state.

**Dynamics (planning model):** The real MuJoCo simulator uses a full coupled inertia matrix. For GPU parallel rollouts, a diagonal mass-matrix approximation is used instead, which keeps each joint independent and avoids an $O(n^3)$ matrix solve per step:

$$\ddot{q}_i = (\tau_i - d \cdot \dot{q}_i) / m_i, \qquad \dot{q}^+ = \dot{q} + \ddot{q} \, \Delta t, \qquad q^+ = q + \dot{q}^+ \, \Delta t$$

with damping $d = 0.1$, mass $m_i = 1.0$ per joint, and $\Delta t = 0.05$ s (semi-implicit Euler).

**Contact model:** Forward kinematics computes the fingertip position as a sum of planar link vectors. If the fingertip is within `CONTACT_RADIUS = 0.06 m` of the object, a push impulse proportional to the positive contact-normal component of the fingertip velocity is applied to the object. The velocity component is clamped to zero so contact can only push, never pull. The contact normal is the unit vector from the fingertip to the object center.

**Running cost (negated Pusher-v5 reward):**

The Pusher-v5 reward is $r = -0.5\|\text{tip} - \text{obj}\| - 1.0\|\text{obj} - \text{goal}\| - 0.1\|a\|^2$. MPPI minimises cost = $-r$:

$$c(s, a) = 0.5 \|\text{tip} - \text{obj}\| + 1.0 \|\text{obj} - \text{goal}\| + 0.1 \|a\|^2$$

The first term encourages the arm to stay near the object; the second penalises object distance from the goal; the third penalises large torques. The CUDA version in `kernels/pusher_kernels.py` uses `sinf`/`cosf`/`sqrtf` and must remain numerically identical to the numpy version in `envs/pusher.py`.

---

## File Reference

### `config.py`
Single `@dataclass` passed by reference to every component. No global state.

| Field | Default | Description |
|---|---|---|
| `env_name` | `"Pusher-v5"` | Gymnasium environment ID |
| `N` | `1000` | Particle filter particle count |
| `process_noise_std` | `0.005` | Process noise std for joint dims (tight: arm well-modelled) |
| `process_noise_std_obj` | `0.05` | Process noise std for object-state dims (loose: contact uncertain) |
| `obs_noise_std` | `0.05` | Observation likelihood noise std for joint dims |
| `obs_noise_std_obj` | `0.1` | Kept for kernel signature; unreachable with OBS_DIM=14 |
| `resample_threshold` | `0.5` | Resample only when ESS < threshold × N |
| `K` | `1024` | MPPI trajectory samples |
| `H` | `30` | MPPI planning horizon |
| `lambda_` | `1.0` | MPPI temperature |
| `sigma` | `0.5` | MPPI perturbation std |
| `max_steps` | `300` | Episode length cap |
| `dt` | `0.05` | Integration timestep (frame_skip=5 × inner_dt=0.01) |
| `threads_per_block` | `256` | CUDA threads per block |
| `device_id` | `0` | CUDA device index |
| `deadline_ms` | `50.0` | Per-step deadline (for future scheduler) |
| `enable_timing` | `True` | Print per-step timing to stdout |
| `K_min` | `64` | Minimum K for adaptive scheduler |
| `K_max` | `4096` | Maximum K; GPU buffers pre-allocated to this size |
| `safety_margin_ms` | `2.0` | Buffer subtracted from deadline before scheduling |

### `dynamics.py`
Abstract base class (`AnalyticalDynamics`) that every environment must implement. Nothing in `particle_filter.py` or `mppi.py` imports from `envs/` directly.

| Method | Role |
|---|---|
| `f_numpy(state, action)` | Single-step CPU dynamics (validation/testing) |
| `cost_numpy(state, action)` | Single-step CPU cost (must match CUDA version) |
| `obs_model(state)` | Maps internal state to predicted observation |
| `sample_initial_particles(obs, N)` | Bootstrap particle cloud from first observation |
| `get_cuda_dynamics_code()` | Returns CUDA C `__device__` source for dynamics and cost |

### `gpu_utils.py`
Stateless CuPy utilities. All GPU math goes through this layer.

| Method | Description |
|---|---|
| `parallel_reduce_sum(arr)` | `float(cp.sum(arr))` |
| `parallel_normalize(weights)` | In-place normalise; resets to uniform on underflow |
| `generate_normal(shape, mean, std)` | GPU Gaussian samples, float32 |
| `inclusive_scan(arr)` | `cp.cumsum` |
| `get_grid_block(N)` | Returns `(grid, block)` for 1-D kernel launches |
| `synchronize()` | `cp.cuda.Device.synchronize()` |

### `particle_filter.py`
Bootstrap SIR particle filter; full particle cloud lives on GPU.

- `initialize(obs)` -- bootstraps N particles from first 23-dim gym obs; object positions sampled from prior (not peeked from obs)
- `update(obs)` -- calls `gym_obs_to_pf_obs()` to strip obs to 14-dim, then runs `pf_weight_update` kernel comparing particle joint-state predictions against measured joint state; normalises weights
- `resample()` -- systematic resampling via `cp.cumsum` + `cp.searchsorted`; resets weights to uniform
- `sample(K)` -- draw K particles proportional to weights for MPPI initialisation
- `propagate(action)` -- `pf_propagate` kernel; called after `env.step()`
- `effective_sample_size()` -- returns $1/\sum w_i^2 \in [1, N]$; computed **before** `resample()` (resampling resets weights, making ESS trivially $N$)

### `mppi.py`
GPU-resident MPPI. Nominal sequence `u_bar` of shape `(H, action_dim)` lives on GPU between steps. Buffers are pre-allocated to `K_max`; `compute_action(states, K=N)` slices views with no runtime reallocation.

Kernels (compiled once via `cp.RawModule`):

| Kernel | Threads | Description |
|---|---|---|
| `mppi_rollout` | 1 per trajectory | Rolls out H steps, accumulates cost into `costs[k]` |
| `compute_importance_weights` | 1 per trajectory | Converts `costs[k]` to weights with numerically stable shift |
| `weighted_eps_update` | 1 per `(t, a)` pair | `u_bar_delta[t,a] = sum_k w_k * eps[k,t,a]` |

### `kernels/pusher_kernels.py`
Raw CUDA C source strings concatenated with the `__device__` code from `envs/pusher.py` before compilation. Exports `ALL_PF_KERNELS` and `ALL_MPPI_KERNELS`.

Additional kernels:

| Kernel | Threads | Description |
|---|---|---|
| `pf_propagate` | 1 per particle | Applies `f_pusher` + process noise (tight for joints, loose for object) |
| `pf_weight_update` | 1 per particle | Compares particle's predicted joint state (14-dim) against measured joint state; multiplies weight by Gaussian likelihood. Object position is hidden — only joint q/qdot are compared. |

### `runner.py`
CPU control loop. Constructs all components, runs the episode, saves `timing_log.npy`.

Control loop order per step:
1. `pf.update(obs)` -- weight update
2. `ess = pf.effective_sample_size()` -- before resampling
3. `pf.resample()`
4. `states = pf.sample(K)` -- seed MPPI
5. `action = mppi.compute_action(states)` + `Device.synchronize()`
6. `obs, reward = env.step(action)`
7. `pf.propagate(action)`

Timing log entry per step:

| Key | Description |
|---|---|
| `step` | Step index |
| `T_total_ms` | Full step wall time |
| `T_gpu_ms` | GPU work + synchronize time |
| `T_env_ms` | `env.step()` time |
| `ESS` | Effective sample size |
| `K_used` | Active MPPI trajectory count |
| `reward` | Gymnasium reward |
| `deadline_ms` | Config deadline |
| `safety_margin_ms` | Config safety margin |

---

## Installation

```bash
conda create -n hpc python=3.10
conda activate hpc
pip install gymnasium[mujoco]
pip install cupy-cuda12x   # match your CUDA version (cuda11x, cuda12x, etc.)
pip install numpy
```

GPU requirement: NVIDIA GPU with compute capability >= 6.0 (Pascal or newer).

---

## Running

```bash
cd mppi_pf_gpu
python runner.py [flags]
```

| Flag | Default | Description |
|---|---|---|
| `--K` | `1024` | MPPI trajectory samples |
| `--N` | `1000` | Particle filter particles |
| `--H` | `30` | Planning horizon |
| `--deadline` | `50.0` | Per-step deadline in ms |
| `--sigma` | `0.5` | MPPI perturbation std |
| `--lambda_` | `1.0` | MPPI temperature |
| `--steps` | `300` | Max steps per episode |
| `--device` | `0` | CUDA device index |
| `--render` | off | Open MuJoCo viewer |
| `--no-timing` | off | Suppress per-step stdout output |

```bash
# Default run
python runner.py

# High quality with viewer
python runner.py --K 2048 --N 2000 --H 40 --render

# Quick pipeline check
python runner.py --K 64 --N 200 --H 10

# Tight deadline stress test
python runner.py --K 4096 --deadline 30.0
```

Per-step timing is printed to stdout when `--no-timing` is not set. After the episode, `timing_log.npy` is written to the working directory and can be loaded for offline analysis:

```python
import numpy as np
log = np.load('timing_log.npy', allow_pickle=True)
```
