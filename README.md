# Hybrid-Probabilistic-Control

GPU-accelerated hybrid controller combining a **Particle Filter** (state estimation) with **MPPI** (stochastic optimal control) under real-time deadline constraints. Implemented in Python/CuPy with raw CUDA kernels. Tested on MuJoCo **Pusher-v5** via Gymnasium.

The environment is made **partially observable**: the object position (`obs[17:19]`) is included in the PF observation but with a loose noise scale, so the particle filter must converge on the true position from a broad prior. This makes the particle filter genuinely necessary for accurate state estimation.

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
    ├── mppi.py                 # GPU-resident MPPI controller with spline CP sampling
    ├── runner.py               # CPU orchestration loop with timing instrumentation
    ├── envs/
    │   └── pusher.py           # PusherDynamics: RNEA numpy + CUDA implementations
    └── kernels/
        └── pusher_kernels.py   # Raw CUDA C kernel source strings
```

---

## Algorithm

The system solves a discrete-time partially observable control problem:

$$s_{t+1} = f(s_t, a_t) + \mathcal{N}(0, \Sigma_p), \qquad o_t = h(s_t) + \mathcal{N}(0, \Sigma_o)$$

A particle filter estimates the posterior over $s_t$; MPPI plans optimal actions over that posterior.

### Particle Filter (State Estimation)

The posterior $p(s_t \mid o_{0:t}, a_{0:t-1})$ is approximated by $N = 1000$ weighted particles $\{(s_i, w_i)\}$.

**Rao-Blackwellisation:** Each step, the true observed joint state (q, qdot) is injected into all particles with tiny jitter. Only the hidden state (object position) differs between particles. This prevents the approximate dynamics model from drifting and ensures the contact signal is the dominant discriminator in the weight update.

Each step has three stages:

**(1) Propagation** — draw each particle forward through the dynamics (parallel, one thread per particle):

$$s_t^{(i)} \sim p(s_t \mid s_{t-1}^{(i)}, a_{t-1}) = f(s_{t-1}^{(i)}, a_{t-1}) + \mathcal{N}(0, \sigma_p^2 I)$$

**(2) Weighting** — multiply each particle's weight by its observation likelihood. Only the hidden object-position dimensions contribute to discrimination; joint dims are injected and thus identical across particles:

$$w_i \propto w_i \cdot p(o_t \mid s_t^{(i)}) = w_i \cdot \exp\left(-\frac{\|h(s_t^{(i)}) - o_t\|^2}{2\sigma_o^2}\right)$$

Weights are renormalised to sum to one.

**(3) Resampling** — when ESS $= 1/\sum w_i^2$ drops below $0.5 \times N$, draw $N$ replacement particles using systematic resampling via `cp.cumsum` + `cp.searchsorted`. This runs entirely on-device.

### MPPI (Stochastic Optimal Control)

MPPI frames finite-horizon control as inference. Given a nominal sequence $\bar{u} = \{u_0, \ldots, u_{H-1}\}$ (initially zero, warm-started on subsequent steps):

**(1) Spline control-point sampling.** Sample $K = 1024$ perturbation sequences as $N_{CP} = 6$ control points per trajectory, then linearly interpolate to $H = 50$ steps. This reduces the effective noise dimension from $H \times 7 = 350$ to $N_{CP} \times 7 = 42$ and guarantees band-limited, smooth action traces. AR(1) temporal correlation ($\beta = 0.8$) is applied in control-point space for additional smoothness:

$$\varepsilon_{cp}^{(k)} \sim \mathcal{N}(0, \sigma^2 \cdot \text{diag}(\sigma_\text{joint}))^{N_{CP}}, \quad \varepsilon^{(k)} = \text{lerp}(\varepsilon_{cp}^{(k)}, H)$$

Joint 6 (wrist_roll) has $\sigma_\text{joint}[6] = 0$ and its nominal plan $\bar{u}_{:,6}$ is pinned to zero — it cannot transfer force to the object ($J_6 = 0$ always due to zero FK offset).

**(2)** Form perturbed candidates: $u^{(k)} = \text{clip}(\bar{u} + \varepsilon^{(k)}, -2, 2)$

**(3)** Seed all $K$ rollouts from the PF weighted mean state (single estimate, tiled). Using diverse particle samples as initial states was found to inject initial-condition noise that dominated the action-quality signal.

**(4)** Roll each candidate out through the RNEA dynamics and accumulate cost (parallel, one GPU thread per trajectory):

$$S_k = \sum_{t=0}^{H-1} c(s_t^{(k)}, u_t^{(k)}) + c_\text{terminal}(s_H^{(k)})$$

**(5)** Compute importance weights with numerically stable shift:

$$h_k = \frac{\exp\left(-(S_k - S_\min)/\lambda\right)}{\sum_j \exp\left(-(S_j - S_\min)/\lambda\right)}$$

**Elite filtering:** Only the top 30% lowest-cost trajectories contribute to the update — weights for the remaining 70% are zeroed before normalisation. This prevents low-quality rollouts from diluting the update signal.

**(6)** Update the nominal sequence:

$$\bar{u} \leftarrow \text{clip}\!\left(\bar{u} + \sum_{k=1}^{K} h_k \cdot \varepsilon^{(k)},\ -2,\ 2\right)$$

**(7)** Execute only $u_0$ (with EMA smoothing, $\alpha = 0.5$), then shift the horizon:

$$\{u_0, \ldots, u_{H-1}\} \to \{u_1, \ldots, u_{H-1}, 0\}$$

### Cost Function

The running cost is a two-term design with no action regularisation (spline CPs and frozen wrist handle smoothness and budget):

**Running cost (per step):**

$$c(s, a) = 10 \cdot d_\text{obj \to goal} + 5 \cdot d_\text{fork \to behind\_pt}$$

where `behind_pt` is a point 12 cm behind the object along the push axis ($\hat{d} = \text{unit}(\text{goal} - \text{obj})$):

$$\text{behind\_pt} = \text{obj\_pos} - 0.12 \cdot \hat{d}$$

This single distance term encodes both fork-object proximity **and** approach direction — being on the wrong side of the object naturally increases the distance to the behind-point.

**Terminal cost (applied once at step H):**

$$c_\text{terminal}(s_H) = 8 \times (10 \cdot d_\text{obj \to goal} + 5 \cdot d_\text{fork \to behind\_pt})$$

The terminal cost forces the planner to end trajectories in good configurations rather than just optimizing average cost.

### Observation Delay

The PF operates with a configurable delay ($d = 3$ steps). Observations are buffered; the PF updates against the observation from $d$ steps ago. For MPPI, the PF mean estimate is propagated forward through the $d$ recent actions on GPU (single particle, zero noise) to produce a current-time state estimate.

### Per-Step Execution Order

```
CPU (runner.py)                         GPU (CUDA kernels)
────────────────                        ──────────────────────────────────
gym obs (23-dim)
  → inject_observation: overwrite q/qdot + FK fork into all particles
  → pf.propagate(delayed_action)  ────► pf_propagate           (N threads)
  → pf.update(delayed_obs)        ────► pf_weight_update        (N threads)
  → resample if ESS < 0.5*N       ────► cumsum + searchsorted   (on-device)
  → estimate_gpu → propagate mean  ──► pf_propagate × d steps  (1 thread)
  → tile mean → K rollout starts   ──► (copy)
mppi.compute_action(states)        ──► mppi_rollout             (K threads × H steps)
                                       compute_importance_weights (K threads)
                                       weighted_eps_update       (H×7 threads)
                      action u_bar[0] ◄─
env.step(action) ──► obs, reward
```

Only ~29 floats cross the CPU-GPU bus per step regardless of $K$ or $N$.

---

## Pusher-v5 Environment

Pusher-v5 is a MuJoCo environment from Gymnasium in which a 7-DOF robotic arm must push a small cylinder to a fixed goal position on a table. The arm is controlled by 7 joint torques clipped to [-2, 2]; the object moves only through contact with the wrist fork (geoms 13-15). There is no gripper.

**Gymnasium observation (23-dim):** `[q(7), qdot(7), fingertip_xyz(3), obj_xyz(3), goal_xyz(3)]`

### Dynamics Model

The planning model uses **full RNEA (Recursive Newton-Euler Algorithm)** — not a simplified diagonal mass approximation:

- **Mass matrix:** 7×7 coupled mass matrix $M(q)$ computed via CRBA (column-wise RNEA), plus armature diagonal.
- **Forward dynamics:** Cholesky solve of $M(q)\ddot{q} = \tau - d\dot{q} - c(q, \dot{q})$.
- **Integration:** 5 semi-implicit Euler substeps per control step ($\Delta t_\text{inner} = 0.01$ s).
- **Joint limits:** All 7 joints have position limits matching the MJCF; velocity is zeroed on contact with limits.
- **Forward kinematics:** Full 7-DOF FK chain from arm base through all joint transforms. Returns the `r_wrist_roll_link` body origin, which is where the collision-enabled fork geoms are centered.
- **Contact:** When the 3D fork-object distance < 0.17 m, a push force proportional to the positive contact-normal component of the fork velocity is applied to the object. Newton's 3rd law reaction torques are applied to the arm via $J^T F$.

Both CPU (numpy) and GPU (CUDA) implementations exist and are kept in sync. RNEA link parameters (mass, CoM, inertia) are computed from the MJCF geom specifications at import time and baked into the CUDA source as `__device__ const` arrays.

**Wrist roll (joint 6) is frozen:** The FK offset from wrist_flex to wrist_roll is `[0,0,0]`, so $J_6 = 0$ always — action[6] cannot transfer force to the object. Sigma is zero, `u_bar[:,6]` is pinned, and the output action[6] is zeroed.

### Contact Stiffening

At episode start, `runner.py` modifies the MuJoCo model to improve contact fidelity:
- Object mass: 0.5 kg (heavier, needs real pushing)
- Slide joint damping: 2.0 (object stops quickly after contact)
- Newton solver with 100 iterations and tight tolerance
- Near-rigid contact parameters: `solref=[0.001, 1.0]`, `solimp=[0.99, 0.9999, 0.0001, 0.5, 2.0]`
- Tangential friction enabled (`condim=3`)

---

## File Reference

### `config.py`
Single `@dataclass` passed by reference to every component.

| Field | Default | Description |
|---|---|---|
| `N` | `1000` | Particle filter particle count |
| `process_noise_std` | `0.0` | Process noise std for joint dims (zero — q/qdot injected each step) |
| `process_noise_std_obj` | `0.01` | Process noise std for object-state dims |
| `obs_noise_std` | `0.01` | Observation likelihood noise std for joint dims |
| `obs_noise_std_obj` | `0.05` | Observation likelihood noise std for object position dims |
| `inject_noise_std` | `0.001` | Jitter on injected q/qdot to prevent degeneracy |
| `resample_threshold` | `0.5` | Resample when ESS < threshold × N |
| `K` | `1024` | MPPI trajectory samples |
| `H` | `50` | MPPI planning horizon (2.5 s look-ahead) |
| `lambda_` | `0.5` | MPPI temperature |
| `sigma` | `0.5` | MPPI perturbation global scale |
| `noise_beta` | `0.8` | AR(1) temporal correlation in CP space |
| `action_alpha` | `0.5` | EMA smoothing on output action |
| `N_CP` | `6` | Spline control points per rollout (noise dim = N_CP × 7) |
| `sigma_joint_weights` | `(1.5,1.2,1.0,2.0,0.8,0.8,0.0)` | Per-joint sigma multipliers (joint 6 frozen) |
| `elite_frac` | `0.3` | Fraction of lowest-cost rollouts used for u_bar update |
| `obs_delay` | `3` | PF receives observations from d steps ago |
| `sensor_noise_std` | `0.02` | Additive sensor noise std on observations |
| `max_steps` | `300` | Episode length cap |
| `dt` | `0.05` | Control timestep (frame_skip=5 × inner_dt=0.01) |
| `threads_per_block` | `256` | CUDA threads per block |
| `K_max` | `4096` | Max K; GPU buffers pre-allocated to this size |

### `dynamics.py`
Abstract base class (`AnalyticalDynamics`) that every environment must implement.

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
| `parallel_normalize(weights)` | In-place normalise; resets to uniform on underflow |
| `generate_normal(shape, mean, std)` | GPU Gaussian samples, float32 |
| `inclusive_scan(arr)` | `cp.cumsum` |
| `get_grid_block(N)` | Returns `(grid, block)` for 1-D kernel launches |

### `particle_filter.py`
Bootstrap SIR particle filter; full particle cloud lives on GPU.

- `initialize(obs)` — bootstraps N particles; object positions sampled from Pusher-v5 initial prior
- `inject_observation(obs)` — overwrites q/qdot + FK fork position in all particles (Rao-Blackwellisation)
- `propagate(action)` — `pf_propagate` kernel; advances all particles one step
- `update(obs)` — `pf_weight_update` kernel; compares particle predictions against observation
- `resample()` — systematic resampling via `cp.cumsum` + `cp.searchsorted`
- `estimate_gpu()` — weighted mean state on GPU for MPPI initialisation
- `effective_sample_size()` — returns $1/\sum w_i^2$

### `mppi.py`
GPU-resident MPPI. Nominal sequence `u_bar` of shape `(H, 7)` lives on GPU between steps. Buffers pre-allocated to `K_max`.

Key features:
- **Spline CP sampling:** noise sampled as `(K, N_CP, 7)` then linearly interpolated to `(K, H, 7)` with precomputed tables
- **AR(1) in CP space:** temporal correlation applied across control points
- **Elite filtering:** bottom 70% of rollouts zeroed before normalisation
- **Frozen wrist:** `u_bar[:,6]` pinned to zero, `action[6]` zeroed after EMA
- **EMA output smoothing:** `action = α·prev + (1-α)·raw` with α=0.5

Kernels (compiled once via `cp.RawModule`):

| Kernel | Threads | Description |
|---|---|---|
| `mppi_rollout` | 1 per trajectory | Rolls out H steps of RNEA dynamics, accumulates running + terminal cost |
| `compute_importance_weights` | 1 per trajectory | Numerically stable cost → weight conversion |
| `weighted_eps_update` | 1 per `(t, a)` pair | `u_bar_delta[t,a] = Σ_k w_k · eps[k,t,a]` |

### `kernels/pusher_kernels.py`
Raw CUDA C source strings concatenated with `__device__` code from `envs/pusher.py` before compilation.

| Kernel | Threads | Description |
|---|---|---|
| `pf_propagate` | 1 per particle | Applies `f_pusher` + process noise |
| `pf_weight_update` | 1 per particle | Gaussian likelihood over obj_pos dimensions |

### `runner.py`
CPU control loop with observation delay handling and diagnostic output.

Per-step timing entry:

| Key | Description |
|---|---|
| `T_total_ms` | Full step wall time |
| `T_gpu_ms` | GPU work + synchronize time |
| `T_env_ms` | `env.step()` time |
| `ESS` | Effective sample size |
| `K_used` | Active MPPI trajectory count |

---

## Installation

```bash
conda create -n hpc python=3.10
conda activate hpc
pip install gymnasium[mujoco] cupy-cuda12x numpy
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
| `--H` | `50` | Planning horizon |
| `--sigma` | `0.5` | MPPI perturbation global scale |
| `--lambda_` | `0.5` | MPPI temperature |
| `--steps` | `300` | Max steps per episode |
| `--device` | `0` | CUDA device index |
| `--render` | off | Open MuJoCo viewer |
| `--record` | off | Record MP4 video to `./videos/` |
| `--no-pf` | off | Bypass PF; give MPPI perfect state from obs |
| `--no-timing` | off | Suppress per-step stdout output |

```bash
# Default run (MPPI + PF)
python runner.py

# MPPI only, perfect state information
python runner.py --no-pf

# With MuJoCo viewer
python runner.py --render

# Record video
python runner.py --record

# Quick pipeline check
python runner.py --K 64 --N 200 --H 10
```

After the episode, `timing_log.npy` is written to the working directory:

```python
import numpy as np
log = np.load('timing_log.npy', allow_pickle=True)
```