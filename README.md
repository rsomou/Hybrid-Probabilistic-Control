# Hybrid-Probabilistic-Control

GPU-accelerated hybrid controller combining a **Particle Filter** (state estimation) with **MPPI** (stochastic optimal control) and a **deadline-aware adaptive scheduler**. Tested on MuJoCo **Pusher-v5** via Gymnasium.

The environment is **partially observable**: the object position is included in the PF observation but with a loose noise scale, so the particle filter must converge on the true position from a broad prior.

---

## Repository Structure

```
Hybrid-Probabilistic-Control/
└── mppi_pf_gpu/
    ├── config.py               All hyperparameters (PF, MPPI, scheduler)
    ├── dynamics.py             Abstract dynamics interface
    ├── gpu_utils.py            CuPy utility layer (reductions, RNG, scan)
    ├── particle_filter.py      GPU-resident bootstrap particle filter
    ├── mppi.py                 GPU-resident MPPI controller
    ├── deadline_scheduler.py   Adaptive K scheduler (deadline-aware)
    ├── runner.py               CPU orchestration loop + timing
    ├── envs/
    │   └── pusher.py           Pusher-v5 dynamics: RNEA (numpy + CUDA)
    └── kernels/
        └── pusher_kernels.py   Raw CUDA C kernel source strings
```

---

## Algorithm Overview

The system solves a partially observable control problem where the true object position is hidden. A particle filter estimates it; MPPI plans optimal actions over that estimate.

### Particle Filter (State Estimation)

The posterior over the state is approximated by N = 1000 weighted particles.

**Rao-Blackwellization:** Each step, the true observed joint state (q, qdot) is injected into all particles with tiny jitter. Only the hidden object position differs between particles. This keeps the one-step contact signal as the dominant discriminator for weight updates.

Each step has three stages:

1. **Propagation** - Advance each particle forward through the dynamics model on GPU (one thread per particle), then add Gaussian process noise to the object position dimensions.

2. **Weighting** - Multiply each particle's weight by a Gaussian likelihood comparing its predicted object position against the observed position. Joint dimensions are skipped (they are identical across particles after injection). Weights are renormalized to sum to one.

3. **Resampling** - When the effective sample size ESS = 1 / sum(w_i^2) drops below 0.5 * N, draw N replacement particles using systematic resampling via cumulative sum + binary search (entirely on GPU).

### MPPI (Stochastic Optimal Control)

MPPI treats finite-horizon control as inference. Given a nominal action sequence u_bar (warm-started from the previous step):

1. **Sample perturbations.** Draw K = 1024 noise sequences. Noise is sampled as (K, N_CP, 7) control points, then linearly interpolated to (K, H, 7) steps. AR(1) temporal correlation (beta = 0.5) is applied for smoothness. Per-joint sigma weights control exploration per joint; joint 6 (wrist_roll) is frozen at sigma = 0.

2. **Form candidates.** Perturbed actions: u_k = clip(u_bar + eps_k, -2, 2).

3. **Seed rollouts.** All K rollouts start from the PF weighted mean state (single estimate, tiled to K copies). Using diverse particle samples as initial states injected noise that dominated the action signal.

4. **Roll out dynamics.** Each of the K threads rolls out H steps of RNEA dynamics on GPU, accumulating running cost plus a terminal cost at step H.

5. **Compute importance weights.** Numerically stable conversion: w_k = exp(-(S_k - S_min) / lambda). **Elite filtering** zeroes the bottom 70% of rollouts before normalization so only the top 30% lowest-cost trajectories contribute.

6. **Update nominal sequence.** u_bar += sum_k(w_k * eps_k), then clip to action bounds. Joint 6 is pinned to zero.

7. **Execute and shift.** Apply only u_bar[0] (with EMA smoothing, alpha = 0.3), then shift the horizon forward.

### Cost Function

Two-term running cost with no action regularization:

- **Goal term:** 200 * d(obj, goal)^2 - drives the object toward the target
- **Approach term:** 15 * d(fork, obj) in 3D - pulls the fork toward the object (includes z so the arm descends to table height)
- **Joint limit barrier:** 0.5 * sum of exponential penalties near each joint limit

The terminal cost at step H is 10x the running cost, forcing the planner to end trajectories in good configurations.

### Observation Delay

The PF operates with a configurable delay (d = 3 steps). Observations are buffered; the PF updates against the observation from d steps ago. For MPPI, the PF mean is propagated forward through the d recent actions on GPU to produce a current-time state estimate.

---

## CPU-GPU Interaction

Each control step follows this sequence. All heavy computation runs on the GPU; the CPU only orchestrates the order of kernel launches and handles the environment step.

```
CPU (runner.py)                          GPU (CUDA kernels)

 1. Receive gym observation              
    (23 floats from env)                 
                                         
 2. Inject observed q/qdot       -----> [pf_propagate]        N threads
    into all particles                   one thread per particle
                                         advances dynamics + adds noise
                                         
 3. Weight update                -----> [pf_weight_update]    N threads
    compare predicted obj_pos            one thread per particle
    against observed obj_pos             computes Gaussian log-likelihood
                                         
 4. Resample if ESS too low      -----> [cumsum + searchsorted]
    (entirely on-device)                 systematic resampling
                                         
 5. Estimate PF mean state       -----> [weighted average]
    Propagate through d recent           pf_propagate x d (1 thread)
    actions for delay compensation       
                                         
 6. Tile mean state to K copies   ----> [GPU copy]
    to seed MPPI rollouts                
                                         
 7. MPPI planning                -----> [mppi_rollout]        K threads
    K parallel trajectory rollouts       each thread: H steps of RNEA
    accumulate cost per trajectory       dynamics + cost accumulation
                                         
                                 -----> [importance_weights]  K threads
                                         cost-to-weight conversion
                                         
                                 -----> [weighted_eps_update] H*7 threads
                                         u_bar += weighted noise
                                         
 8. Receive action               <----- u_bar[0] (7 floats)
    (7 floats from GPU)                  
                                         
 9. env.step(action)                     
    get next observation                 
```

**Bus traffic per step:** about 30 floats cross the CPU-GPU boundary regardless of K or N.

---

## Deadline-Aware Adaptive Scheduler

When `--deadline-aware` is enabled, the scheduler tunes the number of MPPI trajectory samples (K) each step to meet a wall-clock deadline. This replaces the fixed K with an adaptive K that balances planning quality against time pressure.

### How It Works

The scheduler maintains three signals:

1. **Latency model.** An exponentially smoothed estimate of GPU cost per trajectory (ms/trajectory), updated each step from the measured planning time divided by K used.

2. **Overhead model.** A smoothed estimate of non-MPPI time (particle filter, env step, delay propagation). This is subtracted from the deadline to get the MPPI budget.

3. **Uncertainty signal.** The MPPI weight distribution from the previous step, summarized as w_eff / K (the effective sample size ratio). When weights are diffuse (ratio near 1.0), the planner is uncertain and benefits from more samples. When concentrated (ratio near 0), the planner is confident and can use fewer.

### Per-Step Decision

```
budget_ms = deadline - overhead - safety_margin
K_base    = budget_ms / cost_per_trajectory
K_scaled  = K_base * uncertainty_multiplier
K_next    = clamp(round_to_64(K_scaled), K_min, K_max)
```

The uncertainty multiplier maps the w_eff ratio linearly:
- Concentrated weights (confident): scale K down to 0.5x
- Diffuse weights (uncertain): scale K up to 1.5x

K is rounded to multiples of 64 for GPU warp alignment.

### Scheduler Parameters

| Parameter | Default | Description |
|---|---|---|
| `--deadline` | 50.0 | Per-step wall-clock deadline (ms) |
| `--deadline-aware` | off | Enable adaptive K scheduling |
| `K_min` | 64 | Minimum K (quality floor) |
| `K_max` | 4096 | Maximum K (GPU memory ceiling) |
| `safety_margin_ms` | 2.0 | Headroom subtracted from deadline |

### Example Output

With the scheduler enabled, per-step output shows the adapted K and deadline status:

```
Step   42 | R= -0.612 | T= 47.3ms | ESS=  842/1000 | K= 1280 [OK]
Step   43 | R= -0.589 | T= 51.2ms | ESS=  756/1000 | K= 1344 [OVER]
Step   44 | R= -0.601 | T= 43.8ms | ESS=  901/1000 | K=  960 [OK]
```

The episode summary includes scheduler statistics:

```
Scheduler     : K mean=1152  range=[640, 2048]  std=312
                cost/traj=0.0342 ms  overhead=4.21 ms
```

---

## Pusher-v5 Environment

Pusher-v5 is a MuJoCo environment where a 7-DOF robotic arm pushes a small cylinder to a fixed goal on a table. The arm is controlled by 7 joint torques clipped to [-2, 2]; the object moves only through contact with the wrist fork (geoms 13-15).

**Observation (23-dim):** q(7), qdot(7), fingertip_xyz(3), obj_xyz(3), goal_xyz(3)

### Dynamics Model

The planning model uses **full RNEA (Recursive Newton-Euler Algorithm)**:

- **Mass matrix:** 7x7 coupled mass matrix M(q) computed via CRBA, plus armature diagonal
- **Forward dynamics:** Cholesky solve of M(q) * qddot = tau - damping * qdot - bias(q, qdot)
- **Integration:** 5 semi-implicit Euler substeps per control step (inner dt = 0.01 s)
- **Joint limits:** All 7 joints have position limits matching the MJCF; velocity zeroed on contact
- **FK:** Full 7-DOF forward kinematics returning the r_wrist_roll_link body origin (fork collision center)
- **Contact:** When 3D fork-to-object distance < 0.17 m, a push force proportional to fork velocity in the contact normal direction is applied to the object

Both CPU (numpy) and GPU (CUDA) implementations exist. RNEA link parameters are computed from MJCF geom specs at import time and baked into the CUDA source as device constants.

### Contact Stiffening

At episode start, runner.py modifies the MuJoCo model:

- Object mass: 0.5 kg
- Slide joint damping: 2.0
- Newton solver, 100 iterations, tolerance 1e-10
- Near-rigid contacts: solref = [0.001, 1.0], solimp = [0.99, 0.9999, 0.0001, 0.5, 2.0]
- Tangential friction enabled (condim = 3)

---

## Configuration Reference

All hyperparameters live in `config.py` as a single dataclass.

| Field | Default | Description |
|---|---|---|
| `N` | 1000 | Particle filter particle count |
| `process_noise_std` | 0.0 | Process noise for joint dims (zero since q/qdot is injected) |
| `process_noise_std_obj` | 0.01 | Process noise for object position dims |
| `obs_noise_std` | 0.01 | Likelihood noise for joint dims |
| `obs_noise_std_obj` | 0.05 | Likelihood noise for object position |
| `inject_noise_std` | 0.001 | Jitter on injected q/qdot |
| `resample_threshold` | 0.5 | Resample when ESS < threshold * N |
| `K` | 1024 | MPPI trajectory samples (initial; adapted if scheduler active) |
| `H` | 20 | MPPI planning horizon |
| `lambda_` | 200.0 | MPPI temperature |
| `sigma` | 0.8 | MPPI perturbation scale |
| `noise_beta` | 0.5 | AR(1) temporal correlation |
| `action_alpha` | 0.3 | EMA smoothing on output action |
| `N_CP` | 6 | Spline control points per rollout |
| `sigma_joint_weights` | (1.5, 1.2, 1.0, 2.0, 0.8, 0.8, 0.0) | Per-joint sigma multipliers |
| `elite_frac` | 0.3 | Fraction of lowest-cost rollouts used |
| `obs_delay` | 3 | PF observation delay (steps) |
| `sensor_noise_std` | 0.02 | Additive sensor noise |
| `max_steps` | 300 | Episode length cap |
| `dt` | 0.05 | Control timestep (5 inner steps at 0.01 s) |
| `deadline_ms` | 50.0 | Per-step deadline (ms) |
| `K_min` | 64 | Minimum K for scheduler |
| `K_max` | 4096 | Maximum K for scheduler |
| `safety_margin_ms` | 2.0 | Scheduler headroom |

---

## Installation

```bash
conda create -n hpc python=3.10
conda activate hpc
pip install gymnasium[mujoco] cupy-cuda12x numpy
```

Requires an NVIDIA GPU with compute capability >= 6.0 (Pascal or newer).

---

## Running

```bash
cd mppi_pf_gpu
python runner.py [flags]
```

| Flag | Default | Description |
|---|---|---|
| `--K` | 1024 | MPPI trajectory samples |
| `--N` | 1000 | Particle filter particles |
| `--H` | 20 | Planning horizon |
| `--sigma` | 0.5 | Perturbation scale |
| `--lambda_` | 200.0 | MPPI temperature |
| `--deadline` | 50.0 | Per-step deadline (ms) |
| `--steps` | 300 | Max steps per episode |
| `--device` | 0 | CUDA device index |
| `--render` | off | Open MuJoCo viewer |
| `--record` | off | Record MP4 to ./videos/ |
| `--no-pf` | off | Bypass PF, give MPPI perfect state |
| `--no-timing` | off | Suppress per-step output |
| `--deadline-aware` | off | Enable adaptive K scheduling |

### Examples

```bash
# Default (MPPI + PF, fixed K)
python runner.py

# MPPI only, perfect state
python runner.py --no-pf

# Deadline-aware scheduling (K adapts to meet 50ms deadline)
python runner.py --deadline-aware --deadline 50

# With viewer
python runner.py --render

# Record video
python runner.py --record

# Quick test
python runner.py --K 64 --N 200 --H 10
```

### Timing Log

After each episode, `timing_log.npy` is saved:

```python
import numpy as np
log = np.load('timing_log.npy', allow_pickle=True)
# Each entry: step, T_total_ms, T_gpu_ms, T_env_ms, ESS, K_used, reward
```
