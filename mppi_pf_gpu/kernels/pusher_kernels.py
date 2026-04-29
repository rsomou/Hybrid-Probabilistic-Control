"""
kernels/pusher_kernels.py
CUDA kernel source strings for the Pusher-v5 environment.

All strings here are concatenated with the dynamics device code from
envs/pusher.py before being passed to cp.RawKernel for JIT compilation.
They must NOT be compiled standalone — they rely on device functions and
#defines provided by PusherDynamics.get_cuda_dynamics_code().

Layout reference
----------------
Particles / states:  row-major (N, STATE_DIM)  or  (K, STATE_DIM)
Perturbations eps:   row-major (K, H, ACTION_DIM)
u_bar:               row-major (H, ACTION_DIM)
"""

# --------------------------------------------------------------------------- #
# MPPI rollout kernel
# Each CUDA thread handles one trajectory sample k ∈ [0, K).
# Rolls out H steps and accumulates the running cost.
# --------------------------------------------------------------------------- #
MPPI_ROLLOUT_KERNEL = r"""
extern "C" __global__
void mppi_rollout(
    const float* __restrict__ initial_states,  // (K, STATE_DIM)
    const float* __restrict__ u_bar,           // (H, ACTION_DIM)
    const float* __restrict__ eps,             // (K, H, ACTION_DIM)
    const float* __restrict__ action_low,      // (ACTION_DIM,)
    const float* __restrict__ action_high,     // (ACTION_DIM,)
    const float* __restrict__ target,          // (2,) target position
    float*       __restrict__ costs,           // (K,)  output
    float dt,
    int K,
    int H
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    // Load initial state into registers
    float state[STATE_DIM];
    for (int i = 0; i < STATE_DIM; i++) {
        state[i] = initial_states[k * STATE_DIM + i];
    }

    float total_cost = 0.0f;

    for (int t = 0; t < H; t++) {
        // Compute clipped perturbed action: u_bar[t] + eps[k, t, :]
        float action[ACTION_DIM];
        int eps_base = (k * H + t) * ACTION_DIM;
        int u_base   = t * ACTION_DIM;
        for (int a = 0; a < ACTION_DIM; a++) {
            float u = u_bar[u_base + a] + eps[eps_base + a];
            action[a] = fminf(fmaxf(u, action_low[a]), action_high[a]);
        }

        // Accumulate running cost before state transition
        total_cost += cost_pusher(state, action, target);

        // Advance state in place
        f_pusher(state, action, dt);
    }

    costs[k] = total_cost;
}
"""

# --------------------------------------------------------------------------- #
# MPPI importance-weight kernel
# Converts per-trajectory costs to unnormalised importance weights using the
# numerically-stable shift:  w_k = exp(-(S_k - S_min) / lambda)
# --------------------------------------------------------------------------- #
COMPUTE_IMPORTANCE_WEIGHTS_KERNEL = r"""
extern "C" __global__
void compute_importance_weights(
    const float* __restrict__ costs,    // (K,)
    float*       __restrict__ weights,  // (K,)  output
    float lambda_,
    float min_cost,                     // pre-computed, passed from host
    int K
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    weights[k] = expf(-(costs[k] - min_cost) / lambda_);
}
"""

# --------------------------------------------------------------------------- #
# MPPI weighted epsilon accumulation kernel
# Computes u_bar_delta[t, a] = sum_k( w_k * eps[k, t, a] )
# Each thread handles one (t, a) pair — the inner loop runs over K.
# For large K a parallel reduction would be faster; this is correct and simple.
# --------------------------------------------------------------------------- #
WEIGHTED_EPS_UPDATE_KERNEL = r"""
extern "C" __global__
void weighted_eps_update(
    const float* __restrict__ weights,      // (K,)   normalised
    const float* __restrict__ eps,          // (K, H, ACTION_DIM)
    float*       __restrict__ u_bar_delta,  // (H, ACTION_DIM)  output
    int K,
    int H
) {
    // Each thread owns one (t, a) index
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * ACTION_DIM;
    if (idx >= total) return;

    float s = 0.0f;
    for (int k = 0; k < K; k++) {
        // eps layout: (K, H, ACTION_DIM) row-major
        s += weights[k] * eps[k * H * ACTION_DIM + idx];
    }
    u_bar_delta[idx] = s;
}
"""

# --------------------------------------------------------------------------- #
# Particle Filter propagation kernel
# Applies f_pusher to every particle and adds Gaussian process noise.
# --------------------------------------------------------------------------- #
PF_PROPAGATE_KERNEL = r"""
extern "C" __global__
void pf_propagate(
    float*       __restrict__ particles,          // (N, STATE_DIM)  in/out
    const float* __restrict__ action,             // (ACTION_DIM,)
    const float* __restrict__ noise,              // (N, STATE_DIM) pre-generated
    float process_noise_std,      // std for joint dims d < 2*NUM_JOINTS
    float process_noise_std_obj,  // std for object-state dims d >= 2*NUM_JOINTS
    float dt,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Load particle into registers
    float state[STATE_DIM];
    for (int d = 0; d < STATE_DIM; d++) {
        state[d] = particles[i * STATE_DIM + d];
    }

    // Copy action into a local array (required by f_pusher signature)
    float act[ACTION_DIM];
    for (int a = 0; a < ACTION_DIM; a++) {
        act[a] = action[a];
    }

    // Advance dynamics in place
    f_pusher(state, act, dt);

    // Write back with per-component additive process noise:
    //   joint dims (d < 2*NUM_JOINTS = 14): use tight process_noise_std
    //   object-state dims (d >= 14):        use loose process_noise_std_obj
    for (int d = 0; d < STATE_DIM; d++) {
        float ns = (d < 2 * NUM_JOINTS) ? process_noise_std : process_noise_std_obj;
        particles[i * STATE_DIM + d] = state[d] + ns * noise[i * STATE_DIM + d];
    }
}
"""

# --------------------------------------------------------------------------- #
# Particle Filter weight-update kernel
# Computes log-likelihood of each particle given the current observation and
# multiplies the existing weight (in log-sum-exp fashion via direct product).
#
# PF obs layout (OBS_DIM = 16) — built by gym_obs_to_pf_obs() from the
# true Pusher-v5 23-dim gym obs:
#   [0:7]   q      — raw joint angles   (gym obs[0:7])
#   [7:14]  qdot   — joint velocities   (gym obs[7:14])
#   [14]    obj_x  — object position x  (gym obs[17])
#   [15]    obj_y  — object position y  (gym obs[18])
#
# Particle state layout (STATE_DIM = 18):
#   [0:7]   q        state[0:7]
#   [7:14]  qdot     state[7:14]
#   [14:16] obj_pos  state[14:16]  ← where particles differ → ESS < N
#   [16:18] obj_vel  state[16:18]
#
# KEY: predicted_obs[d] == state[d] for ALL d=0..15 — no trig, no branches.
# Two noise scales:
#   obs_noise_std     (tight, 0.005)  for joint dims d < 14
#   obs_noise_std_obj (loose, 0.02)   for obj_pos dims d >= 14
# --------------------------------------------------------------------------- #
PF_WEIGHT_UPDATE_KERNEL = r"""
/* Extract the d-th predicted observation for particle i.
   PF obs is [q(0..6), qdot(7..13), obj_x(14), obj_y(15)].
   particle state[0:16] == pf_obs[0:16], so this is a direct array read.
*/
__device__ float particle_to_obs(const float* particles, int i, int d)
{
    return particles[i * STATE_DIM + d];
}

extern "C" __global__
void pf_weight_update(
    const float* __restrict__ particles,   // (N, STATE_DIM)
    const float* __restrict__ observation, // (OBS_DIM,)
    float*       __restrict__ weights,     // (N,)  in/out (multiplied)
    float obs_noise_std,       // tight std for joint dims  d < 14
    float obs_noise_std_obj,   // looser std for obj_pos dims d >= 14
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float inv_var_joint = 1.0f / (obs_noise_std     * obs_noise_std);
    float inv_var_obj   = 1.0f / (obs_noise_std_obj * obs_noise_std_obj);
    float log_lik       = 0.0f;

    // Compare predicted vs actual observation for all OBS_DIM components
    for (int d = 0; d < OBS_DIM; d++) {
        float pred    = particle_to_obs(particles, i, d);
        float diff    = pred - observation[d];
        // d < 2*NUM_JOINTS (=14): joint q+qdot use tight noise
        // d >= 14:                 obj_pos uses looser noise
        float inv_var = (d < 2 * NUM_JOINTS) ? inv_var_joint : inv_var_obj;
        log_lik      -= 0.5f * diff * diff * inv_var;
    }

    weights[i] *= expf(log_lik);
}
"""

# --------------------------------------------------------------------------- #
# Exported bundle: everything except the dynamics device code.
# Consumed by particle_filter.py and mppi.py.
# --------------------------------------------------------------------------- #
ALL_PF_KERNELS = PF_PROPAGATE_KERNEL + PF_WEIGHT_UPDATE_KERNEL

ALL_MPPI_KERNELS = (
    MPPI_ROLLOUT_KERNEL
    + COMPUTE_IMPORTANCE_WEIGHTS_KERNEL
    + WEIGHTED_EPS_UPDATE_KERNEL
)
