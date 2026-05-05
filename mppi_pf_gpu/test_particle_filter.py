#!/usr/bin/env python3
"""
test_particle_filter.py — Unit and integration tests for ParticleFilter.

Tests are grouped into:
  1. Initialization       — shapes, prior coverage, uniform weights
  2. inject_observation   — q/qdot / fork injection correctness
  3. propagate            — dynamics step changes particle state
  4. update (weights)     — likelihood weighting works as expected
  5. resample             — systematic resampling correctness
  6. ESS                  — effective sample size edge cases
  7. estimate             — weighted-mean state extraction
  8. convergence          — PF converges to the true object position

Run:
    cd mppi_pf_gpu
    python test_particle_filter.py

No gymnasium / MuJoCo environment required — all tests use synthetic
observations built from the known analytical FK.
"""

import sys
import os
import numpy as np
import cupy as cp

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from gpu_utils import GPUUtils
from particle_filter import ParticleFilter
from envs.pusher import PusherDynamics, STATE_DIM, TABLE_Z

# ------------------------------------------------------------------ #
# Shared fixtures
# ------------------------------------------------------------------ #

def make_components(N: int = 500):
    """Return (dynamics, config, gpu, pf) with N particles."""
    config = Config(N=N)
    dynamics = PusherDynamics(dt=config.dt)
    dynamics.set_target(np.array([0.45, -0.05], dtype=np.float32))
    gpu = GPUUtils(config)
    pf  = ParticleFilter(dynamics, config, gpu)
    return dynamics, config, gpu, pf


def make_obs(q: np.ndarray = None, obj_pos: np.ndarray = None) -> np.ndarray:
    """
    Build a synthetic 23-dim Pusher-v5 observation.

    Pusher-v5 obs layout:
      [0:7]   q            joint angles
      [7:14]  qdot         joint velocities (set to zero)
      [14:17] fingertip    FK(q)  (not used by PF)
      [17:20] obj_pos      (x, y, z)  ← the hidden state
      [20:23] goal_pos     fixed at (0.45, -0.05, -0.323)
    """
    if q is None:
        q = np.zeros(7, dtype=np.float32)
    if obj_pos is None:
        obj_pos = np.array([0.35, -0.20, TABLE_Z], dtype=np.float32)

    fk_x, fk_y, fk_z = PusherDynamics._forward_kinematics(q)

    obs = np.zeros(23, dtype=np.float32)
    obs[0:7]   = q
    obs[7:14]  = 0.0                                          # qdot = 0
    obs[14:17] = [fk_x, fk_y, fk_z]                          # fingertip marker
    obs[17:20] = [obj_pos[0], obj_pos[1], TABLE_Z]            # object xyz
    obs[20:23] = [0.45, -0.05, -0.323]                        # goal (fixed)
    return obs


# ================================================================== #
# 1. Initialization
# ================================================================== #

def test_initialize_shapes():
    """After initialize(), particles and weights have the right shapes."""
    dynamics, config, gpu, pf = make_components(N=200)
    obs = make_obs()
    pf.initialize(obs)

    assert pf.particles.shape == (200, STATE_DIM), \
        f"Expected (200, {STATE_DIM}), got {pf.particles.shape}"
    assert pf.weights.shape == (200,), \
        f"Expected (200,), got {pf.weights.shape}"
    print("  PASS test_initialize_shapes")


def test_initialize_weights_uniform():
    """After initialize(), weights are uniform (= 1/N each)."""
    dynamics, config, gpu, pf = make_components(N=200)
    pf.initialize(make_obs())
    w = cp.asnumpy(pf.weights)
    np.testing.assert_allclose(w, 1.0 / 200, rtol=1e-5,
                               err_msg="Initial weights must be exactly 1/N")
    print("  PASS test_initialize_weights_uniform")


def test_initialize_qqdot_near_obs():
    """After initialize(), mean q/qdot of particles is near obs q/qdot."""
    q_true = np.array([0.1, -0.2, 0.3, -0.5, 0.4, -0.3, 0.2], dtype=np.float32)
    dynamics, config, gpu, pf = make_components(N=1000)
    pf.initialize(make_obs(q=q_true))
    particles = cp.asnumpy(pf.particles)

    mean_q = particles[:, 0:7].mean(axis=0)
    np.testing.assert_allclose(mean_q, q_true, atol=0.05,
                               err_msg="Mean q of initialized particles should be near obs q")
    print("  PASS test_initialize_qqdot_near_obs")


def test_initialize_obj_pos_covers_prior():
    """After initialize(), object positions span the expected prior range."""
    dynamics, config, gpu, pf = make_components(N=2000)
    pf.initialize(make_obs())
    particles = cp.asnumpy(pf.particles)

    obj_x = particles[:, 14]
    obj_y = particles[:, 15]

    # Prior: x ~ U(0.25, 0.65),  y ~ U(-0.35, -0.05)
    assert obj_x.min() >= 0.20, f"obj_x min too low: {obj_x.min():.3f}"
    assert obj_x.max() <= 0.70, f"obj_x max too high: {obj_x.max():.3f}"
    assert obj_y.min() >= -0.40, f"obj_y min too low: {obj_y.min():.3f}"
    assert obj_y.max() <= 0.00,  f"obj_y max too high: {obj_y.max():.3f}"

    # With N=2000 draws from U(0.25,0.65) the range should span at least 0.30
    assert obj_x.max() - obj_x.min() > 0.25, \
        f"obj_x spread too narrow: {obj_x.max()-obj_x.min():.3f}"
    print("  PASS test_initialize_obj_pos_covers_prior")


def test_initialize_fork_matches_fk():
    """After initialize(), every particle's fork position = FK(q)."""
    q_true = np.array([0.1, -0.2, 0.3, -0.5, 0.4, -0.3, 0.2], dtype=np.float32)
    dynamics, config, gpu, pf = make_components(N=300)
    pf.initialize(make_obs(q=q_true))
    particles = cp.asnumpy(pf.particles)

    fk_x, fk_y, fk_z = PusherDynamics._forward_kinematics(q_true)
    np.testing.assert_allclose(particles[:, 18], fk_x, atol=1e-5,
                               err_msg="Fork x must match FK")
    np.testing.assert_allclose(particles[:, 19], fk_y, atol=1e-5,
                               err_msg="Fork y must match FK")
    np.testing.assert_allclose(particles[:, 20], fk_z, atol=1e-5,
                               err_msg="Fork z must match FK")
    print("  PASS test_initialize_fork_matches_fk")


# ================================================================== #
# 2. inject_observation
# ================================================================== #

def test_inject_sets_qqdot():
    """inject_observation() overwrites particle q/qdot with obs values (+ small jitter)."""
    dynamics, config, gpu, pf = make_components(N=500)
    obs = make_obs()
    pf.initialize(obs)

    # Disturb q/qdot first to make the injection visible
    pf.particles[:, 0:14] = cp.float32(99.0)

    new_q = np.array([0.5, -0.3, 0.2, -0.8, 0.1, -0.1, 0.3], dtype=np.float32)
    new_obs = make_obs(q=new_q)
    pf.inject_observation(new_obs)
    particles = cp.asnumpy(pf.particles)

    # Mean should be very close to new_q (jitter std = 0.001)
    np.testing.assert_allclose(particles[:, 0:7].mean(axis=0), new_q, atol=0.01,
                               err_msg="Mean q after inject must match obs q")
    # qdot = 0 in the obs; mean should be near 0
    np.testing.assert_allclose(particles[:, 7:14].mean(axis=0), 0.0, atol=0.01,
                               err_msg="Mean qdot after inject must be near 0")
    print("  PASS test_inject_sets_qqdot")


def test_inject_sets_fork_to_fk():
    """inject_observation() sets fork xyz to FK(obs q), NOT obs[14:17]."""
    dynamics, config, gpu, pf = make_components(N=300)
    q = np.array([0.3, -0.1, 0.2, -0.4, 0.1, -0.2, 0.1], dtype=np.float32)
    obs = make_obs(q=q)

    # Corrupt obs[14:17] (fingertip marker) — inject should ignore this
    obs[14:17] = [99.0, 99.0, 99.0]

    pf.initialize(make_obs())          # initialize with default obs
    pf.inject_observation(obs)
    particles = cp.asnumpy(pf.particles)

    fk_x, fk_y, fk_z = PusherDynamics._forward_kinematics(q)
    np.testing.assert_allclose(particles[:, 18], fk_x, atol=1e-4,
                               err_msg="Fork x must be FK(q), not obs[14]")
    np.testing.assert_allclose(particles[:, 19], fk_y, atol=1e-4,
                               err_msg="Fork y must be FK(q), not obs[15]")
    np.testing.assert_allclose(particles[:, 20], fk_z, atol=1e-4,
                               err_msg="Fork z must be FK(q), not obs[16]")
    print("  PASS test_inject_sets_fork_to_fk")


def test_inject_does_not_change_obj_pos():
    """inject_observation() must NOT touch obj_pos (dims 14:16 of state)."""
    dynamics, config, gpu, pf = make_components(N=300)
    pf.initialize(make_obs())

    # Record obj_pos before injection
    before = cp.asnumpy(pf.particles[:, 14:16].copy())

    pf.inject_observation(make_obs(obj_pos=np.array([0.99, -0.99])))
    after = cp.asnumpy(pf.particles[:, 14:16])

    np.testing.assert_array_equal(before, after,
        err_msg="inject_observation must not modify obj_pos dims [14:16]")
    print("  PASS test_inject_does_not_change_obj_pos")


# ================================================================== #
# 3. propagate
# ================================================================== #

def test_propagate_changes_joints():
    """propagate() with a nonzero action changes q/qdot in particles."""
    dynamics, config, gpu, pf = make_components(N=300)
    obs = make_obs()
    pf.initialize(obs)
    pf.inject_observation(obs)

    q_before = cp.asnumpy(pf.particles[:, 0:7].copy())

    action = np.array([0.5, -0.5, 0.3, -0.3, 0.2, -0.2, 0.1], dtype=np.float32)
    pf.propagate(action)
    cp.cuda.Device(0).synchronize()

    q_after = cp.asnumpy(pf.particles[:, 0:7])

    # Joint angles should change after applying a non-trivial torque
    diff = np.abs(q_after - q_before).mean()
    assert diff > 1e-6, f"propagate() with nonzero action should change q; mean |diff|={diff:.2e}"
    print("  PASS test_propagate_changes_joints")


def test_propagate_zero_action_still_integrates():
    """propagate() with zero action still changes state (dynamics integration is not trivial at qdot≠0)."""
    dynamics, config, gpu, pf = make_components(N=200)
    obs = make_obs()
    pf.initialize(obs)
    # Give particles nonzero qdot
    pf.particles[:, 7:14] = cp.float32(0.5)
    state_before = cp.asnumpy(pf.particles[:, 0:7].copy())

    pf.propagate(np.zeros(7, dtype=np.float32))
    cp.cuda.Device(0).synchronize()

    state_after = cp.asnumpy(pf.particles[:, 0:7])
    diff = np.abs(state_after - state_before).mean()
    assert diff > 1e-7, f"Zero action + nonzero qdot should still change q; diff={diff:.2e}"
    print("  PASS test_propagate_zero_action_still_integrates")


# ================================================================== #
# 4. update (weights)
# ================================================================== #

def test_update_changes_weights():
    """update() changes weights from uniform to non-uniform."""
    dynamics, config, gpu, pf = make_components(N=500)
    obs = make_obs(obj_pos=np.array([0.35, -0.20]))
    pf.initialize(obs)                    # obj_pos spread over full prior
    pf.inject_observation(obs)
    # Object positions are spread over the prior — most are far from truth.
    # After update with the true obs, weights should diverge from 1/N.
    pf.update(obs)
    cp.cuda.Device(0).synchronize()

    w = cp.asnumpy(pf.weights)
    # Weights must be normalised and no longer perfectly uniform
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-5,
                               err_msg="Weights must sum to 1 after update")
    assert w.std() > 1e-8, "Weights must diverge from uniform after update"
    print("  PASS test_update_changes_weights")


def test_update_favors_particles_near_true_obj():
    """
    Particles whose obj_pos is near the true value get higher weights.

    Setup: N/2 particles placed exactly at true_obj_pos, N/2 placed 0.5m away.
    After update with the true obs, the near-group total weight must exceed
    the far-group total weight.
    """
    N = 500
    true_obj = np.array([0.35, -0.20], dtype=np.float32)
    far_obj  = np.array([0.35 + 0.5, -0.20], dtype=np.float32)  # 0.5m away

    dynamics, config, gpu, pf = make_components(N=N)
    obs = make_obs(obj_pos=true_obj)
    pf.initialize(obs)
    pf.inject_observation(obs)

    # Manually override obj_pos: first half near, second half far
    pf.particles[:N//2, 14] = np.float32(true_obj[0])
    pf.particles[:N//2, 15] = np.float32(true_obj[1])
    pf.particles[N//2:, 14] = np.float32(far_obj[0])
    pf.particles[N//2:, 15] = np.float32(far_obj[1])

    pf.weights = cp.ones(N, dtype=cp.float32) / N   # reset to uniform

    pf.update(obs)
    cp.cuda.Device(0).synchronize()

    w = cp.asnumpy(pf.weights)
    near_total = w[:N//2].sum()
    far_total  = w[N//2:].sum()

    assert near_total > far_total, (
        f"Near-group weight ({near_total:.4f}) must exceed far-group ({far_total:.4f})"
    )
    print(f"  PASS test_update_favors_particles_near_true_obj  "
          f"(near={near_total:.4f} vs far={far_total:.4f})")


# ================================================================== #
# 5. resample
# ================================================================== #

def test_resample_resets_to_uniform_weights():
    """After resample(), all weights are exactly 1/N."""
    N = 400
    dynamics, config, gpu, pf = make_components(N=N)
    pf.initialize(make_obs())
    # Force very skewed weights
    w = np.zeros(N, dtype=np.float32)
    w[0] = 1.0
    pf.weights = cp.asarray(w)

    pf.resample()
    cp.cuda.Device(0).synchronize()

    w_after = cp.asnumpy(pf.weights)
    np.testing.assert_allclose(w_after, 1.0 / N, rtol=1e-5,
                               err_msg="Resample must restore uniform weights")
    print("  PASS test_resample_resets_to_uniform_weights")


def test_resample_concentrates_on_high_weight_particle():
    """
    When all weight is on one particle, resample duplicates that particle N times.
    """
    N = 300
    dynamics, config, gpu, pf = make_components(N=N)
    pf.initialize(make_obs())

    # Place a unique sentinel obj_pos on particle 42
    sentinel_x = np.float32(0.987)
    sentinel_y = np.float32(-0.123)
    pf.particles[42, 14] = sentinel_x
    pf.particles[42, 15] = sentinel_y

    # All weight on particle 42
    w = np.zeros(N, dtype=np.float32)
    w[42] = 1.0
    pf.weights = cp.asarray(w)

    pf.resample()
    cp.cuda.Device(0).synchronize()

    particles = cp.asnumpy(pf.particles)
    # Every particle should now have the sentinel obj_pos
    np.testing.assert_allclose(particles[:, 14], sentinel_x, atol=1e-5,
                               err_msg="All particles should match the sentinel x after resample")
    np.testing.assert_allclose(particles[:, 15], sentinel_y, atol=1e-5,
                               err_msg="All particles should match the sentinel y after resample")
    print("  PASS test_resample_concentrates_on_high_weight_particle")


# ================================================================== #
# 6. ESS
# ================================================================== #

def test_ess_uniform_equals_N():
    """ESS with uniform weights = N."""
    N = 400
    dynamics, config, gpu, pf = make_components(N=N)
    pf.initialize(make_obs())
    ess = pf.effective_sample_size()
    np.testing.assert_allclose(ess, float(N), rtol=1e-4,
                               err_msg=f"ESS with uniform weights must equal N={N}")
    print(f"  PASS test_ess_uniform_equals_N  (ESS={ess:.1f})")


def test_ess_concentrated_near_one():
    """ESS ≈ 1 when all weight is on a single particle."""
    N = 400
    dynamics, config, gpu, pf = make_components(N=N)
    pf.initialize(make_obs())
    w = np.zeros(N, dtype=np.float32)
    w[0] = 1.0
    pf.weights = cp.asarray(w)
    ess = pf.effective_sample_size()
    assert ess < 2.0, f"ESS with single-particle weight must be ~1; got {ess:.3f}"
    print(f"  PASS test_ess_concentrated_near_one  (ESS={ess:.4f})")


def test_ess_triggers_resample_below_threshold():
    """When ESS < threshold * N, the runner resamples. Verify threshold logic."""
    N = 400
    dynamics, config, gpu, pf = make_components(N=N)
    pf.initialize(make_obs())
    # Put half the weight on one particle → ESS < N/2
    w = np.full(N, 1.0 / (2.0 * N), dtype=np.float32)
    w[0] = 0.5
    w /= w.sum()
    pf.weights = cp.asarray(w.astype(np.float32))

    ess = pf.effective_sample_size()
    threshold = config_for_test().resample_threshold * N

    assert ess < threshold, (
        f"ESS={ess:.1f} should be below resample threshold={threshold:.1f}"
    )
    print(f"  PASS test_ess_triggers_resample_below_threshold  (ESS={ess:.1f} < {threshold:.1f})")


def config_for_test():
    return Config(N=400)


# ================================================================== #
# 7. estimate
# ================================================================== #

def test_estimate_matches_manual_weighted_mean():
    """estimate() returns the correct weighted mean state."""
    N = 10
    dynamics, config, gpu, pf = make_components(N=N)
    pf.initialize(make_obs())

    # Set known particles and weights
    particles_cpu = np.zeros((N, STATE_DIM), dtype=np.float32)
    for i in range(N):
        particles_cpu[i, 14] = float(i) * 0.01   # obj_x = 0, 0.01, ..., 0.09
        particles_cpu[i, 15] = -float(i) * 0.01  # obj_y

    weights_cpu = np.arange(1, N + 1, dtype=np.float32)
    weights_cpu /= weights_cpu.sum()

    pf.particles = cp.asarray(particles_cpu)
    pf.weights   = cp.asarray(weights_cpu)

    estimated = pf.estimate()  # returns CPU numpy

    expected = np.average(particles_cpu, axis=0, weights=weights_cpu)
    np.testing.assert_allclose(estimated, expected.astype(np.float32), atol=1e-5,
                               err_msg="estimate() must return the weighted mean state")
    print(f"  PASS test_estimate_matches_manual_weighted_mean  "
          f"(obj_x={estimated[14]:.4f} expected {expected[14]:.4f})")


def test_estimate_gpu_shape_and_value():
    """estimate_gpu() returns (1, STATE_DIM) on GPU, matching CPU estimate."""
    N = 200
    dynamics, config, gpu, pf = make_components(N=N)
    pf.initialize(make_obs())

    cpu_est = pf.estimate()
    gpu_est = pf.estimate_gpu()

    assert gpu_est.shape == (1, STATE_DIM), \
        f"estimate_gpu shape should be (1, {STATE_DIM}), got {gpu_est.shape}"
    np.testing.assert_allclose(cp.asnumpy(gpu_est[0]), cpu_est, atol=1e-5,
                               err_msg="estimate_gpu must match estimate (CPU)")
    print("  PASS test_estimate_gpu_shape_and_value")


# ================================================================== #
# 8. Convergence — the core correctness test
# ================================================================== #

def test_convergence_to_true_obj_pos():
    """
    The PF must converge on the true object position after a few steps
    of inject → propagate → update → resample, given the obj_pos IS included
    in the likelihood (dims 14:16 of pf_obs, scaled by obs_noise_std_obj).

    Setup:
      - true_obj_pos is within the initial prior but initially unknown.
      - All particles start with obj_pos uniformly drawn over the prior.
      - Each step: inject true q/qdot, propagate (zero action), update with
        true obs, resample when ESS < 0.5*N.
      - After 20 steps, the weighted mean obj_pos must be within 0.1m of truth.
    """
    N = 1000
    TRUE_OBJ = np.array([0.38, -0.22], dtype=np.float32)   # within prior bounds

    dynamics, config, gpu, pf = make_components(N=N)
    q_zero = np.zeros(7, dtype=np.float32)
    obs    = make_obs(q=q_zero, obj_pos=TRUE_OBJ)

    pf.initialize(obs)   # obj_pos drawn from U(0.25,0.65) x U(-0.35,-0.05)
    zero_action = np.zeros(7, dtype=np.float32)

    initial_mean = pf.estimate()
    initial_err  = float(np.linalg.norm(initial_mean[14:16] - TRUE_OBJ))

    for step in range(25):
        pf.inject_observation(obs)
        pf.propagate(zero_action)
        pf.update(obs)
        if float(pf.effective_sample_size()) < config.resample_threshold * N:
            pf.resample()

    cp.cuda.Device(0).synchronize()
    final_mean = pf.estimate()
    final_err  = float(np.linalg.norm(final_mean[14:16] - TRUE_OBJ))

    assert final_err < 0.10, (
        f"PF did not converge: obj_pos error {final_err:.3f}m > 0.10m  "
        f"(initial err was {initial_err:.3f}m)"
    )
    assert final_err < initial_err * 0.5, (
        f"PF should at least halve the initial error: "
        f"initial={initial_err:.3f}m final={final_err:.3f}m"
    )
    print(f"  PASS test_convergence_to_true_obj_pos  "
          f"(initial err={initial_err:.3f}m → final err={final_err:.4f}m  "
          f"true={TRUE_OBJ}, est={final_mean[14:16]})")


def test_convergence_ess_drops_then_recovers():
    """
    Verify the ESS dynamic: starts at N (uniform), drops sharply after
    the first informative update, then stabilises after resampling.
    """
    N = 500
    TRUE_OBJ = np.array([0.40, -0.18], dtype=np.float32)
    dynamics, config, gpu, pf = make_components(N=N)
    obs = make_obs(obj_pos=TRUE_OBJ)
    pf.initialize(obs)

    ess_before = pf.effective_sample_size()
    pf.inject_observation(obs)
    pf.propagate(np.zeros(7, dtype=np.float32))
    pf.update(obs)
    cp.cuda.Device(0).synchronize()
    ess_after_update = pf.effective_sample_size()

    assert ess_before > ess_after_update, (
        f"ESS should drop after first informative update: "
        f"before={ess_before:.1f} after={ess_after_update:.1f}"
    )

    pf.resample()
    ess_after_resample = pf.effective_sample_size()

    assert ess_after_resample > ess_after_update, (
        f"ESS should recover after resample: "
        f"post-update={ess_after_update:.1f} post-resample={ess_after_resample:.1f}"
    )
    print(f"  PASS test_convergence_ess_drops_then_recovers  "
          f"(N→{ess_before:.0f} →update→ {ess_after_update:.1f} →resample→ {ess_after_resample:.1f})")


# ================================================================== #
# Runner
# ================================================================== #

TESTS = [
    # 1. Initialization
    test_initialize_shapes,
    test_initialize_weights_uniform,
    test_initialize_qqdot_near_obs,
    test_initialize_obj_pos_covers_prior,
    test_initialize_fork_matches_fk,
    # 2. inject_observation
    test_inject_sets_qqdot,
    test_inject_sets_fork_to_fk,
    test_inject_does_not_change_obj_pos,
    # 3. propagate
    test_propagate_changes_joints,
    test_propagate_zero_action_still_integrates,
    # 4. update
    test_update_changes_weights,
    test_update_favors_particles_near_true_obj,
    # 5. resample
    test_resample_resets_to_uniform_weights,
    test_resample_concentrates_on_high_weight_particle,
    # 6. ESS
    test_ess_uniform_equals_N,
    test_ess_concentrated_near_one,
    test_ess_triggers_resample_below_threshold,
    # 7. estimate
    test_estimate_matches_manual_weighted_mean,
    test_estimate_gpu_shape_and_value,
    # 8. convergence
    test_convergence_to_true_obj_pos,
    test_convergence_ess_drops_then_recovers,
]


def main():
    passed = 0
    failed = 0
    errors = []

    print("\n" + "=" * 65)
    print("  Particle Filter Test Suite")
    print("=" * 65)

    for test_fn in TESTS:
        name = test_fn.__name__
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            failed += 1
            errors.append((name, exc))
            print(f"  FAIL {name}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 65)
    print(f"  Results: {passed} passed, {failed} failed out of {len(TESTS)} tests")
    print("=" * 65 + "\n")

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
