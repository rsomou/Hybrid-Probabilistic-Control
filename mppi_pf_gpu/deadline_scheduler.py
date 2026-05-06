"""
deadline_scheduler.py
Deadline-aware adaptive K scheduler for MPPI.

Tunes the number of trajectory samples K per step based on:
  1. Remaining wall-clock budget  (deadline_ms - estimated overhead)
  2. Weight-distribution variance (uncertainty proxy from MPPI weights)
  3. Measured GPU latency from previous steps (latency model)

The scheduler maintains an exponentially-smoothed estimate of per-trajectory
GPU cost (ms/trajectory) and uses it to predict how many trajectories will
fit within the deadline.  When MPPI weights are diffuse (high w_eff / K ratio),
the planner is uncertain and benefits from more samples; when concentrated
(low w_eff / K), the planner is confident and can use fewer samples.

Usage
-----
    scheduler = DeadlineScheduler(config)
    for t in range(max_steps):
        K_next = scheduler.get_K(T_gpu_prev_ms, weights, T_overhead_ms)
        action = mppi.compute_action(states, K=K_next)
        scheduler.log(K_next, T_gpu_ms, weights)
"""

import numpy as np


class DeadlineScheduler:
    """
    Adaptive K scheduler that fits MPPI planning within a per-step deadline.

    Parameters
    ----------
    config : Config
        Must have: K, K_min, K_max, deadline_ms, safety_margin_ms
    """

    def __init__(self, config):
        self.K_min = config.K_min
        self.K_max = config.K_max
        self.K_default = config.K
        self.deadline_ms = config.deadline_ms
        self.safety_margin_ms = config.safety_margin_ms

        # ---- Latency model ------------------------------------------------
        # Exponentially smoothed estimate of per-trajectory GPU cost.
        # Updated after each step with the observed (T_gpu_ms / K_used).
        # Initial estimate: assume deadline is achievable with default K.
        self._cost_per_traj_ms = self.deadline_ms / self.K_default
        self._cost_ema_alpha = 0.3   # smoothing factor (higher = more reactive)

        # ---- Overhead model -----------------------------------------------
        # Smoothed estimate of non-MPPI overhead (PF, env step, delay prop).
        # Subtracted from deadline to get the MPPI budget.
        self._overhead_ms = 2.0      # conservative initial estimate
        self._overhead_ema_alpha = 0.3

        # ---- Uncertainty scaling ------------------------------------------
        # When w_eff/K is low (concentrated weights → confident), scale K down.
        # When w_eff/K is high (diffuse weights → uncertain), scale K up.
        # The scaling maps w_eff_ratio ∈ [0, 1] to a multiplier ∈ [0.5, 1.5].
        self._uncertainty_scale_min = 0.5   # multiplier when fully confident
        self._uncertainty_scale_max = 1.5   # multiplier when fully uncertain

        # ---- History for diagnostics --------------------------------------
        self._step = 0
        self._history = []

    def get_K(
        self,
        T_gpu_prev_ms: float = None,
        weights: np.ndarray = None,
        T_overhead_ms: float = None,
    ) -> int:
        """
        Compute K for the next MPPI step.

        Parameters
        ----------
        T_gpu_prev_ms : float or None
            GPU time of the previous MPPI step (ms).
            None on the first step (uses default K).
        weights : (K,) numpy array or None
            Normalised MPPI importance weights from the previous step.
            Used to compute w_eff as an uncertainty proxy.
        T_overhead_ms : float or None
            Non-MPPI overhead time (PF + env + delay propagation) in ms.
            Used to estimate how much of the deadline is available for MPPI.

        Returns
        -------
        K_next : int — clamped to [K_min, K_max], rounded to nearest 64
        """
        # First step: no history, use default K
        if T_gpu_prev_ms is None or self._step == 0:
            self._step += 1
            return self.K_default

        # ---- 1. Update latency model from previous step -------------------
        K_prev = self._history[-1]["K"] if self._history else self.K_default
        if K_prev > 0 and T_gpu_prev_ms > 0:
            observed_cost = T_gpu_prev_ms / K_prev
            self._cost_per_traj_ms = (
                self._cost_ema_alpha * observed_cost
                + (1.0 - self._cost_ema_alpha) * self._cost_per_traj_ms
            )

        # ---- 2. Update overhead model -------------------------------------
        if T_overhead_ms is not None:
            self._overhead_ms = (
                self._overhead_ema_alpha * T_overhead_ms
                + (1.0 - self._overhead_ema_alpha) * self._overhead_ms
            )

        # ---- 3. Compute MPPI time budget ----------------------------------
        budget_ms = self.deadline_ms - self._overhead_ms - self.safety_margin_ms
        budget_ms = max(budget_ms, 1.0)  # at least 1ms

        # ---- 4. Compute base K from latency model -------------------------
        # How many trajectories can we afford in the budget?
        if self._cost_per_traj_ms > 0:
            K_budget = int(budget_ms / self._cost_per_traj_ms)
        else:
            K_budget = self.K_max

        # ---- 5. Apply uncertainty scaling ---------------------------------
        if weights is not None and len(weights) > 0:
            # Effective sample size: w_eff = 1 / sum(w_i^2)
            w_eff = 1.0 / (np.sum(weights ** 2) + 1e-12)
            w_eff_ratio = w_eff / len(weights)  # ∈ (0, 1]

            # Map: low ratio (concentrated) → scale down
            #       high ratio (diffuse)     → scale up
            # Linear interpolation between scale_min and scale_max
            scale = (
                self._uncertainty_scale_min
                + (self._uncertainty_scale_max - self._uncertainty_scale_min)
                * w_eff_ratio
            )
        else:
            scale = 1.0

        K_scaled = int(K_budget * scale)

        # ---- 6. Clamp and round to multiple of 64 (GPU warp alignment) ----
        K_next = max(self.K_min, min(self.K_max, K_scaled))
        K_next = max(self.K_min, (K_next // 64) * 64)  # round down to 64

        self._step += 1
        return K_next

    def log(self, K_used: int, T_gpu_ms: float, weights: np.ndarray = None):
        """
        Record step data for diagnostics.

        Parameters
        ----------
        K_used : int — K actually used this step
        T_gpu_ms : float — observed GPU time
        weights : (K,) array or None — MPPI weights (for w_eff logging)
        """
        w_eff = 0.0
        if weights is not None and len(weights) > 0:
            w_eff = 1.0 / (np.sum(weights ** 2) + 1e-12)

        self._history.append({
            "step": self._step - 1,
            "K": K_used,
            "T_gpu_ms": T_gpu_ms,
            "w_eff": w_eff,
            "cost_per_traj_ms": self._cost_per_traj_ms,
            "overhead_ms": self._overhead_ms,
        })

    def summary(self) -> dict:
        """Return summary statistics for the episode."""
        if not self._history:
            return {}
        Ks = [h["K"] for h in self._history]
        Ts = [h["T_gpu_ms"] for h in self._history]
        return {
            "K_mean": float(np.mean(Ks)),
            "K_min_used": min(Ks),
            "K_max_used": max(Ks),
            "K_std": float(np.std(Ks)),
            "T_gpu_mean_ms": float(np.mean(Ts)),
            "T_gpu_max_ms": float(np.max(Ts)),
            "cost_per_traj_final_ms": self._cost_per_traj_ms,
            "overhead_final_ms": self._overhead_ms,
            "n_steps": len(self._history),
        }
