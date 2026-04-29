"""
dynamics.py
Abstract base class that every environment must implement.

Exposes:
  - numpy dynamics (CPU, for testing/validation)
  - CUDA kernel source string (for GPU compilation via cp.RawKernel)

The split means GPU kernels never need to be touched when adding a new
environment — only this interface needs to be satisfied.
"""

from abc import ABC, abstractmethod
import numpy as np


class AnalyticalDynamics(ABC):
    """
    Interface contract between the environment model and the GPU controllers.

    Implementors must provide:
      - f_numpy      : single-step dynamics on CPU
      - cost_numpy   : single-step cost on CPU
      - obs_model    : state → observation mapping
      - sample_initial_particles : bootstrap particle filter from first obs
      - get_cuda_dynamics_code   : CUDA C source as a Python string

    Properties required:
      - state_dim, action_dim, obs_dim
      - action_low, action_high  (numpy arrays, shape (action_dim,))
    """

    # ------------------------------------------------------------------ #
    # Core dynamics & cost
    # ------------------------------------------------------------------ #

    @abstractmethod
    def f_numpy(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        CPU dynamics: single state transition.

        Args:
            state:  (state_dim,) current state
            action: (action_dim,) control input

        Returns:
            next_state: (state_dim,) after one dt step
        """

    @abstractmethod
    def cost_numpy(self, state: np.ndarray, action: np.ndarray) -> float:
        """
        CPU cost: single-step running cost.

        Args:
            state:  (state_dim,) current state
            action: (action_dim,) control input

        Returns:
            scalar cost for this (state, action) pair
        """

    # ------------------------------------------------------------------ #
    # Observation model
    # ------------------------------------------------------------------ #

    @abstractmethod
    def obs_model(self, state: np.ndarray) -> np.ndarray:
        """
        Observation model: maps internal state → observation vector.

        Optionally adds sensor noise if the subclass wishes (e.g. for
        synthetic data generation). For likelihood computation in the
        particle filter this is called without noise.

        Args:
            state: (state_dim,) internal state

        Returns:
            obs: (obs_dim,) predicted observation
        """

    # ------------------------------------------------------------------ #
    # Particle filter initialisation
    # ------------------------------------------------------------------ #

    @abstractmethod
    def sample_initial_particles(self, obs: np.ndarray, N: int) -> np.ndarray:
        """
        Bootstrap initial particle set from the first environment observation.

        Args:
            obs: (obs_dim,) first observation returned by env.reset()
            N:   number of particles

        Returns:
            particles: (N, state_dim) initial particle cloud
        """

    # ------------------------------------------------------------------ #
    # CUDA source
    # ------------------------------------------------------------------ #

    @abstractmethod
    def get_cuda_dynamics_code(self) -> str:
        """
        Return a CUDA C source string containing at minimum:

          __device__ void f_<env>(float* state, const float* action, float dt)
          __device__ float cost_<env>(const float* state, const float* action,
                                      const float* target)

        The string is concatenated with kernel code and compiled once via
        cp.RawKernel in ParticleFilter.__init__ and MPPI.__init__.

        All #define constants (STATE_DIM, ACTION_DIM, …) must be included
        at the top of the returned string so the kernel code can use them.
        """

    # ------------------------------------------------------------------ #
    # Shape properties
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimensionality of the internal state vector."""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimensionality of the action / control vector."""

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """Dimensionality of the observation vector returned by obs_model."""

    # ------------------------------------------------------------------ #
    # Action bounds
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def action_low(self) -> np.ndarray:
        """Lower bound on actions, shape (action_dim,)."""

    @property
    @abstractmethod
    def action_high(self) -> np.ndarray:
        """Upper bound on actions, shape (action_dim,)."""
