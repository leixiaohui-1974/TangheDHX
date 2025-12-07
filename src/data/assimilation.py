# -*- coding: utf-8 -*-
"""
Data Assimilation Module for Digital Twin System.

This module provides data assimilation algorithms to integrate
observation data with model predictions, including:
- Kalman Filter (KF) and Extended Kalman Filter (EKF)
- Ensemble Kalman Filter (EnKF)
- Particle Filter
- Variational data assimilation (3D-Var, 4D-Var)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
from collections import deque
from enum import Enum
import numpy as np
from scipy import linalg

logger = logging.getLogger(__name__)


class AssimilationMethod(Enum):
    """Data assimilation methods."""
    KALMAN_FILTER = "kalman_filter"
    EXTENDED_KALMAN_FILTER = "extended_kalman_filter"
    ENSEMBLE_KALMAN_FILTER = "ensemble_kalman_filter"
    PARTICLE_FILTER = "particle_filter"
    THREE_D_VAR = "3d_var"
    FOUR_D_VAR = "4d_var"
    OPTIMAL_INTERPOLATION = "optimal_interpolation"


@dataclass
class AssimilationState:
    """State vector for data assimilation."""
    values: np.ndarray
    covariance: np.ndarray
    timestamp: float = 0.0
    innovation: Optional[np.ndarray] = None
    analysis_increment: Optional[np.ndarray] = None


@dataclass
class Observation:
    """Observation data structure."""
    values: np.ndarray
    operator: np.ndarray  # H matrix mapping state to observation space
    error_covariance: np.ndarray  # R matrix
    timestamp: float = 0.0
    source: str = "sensor"
    quality: float = 1.0


@dataclass
class AssimilationResult:
    """Result of data assimilation cycle."""
    analysis_state: np.ndarray
    analysis_covariance: np.ndarray
    innovation: np.ndarray
    innovation_covariance: np.ndarray
    kalman_gain: np.ndarray
    chi_squared: float
    timestamp: float = 0.0
    method: AssimilationMethod = AssimilationMethod.KALMAN_FILTER


class KalmanFilter:
    """
    Standard Kalman Filter for linear systems.

    Implements the classic prediction-update cycle for
    optimal state estimation.
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        process_noise: Optional[np.ndarray] = None,
        measurement_noise: Optional[np.ndarray] = None
    ):
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        # State estimate and covariance
        self._state = np.zeros(state_dim)
        self._covariance = np.eye(state_dim)

        # Process noise (Q matrix)
        self._Q = process_noise if process_noise is not None else np.eye(state_dim) * 0.01

        # Measurement noise (R matrix)
        self._R = measurement_noise if measurement_noise is not None else np.eye(obs_dim) * 0.1

        # State transition matrix (F/A matrix)
        self._F = np.eye(state_dim)

        # Observation matrix (H matrix)
        self._H = np.eye(obs_dim, state_dim)

        # Control input matrix (B matrix)
        self._B = np.zeros((state_dim, 1))

        # History
        self._history: List[AssimilationResult] = []

        logger.debug("KalmanFilter initialized: state_dim=%d, obs_dim=%d",
                    state_dim, obs_dim)

    def set_transition_matrix(self, F: np.ndarray) -> None:
        """Set state transition matrix."""
        if F.shape != (self.state_dim, self.state_dim):
            raise ValueError(f"F must be ({self.state_dim}, {self.state_dim})")
        self._F = F

    def set_observation_matrix(self, H: np.ndarray) -> None:
        """Set observation matrix."""
        if H.shape[1] != self.state_dim:
            raise ValueError(f"H columns must match state_dim ({self.state_dim})")
        self._H = H
        self.obs_dim = H.shape[0]

    def predict(
        self,
        control_input: Optional[np.ndarray] = None,
        dt: float = 1.0
    ) -> AssimilationState:
        """
        Prediction step: propagate state forward.

        Args:
            control_input: Optional control input vector
            dt: Time step

        Returns:
            Predicted state
        """
        # State prediction: x_pred = F * x + B * u
        x_pred = self._F @ self._state
        if control_input is not None:
            x_pred += self._B @ control_input

        # Covariance prediction: P_pred = F * P * F' + Q
        P_pred = self._F @ self._covariance @ self._F.T + self._Q

        self._state = x_pred
        self._covariance = P_pred

        return AssimilationState(
            values=x_pred.copy(),
            covariance=P_pred.copy(),
            timestamp=dt
        )

    def update(
        self,
        observation: Observation,
        timestamp: float = 0.0
    ) -> AssimilationResult:
        """
        Update step: correct prediction with observation.

        Args:
            observation: Observation data
            timestamp: Current timestamp

        Returns:
            Assimilation result
        """
        z = observation.values
        H = observation.operator if observation.operator is not None else self._H
        R = observation.error_covariance if observation.error_covariance is not None else self._R

        # Innovation (measurement residual): y = z - H * x
        innovation = z - H @ self._state

        # Innovation covariance: S = H * P * H' + R
        S = H @ self._covariance @ H.T + R

        # Kalman gain: K = P * H' * S^(-1)
        try:
            K = self._covariance @ H.T @ linalg.inv(S)
        except linalg.LinAlgError:
            # Use pseudo-inverse if singular
            K = self._covariance @ H.T @ linalg.pinv(S)

        # State update: x = x + K * y
        self._state = self._state + K @ innovation

        # Covariance update: P = (I - K * H) * P
        I = np.eye(self.state_dim)
        self._covariance = (I - K @ H) @ self._covariance

        # Ensure symmetry
        self._covariance = (self._covariance + self._covariance.T) / 2

        # Chi-squared statistic for innovation consistency
        chi_squared = float(innovation.T @ linalg.inv(S) @ innovation)

        result = AssimilationResult(
            analysis_state=self._state.copy(),
            analysis_covariance=self._covariance.copy(),
            innovation=innovation,
            innovation_covariance=S,
            kalman_gain=K,
            chi_squared=chi_squared,
            timestamp=timestamp,
            method=AssimilationMethod.KALMAN_FILTER
        )

        self._history.append(result)

        return result

    def assimilate(
        self,
        observation: Observation,
        control_input: Optional[np.ndarray] = None,
        dt: float = 1.0
    ) -> AssimilationResult:
        """
        Full assimilation cycle: predict then update.

        Args:
            observation: Observation data
            control_input: Optional control input
            dt: Time step

        Returns:
            Assimilation result
        """
        self.predict(control_input, dt)
        return self.update(observation, observation.timestamp)

    def get_state(self) -> np.ndarray:
        """Get current state estimate."""
        return self._state.copy()

    def get_covariance(self) -> np.ndarray:
        """Get current covariance estimate."""
        return self._covariance.copy()

    def set_state(self, state: np.ndarray, covariance: Optional[np.ndarray] = None) -> None:
        """Set filter state."""
        self._state = state.copy()
        if covariance is not None:
            self._covariance = covariance.copy()

    def reset(self) -> None:
        """Reset filter to initial state."""
        self._state = np.zeros(self.state_dim)
        self._covariance = np.eye(self.state_dim)
        self._history.clear()


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear systems.

    Uses Jacobian linearization for nonlinear state transition
    and observation models.
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        state_transition: Callable[[np.ndarray, float], np.ndarray],
        observation_model: Callable[[np.ndarray], np.ndarray],
        state_jacobian: Callable[[np.ndarray, float], np.ndarray],
        observation_jacobian: Callable[[np.ndarray], np.ndarray],
        process_noise: Optional[np.ndarray] = None,
        measurement_noise: Optional[np.ndarray] = None
    ):
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        # Nonlinear functions
        self._f = state_transition
        self._h = observation_model
        self._F_jacobian = state_jacobian
        self._H_jacobian = observation_jacobian

        # State
        self._state = np.zeros(state_dim)
        self._covariance = np.eye(state_dim)

        # Noise matrices
        self._Q = process_noise if process_noise is not None else np.eye(state_dim) * 0.01
        self._R = measurement_noise if measurement_noise is not None else np.eye(obs_dim) * 0.1

        self._history: List[AssimilationResult] = []

        logger.debug("ExtendedKalmanFilter initialized")

    def predict(self, dt: float = 1.0) -> AssimilationState:
        """Prediction step with nonlinear model."""
        # State prediction using nonlinear model
        x_pred = self._f(self._state, dt)

        # Jacobian at current state
        F = self._F_jacobian(self._state, dt)

        # Covariance prediction
        P_pred = F @ self._covariance @ F.T + self._Q

        self._state = x_pred
        self._covariance = P_pred

        return AssimilationState(
            values=x_pred.copy(),
            covariance=P_pred.copy(),
            timestamp=dt
        )

    def update(
        self,
        observation: Observation,
        timestamp: float = 0.0
    ) -> AssimilationResult:
        """Update step with nonlinear observation model."""
        z = observation.values
        R = observation.error_covariance if observation.error_covariance is not None else self._R

        # Predicted observation
        z_pred = self._h(self._state)

        # Jacobian at predicted state
        H = self._H_jacobian(self._state)

        # Innovation
        innovation = z - z_pred

        # Innovation covariance
        S = H @ self._covariance @ H.T + R

        # Kalman gain
        try:
            K = self._covariance @ H.T @ linalg.inv(S)
        except linalg.LinAlgError:
            K = self._covariance @ H.T @ linalg.pinv(S)

        # State update
        self._state = self._state + K @ innovation

        # Covariance update
        I = np.eye(self.state_dim)
        self._covariance = (I - K @ H) @ self._covariance
        self._covariance = (self._covariance + self._covariance.T) / 2

        # Chi-squared
        chi_squared = float(innovation.T @ linalg.inv(S) @ innovation)

        result = AssimilationResult(
            analysis_state=self._state.copy(),
            analysis_covariance=self._covariance.copy(),
            innovation=innovation,
            innovation_covariance=S,
            kalman_gain=K,
            chi_squared=chi_squared,
            timestamp=timestamp,
            method=AssimilationMethod.EXTENDED_KALMAN_FILTER
        )

        self._history.append(result)

        return result

    def get_state(self) -> np.ndarray:
        """Get current state estimate."""
        return self._state.copy()

    def set_state(self, state: np.ndarray, covariance: Optional[np.ndarray] = None) -> None:
        """Set filter state."""
        self._state = state.copy()
        if covariance is not None:
            self._covariance = covariance.copy()


class EnsembleKalmanFilter:
    """
    Ensemble Kalman Filter (EnKF) for large-scale systems.

    Uses an ensemble of model states to represent the
    probability distribution.
    """

    def __init__(
        self,
        state_dim: int,
        ensemble_size: int = 50,
        state_transition: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
        observation_model: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        inflation_factor: float = 1.05
    ):
        self.state_dim = state_dim
        self.ensemble_size = ensemble_size
        self.inflation_factor = inflation_factor

        # Ensemble members (state_dim x ensemble_size)
        self._ensemble = np.random.randn(state_dim, ensemble_size) * 0.1

        # Model functions
        self._f = state_transition or (lambda x, dt: x)
        self._h = observation_model or (lambda x: x)

        # Localization (optional)
        self._localization_matrix: Optional[np.ndarray] = None

        self._history: List[Dict[str, Any]] = []

        logger.debug("EnsembleKalmanFilter initialized: ensemble_size=%d", ensemble_size)

    def predict(self, dt: float = 1.0) -> None:
        """Propagate ensemble forward."""
        for i in range(self.ensemble_size):
            self._ensemble[:, i] = self._f(self._ensemble[:, i], dt)

        # Add process noise
        noise = np.random.randn(self.state_dim, self.ensemble_size) * 0.01
        self._ensemble += noise

    def update(
        self,
        observation: np.ndarray,
        obs_error_std: float = 0.1,
        timestamp: float = 0.0
    ) -> Dict[str, Any]:
        """
        Update ensemble with observations.

        Args:
            observation: Observation vector
            obs_error_std: Observation error standard deviation
            timestamp: Current timestamp

        Returns:
            Update statistics
        """
        N = self.ensemble_size

        # Ensemble mean
        x_mean = np.mean(self._ensemble, axis=1)

        # Ensemble perturbations
        X_pert = self._ensemble - x_mean[:, np.newaxis]

        # Covariance inflation
        X_pert *= self.inflation_factor

        # Predicted observations for each ensemble member
        Y = np.array([self._h(self._ensemble[:, i]) for i in range(N)]).T
        y_mean = np.mean(Y, axis=1)
        Y_pert = Y - y_mean[:, np.newaxis]

        # Cross-covariance Pxy
        Pxy = X_pert @ Y_pert.T / (N - 1)

        # Observation covariance Pyy
        Pyy = Y_pert @ Y_pert.T / (N - 1)

        # Add observation error
        R = np.eye(len(observation)) * (obs_error_std ** 2)
        Pyy += R

        # Kalman gain
        try:
            K = Pxy @ linalg.inv(Pyy)
        except linalg.LinAlgError:
            K = Pxy @ linalg.pinv(Pyy)

        # Apply localization if available
        if self._localization_matrix is not None:
            K = K * self._localization_matrix

        # Perturbed observations for each member
        obs_perturbed = observation[:, np.newaxis] + np.random.randn(
            len(observation), N) * obs_error_std

        # Innovation for each member
        innovations = obs_perturbed - Y

        # Update ensemble
        self._ensemble = self._ensemble + K @ innovations

        # Calculate statistics
        analysis_mean = np.mean(self._ensemble, axis=1)
        analysis_std = np.std(self._ensemble, axis=1)
        innovation = observation - y_mean

        result = {
            'analysis_mean': analysis_mean,
            'analysis_std': analysis_std,
            'innovation': innovation,
            'spread': np.mean(analysis_std),
            'timestamp': timestamp
        }

        self._history.append(result)

        return result

    def get_state(self) -> np.ndarray:
        """Get ensemble mean as state estimate."""
        return np.mean(self._ensemble, axis=1)

    def get_spread(self) -> np.ndarray:
        """Get ensemble spread (standard deviation)."""
        return np.std(self._ensemble, axis=1)

    def get_ensemble(self) -> np.ndarray:
        """Get full ensemble."""
        return self._ensemble.copy()

    def set_localization(self, matrix: np.ndarray) -> None:
        """Set localization matrix for Kalman gain."""
        self._localization_matrix = matrix

    def reset(self, initial_state: Optional[np.ndarray] = None, spread: float = 0.1) -> None:
        """Reset ensemble."""
        if initial_state is not None:
            self._ensemble = initial_state[:, np.newaxis] + np.random.randn(
                self.state_dim, self.ensemble_size) * spread
        else:
            self._ensemble = np.random.randn(self.state_dim, self.ensemble_size) * spread
        self._history.clear()


class VariationalAssimilator:
    """
    Variational data assimilation (3D-Var/4D-Var).

    Minimizes a cost function combining background error
    and observation departures.
    """

    def __init__(
        self,
        state_dim: int,
        background_error_cov: Optional[np.ndarray] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ):
        self.state_dim = state_dim
        self._B = background_error_cov if background_error_cov is not None else np.eye(state_dim)
        self._max_iterations = max_iterations
        self._tolerance = tolerance

        self._history: List[Dict[str, Any]] = []

        logger.debug("VariationalAssimilator initialized")

    def assimilate_3dvar(
        self,
        background: np.ndarray,
        observations: List[Observation],
        timestamp: float = 0.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        3D-Var assimilation at a single time.

        Args:
            background: Background state (first guess)
            observations: List of observations
            timestamp: Analysis timestamp

        Returns:
            Tuple of (analysis_state, diagnostics)
        """
        x = background.copy()

        # Combine all observations
        H_combined = []
        R_combined = []
        y_combined = []

        for obs in observations:
            H = obs.operator if obs.operator is not None else np.eye(len(obs.values), self.state_dim)
            H_combined.append(H)
            R_combined.append(obs.error_covariance)
            y_combined.append(obs.values)

        if not observations:
            return background, {'converged': True, 'iterations': 0, 'cost': 0.0}

        H = np.vstack(H_combined)
        R = linalg.block_diag(*R_combined)
        y = np.concatenate(y_combined)

        # Analysis: x_a = x_b + K(y - Hx_b)
        # where K = BH'(HBH' + R)^-1

        try:
            B_inv = linalg.inv(self._B)
        except linalg.LinAlgError:
            B_inv = linalg.pinv(self._B)

        try:
            R_inv = linalg.inv(R)
        except linalg.LinAlgError:
            R_inv = linalg.pinv(R)

        # Compute analysis using direct formula
        # A = (B^-1 + H'R^-1H)^-1
        A = linalg.inv(B_inv + H.T @ R_inv @ H)

        # K = AH'R^-1
        K = A @ H.T @ R_inv

        # Innovation
        innovation = y - H @ background

        # Analysis
        x_analysis = background + K @ innovation

        # Cost function value
        dx = x_analysis - background
        dy = y - H @ x_analysis
        cost_b = 0.5 * dx.T @ B_inv @ dx
        cost_o = 0.5 * dy.T @ R_inv @ dy
        total_cost = cost_b + cost_o

        diagnostics = {
            'converged': True,
            'iterations': 1,
            'cost_total': float(total_cost),
            'cost_background': float(cost_b),
            'cost_observation': float(cost_o),
            'innovation_rms': float(np.sqrt(np.mean(innovation ** 2))),
            'timestamp': timestamp
        }

        self._history.append(diagnostics)

        return x_analysis, diagnostics

    def assimilate_4dvar(
        self,
        background: np.ndarray,
        observations_sequence: List[Tuple[float, List[Observation]]],
        state_transition: Callable[[np.ndarray, float], np.ndarray],
        adjoint_model: Optional[Callable] = None,
        timestamp: float = 0.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        4D-Var assimilation over a time window.

        Args:
            background: Background state at initial time
            observations_sequence: List of (time, observations) pairs
            state_transition: Model forward propagation function
            adjoint_model: Optional adjoint model for gradient computation
            timestamp: Initial timestamp

        Returns:
            Tuple of (analysis_state, diagnostics)
        """
        x = background.copy()

        # Gradient descent optimization (simplified without adjoint)
        for iteration in range(self._max_iterations):
            # Forward integration collecting innovations
            total_cost = 0.0
            gradient = np.zeros(self.state_dim)

            x_current = x.copy()
            prev_time = timestamp

            for obs_time, observations in observations_sequence:
                dt = obs_time - prev_time
                if dt > 0:
                    x_current = state_transition(x_current, dt)

                for obs in observations:
                    H = obs.operator if obs.operator is not None else np.eye(
                        len(obs.values), self.state_dim)
                    innovation = obs.values - H @ x_current

                    try:
                        R_inv = linalg.inv(obs.error_covariance)
                    except linalg.LinAlgError:
                        R_inv = linalg.pinv(obs.error_covariance)

                    total_cost += 0.5 * innovation.T @ R_inv @ innovation

                    # Simplified gradient (would need adjoint for exact)
                    gradient += H.T @ R_inv @ innovation

                prev_time = obs_time

            # Background term
            dx = x - background
            try:
                B_inv = linalg.inv(self._B)
            except linalg.LinAlgError:
                B_inv = linalg.pinv(self._B)

            total_cost += 0.5 * dx.T @ B_inv @ dx
            gradient -= B_inv @ dx

            # Gradient descent update
            step_size = 0.1 / (iteration + 1)
            x_new = x + step_size * gradient

            # Check convergence
            if np.linalg.norm(x_new - x) < self._tolerance:
                diagnostics = {
                    'converged': True,
                    'iterations': iteration + 1,
                    'cost': float(total_cost),
                    'timestamp': timestamp
                }
                self._history.append(diagnostics)
                return x_new, diagnostics

            x = x_new

        diagnostics = {
            'converged': False,
            'iterations': self._max_iterations,
            'cost': float(total_cost),
            'timestamp': timestamp
        }
        self._history.append(diagnostics)

        return x, diagnostics


class DataAssimilationEngine:
    """
    Unified data assimilation engine for digital twin.

    Provides high-level interface for integrating observations
    with model state.
    """

    def __init__(
        self,
        state_dim: int = 12,  # Default: 3 gates x 4 state vars
        method: AssimilationMethod = AssimilationMethod.KALMAN_FILTER
    ):
        self.state_dim = state_dim
        self.method = method

        # Initialize filters
        self._kf = KalmanFilter(state_dim, state_dim)
        self._enkf = EnsembleKalmanFilter(state_dim)
        self._var = VariationalAssimilator(state_dim)

        # State mapping
        self._state_names = [
            'gate_opening_0', 'gate_opening_1', 'gate_opening_2',
            'flow_rate_0', 'flow_rate_1', 'flow_rate_2',
            'velocity_0', 'velocity_1', 'velocity_2',
            'vibration_0', 'vibration_1', 'vibration_2'
        ]

        # Assimilation history
        self._history: deque = deque(maxlen=1000)

        logger.info("DataAssimilationEngine initialized: method=%s", method.value)

    def model_to_state(self, model: 'TangheSiphonModel') -> np.ndarray:
        """Convert model state to state vector."""
        state = np.concatenate([
            model.gate_openings,
            model.flow_rates,
            model.velocities,
            model.vibration_accel
        ])
        return state

    def state_to_model(self, state: np.ndarray, model: 'TangheSiphonModel') -> None:
        """Update model with state vector."""
        num_gates = model.num_gates
        model.gate_openings = state[0:num_gates].copy()
        model.flow_rates = state[num_gates:2*num_gates].copy()
        model.velocities = state[2*num_gates:3*num_gates].copy()
        model.vibration_accel = state[3*num_gates:4*num_gates].copy()

    def create_observation(
        self,
        values: np.ndarray,
        observed_vars: List[str],
        error_std: float = 0.1,
        timestamp: float = 0.0
    ) -> Observation:
        """
        Create observation from sensor readings.

        Args:
            values: Observation values
            observed_vars: Names of observed variables
            error_std: Observation error standard deviation
            timestamp: Observation timestamp

        Returns:
            Observation object
        """
        # Create observation operator (H matrix)
        obs_dim = len(values)
        H = np.zeros((obs_dim, self.state_dim))

        for i, var_name in enumerate(observed_vars):
            if var_name in self._state_names:
                state_idx = self._state_names.index(var_name)
                H[i, state_idx] = 1.0

        # Observation error covariance
        R = np.eye(obs_dim) * (error_std ** 2)

        return Observation(
            values=values,
            operator=H,
            error_covariance=R,
            timestamp=timestamp
        )

    def assimilate(
        self,
        model: 'TangheSiphonModel',
        observations: Dict[str, float],
        error_std: float = 0.1,
        dt: float = 0.1
    ) -> Dict[str, Any]:
        """
        Perform data assimilation cycle.

        Args:
            model: Physics model
            observations: Dict of variable_name -> observed_value
            error_std: Observation error standard deviation
            dt: Time step since last assimilation

        Returns:
            Assimilation diagnostics
        """
        # Get background state from model
        background = self.model_to_state(model)

        # Create observation
        obs_vars = list(observations.keys())
        obs_values = np.array([observations[v] for v in obs_vars])

        observation = self.create_observation(
            obs_values, obs_vars, error_std, model.time
        )

        # Perform assimilation based on method
        if self.method == AssimilationMethod.KALMAN_FILTER:
            self._kf.set_state(background)
            self._kf.predict(dt=dt)
            result = self._kf.update(observation, model.time)
            analysis = result.analysis_state
            diagnostics = {
                'method': 'kalman_filter',
                'chi_squared': result.chi_squared,
                'innovation_norm': float(np.linalg.norm(result.innovation)),
                'timestamp': model.time
            }

        elif self.method == AssimilationMethod.ENSEMBLE_KALMAN_FILTER:
            self._enkf.predict(dt)
            result = self._enkf.update(obs_values, error_std, model.time)
            analysis = result['analysis_mean']
            diagnostics = {
                'method': 'ensemble_kalman_filter',
                'spread': result['spread'],
                'innovation_norm': float(np.linalg.norm(result['innovation'])),
                'timestamp': model.time
            }

        elif self.method == AssimilationMethod.THREE_D_VAR:
            analysis, diag = self._var.assimilate_3dvar(
                background, [observation], model.time
            )
            diagnostics = {
                'method': '3d_var',
                'cost': diag['cost_total'],
                'innovation_rms': diag['innovation_rms'],
                'timestamp': model.time
            }

        else:
            # Default to Kalman filter
            self._kf.set_state(background)
            result = self._kf.update(observation, model.time)
            analysis = result.analysis_state
            diagnostics = {
                'method': 'kalman_filter',
                'timestamp': model.time
            }

        # Update model with analysis
        self.state_to_model(analysis, model)

        # Record history
        self._history.append({
            'timestamp': model.time,
            'analysis': analysis.copy(),
            'observations': observations.copy(),
            **diagnostics
        })

        return diagnostics

    def get_state_uncertainty(self) -> np.ndarray:
        """Get current state uncertainty (std dev)."""
        if self.method == AssimilationMethod.ENSEMBLE_KALMAN_FILTER:
            return self._enkf.get_spread()
        elif self.method == AssimilationMethod.KALMAN_FILTER:
            return np.sqrt(np.diag(self._kf.get_covariance()))
        else:
            return np.zeros(self.state_dim)

    def get_assimilation_history(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent assimilation history."""
        return list(self._history)[-count:]

    def set_method(self, method: AssimilationMethod) -> None:
        """Change assimilation method."""
        self.method = method
        logger.info("Assimilation method changed to: %s", method.value)

    def reset(self) -> None:
        """Reset assimilation engine."""
        self._kf.reset()
        self._enkf.reset()
        self._history.clear()
        logger.info("DataAssimilationEngine reset")
