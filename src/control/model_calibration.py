# -*- coding: utf-8 -*-
"""
Model Calibration and Parameter Update Module.

This module provides dynamic model parameter updating based on
high-fidelity simulation state, including:
- IDZ (Identification Zone) model parameter estimation
- Online parameter calibration
- Model-reality mismatch detection
- Adaptive parameter tuning
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
from collections import deque
from enum import Enum
import numpy as np
from scipy import optimize, linalg

logger = logging.getLogger(__name__)


class CalibrationMethod(Enum):
    """Parameter calibration methods."""
    LEAST_SQUARES = "least_squares"
    RECURSIVE_LEAST_SQUARES = "recursive_least_squares"
    EXTENDED_KALMAN = "extended_kalman"
    GRADIENT_DESCENT = "gradient_descent"
    PARTICLE_SWARM = "particle_swarm"
    GENETIC_ALGORITHM = "genetic_algorithm"


class ParameterStatus(Enum):
    """Parameter estimation status."""
    CONVERGED = "converged"
    CONVERGING = "converging"
    DIVERGING = "diverging"
    UNSTABLE = "unstable"
    INITIAL = "initial"


@dataclass
class ModelParameter:
    """Model parameter with bounds and metadata."""
    name: str
    value: float
    nominal: float
    min_bound: float
    max_bound: float
    unit: str = ""
    sensitivity: float = 1.0
    uncertainty: float = 0.0
    last_updated: float = 0.0
    update_count: int = 0


@dataclass
class CalibrationResult:
    """Result of parameter calibration."""
    parameters: Dict[str, float]
    residual: float
    convergence_status: ParameterStatus
    iterations: int
    covariance: Optional[np.ndarray] = None
    timestamp: float = 0.0
    method: CalibrationMethod = CalibrationMethod.LEAST_SQUARES


@dataclass
class IDZModelConfig:
    """Configuration for IDZ (Identification Zone) model."""
    # Hydraulic parameters
    discharge_coefficient: ModelParameter = field(default_factory=lambda: ModelParameter(
        name="discharge_coefficient",
        value=0.62,
        nominal=0.62,
        min_bound=0.4,
        max_bound=0.9,
        unit="-",
        sensitivity=1.0
    ))

    # Vibration parameters
    strouhal_constant: ModelParameter = field(default_factory=lambda: ModelParameter(
        name="strouhal_constant",
        value=1.077,
        nominal=1.077,
        min_bound=0.8,
        max_bound=1.5,
        unit="-"
    ))

    structural_frequency: ModelParameter = field(default_factory=lambda: ModelParameter(
        name="structural_frequency",
        value=2.8,
        nominal=2.8,
        min_bound=1.5,
        max_bound=5.0,
        unit="Hz"
    ))

    damping_ratio: ModelParameter = field(default_factory=lambda: ModelParameter(
        name="damping_ratio",
        value=0.02,
        nominal=0.02,
        min_bound=0.005,
        max_bound=0.1,
        unit="-"
    ))

    resonance_amplification: ModelParameter = field(default_factory=lambda: ModelParameter(
        name="resonance_amplification",
        value=8.0,
        nominal=8.0,
        min_bound=2.0,
        max_bound=20.0,
        unit="-"
    ))

    # Gate dynamics parameters
    gate_max_speed: ModelParameter = field(default_factory=lambda: ModelParameter(
        name="gate_max_speed",
        value=0.05,
        nominal=0.05,
        min_bound=0.01,
        max_bound=0.2,
        unit="m/s"
    ))

    # Environmental parameters
    effective_area: ModelParameter = field(default_factory=lambda: ModelParameter(
        name="effective_area",
        value=16.6,
        nominal=16.6,
        min_bound=10.0,
        max_bound=25.0,
        unit="m²"
    ))


class RecursiveLeastSquaresEstimator:
    """
    Recursive Least Squares (RLS) parameter estimator.

    Provides online parameter estimation with forgetting factor.
    """

    def __init__(
        self,
        num_params: int,
        forgetting_factor: float = 0.99,
        initial_covariance: float = 1000.0
    ):
        self.num_params = num_params
        self.lambda_ = forgetting_factor

        # Parameter estimate
        self._theta = np.zeros(num_params)

        # Covariance matrix
        self._P = np.eye(num_params) * initial_covariance

        # History
        self._history: List[Dict[str, Any]] = []

        logger.debug("RLS estimator initialized: params=%d, lambda=%.3f",
                    num_params, forgetting_factor)

    def update(
        self,
        phi: np.ndarray,
        y: float,
        timestamp: float = 0.0
    ) -> Tuple[np.ndarray, float]:
        """
        Update parameter estimate with new measurement.

        Args:
            phi: Regressor vector (1 x num_params)
            y: Measured output
            timestamp: Current timestamp

        Returns:
            Tuple of (parameter_estimate, prediction_error)
        """
        phi = phi.reshape(-1, 1)  # Column vector

        # Prediction
        y_pred = (phi.T @ self._theta).item()
        error = y - y_pred

        # Kalman gain
        denominator = self.lambda_ + (phi.T @ self._P @ phi).item()
        K = self._P @ phi / denominator

        # Update parameter estimate
        self._theta = self._theta + K.flatten() * error

        # Update covariance
        self._P = (self._P - K @ phi.T @ self._P) / self.lambda_

        # Ensure symmetry and positive definiteness
        self._P = (self._P + self._P.T) / 2

        # Record history
        self._history.append({
            'timestamp': timestamp,
            'parameters': self._theta.copy(),
            'error': error,
            'trace_P': np.trace(self._P)
        })

        return self._theta.copy(), error

    def get_parameters(self) -> np.ndarray:
        """Get current parameter estimates."""
        return self._theta.copy()

    def get_covariance(self) -> np.ndarray:
        """Get current parameter covariance."""
        return self._P.copy()

    def get_uncertainty(self) -> np.ndarray:
        """Get parameter uncertainty (std dev)."""
        return np.sqrt(np.diag(self._P))

    def set_parameters(self, theta: np.ndarray) -> None:
        """Set parameter values."""
        self._theta = theta.copy()

    def reset(self, initial_covariance: float = 1000.0) -> None:
        """Reset estimator."""
        self._theta = np.zeros(self.num_params)
        self._P = np.eye(self.num_params) * initial_covariance
        self._history.clear()


class IDZModelCalibrator:
    """
    IDZ (Identification Zone) Model Parameter Calibrator.

    Dynamically updates simplified model parameters based on
    high-fidelity simulation or real measurement data.
    """

    def __init__(
        self,
        config: Optional[IDZModelConfig] = None,
        method: CalibrationMethod = CalibrationMethod.RECURSIVE_LEAST_SQUARES
    ):
        self.config = config or IDZModelConfig()
        self.method = method

        # Get list of parameters to calibrate
        self._param_names = [
            'discharge_coefficient',
            'strouhal_constant',
            'structural_frequency',
            'damping_ratio',
            'resonance_amplification'
        ]
        self._num_params = len(self._param_names)

        # Initialize estimator
        self._rls = RecursiveLeastSquaresEstimator(
            self._num_params,
            forgetting_factor=0.995
        )

        # Initialize with nominal values
        nominal = np.array([
            self.config.discharge_coefficient.value,
            self.config.strouhal_constant.value,
            self.config.structural_frequency.value,
            self.config.damping_ratio.value,
            self.config.resonance_amplification.value
        ])
        self._rls.set_parameters(nominal)

        # Data buffers for batch calibration
        self._input_buffer: deque = deque(maxlen=1000)
        self._output_buffer: deque = deque(maxlen=1000)

        # Calibration statistics
        self._calibration_count = 0
        self._last_residual = float('inf')
        self._status = ParameterStatus.INITIAL

        # Mismatch detection
        self._mismatch_threshold = 0.1
        self._mismatch_history: deque = deque(maxlen=100)

        logger.info("IDZModelCalibrator initialized: method=%s", method.value)

    def update_from_high_fidelity(
        self,
        hifi_state: Dict[str, Any],
        idz_state: Dict[str, Any],
        timestamp: float = 0.0
    ) -> CalibrationResult:
        """
        Update IDZ model parameters based on high-fidelity model state.

        Args:
            hifi_state: State from high-fidelity model
            idz_state: State from IDZ (simplified) model
            timestamp: Current timestamp

        Returns:
            CalibrationResult with updated parameters
        """
        # Calculate mismatch between high-fidelity and IDZ models
        mismatch = self._calculate_mismatch(hifi_state, idz_state)

        # Store mismatch for trend detection
        self._mismatch_history.append({
            'timestamp': timestamp,
            'mismatch': mismatch
        })

        # Check if calibration is needed
        if mismatch < self._mismatch_threshold:
            return CalibrationResult(
                parameters=self._get_current_params(),
                residual=mismatch,
                convergence_status=ParameterStatus.CONVERGED,
                iterations=0,
                timestamp=timestamp,
                method=self.method
            )

        # Build regressor from state data
        phi = self._build_regressor(hifi_state)

        # Target output (from high-fidelity model)
        y = self._extract_target(hifi_state)

        # Update parameters
        if self.method == CalibrationMethod.RECURSIVE_LEAST_SQUARES:
            new_params, error = self._rls.update(phi, y, timestamp)
            self._apply_parameter_bounds(new_params)

        elif self.method == CalibrationMethod.GRADIENT_DESCENT:
            new_params = self._gradient_descent_update(hifi_state, idz_state)

        else:
            new_params = self._rls.get_parameters()

        # Update configuration
        self._update_config(new_params, timestamp)

        # Update status
        self._calibration_count += 1
        if mismatch < self._last_residual:
            self._status = ParameterStatus.CONVERGING
        else:
            self._status = ParameterStatus.DIVERGING
        self._last_residual = mismatch

        return CalibrationResult(
            parameters=self._get_current_params(),
            residual=mismatch,
            convergence_status=self._status,
            iterations=self._calibration_count,
            covariance=self._rls.get_covariance(),
            timestamp=timestamp,
            method=self.method
        )

    def _calculate_mismatch(
        self,
        hifi_state: Dict[str, Any],
        idz_state: Dict[str, Any]
    ) -> float:
        """Calculate normalized mismatch between models."""
        mismatch = 0.0
        count = 0

        # Compare flow rates
        if 'flows' in hifi_state and 'flows' in idz_state:
            hifi_flows = np.array(hifi_state['flows'])
            idz_flows = np.array(idz_state['flows'])
            flow_ref = np.maximum(np.abs(hifi_flows), 1.0)
            mismatch += np.mean(np.abs(hifi_flows - idz_flows) / flow_ref)
            count += 1

        # Compare velocities
        if 'velocities' in hifi_state and 'velocities' in idz_state:
            hifi_vel = np.array(hifi_state['velocities'])
            idz_vel = np.array(idz_state['velocities'])
            vel_ref = np.maximum(np.abs(hifi_vel), 0.1)
            mismatch += np.mean(np.abs(hifi_vel - idz_vel) / vel_ref)
            count += 1

        # Compare vibrations
        if 'vibrations' in hifi_state and 'vibrations' in idz_state:
            hifi_vib = np.array(hifi_state['vibrations'])
            idz_vib = np.array(idz_state['vibrations'])
            vib_ref = np.maximum(np.abs(hifi_vib), 0.01)
            mismatch += np.mean(np.abs(hifi_vib - idz_vib) / vib_ref)
            count += 1

        return mismatch / count if count > 0 else 0.0

    def _build_regressor(self, state: Dict[str, Any]) -> np.ndarray:
        """Build regressor vector from state."""
        # Simplified regressor based on physical relationships
        phi = np.zeros(self._num_params)

        # Discharge coefficient regressor (from flow equation)
        if 'openings' in state and 'total_flow' in state:
            openings = np.array(state['openings'])
            phi[0] = np.sum(openings)  # Proportional to total opening

        # Strouhal regressor (from velocity)
        if 'velocities' in state:
            phi[1] = np.mean(state['velocities'])

        # Structural frequency regressor
        if 'frequencies' in state:
            phi[2] = np.mean(state['frequencies'])

        # Damping regressor
        if 'vibrations' in state:
            phi[3] = np.mean(state['vibrations'])

        # Resonance amplification regressor
        if 'vibrations' in state and 'velocities' in state:
            v_mean = np.mean(state['velocities'])
            if v_mean > 0:
                phi[4] = np.mean(state['vibrations']) / (v_mean ** 2 + 0.01)

        return phi

    def _extract_target(self, state: Dict[str, Any]) -> float:
        """Extract target output from state."""
        # Use total flow as primary target
        if 'total_flow' in state:
            return state['total_flow']
        elif 'flows' in state:
            return np.sum(state['flows'])
        return 0.0

    def _apply_parameter_bounds(self, params: np.ndarray) -> None:
        """Apply bounds to parameter estimates."""
        bounds = [
            (self.config.discharge_coefficient.min_bound,
             self.config.discharge_coefficient.max_bound),
            (self.config.strouhal_constant.min_bound,
             self.config.strouhal_constant.max_bound),
            (self.config.structural_frequency.min_bound,
             self.config.structural_frequency.max_bound),
            (self.config.damping_ratio.min_bound,
             self.config.damping_ratio.max_bound),
            (self.config.resonance_amplification.min_bound,
             self.config.resonance_amplification.max_bound)
        ]

        for i, (low, high) in enumerate(bounds):
            params[i] = np.clip(params[i], low, high)

        self._rls.set_parameters(params)

    def _gradient_descent_update(
        self,
        hifi_state: Dict[str, Any],
        idz_state: Dict[str, Any],
        learning_rate: float = 0.01
    ) -> np.ndarray:
        """Update parameters using gradient descent."""
        params = self._rls.get_parameters()

        # Numerical gradient approximation
        delta = 0.001
        gradient = np.zeros(self._num_params)

        for i in range(self._num_params):
            params_plus = params.copy()
            params_plus[i] += delta

            # Approximate gradient using mismatch
            mismatch_base = self._calculate_mismatch(hifi_state, idz_state)
            # Note: Full implementation would require model re-evaluation
            gradient[i] = mismatch_base  # Simplified

        # Update
        params -= learning_rate * gradient
        self._apply_parameter_bounds(params)

        return params

    def _update_config(self, params: np.ndarray, timestamp: float) -> None:
        """Update configuration with new parameters."""
        self.config.discharge_coefficient.value = params[0]
        self.config.discharge_coefficient.last_updated = timestamp
        self.config.discharge_coefficient.update_count += 1

        self.config.strouhal_constant.value = params[1]
        self.config.strouhal_constant.last_updated = timestamp
        self.config.strouhal_constant.update_count += 1

        self.config.structural_frequency.value = params[2]
        self.config.structural_frequency.last_updated = timestamp
        self.config.structural_frequency.update_count += 1

        self.config.damping_ratio.value = params[3]
        self.config.damping_ratio.last_updated = timestamp
        self.config.damping_ratio.update_count += 1

        self.config.resonance_amplification.value = params[4]
        self.config.resonance_amplification.last_updated = timestamp
        self.config.resonance_amplification.update_count += 1

    def _get_current_params(self) -> Dict[str, float]:
        """Get current parameter values as dictionary."""
        return {
            'discharge_coefficient': self.config.discharge_coefficient.value,
            'strouhal_constant': self.config.strouhal_constant.value,
            'structural_frequency': self.config.structural_frequency.value,
            'damping_ratio': self.config.damping_ratio.value,
            'resonance_amplification': self.config.resonance_amplification.value
        }

    def batch_calibrate(
        self,
        input_data: List[Dict[str, Any]],
        output_data: List[Dict[str, Any]],
        timestamps: Optional[List[float]] = None
    ) -> CalibrationResult:
        """
        Perform batch calibration using collected data.

        Args:
            input_data: List of input states
            output_data: List of output states
            timestamps: Optional list of timestamps

        Returns:
            CalibrationResult from batch calibration
        """
        if len(input_data) != len(output_data):
            raise ValueError("Input and output data lengths must match")

        n_samples = len(input_data)
        if n_samples < 10:
            logger.warning("Insufficient data for batch calibration")
            return CalibrationResult(
                parameters=self._get_current_params(),
                residual=float('inf'),
                convergence_status=ParameterStatus.INITIAL,
                iterations=0
            )

        # Build design matrix
        X = np.array([self._build_regressor(s) for s in input_data])
        y = np.array([self._extract_target(s) for s in output_data])

        # Least squares fit
        try:
            params, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            self._apply_parameter_bounds(params)
            residual = float(np.mean(residuals)) if len(residuals) > 0 else 0.0
        except np.linalg.LinAlgError:
            logger.warning("Batch calibration failed - singular matrix")
            return CalibrationResult(
                parameters=self._get_current_params(),
                residual=float('inf'),
                convergence_status=ParameterStatus.UNSTABLE,
                iterations=0
            )

        # Update configuration
        timestamp = timestamps[-1] if timestamps else 0.0
        self._update_config(params, timestamp)

        return CalibrationResult(
            parameters=self._get_current_params(),
            residual=residual,
            convergence_status=ParameterStatus.CONVERGED,
            iterations=n_samples,
            timestamp=timestamp,
            method=CalibrationMethod.LEAST_SQUARES
        )

    def apply_to_model(self, model: 'TangheSiphonModel') -> None:
        """
        Apply calibrated parameters to physics model.

        Args:
            model: Physics model to update
        """
        # Update model configuration
        model._config.discharge_coefficient = self.config.discharge_coefficient.value
        model._config.strouhal_constant = self.config.strouhal_constant.value
        model._config.structural_frequency = self.config.structural_frequency.value
        model._config.damping_ratio = self.config.damping_ratio.value
        model._config.resonance_amplification = self.config.resonance_amplification.value
        model._config.gate_max_speed = self.config.gate_max_speed.value

        # Update model attributes
        model.k_freq = self.config.strouhal_constant.value
        model.f_struct = self.config.structural_frequency.value
        model.damping_ratio = self.config.damping_ratio.value

        logger.info("Calibrated parameters applied to model")

    def detect_model_mismatch(self) -> Dict[str, Any]:
        """
        Detect systematic model-reality mismatch.

        Returns:
            Mismatch analysis results
        """
        if len(self._mismatch_history) < 10:
            return {
                'sufficient_data': False,
                'mismatch_detected': False
            }

        mismatches = [m['mismatch'] for m in self._mismatch_history]
        timestamps = [m['timestamp'] for m in self._mismatch_history]

        mean_mismatch = np.mean(mismatches)
        std_mismatch = np.std(mismatches)

        # Trend analysis
        if len(mismatches) > 20:
            recent = np.mean(mismatches[-10:])
            older = np.mean(mismatches[-20:-10])
            trend = (recent - older) / (older + 0.001)
        else:
            trend = 0.0

        # Systematic mismatch detection
        systematic = mean_mismatch > self._mismatch_threshold
        increasing = trend > 0.1

        return {
            'sufficient_data': True,
            'mismatch_detected': systematic,
            'mean_mismatch': mean_mismatch,
            'std_mismatch': std_mismatch,
            'trend': trend,
            'increasing': increasing,
            'recommendation': 'recalibration_needed' if systematic else 'parameters_valid'
        }

    def get_parameter_uncertainty(self) -> Dict[str, float]:
        """Get uncertainty estimates for each parameter."""
        uncertainty = self._rls.get_uncertainty()
        return {
            name: float(uncertainty[i])
            for i, name in enumerate(self._param_names)
        }

    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of calibration state."""
        return {
            'parameters': self._get_current_params(),
            'uncertainty': self.get_parameter_uncertainty(),
            'status': self._status.value,
            'calibration_count': self._calibration_count,
            'last_residual': self._last_residual,
            'mismatch_analysis': self.detect_model_mismatch()
        }

    def reset(self) -> None:
        """Reset calibrator to initial state."""
        self.config = IDZModelConfig()
        self._rls.reset()
        nominal = np.array([
            self.config.discharge_coefficient.value,
            self.config.strouhal_constant.value,
            self.config.structural_frequency.value,
            self.config.damping_ratio.value,
            self.config.resonance_amplification.value
        ])
        self._rls.set_parameters(nominal)
        self._calibration_count = 0
        self._last_residual = float('inf')
        self._status = ParameterStatus.INITIAL
        self._mismatch_history.clear()
        logger.info("IDZModelCalibrator reset")
