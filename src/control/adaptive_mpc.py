# -*- coding: utf-8 -*-
"""
Adaptive Model Predictive Control (Adaptive-MPC).

This module implements an adaptive MPC controller that:
- Automatically tunes weights based on operating conditions
- Learns from historical performance
- Adapts to changing system dynamics
- Handles uncertainty in model parameters
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Deque
from collections import deque

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from src.config import get_config, ControlConfig, PhysicsConfig

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveState:
    """State for adaptive parameter tuning."""
    # Performance metrics
    flow_errors: Deque[float]
    vibration_peaks: Deque[float]
    control_efforts: Deque[float]

    # Adapted parameters
    alpha: float  # Flow tracking weight
    beta: float   # Action penalty weight
    gamma: float  # Spectral avoidance weight

    # Learning rate
    learning_rate: float = 0.01

    # Bounds
    alpha_bounds: Tuple[float, float] = (0.5, 5.0)
    beta_bounds: Tuple[float, float] = (0.01, 1.0)
    gamma_bounds: Tuple[float, float] = (1.0, 50.0)


class AdaptiveMPC:
    """
    Adaptive Model Predictive Control with online parameter tuning.

    This controller extends the basic MPC with:
    - Online weight adaptation based on performance
    - Model uncertainty handling
    - Multi-objective optimization with adaptive priorities
    - Constraint softening for infeasible regions
    """

    def __init__(
        self,
        model: 'TangheSiphonModel',
        control_config: Optional[ControlConfig] = None,
        physics_config: Optional[PhysicsConfig] = None,
        adaptation_enabled: bool = True
    ) -> None:
        """
        Initialize the adaptive MPC controller.

        Args:
            model: Reference to the physics model
            control_config: Control configuration
            physics_config: Physics configuration
            adaptation_enabled: Enable online adaptation
        """
        self.model = model
        self._ctrl_cfg = control_config or get_config().control
        self._phys_cfg = physics_config or get_config().physics

        self.width = self._phys_cfg.gate_width
        self.k_freq = self._phys_cfg.strouhal_constant
        self.f_struct = self._phys_cfg.structural_frequency

        # Adaptive state
        self._adaptation_enabled = adaptation_enabled
        self._state = AdaptiveState(
            flow_errors=deque(maxlen=100),
            vibration_peaks=deque(maxlen=100),
            control_efforts=deque(maxlen=100),
            alpha=self._ctrl_cfg.alpha_flow_tracking,
            beta=self._ctrl_cfg.beta_action_penalty,
            gamma=self._ctrl_cfg.gamma_spectral_avoidance,
        )

        # Model uncertainty estimation
        self._model_error_history: Deque[float] = deque(maxlen=50)
        self._uncertainty_factor: float = 1.0

        # Previous values for tracking
        self._last_target_openings = np.zeros(self._phys_cfg.num_gates)
        self._last_predicted_flow: float = 0.0
        self._last_actual_flow: float = 0.0

        logger.info(
            "AdaptiveMPC initialized: alpha=%.2f, beta=%.2f, gamma=%.2f, "
            "adaptation=%s",
            self._state.alpha, self._state.beta, self._state.gamma,
            "enabled" if adaptation_enabled else "disabled"
        )

    def get_target_openings(
        self,
        target_flow: float,
        current_openings: np.ndarray,
        current_head_diff: float,
        vibration_readings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Solve optimization problem with adaptive weights.

        Args:
            target_flow: Desired total flow rate [m³/s]
            current_openings: Current gate opening heights [m]
            current_head_diff: Current head difference [m]
            vibration_readings: Optional vibration sensor readings [g]

        Returns:
            Optimal gate opening heights [m]
        """
        if len(current_openings) != self._phys_cfg.num_gates:
            raise ValueError(
                f"Expected {self._phys_cfg.num_gates} openings, "
                f"got {len(current_openings)}"
            )

        # Update model uncertainty based on prediction error
        self._update_uncertainty()

        # Adapt weights based on current conditions
        if self._adaptation_enabled:
            self._adapt_weights(target_flow, vibration_readings)

        # Handle edge cases
        if current_head_diff < 0:
            logger.warning("Negative head difference: %.2f m", current_head_diff)
            current_head_diff = max(0.0, current_head_diff)

        # Solve optimization
        result = self._solve_optimization(
            target_flow, current_openings, current_head_diff
        )

        # Update tracking
        self._last_target_openings = result
        predicted_flows, _ = self._predict_physics(result, current_head_diff)
        self._last_predicted_flow = float(np.sum(predicted_flows))

        return result

    def _solve_optimization(
        self,
        target_flow: float,
        current_openings: np.ndarray,
        head_diff: float
    ) -> np.ndarray:
        """Solve the MPC optimization problem."""
        x0 = current_openings.copy()
        max_opening = self._phys_cfg.max_gate_opening
        bounds = [(0.0, max_opening) for _ in range(self._phys_cfg.num_gates)]

        # Use current adaptive weights
        alpha = self._state.alpha
        beta = self._state.beta
        gamma = self._state.gamma

        def cost_function(x: np.ndarray) -> float:
            flows, velocities = self._predict_physics(x, head_diff)

            # 1. Flow Tracking (with uncertainty)
            total_flow = np.sum(flows)
            j_flow = alpha * (total_flow - target_flow) ** 2

            # 2. Action Penalty (smooth control)
            j_action = beta * np.sum((x - current_openings) ** 2)

            # 3. Spectral Avoidance (with uncertainty margin)
            j_spectral = self._compute_spectral_cost(velocities, gamma)

            # 4. Constraint violation penalty (soft constraints)
            j_constraint = self._compute_constraint_penalty(x, velocities)

            return j_flow + j_action + j_spectral + j_constraint

        # Multiple restarts for global optimization
        best_result = None
        best_cost = float('inf')

        for restart in range(3):
            if restart == 0:
                x_init = x0
            elif restart == 1:
                # Try uniform distribution
                x_init = np.ones(self._phys_cfg.num_gates) * (target_flow / 3 / 26.3)
                x_init = np.clip(x_init, 0, max_opening)
            else:
                # Random perturbation
                x_init = x0 + np.random.normal(0, 0.5, self._phys_cfg.num_gates)
                x_init = np.clip(x_init, 0, max_opening)

            result = minimize(
                cost_function,
                x_init,
                bounds=bounds,
                method='SLSQP',
                options={'maxiter': 100, 'ftol': 1e-6}
            )

            if result.success and result.fun < best_cost:
                best_cost = result.fun
                best_result = result

        if best_result is not None and best_result.success:
            logger.debug(
                "AdaptiveMPC optimization: targets=%s, cost=%.4f, "
                "weights=(%.2f, %.2f, %.2f)",
                best_result.x.round(3), best_result.fun,
                alpha, beta, gamma
            )
            return best_result.x

        logger.warning("All MPC optimizations failed, using current openings")
        return current_openings

    def _predict_physics(
        self,
        openings: np.ndarray,
        head_diff: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict flows and velocities with uncertainty."""
        cfg = self._phys_cfg
        flows = []
        velocities = []

        # Apply uncertainty factor to predictions
        cd_effective = cfg.discharge_coefficient * self._uncertainty_factor

        for e in openings:
            q = cd_effective * self.width * e * np.sqrt(2 * cfg.gravity * head_diff)
            v = q / cfg.effective_area
            flows.append(q)
            velocities.append(v)

        return np.array(flows), np.array(velocities)

    def _compute_spectral_cost(
        self,
        velocities: np.ndarray,
        gamma: float
    ) -> float:
        """Compute spectral avoidance cost with safety margin."""
        epsilon = self._ctrl_cfg.spectral_epsilon
        min_v = self._ctrl_cfg.min_velocity_threshold
        j_spectral = 0.0

        # Add safety margin based on uncertainty
        safety_margin = 0.1 * self._uncertainty_factor

        for v in velocities:
            if v < min_v:
                fs = 0.0
            else:
                fs = self.k_freq * v

            # Potential function with safety margin
            dist = abs(fs - self.f_struct)
            effective_dist = max(0, dist - safety_margin)
            psi = 1.0 / (effective_dist ** 2 + epsilon)
            j_spectral += psi

        return gamma * j_spectral

    def _compute_constraint_penalty(
        self,
        openings: np.ndarray,
        velocities: np.ndarray
    ) -> float:
        """Compute soft constraint violation penalty."""
        penalty = 0.0

        # Velocity constraint (avoid very high velocities)
        max_velocity = 4.0  # m/s
        for v in velocities:
            if v > max_velocity:
                penalty += 100.0 * (v - max_velocity) ** 2

        # Opening rate constraint (implicit through action penalty)
        # Already handled by beta term

        return penalty

    def _update_uncertainty(self) -> None:
        """Update model uncertainty estimation."""
        # Compare predicted vs actual flow
        actual_flow = float(np.sum(self.model.flow_rates))

        if self._last_predicted_flow > 0:
            error = abs(actual_flow - self._last_predicted_flow) / self._last_predicted_flow
            self._model_error_history.append(error)

            if len(self._model_error_history) >= 10:
                mean_error = np.mean(list(self._model_error_history))
                # Adjust uncertainty factor based on error
                self._uncertainty_factor = 1.0 / (1.0 + mean_error)
                self._uncertainty_factor = np.clip(self._uncertainty_factor, 0.8, 1.2)

        self._last_actual_flow = actual_flow

    def _adapt_weights(
        self,
        target_flow: float,
        vibration_readings: Optional[np.ndarray]
    ) -> None:
        """Adapt MPC weights based on performance metrics."""
        # Collect current metrics
        actual_flow = float(np.sum(self.model.flow_rates))
        flow_error = abs(actual_flow - target_flow)
        max_vib = float(np.max(self.model.vibration_accel))

        self._state.flow_errors.append(flow_error)
        self._state.vibration_peaks.append(max_vib)

        if len(self._state.flow_errors) < 20:
            return  # Need more data

        # Compute performance statistics
        recent_flow_errors = list(self._state.flow_errors)[-20:]
        recent_vibs = list(self._state.vibration_peaks)[-20:]

        mean_flow_error = np.mean(recent_flow_errors)
        mean_vib = np.mean(recent_vibs)
        max_recent_vib = np.max(recent_vibs)

        # Adapt alpha (flow tracking)
        if mean_flow_error > 20.0:
            # Poor flow tracking - increase alpha
            self._state.alpha *= (1 + self._state.learning_rate)
        elif mean_flow_error < 5.0 and max_recent_vib < 0.2:
            # Good tracking and safe - can decrease alpha
            self._state.alpha *= (1 - self._state.learning_rate * 0.5)

        # Adapt gamma (spectral avoidance)
        if max_recent_vib > 0.4:
            # High vibration - increase gamma significantly
            self._state.gamma *= (1 + self._state.learning_rate * 2)
        elif max_recent_vib < 0.1:
            # Very safe - can decrease gamma
            self._state.gamma *= (1 - self._state.learning_rate * 0.3)

        # Apply bounds
        self._state.alpha = np.clip(
            self._state.alpha,
            *self._state.alpha_bounds
        )
        self._state.gamma = np.clip(
            self._state.gamma,
            *self._state.gamma_bounds
        )

        logger.debug(
            "Weights adapted: alpha=%.3f, gamma=%.3f (error=%.1f, vib=%.3f)",
            self._state.alpha, self._state.gamma, mean_flow_error, max_recent_vib
        )

    def get_weights(self) -> Tuple[float, float, float]:
        """Get current adaptive weights."""
        return self._state.alpha, self._state.beta, self._state.gamma

    def get_uncertainty(self) -> float:
        """Get current model uncertainty factor."""
        return self._uncertainty_factor

    def set_adaptation_enabled(self, enabled: bool) -> None:
        """Enable or disable online adaptation."""
        self._adaptation_enabled = enabled
        logger.info("Adaptation %s", "enabled" if enabled else "disabled")

    def reset(self) -> None:
        """Reset controller state."""
        self._state.flow_errors.clear()
        self._state.vibration_peaks.clear()
        self._state.control_efforts.clear()
        self._state.alpha = self._ctrl_cfg.alpha_flow_tracking
        self._state.beta = self._ctrl_cfg.beta_action_penalty
        self._state.gamma = self._ctrl_cfg.gamma_spectral_avoidance
        self._model_error_history.clear()
        self._uncertainty_factor = 1.0
        self._last_target_openings = np.zeros(self._phys_cfg.num_gates)
        self._last_predicted_flow = 0.0
        logger.debug("AdaptiveMPC reset")
