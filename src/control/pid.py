# -*- coding: utf-8 -*-
"""
PID Controller with Anti-Windup and Bumpless Transfer.

This module provides a robust PID controller as a backup/fallback
controller when MPC is unavailable or during transitions.
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

from src.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class PIDGains:
    """PID controller gains."""
    kp: float  # Proportional gain
    ki: float  # Integral gain
    kd: float  # Derivative gain

    # Anti-windup limits
    integral_min: float = -50.0
    integral_max: float = 50.0

    # Output limits
    output_min: float = 0.0
    output_max: float = 5.0

    # Derivative filter coefficient (0-1, lower = more filtering)
    derivative_filter: float = 0.1


class PIDController:
    """
    Single-channel PID controller with anti-windup.

    Features:
    - Anti-windup (integral clamping and back-calculation)
    - Derivative filtering (low-pass filter on derivative term)
    - Bumpless transfer (for mode switching)
    - Setpoint weighting
    """

    def __init__(self, gains: PIDGains, name: str = "PID") -> None:
        """
        Initialize PID controller.

        Args:
            gains: PID gains and limits
            name: Controller identifier for logging
        """
        self.gains = gains
        self.name = name

        # State
        self._integral: float = 0.0
        self._last_error: float = 0.0
        self._last_derivative: float = 0.0
        self._last_output: float = 0.0
        self._enabled: bool = True

        logger.debug(
            "%s initialized: Kp=%.3f, Ki=%.3f, Kd=%.3f",
            name, gains.kp, gains.ki, gains.kd
        )

    def compute(
        self,
        setpoint: float,
        measurement: float,
        dt: float
    ) -> float:
        """
        Compute PID control output.

        Args:
            setpoint: Desired value
            measurement: Current measured value
            dt: Time step [s]

        Returns:
            Control output
        """
        if not self._enabled or dt <= 0:
            return self._last_output

        error = setpoint - measurement

        # Proportional term
        p_term = self.gains.kp * error

        # Integral term with anti-windup
        self._integral += error * dt
        self._integral = np.clip(
            self._integral,
            self.gains.integral_min,
            self.gains.integral_max
        )
        i_term = self.gains.ki * self._integral

        # Derivative term with filtering
        if dt > 0:
            raw_derivative = (error - self._last_error) / dt
            # Low-pass filter
            alpha = self.gains.derivative_filter
            filtered_derivative = (
                alpha * raw_derivative +
                (1 - alpha) * self._last_derivative
            )
            self._last_derivative = filtered_derivative
        else:
            filtered_derivative = self._last_derivative

        d_term = self.gains.kd * filtered_derivative

        # Total output
        output = p_term + i_term + d_term

        # Output limiting
        output_clamped = np.clip(
            output,
            self.gains.output_min,
            self.gains.output_max
        )

        # Anti-windup: back-calculation
        if output != output_clamped and self.gains.ki > 0:
            # Reduce integral to prevent further windup
            self._integral -= (output - output_clamped) / self.gains.ki

        self._last_error = error
        self._last_output = output_clamped

        return output_clamped

    def set_output(self, output: float) -> None:
        """
        Set output for bumpless transfer.

        Args:
            output: Current output value to track
        """
        self._last_output = output
        # Back-calculate integral to match current output
        if self.gains.ki > 0:
            # Assume P and D terms are zero at transfer
            self._integral = output / self.gains.ki

    def reset(self) -> None:
        """Reset controller state."""
        self._integral = 0.0
        self._last_error = 0.0
        self._last_derivative = 0.0
        self._last_output = 0.0

    def enable(self, enabled: bool = True) -> None:
        """Enable or disable the controller."""
        self._enabled = enabled


class MultiChannelPID:
    """
    Multi-channel PID controller for gate control.

    Provides coordinated control of multiple gates with:
    - Individual PID controllers per gate
    - Flow distribution logic
    - Resonance avoidance through asymmetric control
    """

    def __init__(
        self,
        num_gates: int = 3,
        flow_gains: Optional[PIDGains] = None
    ) -> None:
        """
        Initialize multi-channel PID controller.

        Args:
            num_gates: Number of gates to control
            flow_gains: PID gains for flow control (shared)
        """
        self.num_gates = num_gates

        # Default gains
        if flow_gains is None:
            flow_gains = PIDGains(
                kp=0.05,   # Conservative proportional gain
                ki=0.01,   # Small integral for steady-state
                kd=0.005,  # Small derivative for damping
                output_min=0.0,
                output_max=5.0
            )

        # Create individual controllers
        self.controllers: List[PIDController] = [
            PIDController(flow_gains, name=f"Gate{i}PID")
            for i in range(num_gates)
        ]

        # Flow distribution weights (can be adapted)
        self._distribution_weights = np.ones(num_gates) / num_gates

        # Resonance avoidance
        self._resonance_velocity = 2.6  # m/s (from physics)
        self._avoidance_margin = 0.3  # m/s

        logger.info("MultiChannelPID initialized for %d gates", num_gates)

    def compute(
        self,
        target_flow: float,
        current_flows: np.ndarray,
        current_velocities: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Compute control outputs for all gates.

        Args:
            target_flow: Total desired flow [m³/s]
            current_flows: Current flow per gate [m³/s]
            current_velocities: Current velocity per gate [m/s]
            dt: Time step [s]

        Returns:
            Gate opening commands [m]
        """
        # Distribute target flow
        flow_targets = self._distribute_flow(
            target_flow, current_velocities
        )

        # Compute individual PID outputs
        outputs = np.zeros(self.num_gates)
        for i, (controller, target, current) in enumerate(
            zip(self.controllers, flow_targets, current_flows)
        ):
            outputs[i] = controller.compute(target, current, dt)

        return outputs

    def _distribute_flow(
        self,
        total_flow: float,
        velocities: np.ndarray
    ) -> np.ndarray:
        """
        Distribute flow among gates avoiding resonance.

        Args:
            total_flow: Total desired flow [m³/s]
            velocities: Current velocities per gate [m/s]

        Returns:
            Flow targets per gate [m³/s]
        """
        # Check which gates are near resonance
        near_resonance = np.abs(velocities - self._resonance_velocity) < self._avoidance_margin

        if np.all(near_resonance):
            # All gates near resonance - use asymmetric distribution
            weights = np.array([0.2, 0.5, 0.3])  # Asymmetric
        elif np.any(near_resonance):
            # Some gates near resonance - redistribute to others
            weights = np.where(near_resonance, 0.2, 0.4)
            weights /= np.sum(weights)
        else:
            # Normal operation - equal distribution
            weights = self._distribution_weights

        return total_flow * weights

    def reset(self) -> None:
        """Reset all controllers."""
        for controller in self.controllers:
            controller.reset()

    def set_distribution_weights(self, weights: np.ndarray) -> None:
        """Set flow distribution weights."""
        if len(weights) != self.num_gates:
            raise ValueError(f"Expected {self.num_gates} weights")
        self._distribution_weights = weights / np.sum(weights)


class HybridController:
    """
    Hybrid controller combining MPC and PID.

    Features:
    - Automatic switching based on conditions
    - Bumpless transfer between controllers
    - Fallback to PID when MPC fails
    """

    def __init__(
        self,
        mpc_controller: 'AdaptiveMPC',
        pid_controller: MultiChannelPID,
        model: 'TangheSiphonModel'
    ) -> None:
        """
        Initialize hybrid controller.

        Args:
            mpc_controller: Primary MPC controller
            pid_controller: Backup PID controller
            model: Reference to physics model
        """
        self.mpc = mpc_controller
        self.pid = pid_controller
        self.model = model

        self._use_mpc: bool = True
        self._mpc_failure_count: int = 0
        self._max_mpc_failures: int = 3

        logger.info("HybridController initialized")

    def compute(
        self,
        target_flow: float,
        current_openings: np.ndarray,
        head_diff: float,
        dt: float
    ) -> Tuple[np.ndarray, str]:
        """
        Compute control output using appropriate controller.

        Args:
            target_flow: Desired total flow [m³/s]
            current_openings: Current gate openings [m]
            head_diff: Head difference [m]
            dt: Time step [s]

        Returns:
            Tuple of (control outputs, controller used)
        """
        controller_used = "MPC"

        if self._use_mpc:
            try:
                outputs = self.mpc.get_target_openings(
                    target_flow, current_openings, head_diff
                )
                self._mpc_failure_count = 0

                # Validate outputs
                if np.any(np.isnan(outputs)) or np.any(np.isinf(outputs)):
                    raise ValueError("MPC produced invalid outputs")

            except Exception as e:
                logger.warning("MPC failed: %s", e)
                self._mpc_failure_count += 1

                if self._mpc_failure_count >= self._max_mpc_failures:
                    logger.warning("Switching to PID (MPC failed %d times)",
                                   self._mpc_failure_count)
                    self._use_mpc = False

                # Fall back to PID
                outputs = self.pid.compute(
                    target_flow,
                    self.model.flow_rates,
                    self.model.velocities,
                    dt
                )
                controller_used = "PID"
        else:
            outputs = self.pid.compute(
                target_flow,
                self.model.flow_rates,
                self.model.velocities,
                dt
            )
            controller_used = "PID"

            # Try to recover MPC
            self._mpc_failure_count = max(0, self._mpc_failure_count - 1)
            if self._mpc_failure_count == 0:
                logger.info("Attempting to recover MPC")
                self._use_mpc = True

        return outputs, controller_used

    def force_mode(self, use_mpc: bool) -> None:
        """Force controller mode."""
        self._use_mpc = use_mpc
        self._mpc_failure_count = 0
        logger.info("Controller mode forced to %s", "MPC" if use_mpc else "PID")

    def reset(self) -> None:
        """Reset both controllers."""
        self.mpc.reset()
        self.pid.reset()
        self._use_mpc = True
        self._mpc_failure_count = 0
