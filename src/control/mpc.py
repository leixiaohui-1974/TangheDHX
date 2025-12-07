# -*- coding: utf-8 -*-
"""
Spectral Model Predictive Control (Spectral-MPC).

This module implements an MPC controller that optimizes gate openings
to meet flow demand while avoiding resonance frequencies.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from src.config import get_config, ControlConfig, PhysicsConfig

logger = logging.getLogger(__name__)


class SpectralMPC:
    """
    Spectral Model Predictive Control (Spectral-MPC).

    Optimizes gate openings to meet flow demand while avoiding resonance
    frequencies using a multi-objective optimization approach.

    The cost function includes:
    - Flow tracking: Minimize deviation from target flow
    - Action penalty: Minimize control effort
    - Spectral avoidance: Penalize operation near resonance frequencies

    Attributes:
        model: Reference to the physics model
        width: Gate width [m]
        k_freq: Strouhal constant for frequency calculation
        f_struct: Structural natural frequency [Hz]
    """

    def __init__(
        self,
        model: 'TangheSiphonModel',
        control_config: Optional[ControlConfig] = None,
        physics_config: Optional[PhysicsConfig] = None
    ) -> None:
        """
        Initialize the MPC controller.

        Args:
            model: Reference to the physics model
            control_config: Control configuration. If None, uses global config.
            physics_config: Physics configuration. If None, uses global config.
        """
        self.model = model
        self._ctrl_cfg = control_config or get_config().control
        self._phys_cfg = physics_config or get_config().physics

        self.width = self._phys_cfg.gate_width
        self.k_freq = self._phys_cfg.strouhal_constant
        self.f_struct = self._phys_cfg.structural_frequency

        # Optimization weights
        self.alpha = self._ctrl_cfg.alpha_flow_tracking
        self.beta = self._ctrl_cfg.beta_action_penalty
        self.gamma = self._ctrl_cfg.gamma_spectral_avoidance

        self.last_target_openings = np.zeros(self._phys_cfg.num_gates)

        logger.debug(
            "SpectralMPC initialized: alpha=%.2f, beta=%.2f, gamma=%.2f",
            self.alpha, self.beta, self.gamma
        )

    def get_target_openings(
        self,
        target_flow: float,
        current_openings: np.ndarray,
        current_head_diff: float
    ) -> np.ndarray:
        """
        Solve optimization problem to find optimal gate openings.

        Args:
            target_flow: Desired total flow rate [m³/s]
            current_openings: Current gate opening heights [m]
            current_head_diff: Current head difference [m]

        Returns:
            Optimal gate opening heights [m]

        Raises:
            ValueError: If input arrays have wrong dimensions
        """
        if len(current_openings) != self._phys_cfg.num_gates:
            raise ValueError(
                f"Expected {self._phys_cfg.num_gates} openings, "
                f"got {len(current_openings)}"
            )

        if current_head_diff < 0:
            logger.warning("Negative head difference: %.2f m", current_head_diff)
            current_head_diff = 0.0

        # Initial guess: current openings
        x0 = current_openings.copy()

        # Bounds: 0 <= e <= max_opening
        max_opening = self._phys_cfg.max_gate_opening
        bounds = [(0.0, max_opening) for _ in range(self._phys_cfg.num_gates)]

        def cost_function(x: np.ndarray) -> float:
            """Multi-objective cost function."""
            flows, velocities = self._predict_physics(x, current_head_diff)

            # 1. Flow Tracking
            total_flow = np.sum(flows)
            j_flow = self.alpha * (total_flow - target_flow) ** 2

            # 2. Action Penalty
            j_action = self.beta * np.sum((x - current_openings) ** 2)

            # 3. Spectral Potential
            j_spectral = self._compute_spectral_cost(velocities)

            return j_flow + j_action + j_spectral

        # Run optimization
        result: OptimizeResult = minimize(
            cost_function,
            x0,
            bounds=bounds,
            method='SLSQP',
            options={'maxiter': 100, 'ftol': 1e-6}
        )

        if result.success:
            self.last_target_openings = result.x
            logger.debug(
                "MPC optimization succeeded: targets=%s, cost=%.4f",
                result.x.round(3), result.fun
            )
            return result.x
        else:
            logger.warning(
                "MPC optimization failed: %s. Using current openings.",
                result.message
            )
            return current_openings

    def _predict_physics(
        self,
        openings: np.ndarray,
        head_diff: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict flows and velocities for given openings.

        Args:
            openings: Gate opening heights [m]
            head_diff: Head difference [m]

        Returns:
            Tuple of (flows, velocities) arrays
        """
        cfg = self._phys_cfg
        flows = []
        velocities = []

        for e in openings:
            q = (
                cfg.discharge_coefficient
                * self.width
                * e
                * np.sqrt(2 * cfg.gravity * head_diff)
            )
            v = q / cfg.effective_area
            flows.append(q)
            velocities.append(v)

        return np.array(flows), np.array(velocities)

    def _compute_spectral_cost(self, velocities: np.ndarray) -> float:
        """
        Compute spectral avoidance cost using potential function.

        Args:
            velocities: Predicted velocities [m/s]

        Returns:
            Spectral cost value
        """
        epsilon = self._ctrl_cfg.spectral_epsilon
        min_v = self._ctrl_cfg.min_velocity_threshold
        j_spectral = 0.0

        for v in velocities:
            if v < min_v:
                fs = 0.0
            else:
                fs = self.k_freq * v

            # Potential function: 1 / (dist² + ε)
            dist = abs(fs - self.f_struct)
            psi = 1.0 / (dist ** 2 + epsilon)
            j_spectral += psi

        return self.gamma * j_spectral

    def reset(self) -> None:
        """Reset controller state."""
        self.last_target_openings = np.zeros(self._phys_cfg.num_gates)
        logger.debug("MPC controller reset")
