# -*- coding: utf-8 -*-
"""
Gate Controller (Actuator Interface).

This module provides the interface to gate actuators for the digital twin.
"""

import logging
from typing import Optional

import numpy as np

from src.config import get_config, PhysicsConfig

logger = logging.getLogger(__name__)


class GateController:
    """
    Interface to the gate actuators.

    Manages target openings and provides bounds checking.

    Attributes:
        model: Reference to the physics model
        target_openings: Target opening heights for each gate [m]
    """

    def __init__(
        self,
        model: 'TangheSiphonModel',
        config: Optional[PhysicsConfig] = None
    ) -> None:
        """
        Initialize the gate controller.

        Args:
            model: Reference to the physics model
            config: Physics configuration. If None, uses global config.
        """
        self.model = model
        self._config = config or get_config().physics
        self.target_openings = np.copy(model.gate_openings)

        logger.debug(
            "GateController initialized for %d gates",
            model.num_gates
        )

    def set_opening(self, gate_index: int, value: float) -> None:
        """
        Set target opening for a specific gate.

        Args:
            gate_index: Index of the gate (0-based)
            value: Target opening height [m]

        Raises:
            ValueError: If gate_index is out of range or value is invalid
        """
        if not 0 <= gate_index < self.model.num_gates:
            raise ValueError(
                f"Gate index must be 0-{self.model.num_gates - 1}"
            )

        # Clamp value to valid range
        min_opening = self._config.min_gate_opening
        max_opening = self._config.max_gate_opening
        clamped_value = np.clip(value, min_opening, max_opening)

        if clamped_value != value:
            logger.debug(
                "Gate %d target clamped: %.3f -> %.3f",
                gate_index, value, clamped_value
            )

        self.target_openings[gate_index] = clamped_value

    def get_target_openings(self) -> np.ndarray:
        """
        Get current target openings.

        Returns:
            Array of target opening heights [m]
        """
        return self.target_openings.copy()

    def set_all_openings(self, values: np.ndarray) -> None:
        """
        Set target openings for all gates.

        Args:
            values: Array of target opening heights [m]

        Raises:
            ValueError: If values has wrong length
        """
        if len(values) != self.model.num_gates:
            raise ValueError(
                f"Expected {self.model.num_gates} values, got {len(values)}"
            )

        for i, value in enumerate(values):
            self.set_opening(i, value)

    def reset(self) -> None:
        """Reset controller to initial state."""
        self.target_openings = np.zeros(self.model.num_gates)
        logger.debug("GateController reset")
