# -*- coding: utf-8 -*-
"""
Local Controller (PLC level).

This module implements the fast-loop control logic including
pass-through commands and dithering for vibration mitigation.
"""

import logging
from typing import List, Optional

import numpy as np

from src.config import get_config, ControlConfig

logger = logging.getLogger(__name__)


class LocalController:
    """
    Local Controller (PLC level).

    Handles fast-loop logic including pass-through and dithering
    to mitigate vibration when detected.

    Attributes:
        model: Reference to the physics model
        dithering_active: List of dithering states for each gate
        dithering_phase: List of dithering phase angles for each gate
    """

    def __init__(
        self,
        model: 'TangheSiphonModel',
        config: Optional[ControlConfig] = None
    ) -> None:
        """
        Initialize the local controller.

        Args:
            model: Reference to the physics model
            config: Control configuration. If None, uses global config.
        """
        self.model = model
        self._config = config or get_config().control

        num_gates = model.num_gates
        self.dithering_active: List[bool] = [False] * num_gates
        self.dithering_phase: List[float] = [0.0] * num_gates

        logger.debug(
            "LocalController initialized: dither_amp=%.3f m, dither_freq=%.2f Hz",
            self._config.dithering_amplitude,
            self._config.dithering_frequency
        )

    def update(self, mpc_targets: np.ndarray, dt: float) -> np.ndarray:
        """
        Adjust MPC targets based on local conditions.

        This method applies dithering when vibration exceeds thresholds
        to help break resonance patterns.

        Args:
            mpc_targets: Target openings from MPC [m]
            dt: Time step [s]

        Returns:
            Final actuator commands [m]
        """
        final_cmds = np.copy(mpc_targets)
        cfg = self._config

        for i in range(len(mpc_targets)):
            accel = self.model.vibration_accel[i]

            # Update dithering state based on vibration level
            if accel > cfg.dithering_threshold_high:
                if not self.dithering_active[i]:
                    self.dithering_active[i] = True
                    logger.info(
                        "Gate %d: Dithering activated (vib=%.3f g > %.3f g)",
                        i, accel, cfg.dithering_threshold_high
                    )
            elif accel < cfg.dithering_threshold_low:
                if self.dithering_active[i]:
                    self.dithering_active[i] = False
                    self.dithering_phase[i] = 0.0
                    logger.info(
                        "Gate %d: Dithering deactivated (vib=%.3f g < %.3f g)",
                        i, accel, cfg.dithering_threshold_low
                    )

            # Apply dithering perturbation if active
            if self.dithering_active[i]:
                omega = 2 * np.pi * cfg.dithering_frequency
                self.dithering_phase[i] += omega * dt
                perturbation = cfg.dithering_amplitude * np.sin(self.dithering_phase[i])
                final_cmds[i] += perturbation

        return final_cmds

    def reset(self) -> None:
        """Reset controller state."""
        num_gates = self.model.num_gates
        self.dithering_active = [False] * num_gates
        self.dithering_phase = [0.0] * num_gates
        logger.debug("LocalController reset")
