# -*- coding: utf-8 -*-
"""
Physical Digital Twin of Tanghe Inverted Siphon.

This module implements the core physics simulation for the siphon system,
including hydraulic flow calculations, vortex-induced vibration (FIV),
and resonance detection.
"""

import logging
from typing import Dict, List, Optional, Any

import numpy as np

from src.config import get_config, PhysicsConfig

logger = logging.getLogger(__name__)


class TangheSiphonModel:
    """
    Physical Digital Twin of Tanghe Inverted Siphon.

    This class simulates the hydraulic and structural dynamics of the inverted
    siphon, including gate movements, flow rates, velocities, and flow-induced
    vibrations.

    Attributes:
        num_gates: Number of gates (default: 3)
        width: Gate width in meters (default: 6.0)
        gate_openings: Current opening height [m] for each gate
        flow_rates: Current flow rate [m³/s] for each gate
        velocities: Current local velocity [m/s] for each gate
        vibration_accel: Current vibration acceleration [g] for each gate
        vortex_freqs: Current vortex shedding frequency [Hz] for each gate
    """

    def __init__(self, config: Optional[PhysicsConfig] = None) -> None:
        """
        Initialize the siphon model.

        Args:
            config: Physics configuration. If None, uses global config.
        """
        self._config = config or get_config().physics

        self.num_gates: int = self._config.num_gates
        self.width: float = self._config.gate_width

        # Gate state
        self.gate_openings: np.ndarray = np.zeros(self.num_gates)

        # Environmental conditions
        self.head_upstream: float = self._config.default_head_upstream
        self.head_downstream: float = self._config.default_head_downstream

        # Physics constants (from config)
        self.k_freq: float = self._config.strouhal_constant
        self.f_struct: float = self._config.structural_frequency
        self.damping_ratio: float = self._config.damping_ratio

        # State vectors
        self.flow_rates: np.ndarray = np.zeros(self.num_gates)
        self.velocities: np.ndarray = np.zeros(self.num_gates)
        self.vortex_freqs: np.ndarray = np.zeros(self.num_gates)
        self.vibration_accel: np.ndarray = np.zeros(self.num_gates)
        self.time: float = 0.0

        # Fault injection state
        self.gate_stuck: List[bool] = [False] * self.num_gates
        self.gate_noise: List[float] = [0.0] * self.num_gates

        logger.debug("TangheSiphonModel initialized with %d gates", self.num_gates)

    def step(self, target_openings: np.ndarray, dt: float = 0.1) -> None:
        """
        Advance simulation by dt seconds.

        Args:
            target_openings: Target opening heights for each gate [m]
            dt: Time step in seconds

        Raises:
            ValueError: If target_openings has wrong shape
        """
        if len(target_openings) != self.num_gates:
            raise ValueError(
                f"Expected {self.num_gates} target openings, got {len(target_openings)}"
            )

        self.time += dt

        # 1. Actuator Dynamics (Gate movement)
        self._update_gate_positions(target_openings, dt)

        # 2. Hydraulics
        self._update_hydraulics()

        # 3. Vortex Dynamics (FIV)
        self._update_vibrations()

    def _update_gate_positions(self, target_openings: np.ndarray, dt: float) -> None:
        """Update gate positions based on targets and actuator dynamics."""
        max_speed = self._config.gate_max_speed
        min_opening = self._config.min_gate_opening
        max_opening = self._config.max_gate_opening

        for i in range(self.num_gates):
            if self.gate_stuck[i]:
                continue  # No movement for stuck gates

            error = target_openings[i] - self.gate_openings[i]
            move = np.clip(error, -max_speed * dt, max_speed * dt)
            self.gate_openings[i] += move

            # Apply mechanical limits
            self.gate_openings[i] = np.clip(
                self.gate_openings[i], min_opening, max_opening
            )

    def _update_hydraulics(self) -> None:
        """Update flow rates and velocities based on current gate openings."""
        cfg = self._config
        delta_h = max(0, self.head_upstream - self.head_downstream)

        for i in range(self.num_gates):
            # Calculate flow: Q = Cd * B * e * sqrt(2 * g * dH)
            q = (
                cfg.discharge_coefficient
                * self.width
                * self.gate_openings[i]
                * np.sqrt(2 * cfg.gravity * delta_h)
            )
            self.flow_rates[i] = q

            # Calculate velocity: v = Q / A_eff
            self.velocities[i] = q / cfg.effective_area

    def _update_vibrations(self) -> None:
        """Update vortex frequencies and vibration accelerations."""
        cfg = self._config

        for i in range(self.num_gates):
            v = self.velocities[i]

            if v < 0.1:
                self.vortex_freqs[i] = 0.0
                self.vibration_accel[i] = 0.0
                continue

            # Calculate vortex shedding frequency
            f_s = cfg.strouhal_constant * v
            self.vortex_freqs[i] = f_s

            # Check for lock-in (resonance)
            ratio = f_s / cfg.structural_frequency
            is_lock_in = cfg.lock_in_ratio_min < ratio < cfg.lock_in_ratio_max

            # Calculate vibration amplitude
            base_amp = cfg.base_vibration_coefficient * (v ** 2)

            if is_lock_in:
                # Resonance amplification
                self.vibration_accel[i] = base_amp * cfg.resonance_amplification
                logger.warning(
                    "Gate %d in resonance: f=%.2f Hz, v=%.2f m/s, vib=%.3f g",
                    i, f_s, v, self.vibration_accel[i]
                )
            else:
                self.vibration_accel[i] = base_amp

    def get_state(self) -> Dict[str, Any]:
        """
        Get current system state.

        Returns:
            Dictionary containing all state variables.
        """
        return {
            'timestamp': self.time,
            'openings': self.gate_openings.tolist(),
            'flows': self.flow_rates.tolist(),
            'velocities': self.velocities.tolist(),
            'frequencies': self.vortex_freqs.tolist(),
            'vibrations': self.vibration_accel.tolist(),
            'total_flow': float(np.sum(self.flow_rates))
        }

    def inject_fault(self, gate_index: int, fault_type: str) -> None:
        """
        Inject a fault into the specified gate.

        Args:
            gate_index: Index of the gate (0-2)
            fault_type: Type of fault ('stuck' or 'clear')

        Raises:
            ValueError: If gate_index is out of range or fault_type is invalid
        """
        if not 0 <= gate_index < self.num_gates:
            raise ValueError(f"Gate index must be 0-{self.num_gates - 1}")

        if fault_type == 'stuck':
            self.gate_stuck[gate_index] = True
            logger.warning("Fault injected: Gate %d is now stuck", gate_index)
        elif fault_type == 'clear':
            self.gate_stuck[gate_index] = False
            logger.info("Fault cleared: Gate %d is operational", gate_index)
        else:
            raise ValueError(f"Unknown fault type: {fault_type}")

    def reset(self) -> None:
        """Reset model to initial state."""
        self.gate_openings = np.zeros(self.num_gates)
        self.flow_rates = np.zeros(self.num_gates)
        self.velocities = np.zeros(self.num_gates)
        self.vortex_freqs = np.zeros(self.num_gates)
        self.vibration_accel = np.zeros(self.num_gates)
        self.time = 0.0
        self.gate_stuck = [False] * self.num_gates
        self.gate_noise = [0.0] * self.num_gates
        self.head_upstream = self._config.default_head_upstream
        self.head_downstream = self._config.default_head_downstream
        logger.info("Model reset to initial state")
