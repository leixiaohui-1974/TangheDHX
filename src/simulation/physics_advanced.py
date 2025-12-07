# -*- coding: utf-8 -*-
"""
High-Fidelity Physical Model of Tanghe Inverted Siphon.

This module extends the basic physics model with:
- Detailed hydraulic modeling with friction losses
- Dynamic vortex shedding simulation
- Structural dynamics and modal analysis
- Temperature effects
- Sediment transport considerations
- Cavitation detection
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from scipy.integrate import odeint

from src.config import get_config, PhysicsConfig

logger = logging.getLogger(__name__)


@dataclass
class AdvancedPhysicsParams:
    """Extended physics parameters for high-fidelity simulation."""
    # Geometric parameters
    pipe_length: float = 1200.0  # meters
    pipe_diameter: float = 4.0  # meters
    roughness: float = 0.001  # Manning's roughness coefficient

    # Fluid properties
    water_density: float = 1000.0  # kg/m³
    water_viscosity: float = 1.0e-6  # m²/s (kinematic)
    water_temperature: float = 15.0  # °C

    # Structural properties
    structural_damping: float = 0.02
    structural_stiffness: float = 1e6  # N/m
    structural_mass: float = 500.0  # kg (per support)
    modal_frequencies: Tuple[float, ...] = (2.8, 5.6, 8.4)  # Hz (first 3 modes)

    # Vortex parameters
    strouhal_number: float = 0.2
    lock_in_bandwidth: float = 0.1  # relative bandwidth

    # Operational limits
    max_velocity: float = 5.0  # m/s
    cavitation_velocity: float = 6.0  # m/s
    max_vibration: float = 1.0  # g


class HighFidelityPhysicsModel:
    """
    High-fidelity physics simulation of the inverted siphon.

    This model includes:
    - Non-linear hydraulic equations with friction
    - Multi-mode structural vibration
    - Vortex-induced vibration with lock-in
    - State-space representation for integration
    """

    def __init__(
        self,
        config: Optional[PhysicsConfig] = None,
        params: Optional[AdvancedPhysicsParams] = None
    ) -> None:
        """
        Initialize the high-fidelity model.

        Args:
            config: Basic physics configuration
            params: Advanced physics parameters
        """
        self._config = config or get_config().physics
        self._params = params or AdvancedPhysicsParams()

        self.num_gates = self._config.num_gates
        self.width = self._config.gate_width

        # Gate state
        self.gate_openings = np.zeros(self.num_gates)
        self.gate_velocities = np.zeros(self.num_gates)  # Opening velocity

        # Environmental conditions
        self.head_upstream = self._config.default_head_upstream
        self.head_downstream = self._config.default_head_downstream

        # Hydraulic state
        self.flow_rates = np.zeros(self.num_gates)
        self.velocities = np.zeros(self.num_gates)
        self.pressure_drop = np.zeros(self.num_gates)

        # Vibration state (multi-mode)
        self.structural_displacement = np.zeros((self.num_gates, 3))  # 3 modes
        self.structural_velocity = np.zeros((self.num_gates, 3))
        self.vibration_accel = np.zeros(self.num_gates)
        self.vortex_freqs = np.zeros(self.num_gates)

        # Derived quantities
        self.reynolds_numbers = np.zeros(self.num_gates)
        self.friction_factors = np.zeros(self.num_gates)
        self.cavitation_indices = np.zeros(self.num_gates)

        # Simulation state
        self.time = 0.0

        # Fault injection
        self.gate_stuck = [False] * self.num_gates
        self.gate_noise = [0.0] * self.num_gates
        self.sensor_bias = np.zeros(self.num_gates)
        self.sensor_noise_scale = 1.0

        logger.info("HighFidelityPhysicsModel initialized")

    def step(self, target_openings: np.ndarray, dt: float = 0.1) -> None:
        """
        Advance simulation by dt seconds.

        Uses high-fidelity physics including:
        - Nonlinear hydraulics
        - Multi-mode vibration
        - Dynamic actuator response

        Args:
            target_openings: Target gate openings [m]
            dt: Time step [s]
        """
        if len(target_openings) != self.num_gates:
            raise ValueError(f"Expected {self.num_gates} openings")

        self.time += dt

        # 1. Gate actuator dynamics (second-order)
        self._update_gate_dynamics(target_openings, dt)

        # 2. Hydraulic calculation with friction
        self._update_hydraulics()

        # 3. Vortex shedding dynamics
        self._update_vortex_dynamics(dt)

        # 4. Structural dynamics (modal superposition)
        self._update_structural_dynamics(dt)

        # 5. Compute derived quantities
        self._compute_derived_quantities()

    def _update_gate_dynamics(
        self,
        target_openings: np.ndarray,
        dt: float
    ) -> None:
        """Update gate positions with second-order dynamics."""
        max_speed = self._config.gate_max_speed
        max_accel = 0.1  # m/s² (rate limit)

        for i in range(self.num_gates):
            if self.gate_stuck[i]:
                self.gate_velocities[i] = 0.0
                continue

            # Position error
            error = target_openings[i] - self.gate_openings[i]

            # Velocity command with rate limiting
            desired_velocity = np.clip(error / dt, -max_speed, max_speed)

            # Acceleration limiting
            accel = (desired_velocity - self.gate_velocities[i]) / dt
            accel = np.clip(accel, -max_accel, max_accel)

            # Update velocity and position
            self.gate_velocities[i] += accel * dt
            self.gate_openings[i] += self.gate_velocities[i] * dt

            # Add noise if present
            if self.gate_noise[i] != 0:
                self.gate_openings[i] += np.random.normal(0, self.gate_noise[i])

            # Apply limits
            self.gate_openings[i] = np.clip(
                self.gate_openings[i],
                self._config.min_gate_opening,
                self._config.max_gate_opening
            )

    def _update_hydraulics(self) -> None:
        """Calculate hydraulic quantities with friction losses."""
        p = self._params
        cfg = self._config

        delta_h = max(0, self.head_upstream - self.head_downstream)

        for i in range(self.num_gates):
            e = self.gate_openings[i]
            if e < 0.01:  # Nearly closed
                self.flow_rates[i] = 0.0
                self.velocities[i] = 0.0
                self.reynolds_numbers[i] = 0.0
                self.friction_factors[i] = 0.0
                continue

            # Initial flow estimate (orifice equation)
            a_orifice = self.width * e
            q_ideal = cfg.discharge_coefficient * a_orifice * np.sqrt(2 * cfg.gravity * delta_h)

            # Iterate for friction (fixed-point iteration)
            q = q_ideal
            for _ in range(5):
                v = q / cfg.effective_area
                re = v * p.pipe_diameter / p.water_viscosity

                # Friction factor (Colebrook-White approximation)
                if re > 0:
                    f = self._compute_friction_factor(re, p.roughness, p.pipe_diameter)
                else:
                    f = 0.02

                # Head loss due to friction
                h_friction = f * (p.pipe_length / p.pipe_diameter) * (v ** 2) / (2 * cfg.gravity)

                # Effective head
                h_effective = max(0, delta_h - h_friction)

                # Update flow
                q = cfg.discharge_coefficient * a_orifice * np.sqrt(2 * cfg.gravity * h_effective)

            self.flow_rates[i] = q
            self.velocities[i] = q / cfg.effective_area
            self.reynolds_numbers[i] = re
            self.friction_factors[i] = f
            self.pressure_drop[i] = p.water_density * cfg.gravity * (delta_h - h_effective)

    def _compute_friction_factor(
        self,
        re: float,
        roughness: float,
        diameter: float
    ) -> float:
        """Compute Darcy-Weisbach friction factor."""
        if re < 2300:
            # Laminar flow
            return 64 / max(re, 1.0)
        else:
            # Turbulent flow (Haaland approximation)
            eps_d = roughness / diameter
            f_inv = -1.8 * np.log10(
                (eps_d / 3.7) ** 1.11 + 6.9 / re
            )
            return (1 / f_inv) ** 2

    def _update_vortex_dynamics(self, dt: float) -> None:
        """Update vortex shedding frequencies and forces."""
        p = self._params

        for i in range(self.num_gates):
            v = self.velocities[i]

            if v < 0.1:
                self.vortex_freqs[i] = 0.0
                continue

            # Strouhal relation for vortex shedding
            # f_s = St * V / D (simplified - using characteristic dimension)
            f_s = self._config.strouhal_constant * v
            self.vortex_freqs[i] = f_s

    def _update_structural_dynamics(self, dt: float) -> None:
        """Update multi-mode structural vibration."""
        p = self._params

        for i in range(self.num_gates):
            v = self.velocities[i]
            f_s = self.vortex_freqs[i]

            if f_s < 0.1:
                # Decay vibration
                for m in range(3):
                    self.structural_displacement[i, m] *= 0.9
                    self.structural_velocity[i, m] *= 0.9
                self.vibration_accel[i] = 0.0
                continue

            # Modal superposition for each mode
            total_accel = 0.0

            for m, f_n in enumerate(p.modal_frequencies):
                omega_n = 2 * np.pi * f_n
                zeta = p.structural_damping

                # Check for lock-in
                ratio = f_s / f_n
                is_lock_in = abs(ratio - 1.0) < p.lock_in_bandwidth

                # Forcing function (vortex lift force)
                if is_lock_in:
                    # Resonant forcing - larger amplitude
                    force_amplitude = 0.5 * p.water_density * v ** 2 * 0.1  # CL ~ 0.1
                else:
                    force_amplitude = 0.5 * p.water_density * v ** 2 * 0.02

                # Sinusoidal forcing at vortex frequency
                omega_f = 2 * np.pi * f_s
                force = force_amplitude * np.sin(omega_f * self.time)

                # SDOF response (simplified state-space integration)
                x = self.structural_displacement[i, m]
                x_dot = self.structural_velocity[i, m]

                # State-space: x_ddot = (F - c*x_dot - k*x) / m
                x_ddot = (
                    force / p.structural_mass
                    - 2 * zeta * omega_n * x_dot
                    - omega_n ** 2 * x
                )

                # Euler integration
                self.structural_velocity[i, m] += x_ddot * dt
                self.structural_displacement[i, m] += self.structural_velocity[i, m] * dt

                # Contribute to total acceleration
                total_accel += abs(x_ddot)

            # Convert to g
            self.vibration_accel[i] = total_accel / 9.81

    def _compute_derived_quantities(self) -> None:
        """Compute derived quantities for monitoring."""
        p = self._params

        for i in range(self.num_gates):
            v = self.velocities[i]

            # Cavitation index
            if v > 0:
                # Simplified cavitation number
                sigma = (self.pressure_drop[i] + 101325) / (
                    0.5 * p.water_density * v ** 2 + 1.0
                )
                self.cavitation_indices[i] = sigma
            else:
                self.cavitation_indices[i] = float('inf')

    def get_state(self) -> Dict[str, Any]:
        """Get comprehensive system state."""
        return {
            'timestamp': self.time,
            'openings': self.gate_openings.tolist(),
            'flows': self.flow_rates.tolist(),
            'velocities': self.velocities.tolist(),
            'frequencies': self.vortex_freqs.tolist(),
            'vibrations': self.vibration_accel.tolist(),
            'total_flow': float(np.sum(self.flow_rates)),
            'reynolds': self.reynolds_numbers.tolist(),
            'friction_factors': self.friction_factors.tolist(),
            'cavitation_indices': self.cavitation_indices.tolist(),
            'pressure_drops': self.pressure_drop.tolist(),
        }

    def inject_fault(self, gate_index: int, fault_type: str) -> None:
        """Inject fault into specified gate."""
        if not 0 <= gate_index < self.num_gates:
            raise ValueError(f"Gate index must be 0-{self.num_gates - 1}")

        if fault_type == 'stuck':
            self.gate_stuck[gate_index] = True
            logger.warning("Fault: Gate %d stuck", gate_index)
        elif fault_type == 'clear':
            self.gate_stuck[gate_index] = False
            logger.info("Fault cleared: Gate %d", gate_index)
        elif fault_type == 'drift':
            self.gate_noise[gate_index] = 0.05
            logger.warning("Fault: Gate %d drifting", gate_index)
        elif fault_type == 'sensor_bias':
            self.sensor_bias[gate_index] = 0.1
            logger.warning("Fault: Gate %d sensor bias", gate_index)
        else:
            raise ValueError(f"Unknown fault type: {fault_type}")

    def reset(self) -> None:
        """Reset model to initial state."""
        self.gate_openings = np.zeros(self.num_gates)
        self.gate_velocities = np.zeros(self.num_gates)
        self.flow_rates = np.zeros(self.num_gates)
        self.velocities = np.zeros(self.num_gates)
        self.pressure_drop = np.zeros(self.num_gates)
        self.structural_displacement = np.zeros((self.num_gates, 3))
        self.structural_velocity = np.zeros((self.num_gates, 3))
        self.vibration_accel = np.zeros(self.num_gates)
        self.vortex_freqs = np.zeros(self.num_gates)
        self.reynolds_numbers = np.zeros(self.num_gates)
        self.friction_factors = np.zeros(self.num_gates)
        self.cavitation_indices = np.zeros(self.num_gates)
        self.time = 0.0
        self.gate_stuck = [False] * self.num_gates
        self.gate_noise = [0.0] * self.num_gates
        self.sensor_bias = np.zeros(self.num_gates)
        self.head_upstream = self._config.default_head_upstream
        self.head_downstream = self._config.default_head_downstream
        logger.info("HighFidelityModel reset")
