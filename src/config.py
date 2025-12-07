# -*- coding: utf-8 -*-
"""
Configuration Module

Centralized configuration management for the Tanghe Siphon system.
All physical constants and system parameters are defined here.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class PhysicsConfig:
    """Physical model configuration parameters."""

    # Structural parameters
    num_gates: int = 3
    gate_width: float = 6.0  # meters
    max_gate_opening: float = 5.0  # meters
    min_gate_opening: float = 0.0  # meters

    # Hydraulic parameters
    effective_area: float = 16.6  # m² (A_eff from analysis)
    discharge_coefficient: float = 0.7  # Cd
    gravity: float = 9.81  # m/s²

    # Default head conditions
    default_head_upstream: float = 10.0  # meters
    default_head_downstream: float = 8.0  # meters

    # Frequency and vibration parameters
    # Calibrated: v=2.6 m/s => f=2.8 Hz (Resonance)
    # f = k * v => k = 2.8 / 2.6 = 1.077
    strouhal_constant: float = 1.077  # k_freq
    structural_frequency: float = 2.8  # Hz (Rod natural frequency)
    damping_ratio: float = 0.02

    # Resonance detection
    lock_in_ratio_min: float = 0.9
    lock_in_ratio_max: float = 1.1
    resonance_amplification: float = 8.0  # Amplification factor at lock-in
    base_vibration_coefficient: float = 0.01  # Base amp = coef * v²

    # Actuator dynamics
    gate_max_speed: float = 0.05  # m/s


@dataclass
class ControlConfig:
    """Control system configuration parameters."""

    # MPC weights
    alpha_flow_tracking: float = 1.0
    beta_action_penalty: float = 0.1
    gamma_spectral_avoidance: float = 10.0

    # Spectral potential epsilon (avoid division by zero)
    spectral_epsilon: float = 0.1

    # Velocity threshold for frequency calculation
    min_velocity_threshold: float = 0.1  # m/s

    # Local controller parameters
    dithering_threshold_high: float = 0.15  # g - activate dithering
    dithering_threshold_low: float = 0.05  # g - deactivate dithering
    dithering_amplitude: float = 0.02  # meters (+/- 2cm)
    dithering_frequency: float = 0.5  # Hz


@dataclass
class SensorConfig:
    """Sensor configuration parameters."""

    # ADCP (Acoustic Doppler Current Profiler)
    adcp_noise_std: float = 0.05  # m/s

    # Vibration sensor
    vibration_noise_std: float = 0.005  # g
    frequency_noise_std: float = 0.05  # Hz


@dataclass
class SimulationConfig:
    """Simulation configuration parameters."""

    # Time step
    dt: float = 0.1  # seconds

    # Default target flow
    default_target_flow: float = 100.0  # m³/s


@dataclass
class WebConfig:
    """Web server configuration parameters."""

    host: str = field(default_factory=lambda: os.environ.get('TANGHE_HOST', '0.0.0.0'))
    port: int = field(default_factory=lambda: int(os.environ.get('TANGHE_PORT', '5000')))
    debug: bool = field(default_factory=lambda: os.environ.get('TANGHE_DEBUG', 'false').lower() == 'true')


@dataclass
class Config:
    """Main configuration container."""

    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    web: WebConfig = field(default_factory=WebConfig)


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def reset_config() -> Config:
    """Reset configuration to defaults (useful for testing)."""
    global config
    config = Config()
    return config
