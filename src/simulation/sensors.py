# -*- coding: utf-8 -*-
"""
Sensor Models.

This module implements sensor simulations for the digital twin,
including ADCP velocity sensors and MEMS vibration sensors.
"""

import logging
from typing import Optional

import numpy as np

from src.config import get_config, SensorConfig

logger = logging.getLogger(__name__)


class ADCPSensor:
    """
    Acoustic Doppler Current Profiler (ADCP).

    Measures local flow velocity with configurable noise.

    Attributes:
        model: Reference to the physics model
        gate_index: Index of the monitored gate
        noise_std: Standard deviation of measurement noise [m/s]
    """

    def __init__(
        self,
        model: 'TangheSiphonModel',
        gate_index: int,
        config: Optional[SensorConfig] = None
    ) -> None:
        """
        Initialize the ADCP sensor.

        Args:
            model: Reference to the physics model
            gate_index: Index of the gate to monitor
            config: Sensor configuration. If None, uses global config.

        Raises:
            ValueError: If gate_index is out of range
        """
        if not 0 <= gate_index < model.num_gates:
            raise ValueError(f"Gate index must be 0-{model.num_gates - 1}")

        self.model = model
        self.gate_index = gate_index
        self._config = config or get_config().sensor
        self.noise_std = self._config.adcp_noise_std

        logger.debug(
            "ADCPSensor initialized for gate %d (noise=%.3f m/s)",
            gate_index, self.noise_std
        )

    def read(self) -> float:
        """
        Read measured velocity with noise.

        Returns:
            Measured velocity [m/s]
        """
        true_v = self.model.velocities[self.gate_index]
        noisy_v = true_v + np.random.normal(0, self.noise_std)
        return max(0.0, noisy_v)  # Velocity cannot be negative


class VibrationSensor:
    """
    MEMS Vibration Sensor.

    Measures acceleration and provides spectral analysis.

    Attributes:
        model: Reference to the physics model
        gate_index: Index of the monitored gate
        noise_std: Standard deviation of acceleration noise [g]
    """

    def __init__(
        self,
        model: 'TangheSiphonModel',
        gate_index: int,
        config: Optional[SensorConfig] = None
    ) -> None:
        """
        Initialize the vibration sensor.

        Args:
            model: Reference to the physics model
            gate_index: Index of the gate to monitor
            config: Sensor configuration. If None, uses global config.

        Raises:
            ValueError: If gate_index is out of range
        """
        if not 0 <= gate_index < model.num_gates:
            raise ValueError(f"Gate index must be 0-{model.num_gates - 1}")

        self.model = model
        self.gate_index = gate_index
        self._config = config or get_config().sensor
        self.noise_std = self._config.vibration_noise_std

        logger.debug(
            "VibrationSensor initialized for gate %d (noise=%.4f g)",
            gate_index, self.noise_std
        )

    def read_accel(self) -> float:
        """
        Read peak acceleration.

        Returns:
            Peak acceleration [g]
        """
        true_a = self.model.vibration_accel[self.gate_index]
        noisy_a = true_a + np.random.normal(0, self.noise_std)
        return max(0.0, noisy_a)  # Acceleration magnitude cannot be negative

    def get_spectrum_peak(self) -> float:
        """
        Get the dominant frequency from spectral analysis.

        In a real system, this would perform FFT on a time series.
        Here we use the physics model's vortex frequency directly.

        Returns:
            Dominant frequency [Hz]
        """
        true_f = self.model.vortex_freqs[self.gate_index]
        if true_f == 0:
            return 0.0

        noisy_f = true_f + np.random.normal(0, self._config.frequency_noise_std)
        return max(0.0, noisy_f)
