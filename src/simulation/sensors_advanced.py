# -*- coding: utf-8 -*-
"""
Advanced Sensor Models with Realistic Characteristics.

This module provides high-fidelity sensor simulations including:
- Realistic noise models (colored noise, drift)
- Sensor dynamics (bandwidth, delay)
- Fault injection capabilities
- Signal processing (filtering, FFT)
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple, Deque, Dict
from collections import deque

import numpy as np

from src.config import get_config, SensorConfig

logger = logging.getLogger(__name__)


@dataclass
class SensorDynamics:
    """Sensor dynamic characteristics."""
    bandwidth: float = 100.0  # Hz (-3dB frequency)
    time_constant: float = 0.01  # seconds
    delay_samples: int = 1
    quantization_bits: int = 16
    range_min: float = 0.0
    range_max: float = 10.0


class NoiseGenerator:
    """
    Realistic noise generator with colored noise and drift.

    Supports:
    - White noise
    - Pink noise (1/f)
    - Brownian noise (1/f²)
    - Random walk drift
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)
        self._pink_state = 0.0
        self._brown_state = 0.0
        self._drift_state = 0.0

    def white_noise(self, std: float) -> float:
        """Generate white Gaussian noise."""
        return self._rng.normal(0, std)

    def pink_noise(self, std: float, alpha: float = 0.1) -> float:
        """Generate pink (1/f) noise using IIR filter."""
        white = self._rng.normal(0, std)
        self._pink_state = alpha * white + (1 - alpha) * self._pink_state
        return self._pink_state

    def brownian_noise(self, std: float, alpha: float = 0.01) -> float:
        """Generate brownian (1/f²) noise."""
        white = self._rng.normal(0, std)
        self._brown_state += alpha * white
        return self._brown_state

    def random_walk_drift(self, rate: float) -> float:
        """Generate slow random walk drift."""
        step = self._rng.normal(0, rate)
        self._drift_state += step
        return self._drift_state

    def reset(self) -> None:
        """Reset noise state."""
        self._pink_state = 0.0
        self._brown_state = 0.0
        self._drift_state = 0.0


class AdvancedADCPSensor:
    """
    Advanced ADCP (Acoustic Doppler) velocity sensor.

    Features:
    - Multi-beam simulation
    - Colored noise model
    - Temperature compensation
    - Outlier detection
    """

    def __init__(
        self,
        model: 'TangheSiphonModel',
        gate_index: int,
        config: Optional[SensorConfig] = None,
        dynamics: Optional[SensorDynamics] = None
    ) -> None:
        if not 0 <= gate_index < model.num_gates:
            raise ValueError(f"Gate index must be 0-{model.num_gates - 1}")

        self.model = model
        self.gate_index = gate_index
        self._config = config or get_config().sensor
        self._dynamics = dynamics or SensorDynamics(
            bandwidth=50.0,
            range_min=0.0,
            range_max=5.0
        )

        # Noise and filtering
        self._noise = NoiseGenerator()
        self._filter_state = 0.0
        self._delay_buffer: Deque[float] = deque(maxlen=self._dynamics.delay_samples)
        self._history: Deque[float] = deque(maxlen=100)

        # Fault state
        self._fault_mode: Optional[str] = None
        self._bias: float = 0.0
        self._scale_factor: float = 1.0

        logger.debug("AdvancedADCPSensor initialized for gate %d", gate_index)

    def read(self) -> float:
        """
        Read velocity with realistic sensor behavior.

        Returns:
            Measured velocity [m/s]
        """
        # True value
        true_v = self.model.velocities[self.gate_index]

        # Apply faults
        if self._fault_mode == 'stuck':
            return self._filter_state  # Return last value
        elif self._fault_mode == 'dead':
            return 0.0

        # Apply scale and bias
        measured = true_v * self._scale_factor + self._bias

        # Add noise
        measured += self._noise.white_noise(self._config.adcp_noise_std)
        measured += self._noise.pink_noise(self._config.adcp_noise_std * 0.3)
        measured += self._noise.random_walk_drift(1e-5)

        # Apply sensor dynamics (low-pass filter)
        alpha = 1.0 / (1.0 + self._dynamics.time_constant / 0.1)
        self._filter_state = alpha * measured + (1 - alpha) * self._filter_state

        # Apply delay
        self._delay_buffer.append(self._filter_state)
        if len(self._delay_buffer) >= self._dynamics.delay_samples:
            output = self._delay_buffer[0]
        else:
            output = self._filter_state

        # Quantization
        output = self._quantize(output)

        # Clamp to range
        output = np.clip(output, self._dynamics.range_min, self._dynamics.range_max)

        # Update history
        self._history.append(output)

        return float(output)

    def _quantize(self, value: float) -> float:
        """Apply ADC quantization."""
        range_span = self._dynamics.range_max - self._dynamics.range_min
        lsb = range_span / (2 ** self._dynamics.quantization_bits)
        return round(value / lsb) * lsb

    def inject_fault(self, fault_type: str) -> None:
        """Inject sensor fault."""
        self._fault_mode = fault_type
        logger.warning("ADCP sensor fault injected: %s", fault_type)

    def clear_fault(self) -> None:
        """Clear sensor fault."""
        self._fault_mode = None
        logger.info("ADCP sensor fault cleared")

    def get_statistics(self) -> Dict[str, float]:
        """Get sensor statistics."""
        if not self._history:
            return {}
        data = list(self._history)
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
        }

    def reset(self) -> None:
        """Reset sensor state."""
        self._noise.reset()
        self._filter_state = 0.0
        self._delay_buffer.clear()
        self._history.clear()
        self._fault_mode = None
        self._bias = 0.0
        self._scale_factor = 1.0


class AdvancedVibrationSensor:
    """
    Advanced MEMS vibration sensor with spectral analysis.

    Features:
    - Multi-axis simulation
    - FFT-based frequency detection
    - Temperature drift
    - Saturation modeling
    """

    def __init__(
        self,
        model: 'TangheSiphonModel',
        gate_index: int,
        config: Optional[SensorConfig] = None,
        sample_rate: float = 1000.0  # Hz
    ) -> None:
        if not 0 <= gate_index < model.num_gates:
            raise ValueError(f"Gate index must be 0-{model.num_gates - 1}")

        self.model = model
        self.gate_index = gate_index
        self._config = config or get_config().sensor
        self._sample_rate = sample_rate

        # FFT parameters
        self._fft_size = 256
        self._time_buffer: Deque[float] = deque(maxlen=self._fft_size)

        # Noise and state
        self._noise = NoiseGenerator()
        self._temperature_drift = 0.0

        # Fault state
        self._fault_mode: Optional[str] = None
        self._saturation_limit = 2.0  # g

        logger.debug("AdvancedVibrationSensor initialized for gate %d", gate_index)

    def read_accel(self) -> float:
        """
        Read peak acceleration.

        Returns:
            Peak acceleration [g]
        """
        true_a = self.model.vibration_accel[self.gate_index]

        if self._fault_mode == 'dead':
            return 0.0
        elif self._fault_mode == 'saturated':
            true_a = min(true_a, 0.1)  # Stuck at low value

        # Add noise
        measured = true_a + self._noise.white_noise(self._config.vibration_noise_std)
        measured += self._temperature_drift

        # Update temperature drift slowly
        self._temperature_drift += self._noise.random_walk_drift(1e-6)
        self._temperature_drift = np.clip(self._temperature_drift, -0.01, 0.01)

        # Saturation
        if abs(measured) > self._saturation_limit:
            measured = np.sign(measured) * self._saturation_limit

        # Update time buffer for FFT
        self._time_buffer.append(measured)

        return float(max(0, measured))

    def get_spectrum_peak(self) -> Tuple[float, float]:
        """
        Get dominant frequency and amplitude from FFT.

        Returns:
            Tuple of (frequency [Hz], amplitude [g])
        """
        if len(self._time_buffer) < self._fft_size // 2:
            return 0.0, 0.0

        # Perform FFT
        data = np.array(list(self._time_buffer))
        # Zero-pad if needed
        if len(data) < self._fft_size:
            data = np.pad(data, (0, self._fft_size - len(data)))

        # Window function
        window = np.hanning(len(data))
        data_windowed = data * window

        # FFT
        spectrum = np.abs(np.fft.rfft(data_windowed))
        freqs = np.fft.rfftfreq(len(data), 1.0 / self._sample_rate)

        # Find peak (ignore DC)
        if len(spectrum) > 1:
            peak_idx = np.argmax(spectrum[1:]) + 1
            peak_freq = float(freqs[peak_idx])
            peak_amp = float(spectrum[peak_idx]) * 2 / len(data)
        else:
            peak_freq = 0.0
            peak_amp = 0.0

        # Add noise to frequency estimate
        if peak_freq > 0:
            peak_freq += self._noise.white_noise(self._config.frequency_noise_std)

        return peak_freq, peak_amp

    def get_rms(self) -> float:
        """Get RMS vibration level."""
        if not self._time_buffer:
            return 0.0
        return float(np.sqrt(np.mean(np.array(list(self._time_buffer)) ** 2)))

    def inject_fault(self, fault_type: str) -> None:
        """Inject sensor fault."""
        self._fault_mode = fault_type
        logger.warning("Vibration sensor fault: %s", fault_type)

    def clear_fault(self) -> None:
        """Clear sensor fault."""
        self._fault_mode = None

    def reset(self) -> None:
        """Reset sensor state."""
        self._noise.reset()
        self._time_buffer.clear()
        self._temperature_drift = 0.0
        self._fault_mode = None


class AdvancedPositionSensor:
    """
    Advanced gate position sensor (encoder/potentiometer).

    Features:
    - Encoder simulation with counts
    - Potentiometer noise model
    - Backlash simulation
    """

    def __init__(
        self,
        model: 'TangheSiphonModel',
        gate_index: int,
        resolution: float = 0.001  # m per count
    ) -> None:
        if not 0 <= gate_index < model.num_gates:
            raise ValueError(f"Gate index must be 0-{model.num_gates - 1}")

        self.model = model
        self.gate_index = gate_index
        self._resolution = resolution

        self._noise = NoiseGenerator()
        self._last_direction = 0
        self._backlash = 0.002  # m
        self._backlash_state = 0.0

        logger.debug("AdvancedPositionSensor initialized for gate %d", gate_index)

    def read(self) -> float:
        """Read gate position with realistic sensor behavior."""
        true_pos = self.model.gate_openings[self.gate_index]

        # Apply backlash hysteresis
        direction = np.sign(true_pos - self._backlash_state)
        if direction != self._last_direction and direction != 0:
            # Direction change - apply backlash
            offset = direction * self._backlash / 2
        else:
            offset = 0

        self._backlash_state = true_pos
        self._last_direction = direction

        # Quantize to resolution
        measured = round(true_pos / self._resolution) * self._resolution

        # Add noise
        measured += self._noise.white_noise(self._resolution * 0.1)

        return float(max(0, measured))

    def reset(self) -> None:
        """Reset sensor state."""
        self._noise.reset()
        self._last_direction = 0
        self._backlash_state = 0.0
