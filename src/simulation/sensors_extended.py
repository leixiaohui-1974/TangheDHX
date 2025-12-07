# -*- coding: utf-8 -*-
"""
Extended Sensor Models for Digital Twin System.

This module provides comprehensive sensor simulations including:
- Water level sensors (ultrasonic, pressure-based)
- Pressure sensors
- Flow meters (electromagnetic, ultrasonic)
- Temperature sensors
- Water quality sensors
- Multi-sensor fusion capabilities
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Deque, Tuple, Any
from collections import deque
from enum import Enum
import numpy as np

from src.config import get_config, SensorConfig
from src.simulation.sensors_advanced import NoiseGenerator, SensorDynamics

logger = logging.getLogger(__name__)


class SensorStatus(Enum):
    """Sensor operational status."""
    NORMAL = "normal"
    DEGRADED = "degraded"
    FAULT = "fault"
    OFFLINE = "offline"
    CALIBRATING = "calibrating"


@dataclass
class SensorHealth:
    """Sensor health and diagnostic information."""
    status: SensorStatus = SensorStatus.NORMAL
    signal_quality: float = 1.0  # 0-1, quality indicator
    last_calibration: float = 0.0  # timestamp
    operating_hours: float = 0.0
    fault_count: int = 0
    drift_accumulated: float = 0.0
    temperature: float = 25.0  # Sensor operating temperature


@dataclass
class SensorReading:
    """Standardized sensor reading with metadata."""
    value: float
    timestamp: float
    unit: str
    quality: float = 1.0
    status: SensorStatus = SensorStatus.NORMAL
    raw_value: Optional[float] = None
    uncertainty: float = 0.0


class WaterLevelSensor:
    """
    Water level sensor simulation.

    Supports multiple measurement principles:
    - Ultrasonic (non-contact)
    - Pressure transducer
    - Radar

    Features:
    - Temperature compensation
    - Wave filtering
    - Stilling well simulation
    """

    def __init__(
        self,
        model: 'TangheSiphonModel',
        location: str = 'upstream',  # 'upstream' or 'downstream'
        sensor_type: str = 'ultrasonic',
        config: Optional[SensorConfig] = None
    ) -> None:
        self.model = model
        self.location = location
        self.sensor_type = sensor_type
        self._config = config or get_config().sensor

        # Sensor characteristics based on type
        self._noise_std = {
            'ultrasonic': 0.005,  # m
            'pressure': 0.002,   # m
            'radar': 0.003       # m
        }.get(sensor_type, 0.005)

        # Internal state
        self._noise = NoiseGenerator()
        self._filter_buffer: Deque[float] = deque(maxlen=10)
        self._health = SensorHealth()
        self._calibration_offset = 0.0
        self._temperature_coeff = 0.0001  # m/°C drift coefficient

        # Stilling well simulation (for wave damping)
        self._stilling_well_tau = 2.0  # Time constant [s]
        self._filtered_level = 0.0

        logger.debug(
            "WaterLevelSensor initialized: location=%s, type=%s",
            location, sensor_type
        )

    def read(self, ambient_temperature: float = 25.0) -> SensorReading:
        """
        Read water level with realistic sensor behavior.

        Args:
            ambient_temperature: Ambient temperature [°C]

        Returns:
            SensorReading with water level [m]
        """
        # Get true level based on location
        if self.location == 'upstream':
            true_level = self.model.head_upstream
        else:
            true_level = self.model.head_downstream

        # Check sensor status
        if self._health.status == SensorStatus.OFFLINE:
            return SensorReading(
                value=float('nan'),
                timestamp=self.model.time,
                unit='m',
                quality=0.0,
                status=SensorStatus.OFFLINE
            )

        # Temperature drift compensation
        temp_drift = (ambient_temperature - 25.0) * self._temperature_coeff

        # Stilling well filter (low-pass for wave damping)
        alpha = 0.1 / self._stilling_well_tau
        self._filtered_level = alpha * true_level + (1 - alpha) * self._filtered_level

        # Add sensor noise based on type
        measured = self._filtered_level
        measured += self._noise.white_noise(self._noise_std)
        measured += self._noise.pink_noise(self._noise_std * 0.2)
        measured += temp_drift
        measured += self._calibration_offset
        measured += self._health.drift_accumulated

        # Update drift slowly
        self._health.drift_accumulated += self._noise.random_walk_drift(1e-6)

        # Quality estimation
        quality = self._estimate_quality(measured, true_level)

        return SensorReading(
            value=float(measured),
            timestamp=self.model.time,
            unit='m',
            quality=quality,
            status=self._health.status,
            raw_value=true_level,
            uncertainty=self._noise_std * 2
        )

    def _estimate_quality(self, measured: float, true: float) -> float:
        """Estimate measurement quality."""
        error = abs(measured - true)
        if error < self._noise_std:
            return 1.0
        elif error < self._noise_std * 3:
            return 0.8
        elif error < self._noise_std * 5:
            return 0.5
        else:
            return 0.2

    def calibrate(self, reference_level: float) -> None:
        """Calibrate sensor with reference measurement."""
        if self.location == 'upstream':
            true_level = self.model.head_upstream
        else:
            true_level = self.model.head_downstream

        self._calibration_offset = reference_level - true_level
        self._health.last_calibration = self.model.time
        self._health.drift_accumulated = 0.0
        logger.info("Water level sensor calibrated: offset=%.4f m", self._calibration_offset)

    def get_health(self) -> SensorHealth:
        """Get sensor health status."""
        return self._health

    def inject_fault(self, fault_type: str) -> None:
        """Inject sensor fault for testing."""
        if fault_type == 'drift':
            self._health.drift_accumulated = 0.1
        elif fault_type == 'stuck':
            self._health.status = SensorStatus.FAULT
        elif fault_type == 'offline':
            self._health.status = SensorStatus.OFFLINE
        self._health.fault_count += 1
        logger.warning("Water level sensor fault injected: %s", fault_type)

    def reset(self) -> None:
        """Reset sensor state."""
        self._noise.reset()
        self._filter_buffer.clear()
        self._health = SensorHealth()
        self._calibration_offset = 0.0
        self._filtered_level = 0.0


class PressureSensor:
    """
    Pressure sensor simulation for pipeline monitoring.

    Features:
    - Absolute/gauge pressure modes
    - Temperature compensation
    - Overrange protection
    - Dampening for pressure spikes
    """

    def __init__(
        self,
        model: 'TangheSiphonModel',
        location: str = 'pipe',
        range_min: float = 0.0,    # bar
        range_max: float = 10.0,   # bar
        config: Optional[SensorConfig] = None
    ) -> None:
        self.model = model
        self.location = location
        self._range_min = range_min
        self._range_max = range_max
        self._config = config or get_config().sensor

        # Sensor characteristics
        self._accuracy = 0.001  # 0.1% of full scale
        self._noise_std = (range_max - range_min) * self._accuracy

        # Internal state
        self._noise = NoiseGenerator()
        self._damping_buffer: Deque[float] = deque(maxlen=5)
        self._health = SensorHealth()
        self._zero_offset = 0.0
        self._span_factor = 1.0

        logger.debug("PressureSensor initialized: range=[%.1f, %.1f] bar", range_min, range_max)

    def read(self) -> SensorReading:
        """Read pressure measurement."""
        # Calculate pressure from head difference
        delta_h = self.model.head_upstream - self.model.head_downstream
        # Convert to pressure: P = rho * g * h (in bar)
        true_pressure = 1000 * 9.81 * delta_h / 100000  # Pa to bar

        if self._health.status == SensorStatus.OFFLINE:
            return SensorReading(
                value=float('nan'),
                timestamp=self.model.time,
                unit='bar',
                quality=0.0,
                status=SensorStatus.OFFLINE
            )

        # Apply sensor characteristics
        measured = true_pressure * self._span_factor + self._zero_offset

        # Add noise
        measured += self._noise.white_noise(self._noise_std)

        # Damping filter
        self._damping_buffer.append(measured)
        measured = float(np.mean(list(self._damping_buffer)))

        # Range limiting
        measured = np.clip(measured, self._range_min, self._range_max)

        # Quality estimation
        quality = 1.0 if self._range_min <= measured <= self._range_max else 0.5

        return SensorReading(
            value=float(measured),
            timestamp=self.model.time,
            unit='bar',
            quality=quality,
            status=self._health.status,
            raw_value=true_pressure,
            uncertainty=self._noise_std * 2
        )

    def calibrate_zero(self, reference: float = 0.0) -> None:
        """Zero calibration."""
        self._zero_offset = reference
        self._health.last_calibration = self.model.time
        logger.info("Pressure sensor zero calibrated")

    def calibrate_span(self, reference_pressure: float, true_pressure: float) -> None:
        """Span calibration."""
        if true_pressure != 0:
            self._span_factor = reference_pressure / true_pressure
        self._health.last_calibration = self.model.time
        logger.info("Pressure sensor span calibrated: factor=%.4f", self._span_factor)

    def reset(self) -> None:
        """Reset sensor state."""
        self._noise.reset()
        self._damping_buffer.clear()
        self._health = SensorHealth()
        self._zero_offset = 0.0
        self._span_factor = 1.0


class FlowMeter:
    """
    Flow meter simulation with multiple measurement principles.

    Types:
    - Electromagnetic (mag meter)
    - Ultrasonic (transit-time, Doppler)
    - Differential pressure (orifice plate)

    Features:
    - Bi-directional flow measurement
    - Empty pipe detection
    - Totalization
    """

    def __init__(
        self,
        model: 'TangheSiphonModel',
        gate_index: int = 0,
        meter_type: str = 'electromagnetic',
        pipe_diameter: float = 2.0,  # m
        config: Optional[SensorConfig] = None
    ) -> None:
        if not 0 <= gate_index < model.num_gates:
            raise ValueError(f"Gate index must be 0-{model.num_gates - 1}")

        self.model = model
        self.gate_index = gate_index
        self.meter_type = meter_type
        self.pipe_diameter = pipe_diameter
        self._config = config or get_config().sensor

        # Meter characteristics
        self._accuracy = {
            'electromagnetic': 0.005,  # 0.5%
            'ultrasonic': 0.01,        # 1%
            'differential_pressure': 0.02  # 2%
        }.get(meter_type, 0.01)

        # Internal state
        self._noise = NoiseGenerator()
        self._health = SensorHealth()
        self._totalizer = 0.0  # Total volume [m³]
        self._last_timestamp = 0.0
        self._filter_buffer: Deque[float] = deque(maxlen=10)
        self._k_factor = 1.0  # Calibration factor

        # Empty pipe detection
        self._empty_pipe_threshold = 0.1  # m/s

        logger.debug("FlowMeter initialized: type=%s, gate=%d", meter_type, gate_index)

    def read(self) -> SensorReading:
        """Read flow rate measurement."""
        true_flow = self.model.flow_rates[self.gate_index]

        if self._health.status == SensorStatus.OFFLINE:
            return SensorReading(
                value=float('nan'),
                timestamp=self.model.time,
                unit='m³/s',
                quality=0.0,
                status=SensorStatus.OFFLINE
            )

        # Check empty pipe
        velocity = self.model.velocities[self.gate_index]
        if velocity < self._empty_pipe_threshold:
            status = SensorStatus.DEGRADED
            quality = 0.5
        else:
            status = self._health.status
            quality = 1.0

        # Apply calibration
        measured = true_flow * self._k_factor

        # Add noise proportional to flow
        noise_std = max(0.01, self._accuracy * abs(measured))
        measured += self._noise.white_noise(noise_std)

        # Filter
        self._filter_buffer.append(measured)
        measured = float(np.mean(list(self._filter_buffer)))

        # Update totalizer
        dt = self.model.time - self._last_timestamp
        if dt > 0 and dt < 1.0:
            self._totalizer += measured * dt
        self._last_timestamp = self.model.time

        return SensorReading(
            value=float(max(0, measured)),
            timestamp=self.model.time,
            unit='m³/s',
            quality=quality,
            status=status,
            raw_value=true_flow,
            uncertainty=noise_std * 2
        )

    def get_totalizer(self) -> float:
        """Get total volume passed through meter [m³]."""
        return self._totalizer

    def reset_totalizer(self) -> None:
        """Reset the volume totalizer."""
        self._totalizer = 0.0
        logger.info("Flow meter totalizer reset")

    def calibrate(self, reference_flow: float, measured_flow: float) -> None:
        """Calibrate with reference measurement."""
        if measured_flow != 0:
            self._k_factor = reference_flow / measured_flow
        self._health.last_calibration = self.model.time
        logger.info("Flow meter calibrated: k_factor=%.4f", self._k_factor)

    def reset(self) -> None:
        """Reset sensor state."""
        self._noise.reset()
        self._health = SensorHealth()
        self._totalizer = 0.0
        self._filter_buffer.clear()
        self._k_factor = 1.0


class TemperatureSensor:
    """
    Temperature sensor simulation.

    Types:
    - RTD (Pt100/Pt1000)
    - Thermocouple
    - Thermistor

    Features:
    - Response time modeling
    - Self-heating compensation
    - Cold junction compensation (thermocouple)
    """

    def __init__(
        self,
        model: 'TangheSiphonModel',
        sensor_type: str = 'rtd',
        range_min: float = -20.0,  # °C
        range_max: float = 100.0,  # °C
        config: Optional[SensorConfig] = None
    ) -> None:
        self.model = model
        self.sensor_type = sensor_type
        self._range_min = range_min
        self._range_max = range_max
        self._config = config or get_config().sensor

        # Sensor characteristics
        self._accuracy = {
            'rtd': 0.1,        # °C
            'thermocouple': 0.5,
            'thermistor': 0.2
        }.get(sensor_type, 0.2)

        self._time_constant = {
            'rtd': 5.0,        # seconds
            'thermocouple': 0.5,
            'thermistor': 2.0
        }.get(sensor_type, 2.0)

        # Internal state
        self._noise = NoiseGenerator()
        self._health = SensorHealth()
        self._filtered_temp = 20.0  # Initial temperature
        self._water_temperature = 15.0  # Simulated water temperature

        logger.debug("TemperatureSensor initialized: type=%s", sensor_type)

    def read(self) -> SensorReading:
        """Read temperature measurement."""
        # Simulate water temperature based on flow (higher flow = slightly lower temp)
        avg_velocity = float(np.mean(self.model.velocities))
        true_temp = self._water_temperature - 0.5 * avg_velocity

        if self._health.status == SensorStatus.OFFLINE:
            return SensorReading(
                value=float('nan'),
                timestamp=self.model.time,
                unit='°C',
                quality=0.0,
                status=SensorStatus.OFFLINE
            )

        # Apply sensor dynamics (thermal lag)
        alpha = 0.1 / self._time_constant
        self._filtered_temp = alpha * true_temp + (1 - alpha) * self._filtered_temp

        # Add noise
        measured = self._filtered_temp
        measured += self._noise.white_noise(self._accuracy * 0.1)

        # Range limiting
        measured = np.clip(measured, self._range_min, self._range_max)

        return SensorReading(
            value=float(measured),
            timestamp=self.model.time,
            unit='°C',
            quality=1.0,
            status=self._health.status,
            raw_value=true_temp,
            uncertainty=self._accuracy
        )

    def set_water_temperature(self, temperature: float) -> None:
        """Set simulated water temperature."""
        self._water_temperature = temperature

    def reset(self) -> None:
        """Reset sensor state."""
        self._noise.reset()
        self._health = SensorHealth()
        self._filtered_temp = 20.0


class WaterQualitySensor:
    """
    Water quality multi-parameter sensor.

    Parameters:
    - Turbidity (NTU)
    - pH
    - Dissolved oxygen (mg/L)
    - Conductivity (μS/cm)
    - Chlorine (mg/L)
    """

    def __init__(
        self,
        model: 'TangheSiphonModel',
        config: Optional[SensorConfig] = None
    ) -> None:
        self.model = model
        self._config = config or get_config().sensor

        # Default water quality values
        self._base_values = {
            'turbidity': 5.0,      # NTU
            'ph': 7.2,             # pH
            'dissolved_oxygen': 8.0,  # mg/L
            'conductivity': 500.0,    # μS/cm
            'chlorine': 0.5           # mg/L
        }

        # Sensor characteristics (accuracy)
        self._accuracy = {
            'turbidity': 0.5,
            'ph': 0.05,
            'dissolved_oxygen': 0.2,
            'conductivity': 5.0,
            'chlorine': 0.02
        }

        self._noise = NoiseGenerator()
        self._health = SensorHealth()

        # Calibration offsets
        self._cal_offsets: Dict[str, float] = {k: 0.0 for k in self._base_values}

        logger.debug("WaterQualitySensor initialized")

    def read(self, parameter: str = 'all') -> Dict[str, SensorReading]:
        """
        Read water quality parameters.

        Args:
            parameter: Specific parameter or 'all'

        Returns:
            Dictionary of SensorReadings
        """
        results = {}

        if self._health.status == SensorStatus.OFFLINE:
            for param in self._base_values:
                results[param] = SensorReading(
                    value=float('nan'),
                    timestamp=self.model.time,
                    unit=self._get_unit(param),
                    quality=0.0,
                    status=SensorStatus.OFFLINE
                )
            return results

        # Flow affects turbidity
        avg_flow = float(np.mean(self.model.flow_rates))
        flow_factor = avg_flow / 50.0 if avg_flow > 0 else 0.0

        params_to_read = [parameter] if parameter != 'all' else list(self._base_values.keys())

        for param in params_to_read:
            if param not in self._base_values:
                continue

            base = self._base_values[param]

            # Apply flow-dependent variations
            if param == 'turbidity':
                true_value = base * (1 + 0.5 * flow_factor)
            elif param == 'dissolved_oxygen':
                true_value = base * (1 + 0.1 * flow_factor)
            else:
                true_value = base

            # Add noise and calibration
            measured = true_value + self._cal_offsets[param]
            measured += self._noise.white_noise(self._accuracy[param])

            results[param] = SensorReading(
                value=float(measured),
                timestamp=self.model.time,
                unit=self._get_unit(param),
                quality=1.0,
                status=self._health.status,
                raw_value=true_value,
                uncertainty=self._accuracy[param] * 2
            )

        return results

    def _get_unit(self, parameter: str) -> str:
        """Get unit for parameter."""
        units = {
            'turbidity': 'NTU',
            'ph': 'pH',
            'dissolved_oxygen': 'mg/L',
            'conductivity': 'μS/cm',
            'chlorine': 'mg/L'
        }
        return units.get(parameter, '')

    def calibrate(self, parameter: str, reference: float) -> None:
        """Calibrate specific parameter."""
        if parameter in self._base_values:
            self._cal_offsets[parameter] = reference - self._base_values[parameter]
            self._health.last_calibration = self.model.time
            logger.info("Water quality sensor calibrated: %s", parameter)

    def reset(self) -> None:
        """Reset sensor state."""
        self._noise.reset()
        self._health = SensorHealth()
        self._cal_offsets = {k: 0.0 for k in self._base_values}


class SensorFusionEngine:
    """
    Multi-sensor data fusion for improved accuracy.

    Implements:
    - Kalman filtering for state estimation
    - Weighted averaging based on sensor quality
    - Outlier detection and rejection
    - Redundancy management
    """

    def __init__(self, num_sensors: int = 3):
        self.num_sensors = num_sensors
        self._weights: np.ndarray = np.ones(num_sensors) / num_sensors
        self._kalman_state = 0.0
        self._kalman_covariance = 1.0
        self._history: Deque[Tuple[float, float]] = deque(maxlen=100)

        logger.debug("SensorFusionEngine initialized for %d sensors", num_sensors)

    def fuse(
        self,
        readings: List[SensorReading],
        method: str = 'weighted_average'
    ) -> SensorReading:
        """
        Fuse multiple sensor readings.

        Args:
            readings: List of sensor readings
            method: Fusion method ('weighted_average', 'kalman', 'median')

        Returns:
            Fused sensor reading
        """
        # Filter out invalid readings
        valid_readings = [r for r in readings if r.status != SensorStatus.OFFLINE
                         and not np.isnan(r.value)]

        if not valid_readings:
            return SensorReading(
                value=float('nan'),
                timestamp=0.0,
                unit='',
                quality=0.0,
                status=SensorStatus.OFFLINE
            )

        values = np.array([r.value for r in valid_readings])
        qualities = np.array([r.quality for r in valid_readings])

        if method == 'weighted_average':
            # Quality-weighted average
            weights = qualities / np.sum(qualities)
            fused_value = float(np.sum(values * weights))
            fused_quality = float(np.mean(qualities))

        elif method == 'kalman':
            # Simple Kalman filter update
            for reading in valid_readings:
                # Prediction (assume constant model)
                pred_cov = self._kalman_covariance + 0.01

                # Update
                measurement_noise = 1.0 / max(0.1, reading.quality)
                kalman_gain = pred_cov / (pred_cov + measurement_noise)

                self._kalman_state += kalman_gain * (reading.value - self._kalman_state)
                self._kalman_covariance = (1 - kalman_gain) * pred_cov

            fused_value = self._kalman_state
            fused_quality = 1.0 / (1.0 + self._kalman_covariance)

        elif method == 'median':
            # Median filter (robust to outliers)
            fused_value = float(np.median(values))
            fused_quality = float(np.mean(qualities))

        else:
            raise ValueError(f"Unknown fusion method: {method}")

        # Calculate combined uncertainty
        uncertainties = [r.uncertainty for r in valid_readings if r.uncertainty > 0]
        combined_uncertainty = np.sqrt(np.sum(np.array(uncertainties) ** 2)) / len(uncertainties) if uncertainties else 0.0

        # Record history
        timestamp = valid_readings[0].timestamp
        self._history.append((timestamp, fused_value))

        return SensorReading(
            value=fused_value,
            timestamp=timestamp,
            unit=valid_readings[0].unit,
            quality=fused_quality,
            status=SensorStatus.NORMAL,
            uncertainty=combined_uncertainty
        )

    def detect_outliers(
        self,
        readings: List[SensorReading],
        threshold: float = 3.0
    ) -> List[int]:
        """
        Detect outlier sensors using Z-score method.

        Args:
            readings: List of sensor readings
            threshold: Z-score threshold for outlier detection

        Returns:
            List of indices of outlier sensors
        """
        values = np.array([r.value for r in readings if not np.isnan(r.value)])

        if len(values) < 3:
            return []

        mean = np.mean(values)
        std = np.std(values)

        if std < 1e-10:
            return []

        outliers = []
        for i, reading in enumerate(readings):
            if np.isnan(reading.value):
                continue
            z_score = abs(reading.value - mean) / std
            if z_score > threshold:
                outliers.append(i)

        return outliers

    def update_weights(self, qualities: List[float]) -> None:
        """Update sensor weights based on quality history."""
        if len(qualities) != self.num_sensors:
            return

        total = sum(qualities)
        if total > 0:
            self._weights = np.array(qualities) / total

    def get_weights(self) -> np.ndarray:
        """Get current sensor weights."""
        return self._weights.copy()

    def reset(self) -> None:
        """Reset fusion engine state."""
        self._weights = np.ones(self.num_sensors) / self.num_sensors
        self._kalman_state = 0.0
        self._kalman_covariance = 1.0
        self._history.clear()


class SensorNetwork:
    """
    Unified sensor network manager.

    Manages all sensors in the digital twin system with:
    - Centralized configuration
    - Coordinated sampling
    - Health monitoring
    - Data logging
    """

    def __init__(self, model: 'TangheSiphonModel'):
        self.model = model

        # Initialize sensors
        self.water_level_upstream = WaterLevelSensor(model, 'upstream', 'ultrasonic')
        self.water_level_downstream = WaterLevelSensor(model, 'downstream', 'ultrasonic')
        self.pressure_sensor = PressureSensor(model, 'pipe')
        self.temperature_sensor = TemperatureSensor(model, 'rtd')
        self.water_quality_sensor = WaterQualitySensor(model)

        # Flow meters for each gate
        self.flow_meters: List[FlowMeter] = [
            FlowMeter(model, i, 'electromagnetic')
            for i in range(model.num_gates)
        ]

        # Fusion engine for redundant sensors
        self.fusion_engine = SensorFusionEngine(num_sensors=3)

        # Data logging
        self._reading_history: Deque[Dict[str, Any]] = deque(maxlen=10000)

        logger.info("SensorNetwork initialized with %d flow meters", len(self.flow_meters))

    def sample_all(self) -> Dict[str, Any]:
        """Sample all sensors and return consolidated readings."""
        readings = {
            'timestamp': self.model.time,
            'water_level_upstream': self.water_level_upstream.read(),
            'water_level_downstream': self.water_level_downstream.read(),
            'pressure': self.pressure_sensor.read(),
            'temperature': self.temperature_sensor.read(),
            'water_quality': self.water_quality_sensor.read('all'),
            'flow_rates': [fm.read() for fm in self.flow_meters],
            'total_flow': sum(fm.read().value for fm in self.flow_meters
                            if not np.isnan(fm.read().value))
        }

        # Log readings
        self._reading_history.append(readings)

        return readings

    def get_health_status(self) -> Dict[str, SensorHealth]:
        """Get health status of all sensors."""
        return {
            'water_level_upstream': self.water_level_upstream.get_health(),
            'water_level_downstream': self.water_level_downstream.get_health(),
            'pressure': self.pressure_sensor._health,
            'temperature': self.temperature_sensor._health,
            'water_quality': self.water_quality_sensor._health,
            'flow_meters': [fm._health for fm in self.flow_meters]
        }

    def get_reading_history(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent reading history."""
        return list(self._reading_history)[-count:]

    def reset_all(self) -> None:
        """Reset all sensors."""
        self.water_level_upstream.reset()
        self.water_level_downstream.reset()
        self.pressure_sensor.reset()
        self.temperature_sensor.reset()
        self.water_quality_sensor.reset()
        for fm in self.flow_meters:
            fm.reset()
        self.fusion_engine.reset()
        self._reading_history.clear()
        logger.info("All sensors reset")
