# -*- coding: utf-8 -*-
"""
Hardware-in-the-Loop (HIL) Testing Framework.

This module provides infrastructure for HIL testing including:
- Real-time simulation synchronization
- Hardware interface abstraction
- Test automation and orchestration
- Performance metrics collection
"""

import logging
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class HILMode(Enum):
    """Hardware-in-the-loop operation modes."""
    SIMULATION = auto()      # Pure software simulation
    HIL_SENSORS = auto()     # Real sensors, simulated actuators
    HIL_ACTUATORS = auto()   # Simulated sensors, real actuators
    FULL_HIL = auto()        # Real sensors and actuators


@dataclass
class HILConfig:
    """Configuration for HIL testing."""
    mode: HILMode = HILMode.SIMULATION
    real_time: bool = True
    dt: float = 0.1  # Time step [s]
    max_timing_jitter: float = 0.02  # Acceptable timing jitter [s]
    sensor_timeout: float = 0.5  # Sensor read timeout [s]
    actuator_timeout: float = 1.0  # Actuator command timeout [s]

    # Logging
    log_all_data: bool = True
    log_interval: int = 10  # Log every N steps


@dataclass
class HILMetrics:
    """Metrics collected during HIL testing."""
    total_steps: int = 0
    timing_violations: int = 0
    sensor_failures: int = 0
    actuator_failures: int = 0

    timing_jitter_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    loop_time_history: deque = field(default_factory=lambda: deque(maxlen=1000))

    def average_jitter(self) -> float:
        if not self.timing_jitter_history:
            return 0.0
        return float(np.mean(list(self.timing_jitter_history)))

    def average_loop_time(self) -> float:
        if not self.loop_time_history:
            return 0.0
        return float(np.mean(list(self.loop_time_history)))


class HardwareInterface(ABC):
    """Abstract base class for hardware interfaces."""

    @abstractmethod
    def connect(self) -> bool:
        """Connect to hardware. Returns True on success."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from hardware."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected."""
        pass


class SensorInterface(HardwareInterface):
    """Abstract interface for sensor hardware."""

    @abstractmethod
    def read_velocity(self, gate_index: int) -> float:
        """Read velocity from sensor [m/s]."""
        pass

    @abstractmethod
    def read_vibration(self, gate_index: int) -> float:
        """Read vibration acceleration [g]."""
        pass

    @abstractmethod
    def read_frequency(self, gate_index: int) -> float:
        """Read dominant vibration frequency [Hz]."""
        pass

    @abstractmethod
    def read_position(self, gate_index: int) -> float:
        """Read gate position [m]."""
        pass


class ActuatorInterface(HardwareInterface):
    """Abstract interface for actuator hardware."""

    @abstractmethod
    def set_position(self, gate_index: int, position: float) -> bool:
        """Command gate position [m]. Returns True on success."""
        pass

    @abstractmethod
    def get_status(self, gate_index: int) -> Dict[str, Any]:
        """Get actuator status."""
        pass

    @abstractmethod
    def emergency_stop(self) -> None:
        """Emergency stop all actuators."""
        pass


class SimulatedSensorInterface(SensorInterface):
    """Simulated sensor interface for testing."""

    def __init__(self, model: 'TangheSiphonModel'):
        self.model = model
        self._connected = False

    def connect(self) -> bool:
        self._connected = True
        logger.info("SimulatedSensorInterface connected")
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("SimulatedSensorInterface disconnected")

    def is_connected(self) -> bool:
        return self._connected

    def read_velocity(self, gate_index: int) -> float:
        if not self._connected:
            raise RuntimeError("Sensor not connected")
        v = self.model.velocities[gate_index]
        return v + np.random.normal(0, 0.05)

    def read_vibration(self, gate_index: int) -> float:
        if not self._connected:
            raise RuntimeError("Sensor not connected")
        a = self.model.vibration_accel[gate_index]
        return max(0, a + np.random.normal(0, 0.005))

    def read_frequency(self, gate_index: int) -> float:
        if not self._connected:
            raise RuntimeError("Sensor not connected")
        f = self.model.vortex_freqs[gate_index]
        return max(0, f + np.random.normal(0, 0.05))

    def read_position(self, gate_index: int) -> float:
        if not self._connected:
            raise RuntimeError("Sensor not connected")
        pos = self.model.gate_openings[gate_index]
        return pos + np.random.normal(0, 0.001)


class SimulatedActuatorInterface(ActuatorInterface):
    """Simulated actuator interface for testing."""

    def __init__(self, model: 'TangheSiphonModel'):
        self.model = model
        self._connected = False
        self._targets = np.zeros(model.num_gates)

    def connect(self) -> bool:
        self._connected = True
        logger.info("SimulatedActuatorInterface connected")
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("SimulatedActuatorInterface disconnected")

    def is_connected(self) -> bool:
        return self._connected

    def set_position(self, gate_index: int, position: float) -> bool:
        if not self._connected:
            return False
        self._targets[gate_index] = position
        return True

    def get_targets(self) -> np.ndarray:
        return self._targets.copy()

    def get_status(self, gate_index: int) -> Dict[str, Any]:
        return {
            'connected': self._connected,
            'target': self._targets[gate_index],
            'actual': self.model.gate_openings[gate_index],
            'error': abs(self._targets[gate_index] - self.model.gate_openings[gate_index]),
        }

    def emergency_stop(self) -> None:
        self._targets = np.zeros(self.model.num_gates)
        logger.warning("Emergency stop activated")


class HILTestRunner:
    """
    Hardware-in-the-Loop test runner.

    Orchestrates HIL testing with:
    - Real-time synchronization
    - Hardware interface management
    - Test scenario execution
    - Metrics collection
    """

    def __init__(
        self,
        model: 'TangheSiphonModel',
        controller: Any,
        config: Optional[HILConfig] = None
    ) -> None:
        """
        Initialize HIL test runner.

        Args:
            model: Physics model (simulation plant)
            controller: Control system
            config: HIL configuration
        """
        self.model = model
        self.controller = controller
        self.config = config or HILConfig()
        self.metrics = HILMetrics()

        # Hardware interfaces
        self.sensor_interface: Optional[SensorInterface] = None
        self.actuator_interface: Optional[ActuatorInterface] = None

        # State
        self._running = False
        self._step_count = 0
        self._last_step_time = 0.0
        self._data_log: List[Dict[str, Any]] = []

        # Thread safety
        self._lock = threading.RLock()

        logger.info("HILTestRunner initialized (mode=%s)", self.config.mode.name)

    def setup(self) -> bool:
        """
        Set up HIL test environment.

        Returns:
            True if setup successful
        """
        # Create appropriate interfaces based on mode
        if self.config.mode == HILMode.SIMULATION:
            self.sensor_interface = SimulatedSensorInterface(self.model)
            self.actuator_interface = SimulatedActuatorInterface(self.model)
        elif self.config.mode == HILMode.HIL_SENSORS:
            # Would use real sensor interface here
            logger.warning("Real sensor interface not implemented, using simulated")
            self.sensor_interface = SimulatedSensorInterface(self.model)
            self.actuator_interface = SimulatedActuatorInterface(self.model)
        elif self.config.mode == HILMode.HIL_ACTUATORS:
            self.sensor_interface = SimulatedSensorInterface(self.model)
            # Would use real actuator interface here
            logger.warning("Real actuator interface not implemented, using simulated")
            self.actuator_interface = SimulatedActuatorInterface(self.model)
        else:  # FULL_HIL
            logger.warning("Full HIL not implemented, using simulation")
            self.sensor_interface = SimulatedSensorInterface(self.model)
            self.actuator_interface = SimulatedActuatorInterface(self.model)

        # Connect interfaces
        sensor_ok = self.sensor_interface.connect()
        actuator_ok = self.actuator_interface.connect()

        return sensor_ok and actuator_ok

    def teardown(self) -> None:
        """Tear down HIL test environment."""
        self._running = False

        if self.sensor_interface:
            self.sensor_interface.disconnect()
        if self.actuator_interface:
            self.actuator_interface.disconnect()

    def run_test(
        self,
        duration: float,
        target_flow_func: Callable[[float], float],
        scenario_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[str, Any]:
        """
        Run HIL test for specified duration.

        Args:
            duration: Test duration [s]
            target_flow_func: Function returning target flow for given time
            scenario_callback: Optional callback for scenario updates

        Returns:
            Test results dictionary
        """
        self._running = True
        self._step_count = 0
        self._data_log.clear()
        self.metrics = HILMetrics()

        dt = self.config.dt
        start_time = time.time()
        sim_time = 0.0

        logger.info("Starting HIL test (duration=%.1fs, dt=%.3fs)", duration, dt)

        try:
            while sim_time < duration and self._running:
                step_start = time.time()

                # Get target flow
                target_flow = target_flow_func(sim_time)

                # Execute scenario callback
                if scenario_callback:
                    scenario_callback(sim_time)

                # Execute control step
                self._execute_step(target_flow, dt)

                # Real-time synchronization
                if self.config.real_time:
                    self._synchronize(dt, step_start)

                sim_time += dt
                self._step_count += 1
                self.metrics.total_steps += 1

        except Exception as e:
            logger.error("HIL test error: %s", e, exc_info=True)
            if self.actuator_interface:
                self.actuator_interface.emergency_stop()
            raise

        finally:
            self._running = False

        # Compile results
        return self._compile_results(duration)

    def _execute_step(self, target_flow: float, dt: float) -> None:
        """Execute single control loop iteration."""
        with self._lock:
            # Read sensors
            try:
                sensor_data = self._read_sensors()
            except Exception as e:
                logger.warning("Sensor read failed: %s", e)
                self.metrics.sensor_failures += 1
                sensor_data = self._get_fallback_sensors()

            # Compute control
            control_output = self._compute_control(target_flow, sensor_data, dt)

            # Command actuators
            try:
                self._command_actuators(control_output)
            except Exception as e:
                logger.warning("Actuator command failed: %s", e)
                self.metrics.actuator_failures += 1

            # Update simulation plant
            if isinstance(self.actuator_interface, SimulatedActuatorInterface):
                targets = self.actuator_interface.get_targets()
                self.model.step(targets, dt)

            # Log data
            if self.config.log_all_data:
                self._log_step(target_flow, sensor_data, control_output)

    def _read_sensors(self) -> Dict[str, np.ndarray]:
        """Read all sensors."""
        velocities = np.array([
            self.sensor_interface.read_velocity(i)
            for i in range(self.model.num_gates)
        ])
        vibrations = np.array([
            self.sensor_interface.read_vibration(i)
            for i in range(self.model.num_gates)
        ])
        frequencies = np.array([
            self.sensor_interface.read_frequency(i)
            for i in range(self.model.num_gates)
        ])
        positions = np.array([
            self.sensor_interface.read_position(i)
            for i in range(self.model.num_gates)
        ])

        return {
            'velocities': velocities,
            'vibrations': vibrations,
            'frequencies': frequencies,
            'positions': positions,
        }

    def _get_fallback_sensors(self) -> Dict[str, np.ndarray]:
        """Get fallback sensor values (from model)."""
        return {
            'velocities': self.model.velocities.copy(),
            'vibrations': self.model.vibration_accel.copy(),
            'frequencies': self.model.vortex_freqs.copy(),
            'positions': self.model.gate_openings.copy(),
        }

    def _compute_control(
        self,
        target_flow: float,
        sensor_data: Dict[str, np.ndarray],
        dt: float
    ) -> np.ndarray:
        """Compute control output."""
        current_openings = sensor_data['positions']
        head_diff = self.model.head_upstream - self.model.head_downstream

        if hasattr(self.controller, 'get_target_openings'):
            # MPC-style controller
            return self.controller.get_target_openings(
                target_flow, current_openings, head_diff
            )
        elif hasattr(self.controller, 'compute'):
            # Hybrid controller
            output, _ = self.controller.compute(
                target_flow, current_openings, head_diff, dt
            )
            return output
        else:
            logger.warning("Unknown controller type")
            return current_openings

    def _command_actuators(self, control_output: np.ndarray) -> None:
        """Command actuators with control output."""
        for i, pos in enumerate(control_output):
            success = self.actuator_interface.set_position(i, pos)
            if not success:
                raise RuntimeError(f"Failed to command gate {i}")

    def _synchronize(self, dt: float, step_start: float) -> None:
        """Synchronize to real-time."""
        elapsed = time.time() - step_start
        sleep_time = dt - elapsed

        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            jitter = abs(sleep_time)
            if jitter > self.config.max_timing_jitter:
                self.metrics.timing_violations += 1
                logger.debug("Timing violation: %.3fs jitter", jitter)

        actual_dt = time.time() - step_start
        self.metrics.timing_jitter_history.append(abs(actual_dt - dt))
        self.metrics.loop_time_history.append(actual_dt)

    def _log_step(
        self,
        target_flow: float,
        sensor_data: Dict[str, np.ndarray],
        control_output: np.ndarray
    ) -> None:
        """Log step data."""
        if self._step_count % self.config.log_interval != 0:
            return

        self._data_log.append({
            'time': self.model.time,
            'target_flow': target_flow,
            'actual_flow': float(np.sum(self.model.flow_rates)),
            'velocities': sensor_data['velocities'].tolist(),
            'vibrations': sensor_data['vibrations'].tolist(),
            'positions': sensor_data['positions'].tolist(),
            'control': control_output.tolist(),
        })

    def _compile_results(self, duration: float) -> Dict[str, Any]:
        """Compile test results."""
        return {
            'duration': duration,
            'total_steps': self.metrics.total_steps,
            'timing_violations': self.metrics.timing_violations,
            'sensor_failures': self.metrics.sensor_failures,
            'actuator_failures': self.metrics.actuator_failures,
            'average_jitter': self.metrics.average_jitter(),
            'average_loop_time': self.metrics.average_loop_time(),
            'data_log': self._data_log,
        }

    def stop(self) -> None:
        """Stop running test."""
        self._running = False
        if self.actuator_interface:
            self.actuator_interface.emergency_stop()
