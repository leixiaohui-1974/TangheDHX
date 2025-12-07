# -*- coding: utf-8 -*-
"""
Extended Actuator Models for Digital Twin System.

This module provides comprehensive actuator simulations including:
- Gate actuators with realistic dynamics
- Pump controllers
- Valve actuators
- Motor drives
- Fault injection and degradation modeling
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Deque, Tuple, Any, Callable
from collections import deque
from enum import Enum
import numpy as np

from src.config import get_config, PhysicsConfig

logger = logging.getLogger(__name__)


class ActuatorStatus(Enum):
    """Actuator operational status."""
    READY = "ready"
    MOVING = "moving"
    STOPPED = "stopped"
    FAULT = "fault"
    EMERGENCY_STOP = "emergency_stop"
    MAINTENANCE = "maintenance"
    CALIBRATING = "calibrating"


class FaultType(Enum):
    """Actuator fault types."""
    NONE = "none"
    STUCK = "stuck"
    DRIFT = "drift"
    SLOW_RESPONSE = "slow_response"
    POSITION_ERROR = "position_error"
    POWER_LOSS = "power_loss"
    MECHANICAL_WEAR = "mechanical_wear"
    OVERHEATING = "overheating"


@dataclass
class ActuatorDynamics:
    """Actuator dynamic characteristics."""
    max_speed: float = 0.05  # m/s or rad/s
    acceleration: float = 0.1  # m/s² or rad/s²
    deceleration: float = 0.15  # m/s² or rad/s²
    dead_band: float = 0.001  # Position dead band
    backlash: float = 0.002  # Mechanical backlash
    time_constant: float = 0.5  # Response time constant [s]
    friction_static: float = 0.02  # Static friction coefficient
    friction_dynamic: float = 0.01  # Dynamic friction coefficient


@dataclass
class ActuatorHealth:
    """Actuator health and diagnostic information."""
    status: ActuatorStatus = ActuatorStatus.READY
    fault_type: FaultType = FaultType.NONE
    operating_hours: float = 0.0
    cycle_count: int = 0
    temperature: float = 40.0  # Operating temperature [°C]
    power_consumption: float = 0.0  # Current power [kW]
    efficiency: float = 0.95  # Current efficiency
    wear_level: float = 0.0  # 0-1, wear indicator
    last_maintenance: float = 0.0  # Timestamp
    fault_history: List[Tuple[float, FaultType]] = field(default_factory=list)


@dataclass
class ActuatorCommand:
    """Command structure for actuator."""
    target_position: float
    speed_limit: Optional[float] = None
    priority: int = 1
    timestamp: float = 0.0
    source: str = "controller"


@dataclass
class ActuatorFeedback:
    """Feedback structure from actuator."""
    position: float
    velocity: float
    torque: float
    status: ActuatorStatus
    timestamp: float
    at_target: bool = False
    error: float = 0.0


class AdvancedGateActuator:
    """
    Advanced gate actuator with realistic dynamics.

    Features:
    - Second-order dynamics with acceleration limits
    - Mechanical wear simulation
    - Power consumption modeling
    - Fault injection capabilities
    - Position and velocity feedback
    """

    def __init__(
        self,
        model: 'TangheSiphonModel',
        gate_index: int,
        dynamics: Optional[ActuatorDynamics] = None,
        config: Optional[PhysicsConfig] = None
    ) -> None:
        if not 0 <= gate_index < model.num_gates:
            raise ValueError(f"Gate index must be 0-{model.num_gates - 1}")

        self.model = model
        self.gate_index = gate_index
        self._config = config or get_config().physics
        self._dynamics = dynamics or ActuatorDynamics(
            max_speed=self._config.gate_max_speed
        )

        # State variables
        self._position = model.gate_openings[gate_index]
        self._velocity = 0.0
        self._target_position = self._position
        self._target_velocity = 0.0

        # Internal state
        self._health = ActuatorHealth()
        self._command_queue: Deque[ActuatorCommand] = deque(maxlen=10)
        self._position_history: Deque[Tuple[float, float]] = deque(maxlen=1000)
        self._last_direction = 0

        # Fault injection state
        self._fault_params: Dict[str, float] = {}

        logger.debug("AdvancedGateActuator initialized for gate %d", gate_index)

    def command(self, target: float, speed_limit: Optional[float] = None) -> None:
        """
        Send position command to actuator.

        Args:
            target: Target position [m]
            speed_limit: Optional speed limit [m/s]
        """
        # Clamp target to valid range
        target = np.clip(target, self._config.min_gate_opening,
                        self._config.max_gate_opening)

        cmd = ActuatorCommand(
            target_position=target,
            speed_limit=speed_limit or self._dynamics.max_speed,
            timestamp=self.model.time
        )

        self._target_position = target

        # Update cycle count if direction changes
        direction = np.sign(target - self._position)
        if direction != self._last_direction and direction != 0:
            self._health.cycle_count += 1
        self._last_direction = direction

        logger.debug("Gate %d command: target=%.3f m", self.gate_index, target)

    def step(self, dt: float) -> ActuatorFeedback:
        """
        Advance actuator simulation by dt seconds.

        Args:
            dt: Time step [s]

        Returns:
            ActuatorFeedback with current state
        """
        # Update operating hours
        self._health.operating_hours += dt / 3600.0

        # Check fault status
        if self._health.fault_type == FaultType.STUCK:
            self._velocity = 0.0
            self._health.status = ActuatorStatus.FAULT
            return self._create_feedback()

        if self._health.fault_type == FaultType.POWER_LOSS:
            self._velocity = 0.0
            self._health.status = ActuatorStatus.FAULT
            return self._create_feedback()

        # Calculate position error
        error = self._target_position - self._position

        # Apply dead band
        if abs(error) < self._dynamics.dead_band:
            self._velocity = 0.0
            self._health.status = ActuatorStatus.READY
            self._update_model()
            return self._create_feedback()

        # Calculate desired velocity
        desired_velocity = np.clip(
            error / self._dynamics.time_constant,
            -self._dynamics.max_speed,
            self._dynamics.max_speed
        )

        # Apply slow response fault
        if self._health.fault_type == FaultType.SLOW_RESPONSE:
            desired_velocity *= 0.3

        # Apply acceleration limits
        velocity_error = desired_velocity - self._velocity
        if velocity_error > 0:
            max_delta = self._dynamics.acceleration * dt
        else:
            max_delta = self._dynamics.deceleration * dt

        velocity_change = np.clip(velocity_error, -max_delta, max_delta)
        self._velocity += velocity_change

        # Apply friction
        if abs(self._velocity) < 0.001:
            # Static friction
            if abs(error) < self._dynamics.friction_static:
                self._velocity = 0.0
        else:
            # Dynamic friction
            self._velocity *= (1 - self._dynamics.friction_dynamic)

        # Apply mechanical wear effect
        wear_factor = 1.0 - 0.2 * self._health.wear_level
        self._velocity *= wear_factor

        # Update position
        new_position = self._position + self._velocity * dt

        # Apply backlash
        if np.sign(self._velocity) != np.sign(self._last_direction) and self._last_direction != 0:
            backlash_offset = np.sign(self._velocity) * self._dynamics.backlash / 2
            new_position += backlash_offset

        # Apply position error fault
        if self._health.fault_type == FaultType.POSITION_ERROR:
            new_position += self._fault_params.get('position_offset', 0.01)

        # Clamp to limits
        self._position = np.clip(
            new_position,
            self._config.min_gate_opening,
            self._config.max_gate_opening
        )

        # Update model
        self._update_model()

        # Calculate power consumption
        self._health.power_consumption = self._calculate_power()

        # Update temperature
        self._update_temperature(dt)

        # Update wear
        self._update_wear(dt)

        # Update status
        if abs(self._velocity) > 0.001:
            self._health.status = ActuatorStatus.MOVING
        else:
            self._health.status = ActuatorStatus.READY

        # Record history
        self._position_history.append((self.model.time, self._position))

        return self._create_feedback()

    def _update_model(self) -> None:
        """Update the physics model with current position."""
        self.model.gate_openings[self.gate_index] = self._position

    def _calculate_power(self) -> float:
        """Calculate current power consumption [kW]."""
        # Simplified power model: P = k * v² + P_idle
        base_power = 0.5  # kW idle
        motion_power = 2.0 * (self._velocity / self._dynamics.max_speed) ** 2
        return base_power + motion_power

    def _update_temperature(self, dt: float) -> None:
        """Update actuator temperature."""
        # Simplified thermal model
        ambient = 25.0
        heating_rate = self._health.power_consumption * 2.0  # °C/s per kW
        cooling_rate = 0.1 * (self._health.temperature - ambient)

        self._health.temperature += (heating_rate - cooling_rate) * dt
        self._health.temperature = np.clip(self._health.temperature, ambient, 100.0)

        # Check overheating
        if self._health.temperature > 80.0:
            self._health.fault_type = FaultType.OVERHEATING
            logger.warning("Gate %d actuator overheating: %.1f°C",
                          self.gate_index, self._health.temperature)

    def _update_wear(self, dt: float) -> None:
        """Update mechanical wear level."""
        # Wear increases with movement and cycles
        wear_rate = abs(self._velocity) * 1e-6
        self._health.wear_level += wear_rate * dt
        self._health.wear_level = min(1.0, self._health.wear_level)

        # Mechanical wear fault
        if self._health.wear_level > 0.8:
            if self._health.fault_type == FaultType.NONE:
                self._health.fault_type = FaultType.MECHANICAL_WEAR
                logger.warning("Gate %d actuator mechanical wear: %.1f%%",
                              self.gate_index, self._health.wear_level * 100)

    def _create_feedback(self) -> ActuatorFeedback:
        """Create feedback structure."""
        error = self._target_position - self._position
        at_target = abs(error) < self._dynamics.dead_band

        return ActuatorFeedback(
            position=self._position,
            velocity=self._velocity,
            torque=self._health.power_consumption * 0.1,  # Simplified
            status=self._health.status,
            timestamp=self.model.time,
            at_target=at_target,
            error=error
        )

    def inject_fault(self, fault_type: FaultType, **params) -> None:
        """Inject actuator fault."""
        self._health.fault_type = fault_type
        self._fault_params = params
        self._health.fault_history.append((self.model.time, fault_type))
        logger.warning("Gate %d actuator fault: %s", self.gate_index, fault_type.value)

    def clear_fault(self) -> None:
        """Clear actuator fault."""
        self._health.fault_type = FaultType.NONE
        self._fault_params = {}
        self._health.status = ActuatorStatus.READY
        logger.info("Gate %d actuator fault cleared", self.gate_index)

    def maintenance(self) -> None:
        """Perform maintenance - reset wear and temperature."""
        self._health.wear_level = 0.0
        self._health.temperature = 25.0
        self._health.last_maintenance = self.model.time
        self._health.fault_type = FaultType.NONE
        self._health.status = ActuatorStatus.READY
        logger.info("Gate %d actuator maintenance completed", self.gate_index)

    def get_health(self) -> ActuatorHealth:
        """Get actuator health status."""
        return self._health

    def get_position_history(self) -> List[Tuple[float, float]]:
        """Get position history."""
        return list(self._position_history)

    def emergency_stop(self) -> None:
        """Execute emergency stop."""
        self._velocity = 0.0
        self._health.status = ActuatorStatus.EMERGENCY_STOP
        logger.warning("Gate %d actuator emergency stop", self.gate_index)

    def reset(self) -> None:
        """Reset actuator state."""
        self._position = 0.0
        self._velocity = 0.0
        self._target_position = 0.0
        self._health = ActuatorHealth()
        self._position_history.clear()
        self._update_model()


class PumpActuator:
    """
    Pump actuator simulation with VFD (Variable Frequency Drive).

    Features:
    - Speed control with ramp rates
    - Power consumption modeling
    - Cavitation detection
    - Efficiency curves
    """

    def __init__(
        self,
        rated_speed: float = 1450.0,  # RPM
        rated_power: float = 100.0,   # kW
        rated_flow: float = 1.0       # m³/s
    ) -> None:
        self.rated_speed = rated_speed
        self.rated_power = rated_power
        self.rated_flow = rated_flow

        # State
        self._current_speed = 0.0
        self._target_speed = 0.0
        self._health = ActuatorHealth()

        # VFD parameters
        self._ramp_rate = 100.0  # RPM/s
        self._min_speed = 200.0  # RPM
        self._max_speed = rated_speed * 1.1

        # Efficiency curve parameters (simplified quadratic)
        self._efficiency_a = -0.0002  # Quadratic term
        self._efficiency_b = 0.001    # Linear term
        self._efficiency_c = 0.5      # Constant term

        # Cavitation threshold
        self._cavitation_npsh = 3.0  # m

        logger.debug("PumpActuator initialized: rated=%.0f RPM, %.0f kW",
                    rated_speed, rated_power)

    def set_speed(self, speed: float) -> None:
        """Set target pump speed [RPM]."""
        self._target_speed = np.clip(speed, 0.0, self._max_speed)

    def set_flow_setpoint(self, flow: float) -> None:
        """Set flow setpoint - calculates required speed."""
        # Affinity law: Q ∝ N
        required_speed = (flow / self.rated_flow) * self.rated_speed
        self.set_speed(required_speed)

    def step(self, dt: float, available_npsh: float = 10.0) -> Dict[str, float]:
        """
        Advance pump simulation.

        Args:
            dt: Time step [s]
            available_npsh: Available net positive suction head [m]

        Returns:
            Dictionary with pump state
        """
        # Update operating hours
        if self._current_speed > self._min_speed:
            self._health.operating_hours += dt / 3600.0

        # Speed ramping
        speed_error = self._target_speed - self._current_speed

        if abs(speed_error) > 1.0:
            ramp = np.sign(speed_error) * self._ramp_rate * dt
            self._current_speed += np.clip(ramp, -abs(speed_error), abs(speed_error))
        else:
            self._current_speed = self._target_speed

        # Enforce minimum speed or stop
        if self._target_speed < self._min_speed:
            self._current_speed = 0.0

        # Calculate flow (affinity law)
        speed_ratio = self._current_speed / self.rated_speed
        flow = speed_ratio * self.rated_flow

        # Calculate head (affinity law: H ∝ N²)
        head = speed_ratio ** 2 * 20.0  # Assuming 20m rated head

        # Calculate power (affinity law: P ∝ N³, modified by efficiency)
        efficiency = self._calculate_efficiency()
        power = speed_ratio ** 3 * self.rated_power / efficiency if efficiency > 0 else 0

        self._health.power_consumption = power
        self._health.efficiency = efficiency

        # Cavitation check
        required_npsh = 0.5 + 0.1 * speed_ratio ** 2
        cavitating = available_npsh < required_npsh

        if cavitating:
            self._health.status = ActuatorStatus.FAULT
            self._health.fault_type = FaultType.MECHANICAL_WEAR
            # Reduce performance
            flow *= 0.7
            head *= 0.8

        # Update status
        if self._current_speed > 0:
            self._health.status = ActuatorStatus.MOVING
        else:
            self._health.status = ActuatorStatus.READY

        return {
            'speed': self._current_speed,
            'flow': flow,
            'head': head,
            'power': power,
            'efficiency': efficiency,
            'cavitating': cavitating,
            'status': self._health.status.value
        }

    def _calculate_efficiency(self) -> float:
        """Calculate pump efficiency based on operating point."""
        speed_ratio = self._current_speed / self.rated_speed
        if speed_ratio < 0.3:
            return 0.3
        # Simplified efficiency curve
        efficiency = (self._efficiency_a * speed_ratio ** 2 +
                     self._efficiency_b * speed_ratio +
                     self._efficiency_c)
        return np.clip(efficiency, 0.3, 0.95)

    def get_health(self) -> ActuatorHealth:
        """Get pump health status."""
        return self._health

    def emergency_stop(self) -> None:
        """Emergency stop pump."""
        self._target_speed = 0.0
        self._current_speed = 0.0
        self._health.status = ActuatorStatus.EMERGENCY_STOP

    def reset(self) -> None:
        """Reset pump state."""
        self._current_speed = 0.0
        self._target_speed = 0.0
        self._health = ActuatorHealth()


class ValveActuator:
    """
    Valve actuator simulation (butterfly, gate, ball valves).

    Features:
    - Position control with travel time
    - Flow characteristic modeling (linear, equal percentage, quick opening)
    - Tight shutoff simulation
    - Seat wear modeling
    """

    def __init__(
        self,
        valve_type: str = 'butterfly',
        size_dn: int = 500,  # mm
        cv_rated: float = 1000.0,  # Flow coefficient
        travel_time: float = 30.0  # Full stroke time [s]
    ) -> None:
        self.valve_type = valve_type
        self.size_dn = size_dn
        self.cv_rated = cv_rated
        self._travel_time = travel_time

        # State
        self._position = 0.0  # 0-100%
        self._target_position = 0.0
        self._health = ActuatorHealth()

        # Valve characteristics
        self._characteristic = self._get_characteristic()
        self._seat_leakage = 0.0  # % leakage when closed

        logger.debug("ValveActuator initialized: type=%s, DN%d", valve_type, size_dn)

    def _get_characteristic(self) -> Callable[[float], float]:
        """Get valve flow characteristic function."""
        if self.valve_type == 'butterfly':
            # Modified equal percentage
            return lambda x: x ** 2 / 100.0
        elif self.valve_type == 'gate':
            # Linear
            return lambda x: x / 100.0
        elif self.valve_type == 'ball':
            # Quick opening
            return lambda x: np.sqrt(x / 100.0)
        else:
            return lambda x: x / 100.0

    def set_position(self, position: float) -> None:
        """Set target valve position [%]."""
        self._target_position = np.clip(position, 0.0, 100.0)

    def step(self, dt: float, delta_p: float = 1.0) -> Dict[str, float]:
        """
        Advance valve simulation.

        Args:
            dt: Time step [s]
            delta_p: Differential pressure [bar]

        Returns:
            Dictionary with valve state
        """
        # Position control
        speed = 100.0 / self._travel_time  # %/s
        position_error = self._target_position - self._position

        if abs(position_error) > 0.1:
            self._health.status = ActuatorStatus.MOVING
            move = np.sign(position_error) * speed * dt
            self._position += np.clip(move, -abs(position_error), abs(position_error))
        else:
            self._position = self._target_position
            self._health.status = ActuatorStatus.READY

        # Calculate Cv at current position
        cv_ratio = self._characteristic(self._position)
        cv_actual = cv_ratio * self.cv_rated

        # Account for seat leakage when closed
        if self._position < 1.0:
            cv_actual = max(cv_actual, self._seat_leakage * self.cv_rated / 100.0)

        # Calculate flow (Q = Cv * sqrt(dP / SG))
        flow = cv_actual * np.sqrt(abs(delta_p))  # Simplified

        # Update cycle count on direction change
        if position_error * self._health.cycle_count > 0:  # Simple detection
            self._health.cycle_count += 1

        # Update wear
        self._health.wear_level += abs(position_error) * dt * 1e-7
        self._health.wear_level = min(1.0, self._health.wear_level)

        # Seat wear affects leakage
        self._seat_leakage = self._health.wear_level * 2.0  # Up to 2% leakage

        return {
            'position': self._position,
            'cv_actual': cv_actual,
            'flow': flow,
            'leakage': self._seat_leakage,
            'status': self._health.status.value
        }

    def get_health(self) -> ActuatorHealth:
        """Get valve health status."""
        return self._health

    def reset(self) -> None:
        """Reset valve state."""
        self._position = 0.0
        self._target_position = 0.0
        self._health = ActuatorHealth()
        self._seat_leakage = 0.0


class ActuatorNetwork:
    """
    Unified actuator network manager.

    Manages all actuators in the digital twin system with:
    - Centralized command distribution
    - Coordinated movement
    - Safety interlocks
    - Health monitoring
    """

    def __init__(self, model: 'TangheSiphonModel'):
        self.model = model

        # Initialize gate actuators
        self.gate_actuators: List[AdvancedGateActuator] = [
            AdvancedGateActuator(model, i)
            for i in range(model.num_gates)
        ]

        # Optional pump and valve
        self.pump: Optional[PumpActuator] = None
        self.inlet_valve: Optional[ValveActuator] = None
        self.outlet_valve: Optional[ValveActuator] = None

        # Safety interlocks
        self._interlocks_enabled = True
        self._emergency_stop_active = False

        # Command history
        self._command_history: Deque[Dict[str, Any]] = deque(maxlen=1000)

        logger.info("ActuatorNetwork initialized with %d gate actuators",
                   len(self.gate_actuators))

    def command_gates(self, targets: np.ndarray) -> List[ActuatorFeedback]:
        """
        Send position commands to all gates.

        Args:
            targets: Target positions for each gate [m]

        Returns:
            List of actuator feedbacks
        """
        if self._emergency_stop_active:
            logger.warning("Command rejected: emergency stop active")
            return [act._create_feedback() for act in self.gate_actuators]

        if len(targets) != len(self.gate_actuators):
            raise ValueError(f"Expected {len(self.gate_actuators)} targets")

        # Apply safety interlocks
        safe_targets = self._apply_interlocks(targets)

        # Send commands
        for actuator, target in zip(self.gate_actuators, safe_targets):
            actuator.command(target)

        # Record command
        self._command_history.append({
            'timestamp': self.model.time,
            'targets': targets.tolist(),
            'safe_targets': safe_targets.tolist()
        })

        return [act._create_feedback() for act in self.gate_actuators]

    def _apply_interlocks(self, targets: np.ndarray) -> np.ndarray:
        """Apply safety interlocks to commands."""
        if not self._interlocks_enabled:
            return targets

        safe_targets = targets.copy()

        # Example interlock: maximum opening rate across all gates
        current = np.array([act._position for act in self.gate_actuators])
        max_change = 0.5  # m maximum change per command

        for i in range(len(safe_targets)):
            change = safe_targets[i] - current[i]
            if abs(change) > max_change:
                safe_targets[i] = current[i] + np.sign(change) * max_change

        return safe_targets

    def step(self, dt: float) -> Dict[str, List[ActuatorFeedback]]:
        """
        Advance all actuators.

        Args:
            dt: Time step [s]

        Returns:
            Dictionary with all actuator feedbacks
        """
        gate_feedbacks = [act.step(dt) for act in self.gate_actuators]

        result = {
            'gates': gate_feedbacks
        }

        if self.pump:
            result['pump'] = self.pump.step(dt)

        if self.inlet_valve:
            result['inlet_valve'] = self.inlet_valve.step(dt)

        if self.outlet_valve:
            result['outlet_valve'] = self.outlet_valve.step(dt)

        return result

    def emergency_stop(self) -> None:
        """Execute emergency stop on all actuators."""
        self._emergency_stop_active = True

        for actuator in self.gate_actuators:
            actuator.emergency_stop()

        if self.pump:
            self.pump.emergency_stop()

        logger.critical("Emergency stop executed on all actuators")

    def reset_emergency_stop(self) -> None:
        """Reset emergency stop condition."""
        self._emergency_stop_active = False
        for actuator in self.gate_actuators:
            actuator.clear_fault()
        logger.info("Emergency stop reset")

    def get_all_health(self) -> Dict[str, Any]:
        """Get health status of all actuators."""
        return {
            'gates': [act.get_health() for act in self.gate_actuators],
            'pump': self.pump.get_health() if self.pump else None,
            'inlet_valve': self.inlet_valve.get_health() if self.inlet_valve else None,
            'outlet_valve': self.outlet_valve.get_health() if self.outlet_valve else None,
            'emergency_stop_active': self._emergency_stop_active
        }

    def maintenance_all(self) -> None:
        """Perform maintenance on all actuators."""
        for actuator in self.gate_actuators:
            actuator.maintenance()
        logger.info("Maintenance completed on all actuators")

    def reset_all(self) -> None:
        """Reset all actuators."""
        self._emergency_stop_active = False
        for actuator in self.gate_actuators:
            actuator.reset()
        if self.pump:
            self.pump.reset()
        if self.inlet_valve:
            self.inlet_valve.reset()
        if self.outlet_valve:
            self.outlet_valve.reset()
        self._command_history.clear()
        logger.info("All actuators reset")
