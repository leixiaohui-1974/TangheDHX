"""
Predictive Maintenance System for Tanghe Inverted Siphon.

Provides comprehensive predictive maintenance capabilities including:
- Equipment degradation modeling (Weibull, Exponential, Linear)
- Health index calculation
- Remaining Useful Life (RUL) estimation
- Failure prediction and early warning
- Maintenance scheduling optimization
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import warnings

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations and Data Classes
# =============================================================================

class HealthStatus(Enum):
    """Equipment health status levels."""
    EXCELLENT = "excellent"      # Health > 90%
    GOOD = "good"               # Health 70-90%
    FAIR = "fair"               # Health 50-70%
    POOR = "poor"               # Health 30-50%
    CRITICAL = "critical"       # Health < 30%


class MaintenancePriority(Enum):
    """Maintenance action priority levels."""
    EMERGENCY = 1       # Immediate action required
    HIGH = 2            # Action within 24 hours
    MEDIUM = 3          # Action within 1 week
    LOW = 4             # Action within 1 month
    ROUTINE = 5         # Scheduled maintenance


class MaintenanceType(Enum):
    """Types of maintenance actions."""
    INSPECTION = "inspection"
    LUBRICATION = "lubrication"
    CALIBRATION = "calibration"
    MINOR_REPAIR = "minor_repair"
    MAJOR_REPAIR = "major_repair"
    REPLACEMENT = "replacement"
    OVERHAUL = "overhaul"


class FailureMode(Enum):
    """Common failure modes for siphon equipment."""
    GATE_STUCK = "gate_stuck"
    GATE_DRIFT = "gate_drift"
    SENSOR_DRIFT = "sensor_drift"
    SENSOR_FAILURE = "sensor_failure"
    ACTUATOR_SLOW = "actuator_slow"
    ACTUATOR_FAILURE = "actuator_failure"
    SEAL_LEAK = "seal_leak"
    CORROSION = "corrosion"
    VIBRATION_EXCESSIVE = "vibration_excessive"
    BLOCKAGE = "blockage"


@dataclass
class EquipmentInfo:
    """Equipment information and specifications."""
    equipment_id: str
    equipment_type: str  # 'gate', 'sensor', 'actuator', 'pump', etc.
    installation_date: datetime
    manufacturer: str = "Unknown"
    model: str = "Unknown"
    expected_lifetime_hours: float = 50000.0
    maintenance_interval_hours: float = 2000.0
    replacement_cost: float = 10000.0
    downtime_cost_per_hour: float = 500.0


@dataclass
class HealthIndex:
    """Comprehensive health index for equipment."""
    overall: float              # 0-100 scale
    mechanical: float = 100.0   # Mechanical health
    electrical: float = 100.0   # Electrical health
    performance: float = 100.0  # Performance efficiency
    reliability: float = 100.0  # Reliability score
    status: HealthStatus = HealthStatus.EXCELLENT
    timestamp: float = 0.0

    def __post_init__(self):
        """Update status based on overall health."""
        if self.overall > 90:
            self.status = HealthStatus.EXCELLENT
        elif self.overall > 70:
            self.status = HealthStatus.GOOD
        elif self.overall > 50:
            self.status = HealthStatus.FAIR
        elif self.overall > 30:
            self.status = HealthStatus.POOR
        else:
            self.status = HealthStatus.CRITICAL


@dataclass
class DegradationState:
    """Current degradation state of equipment."""
    equipment_id: str
    health_index: HealthIndex
    degradation_rate: float     # Health points per hour
    operating_hours: float
    cycle_count: int
    last_maintenance: datetime
    rul_hours: float            # Remaining Useful Life in hours
    failure_probability: float  # Probability of failure in next interval
    confidence: float           # Confidence in prediction (0-1)


@dataclass
class MaintenanceAction:
    """Recommended maintenance action."""
    equipment_id: str
    action_type: MaintenanceType
    priority: MaintenancePriority
    description: str
    estimated_duration_hours: float
    estimated_cost: float
    due_date: datetime
    failure_modes_addressed: List[FailureMode] = field(default_factory=list)
    risk_if_delayed: float = 0.0  # Additional risk percentage if delayed


@dataclass
class FailurePrediction:
    """Failure prediction result."""
    equipment_id: str
    failure_mode: FailureMode
    probability: float          # 0-1
    time_to_failure_hours: float
    confidence: float           # 0-1
    contributing_factors: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)


# =============================================================================
# Degradation Models
# =============================================================================

class DegradationModel(ABC):
    """Abstract base class for degradation models."""

    @abstractmethod
    def calculate_degradation(
        self,
        operating_hours: float,
        initial_health: float = 100.0,
        **kwargs
    ) -> float:
        """Calculate current health based on operating time."""
        pass

    @abstractmethod
    def estimate_rul(
        self,
        current_health: float,
        failure_threshold: float = 30.0,
        **kwargs
    ) -> float:
        """Estimate remaining useful life in hours."""
        pass

    @abstractmethod
    def failure_probability(
        self,
        operating_hours: float,
        interval_hours: float = 100.0,
        **kwargs
    ) -> float:
        """Calculate probability of failure in next interval."""
        pass


class WeibullDegradation(DegradationModel):
    """
    Weibull distribution-based degradation model.

    Commonly used for modeling equipment wear-out failures.
    """

    def __init__(
        self,
        shape: float = 2.0,      # Shape parameter (beta)
        scale: float = 50000.0,  # Scale parameter (eta) in hours
        location: float = 0.0    # Location parameter (gamma)
    ):
        """
        Initialize Weibull degradation model.

        Args:
            shape: Shape parameter beta (>1 for wear-out, <1 for early failure)
            scale: Characteristic life in hours
            location: Minimum life (failure-free period)
        """
        self.shape = shape
        self.scale = scale
        self.location = location
        logger.debug(
            "WeibullDegradation initialized: shape=%.2f, scale=%.0f, location=%.0f",
            shape, scale, location
        )

    def calculate_degradation(
        self,
        operating_hours: float,
        initial_health: float = 100.0,
        **kwargs
    ) -> float:
        """
        Calculate current health using Weibull reliability function.

        Health = initial_health * R(t) where R(t) is reliability function.
        """
        if operating_hours <= self.location:
            return initial_health

        t = operating_hours - self.location
        reliability = np.exp(-((t / self.scale) ** self.shape))
        health = initial_health * reliability

        return max(0.0, min(100.0, health))

    def estimate_rul(
        self,
        current_health: float,
        failure_threshold: float = 30.0,
        **kwargs
    ) -> float:
        """
        Estimate RUL based on current health.

        Solves for time when health reaches threshold.
        """
        if current_health <= failure_threshold:
            return 0.0

        initial_health = kwargs.get('initial_health', 100.0)
        current_hours = kwargs.get('operating_hours', 0.0)

        # Health = initial * exp(-(t/scale)^shape)
        # t = scale * (-ln(health/initial))^(1/shape)
        target_reliability = failure_threshold / initial_health
        if target_reliability <= 0:
            return float('inf')

        try:
            time_to_threshold = self.scale * ((-np.log(target_reliability)) ** (1/self.shape))
            rul = time_to_threshold + self.location - current_hours
            return max(0.0, rul)
        except (ValueError, ZeroDivisionError):
            return 0.0

    def failure_probability(
        self,
        operating_hours: float,
        interval_hours: float = 100.0,
        **kwargs
    ) -> float:
        """
        Calculate probability of failure in next interval.

        P(T <= t+dt | T > t) = 1 - R(t+dt)/R(t)
        """
        if operating_hours < self.location:
            return 0.0

        t = operating_hours - self.location
        t_next = t + interval_hours

        r_current = np.exp(-((t / self.scale) ** self.shape)) if t > 0 else 1.0
        r_next = np.exp(-((t_next / self.scale) ** self.shape))

        if r_current <= 0:
            return 1.0

        prob = 1.0 - (r_next / r_current)
        return max(0.0, min(1.0, prob))


class ExponentialDegradation(DegradationModel):
    """
    Exponential degradation model.

    Models constant failure rate (random failures).
    """

    def __init__(
        self,
        failure_rate: float = 0.00002,  # Failures per hour (lambda)
        degradation_rate: float = 0.001  # Health points lost per hour
    ):
        """
        Initialize exponential degradation model.

        Args:
            failure_rate: Constant failure rate (lambda)
            degradation_rate: Linear health degradation rate
        """
        self.failure_rate = failure_rate
        self.degradation_rate = degradation_rate
        logger.debug(
            "ExponentialDegradation initialized: failure_rate=%.6f, degradation_rate=%.4f",
            failure_rate, degradation_rate
        )

    def calculate_degradation(
        self,
        operating_hours: float,
        initial_health: float = 100.0,
        **kwargs
    ) -> float:
        """Calculate health with exponential reliability decay."""
        reliability = np.exp(-self.failure_rate * operating_hours)
        linear_degradation = self.degradation_rate * operating_hours

        health = initial_health * reliability - linear_degradation
        return max(0.0, min(100.0, health))

    def estimate_rul(
        self,
        current_health: float,
        failure_threshold: float = 30.0,
        **kwargs
    ) -> float:
        """Estimate RUL assuming constant degradation rate."""
        if current_health <= failure_threshold:
            return 0.0

        health_remaining = current_health - failure_threshold

        # Approximate RUL using current degradation rate
        if self.degradation_rate > 0:
            rul = health_remaining / self.degradation_rate
        else:
            # Use MTTF if no linear degradation
            rul = 1.0 / self.failure_rate if self.failure_rate > 0 else float('inf')

        return max(0.0, rul)

    def failure_probability(
        self,
        operating_hours: float,
        interval_hours: float = 100.0,
        **kwargs
    ) -> float:
        """
        Calculate failure probability (memoryless property).

        P(failure in interval) = 1 - exp(-lambda * interval)
        """
        prob = 1.0 - np.exp(-self.failure_rate * interval_hours)
        return max(0.0, min(1.0, prob))


class LinearDegradation(DegradationModel):
    """
    Simple linear degradation model.

    Health decreases linearly with operating time.
    """

    def __init__(
        self,
        degradation_rate: float = 0.002,  # Health points per hour
        initial_health: float = 100.0
    ):
        """
        Initialize linear degradation model.

        Args:
            degradation_rate: Health points lost per operating hour
            initial_health: Starting health level
        """
        self.degradation_rate = degradation_rate
        self.initial_health = initial_health
        logger.debug(
            "LinearDegradation initialized: rate=%.4f, initial=%.1f",
            degradation_rate, initial_health
        )

    def calculate_degradation(
        self,
        operating_hours: float,
        initial_health: float = 100.0,
        **kwargs
    ) -> float:
        """Calculate health with linear degradation."""
        health = initial_health - (self.degradation_rate * operating_hours)
        return max(0.0, min(100.0, health))

    def estimate_rul(
        self,
        current_health: float,
        failure_threshold: float = 30.0,
        **kwargs
    ) -> float:
        """Estimate RUL with linear degradation."""
        if current_health <= failure_threshold:
            return 0.0

        if self.degradation_rate <= 0:
            return float('inf')

        rul = (current_health - failure_threshold) / self.degradation_rate
        return max(0.0, rul)

    def failure_probability(
        self,
        operating_hours: float,
        interval_hours: float = 100.0,
        **kwargs
    ) -> float:
        """Calculate probability of reaching failure threshold in interval."""
        current_health = kwargs.get('current_health', 100.0)
        failure_threshold = kwargs.get('failure_threshold', 30.0)

        health_after_interval = current_health - (self.degradation_rate * interval_hours)

        if health_after_interval <= failure_threshold:
            # Calculate exact probability based on margin
            margin = current_health - failure_threshold
            expected_loss = self.degradation_rate * interval_hours
            if expected_loss > 0:
                return min(1.0, expected_loss / (margin + expected_loss))
            return 0.0

        return 0.0


# =============================================================================
# Equipment Health Monitor
# =============================================================================

class EquipmentHealthMonitor:
    """
    Monitors equipment health and tracks degradation over time.
    """

    def __init__(
        self,
        equipment_info: EquipmentInfo,
        degradation_model: Optional[DegradationModel] = None,
        history_length: int = 1000
    ):
        """
        Initialize health monitor for equipment.

        Args:
            equipment_info: Equipment specifications
            degradation_model: Model for degradation calculation
            history_length: Number of historical readings to keep
        """
        self.equipment_info = equipment_info
        self.degradation_model = degradation_model or WeibullDegradation()

        # Operating statistics
        self.operating_hours: float = 0.0
        self.cycle_count: int = 0
        self.last_maintenance: datetime = equipment_info.installation_date

        # Health tracking
        self._health_history: deque = deque(maxlen=history_length)
        self._current_health = HealthIndex(overall=100.0)

        # Sensor data buffers for health calculation
        self._vibration_buffer: deque = deque(maxlen=100)
        self._temperature_buffer: deque = deque(maxlen=100)
        self._performance_buffer: deque = deque(maxlen=100)

        # Anomaly detection thresholds
        self._vibration_threshold = 2.0  # m/s²
        self._temperature_threshold = 40.0  # °C

        logger.info(
            "EquipmentHealthMonitor initialized for %s (%s)",
            equipment_info.equipment_id,
            equipment_info.equipment_type
        )

    def update(
        self,
        timestamp: float,
        operating: bool = True,
        vibration: Optional[float] = None,
        temperature: Optional[float] = None,
        performance_metric: Optional[float] = None,
        **kwargs
    ) -> HealthIndex:
        """
        Update health monitor with new data.

        Args:
            timestamp: Current timestamp (hours since start)
            operating: Whether equipment is currently operating
            vibration: Vibration measurement (m/s²)
            temperature: Temperature measurement (°C)
            performance_metric: Performance indicator (0-100)
            **kwargs: Additional sensor readings

        Returns:
            Updated health index
        """
        # Update operating hours
        if operating:
            dt = timestamp - (self._health_history[-1].timestamp if self._health_history else 0)
            self.operating_hours += max(0, dt)

        # Buffer sensor data
        if vibration is not None:
            self._vibration_buffer.append(vibration)
        if temperature is not None:
            self._temperature_buffer.append(temperature)
        if performance_metric is not None:
            self._performance_buffer.append(performance_metric)

        # Calculate health components
        mechanical_health = self._calculate_mechanical_health()
        electrical_health = self._calculate_electrical_health()
        performance_health = self._calculate_performance_health()
        reliability_health = self._calculate_reliability_health()

        # Calculate overall health (weighted average)
        overall = (
            0.30 * mechanical_health +
            0.20 * electrical_health +
            0.30 * performance_health +
            0.20 * reliability_health
        )

        # Apply degradation model
        degraded_health = self.degradation_model.calculate_degradation(
            self.operating_hours,
            initial_health=overall
        )

        # Create health index
        self._current_health = HealthIndex(
            overall=degraded_health,
            mechanical=mechanical_health,
            electrical=electrical_health,
            performance=performance_health,
            reliability=reliability_health,
            timestamp=timestamp
        )

        # Store in history
        self._health_history.append(self._current_health)

        return self._current_health

    def _calculate_mechanical_health(self) -> float:
        """Calculate mechanical health from vibration data."""
        if not self._vibration_buffer:
            return 100.0

        avg_vibration = np.mean(list(self._vibration_buffer))
        max_vibration = np.max(list(self._vibration_buffer))

        # Health decreases with vibration
        health = 100.0
        if avg_vibration > self._vibration_threshold:
            health -= (avg_vibration - self._vibration_threshold) * 20
        if max_vibration > self._vibration_threshold * 2:
            health -= 20

        return max(0.0, min(100.0, health))

    def _calculate_electrical_health(self) -> float:
        """Calculate electrical health from temperature data."""
        if not self._temperature_buffer:
            return 100.0

        avg_temp = np.mean(list(self._temperature_buffer))
        max_temp = np.max(list(self._temperature_buffer))

        # Health decreases with high temperature
        health = 100.0
        if avg_temp > self._temperature_threshold:
            health -= (avg_temp - self._temperature_threshold) * 2
        if max_temp > self._temperature_threshold * 1.5:
            health -= 15

        return max(0.0, min(100.0, health))

    def _calculate_performance_health(self) -> float:
        """Calculate performance health from efficiency metrics."""
        if not self._performance_buffer:
            return 100.0

        # Performance buffer contains 0-100 efficiency values
        avg_performance = np.mean(list(self._performance_buffer))
        trend = self._calculate_trend(list(self._performance_buffer))

        health = avg_performance
        # Penalize declining trend
        if trend < -0.1:
            health -= 10

        return max(0.0, min(100.0, health))

    def _calculate_reliability_health(self) -> float:
        """Calculate reliability based on degradation model."""
        failure_prob = self.degradation_model.failure_probability(
            self.operating_hours,
            interval_hours=100.0
        )

        # Convert failure probability to health score
        reliability = (1.0 - failure_prob) * 100.0
        return max(0.0, min(100.0, reliability))

    def _calculate_trend(self, data: List[float]) -> float:
        """Calculate trend (slope) in data."""
        if len(data) < 2:
            return 0.0

        x = np.arange(len(data))
        try:
            slope, _ = np.polyfit(x, data, 1)
            return slope
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    def get_degradation_state(self, timestamp: float) -> DegradationState:
        """Get comprehensive degradation state."""
        rul = self.degradation_model.estimate_rul(
            self._current_health.overall,
            failure_threshold=30.0,
            operating_hours=self.operating_hours
        )

        failure_prob = self.degradation_model.failure_probability(
            self.operating_hours,
            interval_hours=168.0  # One week
        )

        # Calculate degradation rate from history
        degradation_rate = self._estimate_degradation_rate()

        return DegradationState(
            equipment_id=self.equipment_info.equipment_id,
            health_index=self._current_health,
            degradation_rate=degradation_rate,
            operating_hours=self.operating_hours,
            cycle_count=self.cycle_count,
            last_maintenance=self.last_maintenance,
            rul_hours=rul,
            failure_probability=failure_prob,
            confidence=self._calculate_confidence()
        )

    def _estimate_degradation_rate(self) -> float:
        """Estimate current degradation rate from history."""
        if len(self._health_history) < 10:
            return 0.001  # Default rate

        recent = list(self._health_history)[-10:]
        health_values = [h.overall for h in recent]
        time_values = [h.timestamp for h in recent]

        if time_values[-1] - time_values[0] > 0:
            rate = (health_values[0] - health_values[-1]) / (time_values[-1] - time_values[0])
            return max(0.0, rate)

        return 0.001

    def _calculate_confidence(self) -> float:
        """Calculate confidence in predictions based on data quality."""
        # More data = higher confidence
        data_points = len(self._health_history)
        confidence = min(1.0, data_points / 100.0)

        # Reduce confidence if recent data is inconsistent
        if len(self._health_history) > 10:
            recent = [h.overall for h in list(self._health_history)[-10:]]
            variance = np.var(recent)
            if variance > 100:
                confidence *= 0.7

        return confidence

    def record_maintenance(self, timestamp: datetime) -> None:
        """Record that maintenance was performed."""
        self.last_maintenance = timestamp
        self.cycle_count += 1

        # Reset some health components after maintenance
        if self._current_health.overall < 70:
            # Maintenance restores some health
            restored_health = min(90.0, self._current_health.overall + 30)
            self._current_health = HealthIndex(
                overall=restored_health,
                mechanical=min(100.0, self._current_health.mechanical + 20),
                electrical=min(100.0, self._current_health.electrical + 20),
                performance=min(100.0, self._current_health.performance + 20),
                reliability=min(100.0, self._current_health.reliability + 10),
                timestamp=self._current_health.timestamp
            )

        logger.info(
            "Maintenance recorded for %s, health restored to %.1f%%",
            self.equipment_info.equipment_id,
            self._current_health.overall
        )


# =============================================================================
# Failure Prediction Engine
# =============================================================================

class FailurePredictionEngine:
    """
    Predicts equipment failures based on health monitoring and historical data.
    """

    def __init__(
        self,
        failure_threshold: float = 30.0,
        warning_threshold: float = 50.0,
        prediction_horizon_hours: float = 168.0  # One week
    ):
        """
        Initialize failure prediction engine.

        Args:
            failure_threshold: Health level considered as failure
            warning_threshold: Health level for early warning
            prediction_horizon_hours: Time horizon for predictions
        """
        self.failure_threshold = failure_threshold
        self.warning_threshold = warning_threshold
        self.prediction_horizon = prediction_horizon_hours

        # Historical failure data for learning
        self._failure_history: List[Dict[str, Any]] = []

        # Failure mode signatures (patterns that indicate specific failures)
        self._failure_signatures = self._init_failure_signatures()

        logger.info(
            "FailurePredictionEngine initialized: threshold=%.1f, horizon=%.0f hours",
            failure_threshold, prediction_horizon_hours
        )

    def _init_failure_signatures(self) -> Dict[FailureMode, Dict[str, Any]]:
        """Initialize failure mode signatures."""
        return {
            FailureMode.GATE_STUCK: {
                'mechanical_threshold': 40.0,
                'vibration_pattern': 'sudden_stop',
                'indicators': ['position_error', 'motor_current_spike']
            },
            FailureMode.GATE_DRIFT: {
                'performance_threshold': 60.0,
                'trend': 'gradual_decline',
                'indicators': ['calibration_error', 'sensor_drift']
            },
            FailureMode.SENSOR_DRIFT: {
                'electrical_threshold': 50.0,
                'pattern': 'increasing_noise',
                'indicators': ['measurement_variance', 'zero_offset']
            },
            FailureMode.ACTUATOR_SLOW: {
                'mechanical_threshold': 50.0,
                'performance_threshold': 70.0,
                'indicators': ['response_time', 'position_lag']
            },
            FailureMode.SEAL_LEAK: {
                'reliability_threshold': 60.0,
                'indicators': ['pressure_drop', 'flow_anomaly']
            },
            FailureMode.VIBRATION_EXCESSIVE: {
                'mechanical_threshold': 30.0,
                'vibration_level': 3.0,
                'indicators': ['bearing_wear', 'imbalance']
            }
        }

    def predict_failures(
        self,
        health_monitor: EquipmentHealthMonitor
    ) -> List[FailurePrediction]:
        """
        Predict potential failures for equipment.

        Args:
            health_monitor: Equipment health monitor

        Returns:
            List of failure predictions
        """
        predictions = []
        state = health_monitor.get_degradation_state(health_monitor.operating_hours)
        health = state.health_index

        # Check each failure mode
        for failure_mode, signature in self._failure_signatures.items():
            prediction = self._check_failure_mode(
                failure_mode, signature, health, state
            )
            if prediction is not None:
                predictions.append(prediction)

        # Sort by probability (highest first)
        predictions.sort(key=lambda p: p.probability, reverse=True)

        return predictions

    def _check_failure_mode(
        self,
        failure_mode: FailureMode,
        signature: Dict[str, Any],
        health: HealthIndex,
        state: DegradationState
    ) -> Optional[FailurePrediction]:
        """Check if a specific failure mode is likely."""
        probability = 0.0
        contributing_factors = []

        # Check mechanical threshold
        if 'mechanical_threshold' in signature:
            if health.mechanical < signature['mechanical_threshold']:
                probability += 0.3
                contributing_factors.append(
                    f"Low mechanical health: {health.mechanical:.1f}%"
                )

        # Check electrical threshold
        if 'electrical_threshold' in signature:
            if health.electrical < signature['electrical_threshold']:
                probability += 0.2
                contributing_factors.append(
                    f"Low electrical health: {health.electrical:.1f}%"
                )

        # Check performance threshold
        if 'performance_threshold' in signature:
            if health.performance < signature['performance_threshold']:
                probability += 0.25
                contributing_factors.append(
                    f"Low performance: {health.performance:.1f}%"
                )

        # Check reliability threshold
        if 'reliability_threshold' in signature:
            if health.reliability < signature['reliability_threshold']:
                probability += 0.25
                contributing_factors.append(
                    f"Low reliability: {health.reliability:.1f}%"
                )

        # Base probability on overall degradation
        if health.overall < self.warning_threshold:
            probability += 0.2
            contributing_factors.append(
                f"Overall health critical: {health.overall:.1f}%"
            )

        # Only return prediction if probability is significant
        if probability < 0.1:
            return None

        # Calculate time to failure
        ttf = self._estimate_time_to_failure(state, probability)

        # Generate recommendations
        recommendations = self._generate_recommendations(failure_mode, probability)

        return FailurePrediction(
            equipment_id=state.equipment_id,
            failure_mode=failure_mode,
            probability=min(1.0, probability),
            time_to_failure_hours=ttf,
            confidence=state.confidence,
            contributing_factors=contributing_factors,
            recommended_actions=recommendations
        )

    def _estimate_time_to_failure(
        self,
        state: DegradationState,
        probability: float
    ) -> float:
        """Estimate time to failure based on degradation state."""
        if state.degradation_rate > 0:
            health_remaining = state.health_index.overall - self.failure_threshold
            ttf = health_remaining / state.degradation_rate
        else:
            ttf = state.rul_hours

        # Adjust based on failure probability
        ttf *= (1.0 - probability * 0.5)

        return max(0.0, ttf)

    def _generate_recommendations(
        self,
        failure_mode: FailureMode,
        probability: float
    ) -> List[str]:
        """Generate recommendations based on failure mode and probability."""
        recommendations = []

        if probability > 0.5:
            recommendations.append("Schedule immediate inspection")
        elif probability > 0.3:
            recommendations.append("Plan preventive maintenance within 1 week")
        else:
            recommendations.append("Monitor closely, include in next routine check")

        # Mode-specific recommendations
        mode_recommendations = {
            FailureMode.GATE_STUCK: [
                "Check lubrication levels",
                "Inspect mechanical linkages",
                "Test emergency release mechanism"
            ],
            FailureMode.GATE_DRIFT: [
                "Recalibrate position sensors",
                "Check for mechanical wear",
                "Verify actuator feedback loop"
            ],
            FailureMode.SENSOR_DRIFT: [
                "Perform sensor calibration",
                "Check wiring connections",
                "Consider sensor replacement"
            ],
            FailureMode.ACTUATOR_SLOW: [
                "Check hydraulic/pneumatic pressure",
                "Inspect actuator seals",
                "Test control signal integrity"
            ],
            FailureMode.SEAL_LEAK: [
                "Inspect all seals visually",
                "Check pressure readings",
                "Plan seal replacement"
            ],
            FailureMode.VIBRATION_EXCESSIVE: [
                "Check bearing condition",
                "Verify alignment",
                "Balance rotating components"
            ]
        }

        if failure_mode in mode_recommendations:
            recommendations.extend(mode_recommendations[failure_mode])

        return recommendations

    def record_failure(
        self,
        equipment_id: str,
        failure_mode: FailureMode,
        timestamp: datetime,
        health_at_failure: float,
        operating_hours: float
    ) -> None:
        """Record an actual failure for learning."""
        self._failure_history.append({
            'equipment_id': equipment_id,
            'failure_mode': failure_mode,
            'timestamp': timestamp,
            'health_at_failure': health_at_failure,
            'operating_hours': operating_hours
        })

        logger.info(
            "Failure recorded: %s - %s at %.0f hours, health=%.1f%%",
            equipment_id, failure_mode.value, operating_hours, health_at_failure
        )


# =============================================================================
# Maintenance Scheduler
# =============================================================================

class MaintenanceScheduler:
    """
    Optimizes maintenance scheduling based on health predictions and costs.
    """

    def __init__(
        self,
        planning_horizon_days: int = 30,
        max_daily_maintenance_hours: float = 8.0
    ):
        """
        Initialize maintenance scheduler.

        Args:
            planning_horizon_days: Days to plan ahead
            max_daily_maintenance_hours: Maximum maintenance hours per day
        """
        self.planning_horizon = planning_horizon_days
        self.max_daily_hours = max_daily_maintenance_hours

        # Scheduled actions
        self._scheduled_actions: List[MaintenanceAction] = []

        # Cost factors
        self._cost_factors = {
            MaintenanceType.INSPECTION: 100.0,
            MaintenanceType.LUBRICATION: 200.0,
            MaintenanceType.CALIBRATION: 300.0,
            MaintenanceType.MINOR_REPAIR: 1000.0,
            MaintenanceType.MAJOR_REPAIR: 5000.0,
            MaintenanceType.REPLACEMENT: 10000.0,
            MaintenanceType.OVERHAUL: 15000.0
        }

        # Duration factors (hours)
        self._duration_factors = {
            MaintenanceType.INSPECTION: 1.0,
            MaintenanceType.LUBRICATION: 0.5,
            MaintenanceType.CALIBRATION: 2.0,
            MaintenanceType.MINOR_REPAIR: 4.0,
            MaintenanceType.MAJOR_REPAIR: 16.0,
            MaintenanceType.REPLACEMENT: 8.0,
            MaintenanceType.OVERHAUL: 40.0
        }

        logger.info(
            "MaintenanceScheduler initialized: horizon=%d days",
            planning_horizon_days
        )

    def generate_maintenance_plan(
        self,
        health_monitors: Dict[str, EquipmentHealthMonitor],
        failure_predictions: Dict[str, List[FailurePrediction]]
    ) -> List[MaintenanceAction]:
        """
        Generate optimized maintenance plan.

        Args:
            health_monitors: Dict of equipment ID to health monitor
            failure_predictions: Dict of equipment ID to failure predictions

        Returns:
            List of scheduled maintenance actions
        """
        actions = []
        current_time = datetime.now()

        for equipment_id, monitor in health_monitors.items():
            state = monitor.get_degradation_state(monitor.operating_hours)
            predictions = failure_predictions.get(equipment_id, [])

            # Determine required maintenance based on health and predictions
            equipment_actions = self._determine_maintenance_actions(
                equipment_id,
                monitor.equipment_info,
                state,
                predictions,
                current_time
            )

            actions.extend(equipment_actions)

        # Optimize schedule (avoid conflicts, respect capacity)
        optimized_actions = self._optimize_schedule(actions)

        # Store scheduled actions
        self._scheduled_actions = optimized_actions

        return optimized_actions

    def _determine_maintenance_actions(
        self,
        equipment_id: str,
        equipment_info: EquipmentInfo,
        state: DegradationState,
        predictions: List[FailurePrediction],
        current_time: datetime
    ) -> List[MaintenanceAction]:
        """Determine required maintenance actions for equipment."""
        actions = []

        # Check for critical health
        if state.health_index.status == HealthStatus.CRITICAL:
            actions.append(MaintenanceAction(
                equipment_id=equipment_id,
                action_type=MaintenanceType.MAJOR_REPAIR,
                priority=MaintenancePriority.EMERGENCY,
                description=f"Critical health: {state.health_index.overall:.1f}%",
                estimated_duration_hours=self._duration_factors[MaintenanceType.MAJOR_REPAIR],
                estimated_cost=self._cost_factors[MaintenanceType.MAJOR_REPAIR],
                due_date=current_time,
                risk_if_delayed=0.8
            ))

        # Check for high failure probability predictions
        for prediction in predictions:
            if prediction.probability > 0.5:
                action_type = self._select_action_type(prediction)
                priority = self._calculate_priority(prediction)

                actions.append(MaintenanceAction(
                    equipment_id=equipment_id,
                    action_type=action_type,
                    priority=priority,
                    description=f"Predicted {prediction.failure_mode.value}: {prediction.probability:.0%}",
                    estimated_duration_hours=self._duration_factors[action_type],
                    estimated_cost=self._cost_factors[action_type],
                    due_date=current_time + timedelta(hours=prediction.time_to_failure_hours * 0.7),
                    failure_modes_addressed=[prediction.failure_mode],
                    risk_if_delayed=prediction.probability
                ))

        # Check for routine maintenance due
        hours_since_maintenance = (current_time - state.last_maintenance).total_seconds() / 3600
        if hours_since_maintenance > equipment_info.maintenance_interval_hours:
            actions.append(MaintenanceAction(
                equipment_id=equipment_id,
                action_type=MaintenanceType.INSPECTION,
                priority=MaintenancePriority.ROUTINE,
                description="Routine inspection overdue",
                estimated_duration_hours=self._duration_factors[MaintenanceType.INSPECTION],
                estimated_cost=self._cost_factors[MaintenanceType.INSPECTION],
                due_date=current_time + timedelta(days=7),
                risk_if_delayed=0.1
            ))

        return actions

    def _select_action_type(self, prediction: FailurePrediction) -> MaintenanceType:
        """Select appropriate maintenance action type based on prediction."""
        if prediction.probability > 0.8:
            return MaintenanceType.REPLACEMENT
        elif prediction.probability > 0.6:
            return MaintenanceType.MAJOR_REPAIR
        elif prediction.probability > 0.4:
            return MaintenanceType.MINOR_REPAIR
        elif prediction.failure_mode in [FailureMode.SENSOR_DRIFT, FailureMode.GATE_DRIFT]:
            return MaintenanceType.CALIBRATION
        else:
            return MaintenanceType.INSPECTION

    def _calculate_priority(self, prediction: FailurePrediction) -> MaintenancePriority:
        """Calculate maintenance priority based on prediction."""
        if prediction.probability > 0.8 or prediction.time_to_failure_hours < 24:
            return MaintenancePriority.EMERGENCY
        elif prediction.probability > 0.6 or prediction.time_to_failure_hours < 72:
            return MaintenancePriority.HIGH
        elif prediction.probability > 0.4 or prediction.time_to_failure_hours < 168:
            return MaintenancePriority.MEDIUM
        elif prediction.probability > 0.2:
            return MaintenancePriority.LOW
        else:
            return MaintenancePriority.ROUTINE

    def _optimize_schedule(
        self,
        actions: List[MaintenanceAction]
    ) -> List[MaintenanceAction]:
        """
        Optimize maintenance schedule to minimize costs and conflicts.
        """
        if not actions:
            return []

        # Sort by priority and due date
        actions.sort(key=lambda a: (a.priority.value, a.due_date))

        # Group actions by day to respect capacity
        scheduled = []
        daily_hours: Dict[str, float] = {}

        for action in actions:
            day_key = action.due_date.strftime('%Y-%m-%d')
            current_hours = daily_hours.get(day_key, 0.0)

            if current_hours + action.estimated_duration_hours <= self.max_daily_hours:
                scheduled.append(action)
                daily_hours[day_key] = current_hours + action.estimated_duration_hours
            else:
                # Reschedule to next available day
                new_date = action.due_date + timedelta(days=1)
                while daily_hours.get(new_date.strftime('%Y-%m-%d'), 0.0) + action.estimated_duration_hours > self.max_daily_hours:
                    new_date += timedelta(days=1)
                    if (new_date - datetime.now()).days > self.planning_horizon:
                        break

                rescheduled_action = MaintenanceAction(
                    equipment_id=action.equipment_id,
                    action_type=action.action_type,
                    priority=action.priority,
                    description=action.description,
                    estimated_duration_hours=action.estimated_duration_hours,
                    estimated_cost=action.estimated_cost,
                    due_date=new_date,
                    failure_modes_addressed=action.failure_modes_addressed,
                    risk_if_delayed=action.risk_if_delayed
                )
                scheduled.append(rescheduled_action)
                new_day_key = new_date.strftime('%Y-%m-%d')
                daily_hours[new_day_key] = daily_hours.get(new_day_key, 0.0) + action.estimated_duration_hours

        return scheduled

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get summary of maintenance costs."""
        total_cost = sum(a.estimated_cost for a in self._scheduled_actions)
        total_hours = sum(a.estimated_duration_hours for a in self._scheduled_actions)

        by_priority = {}
        for priority in MaintenancePriority:
            priority_actions = [a for a in self._scheduled_actions if a.priority == priority]
            by_priority[priority.name] = {
                'count': len(priority_actions),
                'cost': sum(a.estimated_cost for a in priority_actions),
                'hours': sum(a.estimated_duration_hours for a in priority_actions)
            }

        return {
            'total_actions': len(self._scheduled_actions),
            'total_cost': total_cost,
            'total_hours': total_hours,
            'by_priority': by_priority
        }


# =============================================================================
# Predictive Maintenance System (Main Interface)
# =============================================================================

class PredictiveMaintenanceSystem:
    """
    Integrated predictive maintenance system for the Tanghe Inverted Siphon.

    Combines health monitoring, failure prediction, and maintenance scheduling.
    """

    def __init__(
        self,
        failure_threshold: float = 30.0,
        warning_threshold: float = 50.0,
        planning_horizon_days: int = 30
    ):
        """
        Initialize predictive maintenance system.

        Args:
            failure_threshold: Health level considered as failure
            warning_threshold: Health level for early warning
            planning_horizon_days: Days to plan maintenance ahead
        """
        self.failure_threshold = failure_threshold
        self.warning_threshold = warning_threshold

        # Component modules
        self.failure_engine = FailurePredictionEngine(
            failure_threshold=failure_threshold,
            warning_threshold=warning_threshold
        )
        self.scheduler = MaintenanceScheduler(
            planning_horizon_days=planning_horizon_days
        )

        # Equipment monitors
        self._monitors: Dict[str, EquipmentHealthMonitor] = {}

        # System state
        self._last_update: float = 0.0
        self._system_health: float = 100.0

        logger.info("PredictiveMaintenanceSystem initialized")

    def register_equipment(
        self,
        equipment_info: EquipmentInfo,
        degradation_model: Optional[DegradationModel] = None
    ) -> EquipmentHealthMonitor:
        """
        Register equipment for monitoring.

        Args:
            equipment_info: Equipment specifications
            degradation_model: Optional custom degradation model

        Returns:
            Health monitor for the equipment
        """
        monitor = EquipmentHealthMonitor(
            equipment_info=equipment_info,
            degradation_model=degradation_model
        )

        self._monitors[equipment_info.equipment_id] = monitor

        logger.info(
            "Equipment registered: %s (%s)",
            equipment_info.equipment_id,
            equipment_info.equipment_type
        )

        return monitor

    def update(
        self,
        timestamp: float,
        sensor_data: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Update system with new sensor data.

        Args:
            timestamp: Current timestamp (hours)
            sensor_data: Dict mapping equipment_id to sensor readings

        Returns:
            Update result with health statuses and alerts
        """
        health_updates = {}
        alerts = []

        for equipment_id, readings in sensor_data.items():
            if equipment_id in self._monitors:
                monitor = self._monitors[equipment_id]

                health = monitor.update(
                    timestamp=timestamp,
                    **readings
                )

                health_updates[equipment_id] = {
                    'health': health.overall,
                    'status': health.status.value,
                    'mechanical': health.mechanical,
                    'electrical': health.electrical,
                    'performance': health.performance,
                    'reliability': health.reliability
                }

                # Generate alerts for degraded equipment
                if health.status in [HealthStatus.POOR, HealthStatus.CRITICAL]:
                    alerts.append({
                        'equipment_id': equipment_id,
                        'type': 'health_warning',
                        'severity': 'critical' if health.status == HealthStatus.CRITICAL else 'warning',
                        'message': f"Health at {health.overall:.1f}%",
                        'timestamp': timestamp
                    })

        # Update system health
        if health_updates:
            self._system_health = np.mean([h['health'] for h in health_updates.values()])

        self._last_update = timestamp

        return {
            'timestamp': timestamp,
            'system_health': self._system_health,
            'equipment_health': health_updates,
            'alerts': alerts
        }

    def predict_failures(self) -> Dict[str, List[FailurePrediction]]:
        """
        Predict failures for all monitored equipment.

        Returns:
            Dict mapping equipment_id to failure predictions
        """
        predictions = {}

        for equipment_id, monitor in self._monitors.items():
            equipment_predictions = self.failure_engine.predict_failures(monitor)
            if equipment_predictions:
                predictions[equipment_id] = equipment_predictions

        return predictions

    def generate_maintenance_plan(self) -> List[MaintenanceAction]:
        """
        Generate optimized maintenance plan.

        Returns:
            List of scheduled maintenance actions
        """
        predictions = self.predict_failures()

        plan = self.scheduler.generate_maintenance_plan(
            self._monitors,
            predictions
        )

        return plan

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        equipment_status = {}

        for equipment_id, monitor in self._monitors.items():
            state = monitor.get_degradation_state(monitor.operating_hours)
            equipment_status[equipment_id] = {
                'health': state.health_index.overall,
                'status': state.health_index.status.value,
                'operating_hours': state.operating_hours,
                'rul_hours': state.rul_hours,
                'failure_probability': state.failure_probability,
                'last_maintenance': state.last_maintenance.isoformat()
            }

        # Get maintenance cost summary
        cost_summary = self.scheduler.get_cost_summary()

        return {
            'system_health': self._system_health,
            'last_update': self._last_update,
            'equipment_count': len(self._monitors),
            'equipment_status': equipment_status,
            'maintenance_costs': cost_summary
        }

    def get_equipment_report(self, equipment_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed report for specific equipment."""
        if equipment_id not in self._monitors:
            return None

        monitor = self._monitors[equipment_id]
        state = monitor.get_degradation_state(monitor.operating_hours)
        predictions = self.failure_engine.predict_failures(monitor)

        return {
            'equipment_id': equipment_id,
            'equipment_type': monitor.equipment_info.equipment_type,
            'health_index': {
                'overall': state.health_index.overall,
                'mechanical': state.health_index.mechanical,
                'electrical': state.health_index.electrical,
                'performance': state.health_index.performance,
                'reliability': state.health_index.reliability,
                'status': state.health_index.status.value
            },
            'degradation': {
                'rate': state.degradation_rate,
                'operating_hours': state.operating_hours,
                'cycle_count': state.cycle_count,
                'rul_hours': state.rul_hours
            },
            'failure_predictions': [
                {
                    'mode': p.failure_mode.value,
                    'probability': p.probability,
                    'time_to_failure_hours': p.time_to_failure_hours,
                    'contributing_factors': p.contributing_factors,
                    'recommendations': p.recommended_actions
                }
                for p in predictions
            ],
            'maintenance': {
                'last_maintenance': state.last_maintenance.isoformat(),
                'interval_hours': monitor.equipment_info.maintenance_interval_hours,
                'hours_since_maintenance': (
                    datetime.now() - state.last_maintenance
                ).total_seconds() / 3600
            }
        }
