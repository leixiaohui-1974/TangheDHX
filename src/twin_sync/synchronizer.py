"""
Digital Twin Synchronization Engine.

This module provides comprehensive synchronization between physical systems
and their digital twin representations, including data fusion, latency
compensation, and discrepancy detection.
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class SyncMode(Enum):
    """Synchronization mode."""
    REALTIME = auto()      # Real-time synchronization
    BATCH = auto()         # Batch synchronization
    ON_DEMAND = auto()     # Sync only when requested
    PREDICTIVE = auto()    # Use prediction to compensate for delays


class SyncStatus(Enum):
    """Synchronization status."""
    IDLE = auto()          # Not syncing
    SYNCING = auto()       # Currently synchronizing
    SYNCHRONIZED = auto()  # Fully synchronized
    DEGRADED = auto()      # Partial synchronization
    FAILED = auto()        # Synchronization failed
    RECOVERING = auto()    # Recovering from failure


class SyncQuality(Enum):
    """Quality of synchronization."""
    EXCELLENT = auto()     # < 1% error, < 10ms latency
    GOOD = auto()          # < 5% error, < 50ms latency
    ACCEPTABLE = auto()    # < 10% error, < 200ms latency
    POOR = auto()          # < 20% error, < 500ms latency
    UNACCEPTABLE = auto()  # >= 20% error or >= 500ms latency


class SourceType(Enum):
    """Type of data source."""
    PHYSICAL_SENSOR = auto()    # Direct sensor reading
    PHYSICAL_ACTUATOR = auto()  # Actuator feedback
    SCADA = auto()              # SCADA system
    HISTORIAN = auto()          # Historical database
    MODEL_OUTPUT = auto()       # Digital model output
    ESTIMATION = auto()         # Estimated/inferred value
    MANUAL = auto()             # Manual input


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DataSource:
    """Configuration for a data source."""
    source_id: str
    source_type: SourceType
    name: str
    variables: List[str]
    sample_rate_hz: float = 1.0
    latency_ms: float = 0.0
    reliability: float = 1.0  # 0-1, probability of valid data
    accuracy: float = 0.95    # 0-1, measurement accuracy
    priority: int = 1         # Lower = higher priority
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateVector:
    """State vector with uncertainty."""
    timestamp: float
    values: Dict[str, float]
    uncertainties: Dict[str, float] = field(default_factory=dict)
    source_id: Optional[str] = None
    confidence: float = 1.0
    quality: SyncQuality = SyncQuality.GOOD


@dataclass
class SyncMetrics:
    """Synchronization performance metrics."""
    sync_count: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    avg_discrepancy: float = 0.0
    max_discrepancy: float = 0.0
    fusion_count: int = 0
    recovery_count: int = 0
    last_sync_time: Optional[float] = None
    uptime_seconds: float = 0.0
    quality_history: List[SyncQuality] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if self.sync_count == 0:
            return 0.0
        return self.total_latency_ms / self.sync_count

    @property
    def current_quality(self) -> SyncQuality:
        """Current sync quality based on recent history."""
        if not self.quality_history:
            return SyncQuality.GOOD
        # Use mode of last 10 quality readings
        recent = self.quality_history[-10:]
        from collections import Counter
        return Counter(recent).most_common(1)[0][0]


@dataclass
class DiscrepancyEvent:
    """Record of a discrepancy detection."""
    event_id: str
    timestamp: float
    variable: str
    physical_value: float
    digital_value: float
    discrepancy: float
    discrepancy_pct: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    resolved: bool = False
    resolution_time: Optional[float] = None
    resolution_method: Optional[str] = None


@dataclass
class LatencyEstimate:
    """Latency estimation for a source."""
    source_id: str
    current_latency_ms: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    jitter_ms: float
    sample_count: int
    last_update: float


@dataclass
class ConfidenceScore:
    """Confidence score for a state estimate."""
    variable: str
    confidence: float  # 0-1
    contributing_sources: List[str]
    source_weights: Dict[str, float]
    uncertainty: float
    last_update: float


# =============================================================================
# Data Fusion Engine
# =============================================================================

class DataFusionEngine:
    """
    Multi-source data fusion with uncertainty quantification.

    Uses weighted averaging with Kalman-like update for optimal
    combination of multiple data sources.
    """

    def __init__(self, default_process_noise: float = 0.01):
        """
        Initialize data fusion engine.

        Args:
            default_process_noise: Default process noise for state estimation
        """
        self._sources: Dict[str, DataSource] = {}
        self._state: Dict[str, float] = {}
        self._variance: Dict[str, float] = {}
        self._process_noise = default_process_noise
        self._history: Dict[str, deque] = {}
        self._lock = threading.Lock()

        logger.info("DataFusionEngine initialized")

    def register_source(self, source: DataSource) -> None:
        """Register a data source."""
        with self._lock:
            self._sources[source.source_id] = source
            logger.info(f"Registered data source: {source.source_id} ({source.name})")

    def unregister_source(self, source_id: str) -> bool:
        """Unregister a data source."""
        with self._lock:
            if source_id in self._sources:
                del self._sources[source_id]
                logger.info(f"Unregistered data source: {source_id}")
                return True
            return False

    def get_sources(self) -> List[DataSource]:
        """Get all registered sources."""
        with self._lock:
            return list(self._sources.values())

    def fuse(
        self,
        measurements: Dict[str, Dict[str, float]],
        timestamps: Optional[Dict[str, float]] = None
    ) -> StateVector:
        """
        Fuse measurements from multiple sources.

        Args:
            measurements: Dict of source_id -> {variable: value}
            timestamps: Optional dict of source_id -> timestamp

        Returns:
            Fused state vector
        """
        with self._lock:
            if timestamps is None:
                timestamps = {s: time.time() for s in measurements}

            fused_values: Dict[str, float] = {}
            fused_uncertainties: Dict[str, float] = {}

            # Group measurements by variable
            var_measurements: Dict[str, List[Tuple[str, float, float]]] = {}

            for source_id, values in measurements.items():
                source = self._sources.get(source_id)
                if source is None or not source.enabled:
                    continue

                # Calculate measurement weight based on source properties
                weight = self._calculate_weight(source)

                for var, value in values.items():
                    if var not in var_measurements:
                        var_measurements[var] = []
                    var_measurements[var].append((source_id, value, weight))

            # Fuse each variable
            for var, meas_list in var_measurements.items():
                if not meas_list:
                    continue

                # Weighted average
                total_weight = sum(w for _, _, w in meas_list)
                if total_weight > 0:
                    fused_value = sum(v * w for _, v, w in meas_list) / total_weight

                    # Calculate uncertainty (weighted variance)
                    if len(meas_list) > 1:
                        variance = sum(
                            w * (v - fused_value) ** 2
                            for _, v, w in meas_list
                        ) / total_weight
                        uncertainty = np.sqrt(variance)
                    else:
                        # Single source - use source accuracy
                        source = self._sources.get(meas_list[0][0])
                        uncertainty = abs(fused_value) * (1 - source.accuracy) if source else 0.05

                    # Kalman-like update if we have prior state
                    if var in self._state:
                        fused_value, uncertainty = self._kalman_update(
                            var, fused_value, uncertainty
                        )

                    fused_values[var] = fused_value
                    fused_uncertainties[var] = uncertainty
                    self._state[var] = fused_value
                    self._variance[var] = uncertainty ** 2

                    # Record history
                    if var not in self._history:
                        self._history[var] = deque(maxlen=1000)
                    self._history[var].append((time.time(), fused_value))

            # Calculate overall confidence
            confidence = self._calculate_confidence(fused_values, fused_uncertainties)
            quality = self._assess_quality(fused_uncertainties, timestamps)

            return StateVector(
                timestamp=time.time(),
                values=fused_values,
                uncertainties=fused_uncertainties,
                confidence=confidence,
                quality=quality
            )

    def _calculate_weight(self, source: DataSource) -> float:
        """Calculate fusion weight for a source."""
        # Weight based on reliability, accuracy, and priority
        base_weight = source.reliability * source.accuracy
        priority_factor = 1.0 / source.priority  # Lower priority number = higher weight
        return base_weight * priority_factor

    def _kalman_update(
        self,
        var: str,
        measurement: float,
        measurement_var: float
    ) -> Tuple[float, float]:
        """
        Kalman filter update step.

        Args:
            var: Variable name
            measurement: New measurement
            measurement_var: Measurement variance

        Returns:
            Updated (value, uncertainty)
        """
        prior = self._state.get(var, measurement)
        prior_var = self._variance.get(var, measurement_var) + self._process_noise

        # Kalman gain
        if prior_var + measurement_var > 0:
            K = prior_var / (prior_var + measurement_var)
        else:
            K = 0.5

        # Update
        posterior = prior + K * (measurement - prior)
        posterior_var = (1 - K) * prior_var

        return posterior, np.sqrt(posterior_var)

    def _calculate_confidence(
        self,
        values: Dict[str, float],
        uncertainties: Dict[str, float]
    ) -> float:
        """Calculate overall confidence score."""
        if not values or not uncertainties:
            return 0.0

        # Confidence based on relative uncertainty
        confidences = []
        for var in values:
            if var in uncertainties and values[var] != 0:
                rel_uncertainty = abs(uncertainties[var] / values[var])
                conf = max(0, 1 - rel_uncertainty)
                confidences.append(conf)
            else:
                confidences.append(0.9)  # Default confidence

        return np.mean(confidences) if confidences else 0.5

    def _assess_quality(
        self,
        uncertainties: Dict[str, float],
        timestamps: Dict[str, float]
    ) -> SyncQuality:
        """Assess synchronization quality."""
        if not uncertainties:
            return SyncQuality.POOR

        avg_uncertainty = np.mean(list(uncertainties.values()))

        # Calculate data age
        current_time = time.time()
        max_age = max(current_time - ts for ts in timestamps.values()) if timestamps else 0

        if avg_uncertainty < 0.01 and max_age < 0.01:
            return SyncQuality.EXCELLENT
        elif avg_uncertainty < 0.05 and max_age < 0.05:
            return SyncQuality.GOOD
        elif avg_uncertainty < 0.10 and max_age < 0.2:
            return SyncQuality.ACCEPTABLE
        elif avg_uncertainty < 0.20 and max_age < 0.5:
            return SyncQuality.POOR
        else:
            return SyncQuality.UNACCEPTABLE

    def get_state(self) -> Dict[str, float]:
        """Get current fused state."""
        with self._lock:
            return self._state.copy()

    def get_uncertainty(self) -> Dict[str, float]:
        """Get current uncertainties."""
        with self._lock:
            return {k: np.sqrt(v) for k, v in self._variance.items()}

    def get_history(
        self,
        variable: str,
        duration_seconds: float = 60.0
    ) -> List[Tuple[float, float]]:
        """Get historical values for a variable."""
        with self._lock:
            if variable not in self._history:
                return []

            cutoff = time.time() - duration_seconds
            return [(ts, val) for ts, val in self._history[variable] if ts >= cutoff]

    def reset(self) -> None:
        """Reset fusion state."""
        with self._lock:
            self._state.clear()
            self._variance.clear()
            self._history.clear()
            logger.info("DataFusionEngine reset")


# =============================================================================
# Latency Compensator
# =============================================================================

class LatencyCompensator:
    """
    Compensates for network and sensor latencies.

    Uses prediction and interpolation to estimate current state
    from delayed measurements.
    """

    def __init__(
        self,
        max_compensation_ms: float = 500.0,
        prediction_horizon_ms: float = 100.0
    ):
        """
        Initialize latency compensator.

        Args:
            max_compensation_ms: Maximum latency to compensate for
            prediction_horizon_ms: Prediction horizon for compensation
        """
        self._max_compensation = max_compensation_ms
        self._prediction_horizon = prediction_horizon_ms
        self._latency_estimates: Dict[str, deque] = {}
        self._state_history: Dict[str, deque] = {}
        self._predictor: Optional[Callable] = None
        self._lock = threading.RLock()  # Reentrant lock for nested calls

        logger.info(
            f"LatencyCompensator initialized (max={max_compensation_ms}ms, "
            f"horizon={prediction_horizon_ms}ms)"
        )

    def record_latency(self, source_id: str, latency_ms: float) -> None:
        """Record a latency measurement for a source."""
        with self._lock:
            if source_id not in self._latency_estimates:
                self._latency_estimates[source_id] = deque(maxlen=100)
            self._latency_estimates[source_id].append(latency_ms)

    def get_latency_estimate(self, source_id: str) -> Optional[LatencyEstimate]:
        """Get latency estimate for a source."""
        with self._lock:
            if source_id not in self._latency_estimates:
                return None

            latencies = list(self._latency_estimates[source_id])
            if not latencies:
                return None

            return LatencyEstimate(
                source_id=source_id,
                current_latency_ms=latencies[-1],
                avg_latency_ms=np.mean(latencies),
                min_latency_ms=min(latencies),
                max_latency_ms=max(latencies),
                jitter_ms=np.std(latencies),
                sample_count=len(latencies),
                last_update=time.time()
            )

    def set_predictor(self, predictor: Callable[[Dict[str, float], float], Dict[str, float]]) -> None:
        """
        Set state predictor function.

        Args:
            predictor: Function (state, dt) -> predicted_state
        """
        self._predictor = predictor

    def record_state(self, state: Dict[str, float], timestamp: float) -> None:
        """Record state for interpolation."""
        with self._lock:
            for var, value in state.items():
                if var not in self._state_history:
                    self._state_history[var] = deque(maxlen=1000)
                self._state_history[var].append((timestamp, value))

    def compensate(
        self,
        measurement: Dict[str, float],
        measurement_time: float,
        source_id: str
    ) -> Dict[str, float]:
        """
        Compensate for latency in a measurement.

        Args:
            measurement: Delayed measurement
            measurement_time: Time when measurement was taken
            source_id: Source identifier

        Returns:
            Compensated measurement
        """
        current_time = time.time()
        latency_s = current_time - measurement_time
        latency_ms = latency_s * 1000

        # Record latency
        self.record_latency(source_id, latency_ms)

        # Check if compensation needed
        if latency_ms < 1:
            return measurement

        # Cap at max compensation
        if latency_ms > self._max_compensation:
            logger.warning(
                f"Latency {latency_ms:.1f}ms exceeds max {self._max_compensation}ms "
                f"for source {source_id}"
            )
            latency_s = self._max_compensation / 1000

        compensated = measurement.copy()

        # Use predictor if available
        if self._predictor is not None:
            try:
                compensated = self._predictor(measurement, latency_s)
            except Exception as e:
                logger.warning(f"Prediction failed: {e}, using interpolation")
                compensated = self._interpolate(measurement, latency_s)
        else:
            compensated = self._interpolate(measurement, latency_s)

        return compensated

    def _interpolate(
        self,
        measurement: Dict[str, float],
        dt: float
    ) -> Dict[str, float]:
        """Interpolate state forward in time using linear extrapolation."""
        with self._lock:
            compensated = {}

            for var, value in measurement.items():
                if var in self._state_history and len(self._state_history[var]) >= 2:
                    # Linear extrapolation based on recent history
                    history = list(self._state_history[var])[-10:]
                    if len(history) >= 2:
                        times = [h[0] for h in history]
                        values = [h[1] for h in history]

                        # Fit linear trend
                        coeffs = np.polyfit(times, values, 1)
                        slope = coeffs[0]

                        # Extrapolate
                        compensated[var] = value + slope * dt
                    else:
                        compensated[var] = value
                else:
                    compensated[var] = value

            return compensated

    def get_all_latency_estimates(self) -> Dict[str, LatencyEstimate]:
        """Get latency estimates for all sources."""
        with self._lock:
            estimates = {}
            for source_id in self._latency_estimates:
                est = self.get_latency_estimate(source_id)
                if est is not None:
                    estimates[source_id] = est
            return estimates

    def reset(self) -> None:
        """Reset compensator state."""
        with self._lock:
            self._latency_estimates.clear()
            self._state_history.clear()
            logger.info("LatencyCompensator reset")


# =============================================================================
# Discrepancy Detector
# =============================================================================

class DiscrepancyDetector:
    """
    Detects and tracks discrepancies between physical and digital states.
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        default_threshold: float = 0.1
    ):
        """
        Initialize discrepancy detector.

        Args:
            thresholds: Variable-specific thresholds (as fraction)
            default_threshold: Default threshold for unlisted variables
        """
        self._thresholds = thresholds or {}
        self._default_threshold = default_threshold
        self._events: deque = deque(maxlen=1000)
        self._active_discrepancies: Dict[str, DiscrepancyEvent] = {}
        self._callbacks: List[Callable[[DiscrepancyEvent], None]] = []
        self._lock = threading.Lock()

        logger.info(
            f"DiscrepancyDetector initialized "
            f"(default_threshold={default_threshold*100:.1f}%)"
        )

    def set_threshold(self, variable: str, threshold: float) -> None:
        """Set threshold for a specific variable."""
        self._thresholds[variable] = threshold

    def add_callback(self, callback: Callable[[DiscrepancyEvent], None]) -> None:
        """Add callback for discrepancy events."""
        self._callbacks.append(callback)

    def check(
        self,
        physical_state: Dict[str, float],
        digital_state: Dict[str, float]
    ) -> List[DiscrepancyEvent]:
        """
        Check for discrepancies between physical and digital states.

        Args:
            physical_state: State from physical system
            digital_state: State from digital twin

        Returns:
            List of discrepancy events
        """
        events = []
        current_time = time.time()

        with self._lock:
            # Check each variable
            for var in set(physical_state.keys()) | set(digital_state.keys()):
                physical_val = physical_state.get(var)
                digital_val = digital_state.get(var)

                if physical_val is None or digital_val is None:
                    continue

                # Calculate discrepancy
                discrepancy = abs(physical_val - digital_val)

                # Calculate percentage discrepancy
                ref_value = abs(physical_val) if abs(physical_val) > 1e-6 else 1.0
                discrepancy_pct = discrepancy / ref_value

                # Get threshold
                threshold = self._thresholds.get(var, self._default_threshold)

                if discrepancy_pct > threshold:
                    severity = self._classify_severity(discrepancy_pct, threshold)

                    event = DiscrepancyEvent(
                        event_id=str(uuid.uuid4())[:8],
                        timestamp=current_time,
                        variable=var,
                        physical_value=physical_val,
                        digital_value=digital_val,
                        discrepancy=discrepancy,
                        discrepancy_pct=discrepancy_pct * 100,
                        severity=severity
                    )

                    events.append(event)
                    self._events.append(event)
                    self._active_discrepancies[var] = event

                    # Trigger callbacks
                    for callback in self._callbacks:
                        try:
                            callback(event)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")

                elif var in self._active_discrepancies:
                    # Discrepancy resolved
                    old_event = self._active_discrepancies[var]
                    old_event.resolved = True
                    old_event.resolution_time = current_time
                    old_event.resolution_method = "natural_convergence"
                    del self._active_discrepancies[var]

        return events

    def _classify_severity(self, discrepancy_pct: float, threshold: float) -> str:
        """Classify severity of discrepancy."""
        ratio = discrepancy_pct / threshold

        if ratio < 2:
            return 'low'
        elif ratio < 5:
            return 'medium'
        elif ratio < 10:
            return 'high'
        else:
            return 'critical'

    def resolve(
        self,
        variable: str,
        resolution_method: str = "manual"
    ) -> bool:
        """Manually resolve a discrepancy."""
        with self._lock:
            if variable in self._active_discrepancies:
                event = self._active_discrepancies[variable]
                event.resolved = True
                event.resolution_time = time.time()
                event.resolution_method = resolution_method
                del self._active_discrepancies[variable]
                logger.info(f"Discrepancy resolved for {variable}: {resolution_method}")
                return True
            return False

    def get_active_discrepancies(self) -> List[DiscrepancyEvent]:
        """Get all active discrepancies."""
        with self._lock:
            return list(self._active_discrepancies.values())

    def get_history(self, limit: int = 100) -> List[DiscrepancyEvent]:
        """Get discrepancy history."""
        with self._lock:
            return list(self._events)[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get discrepancy statistics."""
        with self._lock:
            events = list(self._events)

            if not events:
                return {
                    'total_events': 0,
                    'active_count': 0,
                    'resolved_count': 0,
                    'by_severity': {},
                    'by_variable': {}
                }

            by_severity = {}
            by_variable = {}

            for event in events:
                by_severity[event.severity] = by_severity.get(event.severity, 0) + 1
                by_variable[event.variable] = by_variable.get(event.variable, 0) + 1

            return {
                'total_events': len(events),
                'active_count': len(self._active_discrepancies),
                'resolved_count': sum(1 for e in events if e.resolved),
                'avg_discrepancy_pct': np.mean([e.discrepancy_pct for e in events]),
                'max_discrepancy_pct': max(e.discrepancy_pct for e in events),
                'by_severity': by_severity,
                'by_variable': by_variable
            }

    def reset(self) -> None:
        """Reset detector state."""
        with self._lock:
            self._events.clear()
            self._active_discrepancies.clear()
            logger.info("DiscrepancyDetector reset")


# =============================================================================
# Confidence Tracker
# =============================================================================

class ConfidenceTracker:
    """
    Tracks confidence in state estimates across multiple sources.
    """

    def __init__(self, decay_rate: float = 0.99):
        """
        Initialize confidence tracker.

        Args:
            decay_rate: Rate at which confidence decays without updates
        """
        self._decay_rate = decay_rate
        self._scores: Dict[str, ConfidenceScore] = {}
        self._source_performance: Dict[str, deque] = {}
        self._lock = threading.Lock()

        logger.info(f"ConfidenceTracker initialized (decay_rate={decay_rate})")

    def update(
        self,
        variable: str,
        sources: Dict[str, float],
        weights: Dict[str, float],
        uncertainty: float
    ) -> ConfidenceScore:
        """
        Update confidence score for a variable.

        Args:
            variable: Variable name
            sources: Source values {source_id: value}
            weights: Source weights {source_id: weight}
            uncertainty: Estimated uncertainty

        Returns:
            Updated confidence score
        """
        with self._lock:
            # Calculate confidence based on source agreement and uncertainty
            values = list(sources.values())

            if len(values) > 1:
                # Agreement factor: how well sources agree
                std = np.std(values)
                mean = np.mean(values)
                cv = std / abs(mean) if abs(mean) > 1e-6 else std
                agreement_factor = max(0, 1 - cv)
            else:
                agreement_factor = 0.8  # Single source penalty

            # Uncertainty factor
            ref_value = abs(np.mean(values)) if values else 1.0
            rel_uncertainty = uncertainty / ref_value if ref_value > 0 else uncertainty
            uncertainty_factor = max(0, 1 - rel_uncertainty * 2)

            # Combine factors
            confidence = 0.5 * agreement_factor + 0.5 * uncertainty_factor

            # Apply decay if updating existing score
            if variable in self._scores:
                old_score = self._scores[variable]
                time_since_update = time.time() - old_score.last_update
                decay = self._decay_rate ** time_since_update
                confidence = 0.7 * confidence + 0.3 * old_score.confidence * decay

            score = ConfidenceScore(
                variable=variable,
                confidence=min(1.0, max(0.0, confidence)),
                contributing_sources=list(sources.keys()),
                source_weights=weights,
                uncertainty=uncertainty,
                last_update=time.time()
            )

            self._scores[variable] = score
            return score

    def record_source_accuracy(
        self,
        source_id: str,
        actual: float,
        predicted: float
    ) -> None:
        """Record source prediction accuracy for tracking."""
        with self._lock:
            if source_id not in self._source_performance:
                self._source_performance[source_id] = deque(maxlen=100)

            error = abs(actual - predicted)
            rel_error = error / abs(actual) if abs(actual) > 1e-6 else error
            accuracy = max(0, 1 - rel_error)

            self._source_performance[source_id].append(accuracy)

    def get_source_reliability(self, source_id: str) -> float:
        """Get estimated reliability for a source."""
        with self._lock:
            if source_id not in self._source_performance:
                return 0.9  # Default

            accuracies = list(self._source_performance[source_id])
            if not accuracies:
                return 0.9

            return np.mean(accuracies)

    def get_confidence(self, variable: str) -> Optional[ConfidenceScore]:
        """Get confidence score for a variable."""
        with self._lock:
            return self._scores.get(variable)

    def get_all_confidences(self) -> Dict[str, ConfidenceScore]:
        """Get all confidence scores."""
        with self._lock:
            return self._scores.copy()

    def get_low_confidence_variables(self, threshold: float = 0.5) -> List[str]:
        """Get variables with low confidence."""
        with self._lock:
            return [
                var for var, score in self._scores.items()
                if score.confidence < threshold
            ]

    def reset(self) -> None:
        """Reset tracker state."""
        with self._lock:
            self._scores.clear()
            self._source_performance.clear()
            logger.info("ConfidenceTracker reset")


# =============================================================================
# State Synchronizer
# =============================================================================

class StateSynchronizer:
    """
    Core synchronization engine for digital twin state management.
    """

    def __init__(
        self,
        model: Any,
        sync_interval_ms: float = 100.0,
        mode: SyncMode = SyncMode.REALTIME
    ):
        """
        Initialize state synchronizer.

        Args:
            model: Digital twin physics model
            sync_interval_ms: Synchronization interval
            mode: Synchronization mode
        """
        self._model = model
        self._sync_interval = sync_interval_ms / 1000.0
        self._mode = mode

        self._fusion = DataFusionEngine()
        self._latency_comp = LatencyCompensator()
        self._discrepancy = DiscrepancyDetector()
        self._confidence = ConfidenceTracker()

        self._status = SyncStatus.IDLE
        self._metrics = SyncMetrics()
        self._running = False
        self._sync_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Callbacks
        self._on_sync: List[Callable[[StateVector], None]] = []
        self._on_discrepancy: List[Callable[[DiscrepancyEvent], None]] = []

        logger.info(
            f"StateSynchronizer initialized "
            f"(interval={sync_interval_ms}ms, mode={mode.name})"
        )

    @property
    def status(self) -> SyncStatus:
        """Get current sync status."""
        return self._status

    @property
    def mode(self) -> SyncMode:
        """Get sync mode."""
        return self._mode

    @mode.setter
    def mode(self, value: SyncMode) -> None:
        """Set sync mode."""
        self._mode = value
        logger.info(f"Sync mode changed to {value.name}")

    def register_source(self, source: DataSource) -> None:
        """Register a data source."""
        self._fusion.register_source(source)

    def add_sync_callback(self, callback: Callable[[StateVector], None]) -> None:
        """Add callback for sync events."""
        self._on_sync.append(callback)

    def add_discrepancy_callback(self, callback: Callable[[DiscrepancyEvent], None]) -> None:
        """Add callback for discrepancy events."""
        self._on_discrepancy.append(callback)
        self._discrepancy.add_callback(callback)

    def start(self) -> None:
        """Start synchronization."""
        if self._running:
            return

        self._running = True
        self._status = SyncStatus.SYNCING
        self._metrics.uptime_seconds = 0
        self._start_time = time.time()

        if self._mode == SyncMode.REALTIME:
            self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
            self._sync_thread.start()

        logger.info("StateSynchronizer started")

    def stop(self) -> None:
        """Stop synchronization."""
        self._running = False
        self._status = SyncStatus.IDLE

        if self._sync_thread is not None:
            self._sync_thread.join(timeout=2.0)
            self._sync_thread = None

        logger.info("StateSynchronizer stopped")

    def _sync_loop(self) -> None:
        """Main synchronization loop."""
        while self._running:
            try:
                self.sync()
                time.sleep(self._sync_interval)
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
                self._status = SyncStatus.DEGRADED
                time.sleep(1.0)

    def sync(self, measurements: Optional[Dict[str, Dict[str, float]]] = None) -> StateVector:
        """
        Perform synchronization.

        Args:
            measurements: Optional external measurements to incorporate

        Returns:
            Synchronized state vector
        """
        start_time = time.time()

        with self._lock:
            # Get digital model state
            digital_state = self._model.get_state() if hasattr(self._model, 'get_state') else {}

            # Build measurements dict
            all_measurements = {}
            timestamps = {}

            # Add model state as a source
            all_measurements['model'] = self._extract_numeric_state(digital_state)
            timestamps['model'] = time.time()

            # Add external measurements
            if measurements:
                for source_id, values in measurements.items():
                    # Compensate for latency
                    ts = timestamps.get(source_id, time.time() - 0.05)  # Assume 50ms if not specified
                    compensated = self._latency_comp.compensate(values, ts, source_id)
                    all_measurements[source_id] = compensated
                    timestamps[source_id] = ts

            # Fuse data
            fused_state = self._fusion.fuse(all_measurements, timestamps)

            # Check for discrepancies if we have external data
            if measurements:
                physical_state = {}
                for source_id, values in measurements.items():
                    physical_state.update(values)

                model_state = self._extract_numeric_state(digital_state)
                discrepancies = self._discrepancy.check(physical_state, model_state)

                for event in discrepancies:
                    for callback in self._on_discrepancy:
                        try:
                            callback(event)
                        except Exception as e:
                            logger.error(f"Discrepancy callback error: {e}")

            # Update confidence for each variable
            for var in fused_state.values:
                sources = {s: m.get(var, 0) for s, m in all_measurements.items() if var in m}
                weights = {s: 1.0 / len(sources) for s in sources}
                uncertainty = fused_state.uncertainties.get(var, 0.05)
                self._confidence.update(var, sources, weights, uncertainty)

            # Record state for latency compensation
            self._latency_comp.record_state(fused_state.values, fused_state.timestamp)

            # Update metrics
            sync_time_ms = (time.time() - start_time) * 1000
            self._update_metrics(sync_time_ms, fused_state.quality)

            # Update status
            self._status = SyncStatus.SYNCHRONIZED

            # Trigger callbacks
            for callback in self._on_sync:
                try:
                    callback(fused_state)
                except Exception as e:
                    logger.error(f"Sync callback error: {e}")

            return fused_state

    def _extract_numeric_state(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Extract numeric values from state dict."""
        numeric_state = {}
        for key, value in state.items():
            if isinstance(value, (int, float)):
                numeric_state[key] = float(value)
            elif isinstance(value, np.ndarray):
                for i, v in enumerate(value.flatten()):
                    numeric_state[f"{key}_{i}"] = float(v)
            elif isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    if isinstance(v, (int, float)):
                        numeric_state[f"{key}_{i}"] = float(v)
        return numeric_state

    def _update_metrics(self, latency_ms: float, quality: SyncQuality) -> None:
        """Update sync metrics."""
        self._metrics.sync_count += 1
        self._metrics.total_latency_ms += latency_ms
        self._metrics.max_latency_ms = max(self._metrics.max_latency_ms, latency_ms)
        self._metrics.last_sync_time = time.time()
        self._metrics.quality_history.append(quality)

        if hasattr(self, '_start_time'):
            self._metrics.uptime_seconds = time.time() - self._start_time

    def get_metrics(self) -> SyncMetrics:
        """Get synchronization metrics."""
        return self._metrics

    def get_confidence_report(self) -> Dict[str, Any]:
        """Get confidence report for all variables."""
        confidences = self._confidence.get_all_confidences()
        return {
            'variables': {
                var: {
                    'confidence': score.confidence,
                    'uncertainty': score.uncertainty,
                    'sources': score.contributing_sources
                }
                for var, score in confidences.items()
            },
            'low_confidence': self._confidence.get_low_confidence_variables(),
            'avg_confidence': np.mean([s.confidence for s in confidences.values()]) if confidences else 0
        }

    def get_discrepancy_report(self) -> Dict[str, Any]:
        """Get discrepancy report."""
        return self._discrepancy.get_statistics()

    def get_latency_report(self) -> Dict[str, Any]:
        """Get latency report for all sources."""
        estimates = self._latency_comp.get_all_latency_estimates()
        return {
            source_id: {
                'current_ms': est.current_latency_ms,
                'avg_ms': est.avg_latency_ms,
                'jitter_ms': est.jitter_ms
            }
            for source_id, est in estimates.items()
        }

    def reset(self) -> None:
        """Reset synchronizer state."""
        with self._lock:
            self._fusion.reset()
            self._latency_comp.reset()
            self._discrepancy.reset()
            self._confidence.reset()
            self._metrics = SyncMetrics()
            self._status = SyncStatus.IDLE
            logger.info("StateSynchronizer reset")


# =============================================================================
# Twin Sync Manager
# =============================================================================

class TwinSyncManager:
    """
    High-level manager for digital twin synchronization.

    Provides a unified interface for all synchronization operations
    including data fusion, latency compensation, and discrepancy handling.
    """

    def __init__(
        self,
        model: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize twin sync manager.

        Args:
            model: Digital twin physics model
            config: Optional configuration dict
        """
        config = config or {}

        self._model = model
        self._synchronizer = StateSynchronizer(
            model,
            sync_interval_ms=config.get('sync_interval_ms', 100.0),
            mode=SyncMode[config.get('mode', 'REALTIME')]
        )

        self._sources: Dict[str, DataSource] = {}
        self._running = False
        self._last_physical_state: Dict[str, float] = {}
        self._lock = threading.Lock()

        # Initialize default model source
        self._synchronizer.register_source(DataSource(
            source_id='model',
            source_type=SourceType.MODEL_OUTPUT,
            name='Digital Twin Model',
            variables=[],
            sample_rate_hz=10.0,
            latency_ms=0,
            reliability=1.0,
            accuracy=0.95,
            priority=2
        ))

        logger.info("TwinSyncManager initialized")

    def add_physical_source(
        self,
        source_id: str,
        name: str,
        variables: List[str],
        source_type: SourceType = SourceType.PHYSICAL_SENSOR,
        sample_rate_hz: float = 1.0,
        latency_ms: float = 50.0,
        reliability: float = 0.99,
        accuracy: float = 0.98
    ) -> None:
        """Add a physical data source."""
        source = DataSource(
            source_id=source_id,
            source_type=source_type,
            name=name,
            variables=variables,
            sample_rate_hz=sample_rate_hz,
            latency_ms=latency_ms,
            reliability=reliability,
            accuracy=accuracy,
            priority=1  # Physical sources have highest priority
        )

        with self._lock:
            self._sources[source_id] = source
            self._synchronizer.register_source(source)

        logger.info(f"Added physical source: {source_id} ({name})")

    def remove_source(self, source_id: str) -> bool:
        """Remove a data source."""
        with self._lock:
            if source_id in self._sources:
                del self._sources[source_id]
                return True
            return False

    def start(self) -> None:
        """Start synchronization."""
        self._running = True
        self._synchronizer.start()
        logger.info("TwinSyncManager started")

    def stop(self) -> None:
        """Stop synchronization."""
        self._running = False
        self._synchronizer.stop()
        logger.info("TwinSyncManager stopped")

    def update_physical_state(
        self,
        source_id: str,
        state: Dict[str, float],
        timestamp: Optional[float] = None
    ) -> StateVector:
        """
        Update with physical system state.

        Args:
            source_id: Source identifier
            state: Physical state measurements
            timestamp: Optional measurement timestamp

        Returns:
            Synchronized state vector
        """
        with self._lock:
            self._last_physical_state.update(state)

            measurements = {source_id: state}
            return self._synchronizer.sync(measurements)

    def sync_now(self) -> StateVector:
        """Force immediate synchronization."""
        return self._synchronizer.sync()

    def get_synchronized_state(self) -> Dict[str, float]:
        """Get current synchronized state."""
        return self._synchronizer._fusion.get_state()

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status."""
        metrics = self._synchronizer.get_metrics()

        return {
            'running': self._running,
            'status': self._synchronizer.status.name,
            'mode': self._synchronizer.mode.name,
            'sources': {
                sid: {
                    'name': s.name,
                    'type': s.source_type.name,
                    'enabled': s.enabled
                }
                for sid, s in self._sources.items()
            },
            'metrics': {
                'sync_count': metrics.sync_count,
                'avg_latency_ms': metrics.avg_latency_ms,
                'max_latency_ms': metrics.max_latency_ms,
                'current_quality': metrics.current_quality.name,
                'uptime_seconds': metrics.uptime_seconds
            },
            'confidence': self._synchronizer.get_confidence_report(),
            'discrepancies': self._synchronizer.get_discrepancy_report(),
            'latencies': self._synchronizer.get_latency_report()
        }

    def get_discrepancies(self) -> List[DiscrepancyEvent]:
        """Get active discrepancies."""
        return self._synchronizer._discrepancy.get_active_discrepancies()

    def resolve_discrepancy(
        self,
        variable: str,
        use_physical: bool = True
    ) -> bool:
        """
        Resolve a discrepancy by choosing physical or digital value.

        Args:
            variable: Variable name
            use_physical: True to use physical value, False for digital

        Returns:
            True if resolved
        """
        method = "use_physical" if use_physical else "use_digital"
        return self._synchronizer._discrepancy.resolve(variable, method)

    def set_discrepancy_threshold(self, variable: str, threshold: float) -> None:
        """Set discrepancy threshold for a variable."""
        self._synchronizer._discrepancy.set_threshold(variable, threshold)

    def add_sync_callback(self, callback: Callable[[StateVector], None]) -> None:
        """Add callback for sync events."""
        self._synchronizer.add_sync_callback(callback)

    def add_discrepancy_callback(self, callback: Callable[[DiscrepancyEvent], None]) -> None:
        """Add callback for discrepancy events."""
        self._synchronizer.add_discrepancy_callback(callback)

    def reset(self) -> None:
        """Reset manager state."""
        with self._lock:
            self._synchronizer.reset()
            self._last_physical_state.clear()
            logger.info("TwinSyncManager reset")

    def get_fusion_history(
        self,
        variable: str,
        duration_seconds: float = 60.0
    ) -> List[Tuple[float, float]]:
        """Get fusion history for a variable."""
        return self._synchronizer._fusion.get_history(variable, duration_seconds)
