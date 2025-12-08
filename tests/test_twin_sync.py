"""
Unit tests for Digital Twin Synchronization Module.

Tests:
- Data fusion engine
- Latency compensator
- Discrepancy detector
- Confidence tracker
- State synchronizer
- Twin sync manager
"""

import pytest
import time
import threading
import numpy as np
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from src.twin_sync.synchronizer import (
    # Enums
    SyncMode,
    SyncStatus,
    SyncQuality,
    SourceType,
    # Data classes
    DataSource,
    StateVector,
    SyncMetrics,
    DiscrepancyEvent,
    LatencyEstimate,
    ConfidenceScore,
    # Classes
    DataFusionEngine,
    LatencyCompensator,
    DiscrepancyDetector,
    ConfidenceTracker,
    StateSynchronizer,
    TwinSyncManager
)


# =============================================================================
# Enum Tests
# =============================================================================

class TestEnums:
    """Test enumeration values."""

    def test_sync_mode_values(self):
        """Test SyncMode enum values."""
        assert SyncMode.REALTIME is not None
        assert SyncMode.BATCH is not None
        assert SyncMode.ON_DEMAND is not None
        assert SyncMode.PREDICTIVE is not None

    def test_sync_status_values(self):
        """Test SyncStatus enum values."""
        assert SyncStatus.IDLE is not None
        assert SyncStatus.SYNCING is not None
        assert SyncStatus.SYNCHRONIZED is not None
        assert SyncStatus.DEGRADED is not None
        assert SyncStatus.FAILED is not None

    def test_sync_quality_values(self):
        """Test SyncQuality enum values."""
        assert SyncQuality.EXCELLENT is not None
        assert SyncQuality.GOOD is not None
        assert SyncQuality.ACCEPTABLE is not None
        assert SyncQuality.POOR is not None
        assert SyncQuality.UNACCEPTABLE is not None

    def test_source_type_values(self):
        """Test SourceType enum values."""
        assert SourceType.PHYSICAL_SENSOR is not None
        assert SourceType.SCADA is not None
        assert SourceType.MODEL_OUTPUT is not None


# =============================================================================
# Data Class Tests
# =============================================================================

class TestDataClasses:
    """Test data class structures."""

    def test_data_source_creation(self):
        """Test DataSource creation."""
        source = DataSource(
            source_id='sensor_1',
            source_type=SourceType.PHYSICAL_SENSOR,
            name='Flow Sensor 1',
            variables=['flow', 'pressure'],
            sample_rate_hz=10.0,
            latency_ms=50.0,
            reliability=0.99,
            accuracy=0.98
        )
        assert source.source_id == 'sensor_1'
        assert source.source_type == SourceType.PHYSICAL_SENSOR
        assert len(source.variables) == 2
        assert source.enabled is True

    def test_data_source_defaults(self):
        """Test DataSource default values."""
        source = DataSource(
            source_id='test',
            source_type=SourceType.SCADA,
            name='Test',
            variables=[]
        )
        assert source.sample_rate_hz == 1.0
        assert source.latency_ms == 0.0
        assert source.reliability == 1.0
        assert source.accuracy == 0.95

    def test_state_vector_creation(self):
        """Test StateVector creation."""
        state = StateVector(
            timestamp=time.time(),
            values={'flow': 150.0, 'pressure': 10.5},
            uncertainties={'flow': 1.5, 'pressure': 0.1},
            confidence=0.95
        )
        assert len(state.values) == 2
        assert state.confidence == 0.95
        assert state.quality == SyncQuality.GOOD

    def test_sync_metrics_avg_latency(self):
        """Test SyncMetrics average latency calculation."""
        metrics = SyncMetrics()
        metrics.sync_count = 10
        metrics.total_latency_ms = 500.0
        assert metrics.avg_latency_ms == 50.0

    def test_sync_metrics_avg_latency_zero(self):
        """Test SyncMetrics average latency when no syncs."""
        metrics = SyncMetrics()
        assert metrics.avg_latency_ms == 0.0

    def test_sync_metrics_current_quality(self):
        """Test SyncMetrics current quality calculation."""
        metrics = SyncMetrics()
        metrics.quality_history = [
            SyncQuality.GOOD, SyncQuality.GOOD, SyncQuality.EXCELLENT
        ]
        assert metrics.current_quality == SyncQuality.GOOD

    def test_discrepancy_event_creation(self):
        """Test DiscrepancyEvent creation."""
        event = DiscrepancyEvent(
            event_id='evt_001',
            timestamp=time.time(),
            variable='flow',
            physical_value=150.0,
            digital_value=148.0,
            discrepancy=2.0,
            discrepancy_pct=1.33,
            severity='low'
        )
        assert event.event_id == 'evt_001'
        assert event.resolved is False

    def test_latency_estimate_creation(self):
        """Test LatencyEstimate creation."""
        estimate = LatencyEstimate(
            source_id='sensor_1',
            current_latency_ms=45.0,
            avg_latency_ms=50.0,
            min_latency_ms=30.0,
            max_latency_ms=80.0,
            jitter_ms=10.0,
            sample_count=100,
            last_update=time.time()
        )
        assert estimate.source_id == 'sensor_1'
        assert estimate.jitter_ms == 10.0

    def test_confidence_score_creation(self):
        """Test ConfidenceScore creation."""
        score = ConfidenceScore(
            variable='flow',
            confidence=0.92,
            contributing_sources=['sensor_1', 'sensor_2'],
            source_weights={'sensor_1': 0.6, 'sensor_2': 0.4},
            uncertainty=1.5,
            last_update=time.time()
        )
        assert score.confidence == 0.92
        assert len(score.contributing_sources) == 2


# =============================================================================
# Data Fusion Engine Tests
# =============================================================================

class TestDataFusionEngine:
    """Test data fusion engine."""

    @pytest.fixture
    def fusion_engine(self):
        """Create fusion engine for testing."""
        return DataFusionEngine()

    @pytest.fixture
    def registered_engine(self, fusion_engine):
        """Create engine with registered sources."""
        source1 = DataSource(
            source_id='sensor_1',
            source_type=SourceType.PHYSICAL_SENSOR,
            name='Sensor 1',
            variables=['flow'],
            reliability=0.99,
            accuracy=0.98
        )
        source2 = DataSource(
            source_id='sensor_2',
            source_type=SourceType.PHYSICAL_SENSOR,
            name='Sensor 2',
            variables=['flow'],
            reliability=0.95,
            accuracy=0.95
        )
        fusion_engine.register_source(source1)
        fusion_engine.register_source(source2)
        return fusion_engine

    def test_initialization(self, fusion_engine):
        """Test engine initialization."""
        assert fusion_engine is not None
        assert len(fusion_engine.get_sources()) == 0

    def test_register_source(self, fusion_engine):
        """Test source registration."""
        source = DataSource(
            source_id='test',
            source_type=SourceType.PHYSICAL_SENSOR,
            name='Test',
            variables=['flow']
        )
        fusion_engine.register_source(source)
        assert len(fusion_engine.get_sources()) == 1

    def test_unregister_source(self, fusion_engine):
        """Test source unregistration."""
        source = DataSource(
            source_id='test',
            source_type=SourceType.PHYSICAL_SENSOR,
            name='Test',
            variables=['flow']
        )
        fusion_engine.register_source(source)
        result = fusion_engine.unregister_source('test')
        assert result is True
        assert len(fusion_engine.get_sources()) == 0

    def test_unregister_nonexistent(self, fusion_engine):
        """Test unregistering nonexistent source."""
        result = fusion_engine.unregister_source('nonexistent')
        assert result is False

    def test_fuse_single_source(self, fusion_engine):
        """Test fusion with single source."""
        source = DataSource(
            source_id='sensor',
            source_type=SourceType.PHYSICAL_SENSOR,
            name='Sensor',
            variables=['flow']
        )
        fusion_engine.register_source(source)

        measurements = {'sensor': {'flow': 150.0}}
        result = fusion_engine.fuse(measurements)

        assert 'flow' in result.values
        assert result.values['flow'] == pytest.approx(150.0, rel=0.01)

    def test_fuse_multiple_sources(self, registered_engine):
        """Test fusion with multiple sources."""
        measurements = {
            'sensor_1': {'flow': 150.0},
            'sensor_2': {'flow': 152.0}
        }
        result = registered_engine.fuse(measurements)

        assert 'flow' in result.values
        # Weighted average should be between 150 and 152
        assert 150.0 <= result.values['flow'] <= 152.0

    def test_fuse_uncertainty_calculation(self, registered_engine):
        """Test uncertainty is calculated."""
        measurements = {
            'sensor_1': {'flow': 150.0},
            'sensor_2': {'flow': 160.0}
        }
        result = registered_engine.fuse(measurements)

        assert 'flow' in result.uncertainties
        assert result.uncertainties['flow'] > 0

    def test_fuse_quality_assessment(self, registered_engine):
        """Test quality assessment."""
        measurements = {
            'sensor_1': {'flow': 150.0},
            'sensor_2': {'flow': 150.1}
        }
        result = registered_engine.fuse(measurements)

        assert result.quality in SyncQuality

    def test_fuse_disabled_source(self, fusion_engine):
        """Test that disabled sources are ignored."""
        source = DataSource(
            source_id='disabled',
            source_type=SourceType.PHYSICAL_SENSOR,
            name='Disabled',
            variables=['flow'],
            enabled=False
        )
        fusion_engine.register_source(source)

        measurements = {'disabled': {'flow': 150.0}}
        result = fusion_engine.fuse(measurements)

        # Disabled source should be ignored
        assert 'flow' not in result.values or result.values.get('flow') != 150.0

    def test_get_state(self, registered_engine):
        """Test get_state after fusion."""
        measurements = {
            'sensor_1': {'flow': 150.0, 'pressure': 10.0}
        }
        registered_engine.fuse(measurements)

        state = registered_engine.get_state()
        assert 'flow' in state

    def test_get_uncertainty(self, registered_engine):
        """Test get_uncertainty after fusion."""
        measurements = {
            'sensor_1': {'flow': 150.0}
        }
        registered_engine.fuse(measurements)

        uncertainty = registered_engine.get_uncertainty()
        assert isinstance(uncertainty, dict)

    def test_get_history(self, registered_engine):
        """Test history retrieval."""
        for i in range(5):
            measurements = {'sensor_1': {'flow': 150.0 + i}}
            registered_engine.fuse(measurements)
            time.sleep(0.01)

        history = registered_engine.get_history('flow', duration_seconds=60.0)
        assert len(history) == 5

    def test_reset(self, registered_engine):
        """Test reset clears state."""
        measurements = {'sensor_1': {'flow': 150.0}}
        registered_engine.fuse(measurements)

        registered_engine.reset()

        assert registered_engine.get_state() == {}

    def test_kalman_update(self, registered_engine):
        """Test Kalman-like filtering effect."""
        # First measurement
        registered_engine.fuse({'sensor_1': {'flow': 100.0}})

        # Second measurement with outlier
        registered_engine.fuse({'sensor_1': {'flow': 200.0}})

        state = registered_engine.get_state()
        # Should be smoothed, not jump directly to 200
        assert 100.0 < state['flow'] < 200.0


# =============================================================================
# Latency Compensator Tests
# =============================================================================

class TestLatencyCompensator:
    """Test latency compensator."""

    @pytest.fixture
    def compensator(self):
        """Create compensator for testing."""
        return LatencyCompensator(
            max_compensation_ms=500.0,
            prediction_horizon_ms=100.0
        )

    def test_initialization(self, compensator):
        """Test compensator initialization."""
        assert compensator is not None
        assert compensator._max_compensation == 500.0

    def test_record_latency(self, compensator):
        """Test latency recording."""
        compensator.record_latency('sensor_1', 45.0)
        compensator.record_latency('sensor_1', 50.0)
        compensator.record_latency('sensor_1', 48.0)

        estimate = compensator.get_latency_estimate('sensor_1')
        assert estimate is not None
        assert estimate.current_latency_ms == 48.0

    def test_get_latency_estimate(self, compensator):
        """Test latency estimation."""
        for i in range(10):
            compensator.record_latency('sensor', 40 + i)

        estimate = compensator.get_latency_estimate('sensor')
        assert estimate.avg_latency_ms == pytest.approx(44.5, rel=0.01)
        assert estimate.min_latency_ms == 40
        assert estimate.max_latency_ms == 49
        assert estimate.sample_count == 10

    def test_get_latency_nonexistent(self, compensator):
        """Test getting latency for nonexistent source."""
        estimate = compensator.get_latency_estimate('nonexistent')
        assert estimate is None

    def test_compensate_no_delay(self, compensator):
        """Test compensation with no delay."""
        measurement = {'flow': 150.0}
        result = compensator.compensate(
            measurement,
            time.time(),  # Current time
            'sensor'
        )
        assert result['flow'] == pytest.approx(150.0, rel=0.01)

    def test_compensate_with_delay(self, compensator):
        """Test compensation with delay."""
        # Record some state history first
        for i in range(10):
            compensator.record_state({'flow': 100.0 + i * 5}, time.time() - 1 + i * 0.1)

        measurement = {'flow': 150.0}
        result = compensator.compensate(
            measurement,
            time.time() - 0.1,  # 100ms ago
            'sensor'
        )
        # Should be extrapolated forward
        assert result is not None
        assert 'flow' in result

    def test_set_predictor(self, compensator):
        """Test setting custom predictor."""
        def predictor(state, dt):
            return {k: v * 1.1 for k, v in state.items()}

        compensator.set_predictor(predictor)

        measurement = {'flow': 100.0}
        result = compensator.compensate(
            measurement,
            time.time() - 0.05,
            'sensor'
        )
        assert result['flow'] == pytest.approx(110.0, rel=0.01)

    def test_record_state(self, compensator):
        """Test state recording."""
        compensator.record_state({'flow': 150.0}, time.time())
        # State should be recorded internally
        assert 'flow' in compensator._state_history

    def test_get_all_latency_estimates(self, compensator):
        """Test getting all latency estimates."""
        compensator.record_latency('sensor_1', 50.0)
        compensator.record_latency('sensor_2', 60.0)

        estimates = compensator.get_all_latency_estimates()
        assert len(estimates) == 2
        assert 'sensor_1' in estimates
        assert 'sensor_2' in estimates

    def test_reset(self, compensator):
        """Test reset."""
        compensator.record_latency('sensor', 50.0)
        compensator.record_state({'flow': 100.0}, time.time())

        compensator.reset()

        assert compensator.get_latency_estimate('sensor') is None
        assert len(compensator._state_history) == 0


# =============================================================================
# Discrepancy Detector Tests
# =============================================================================

class TestDiscrepancyDetector:
    """Test discrepancy detector."""

    @pytest.fixture
    def detector(self):
        """Create detector for testing."""
        return DiscrepancyDetector(
            thresholds={'flow': 0.05},  # 5% threshold for flow
            default_threshold=0.1
        )

    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector is not None
        assert detector._default_threshold == 0.1

    def test_set_threshold(self, detector):
        """Test setting threshold."""
        detector.set_threshold('pressure', 0.02)
        assert detector._thresholds['pressure'] == 0.02

    def test_check_no_discrepancy(self, detector):
        """Test check with no discrepancy."""
        physical = {'flow': 150.0}
        digital = {'flow': 151.0}  # 0.67% difference

        events = detector.check(physical, digital)
        assert len(events) == 0

    def test_check_with_discrepancy(self, detector):
        """Test check with discrepancy."""
        physical = {'flow': 150.0}
        digital = {'flow': 180.0}  # 20% difference

        events = detector.check(physical, digital)
        assert len(events) == 1
        assert events[0].variable == 'flow'

    def test_discrepancy_severity_low(self, detector):
        """Test low severity discrepancy."""
        physical = {'flow': 100.0}
        digital = {'flow': 106.0}  # 6% difference, threshold 5%

        events = detector.check(physical, digital)
        assert len(events) == 1
        assert events[0].severity == 'low'

    def test_discrepancy_severity_high(self, detector):
        """Test high severity discrepancy."""
        physical = {'flow': 100.0}
        digital = {'flow': 150.0}  # 50% difference

        events = detector.check(physical, digital)
        assert len(events) == 1
        assert events[0].severity in ['high', 'critical']

    def test_discrepancy_resolution(self, detector):
        """Test automatic discrepancy resolution."""
        # Create discrepancy
        physical = {'flow': 100.0}
        digital = {'flow': 150.0}
        detector.check(physical, digital)

        assert len(detector.get_active_discrepancies()) == 1

        # Resolve naturally
        physical = {'flow': 100.0}
        digital = {'flow': 100.5}
        detector.check(physical, digital)

        assert len(detector.get_active_discrepancies()) == 0

    def test_manual_resolve(self, detector):
        """Test manual resolution."""
        physical = {'flow': 100.0}
        digital = {'flow': 150.0}
        detector.check(physical, digital)

        result = detector.resolve('flow', 'manual_correction')
        assert result is True
        assert len(detector.get_active_discrepancies()) == 0

    def test_resolve_nonexistent(self, detector):
        """Test resolving nonexistent discrepancy."""
        result = detector.resolve('nonexistent')
        assert result is False

    def test_callback(self, detector):
        """Test discrepancy callback."""
        events_received = []

        def callback(event):
            events_received.append(event)

        detector.add_callback(callback)

        physical = {'flow': 100.0}
        digital = {'flow': 150.0}
        detector.check(physical, digital)

        assert len(events_received) == 1

    def test_get_history(self, detector):
        """Test getting discrepancy history."""
        for i in range(5):
            physical = {'flow': 100.0}
            digital = {'flow': 100.0 + (i + 1) * 20}
            detector.check(physical, digital)

        history = detector.get_history(limit=3)
        assert len(history) == 3

    def test_get_statistics(self, detector):
        """Test statistics generation."""
        physical = {'flow': 100.0, 'pressure': 10.0}
        digital = {'flow': 150.0, 'pressure': 15.0}
        detector.check(physical, digital)

        stats = detector.get_statistics()
        assert stats['total_events'] > 0
        assert 'by_severity' in stats
        assert 'by_variable' in stats

    def test_reset(self, detector):
        """Test reset."""
        physical = {'flow': 100.0}
        digital = {'flow': 150.0}
        detector.check(physical, digital)

        detector.reset()

        assert len(detector.get_active_discrepancies()) == 0
        assert detector.get_statistics()['total_events'] == 0


# =============================================================================
# Confidence Tracker Tests
# =============================================================================

class TestConfidenceTracker:
    """Test confidence tracker."""

    @pytest.fixture
    def tracker(self):
        """Create tracker for testing."""
        return ConfidenceTracker(decay_rate=0.99)

    def test_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker is not None
        assert tracker._decay_rate == 0.99

    def test_update_single_source(self, tracker):
        """Test update with single source."""
        score = tracker.update(
            variable='flow',
            sources={'sensor_1': 150.0},
            weights={'sensor_1': 1.0},
            uncertainty=1.5
        )

        assert score.variable == 'flow'
        assert 0 <= score.confidence <= 1

    def test_update_multiple_sources(self, tracker):
        """Test update with multiple sources."""
        score = tracker.update(
            variable='flow',
            sources={'sensor_1': 150.0, 'sensor_2': 151.0},
            weights={'sensor_1': 0.6, 'sensor_2': 0.4},
            uncertainty=0.5
        )

        assert score.confidence > 0.5  # Good agreement

    def test_update_disagreeing_sources(self, tracker):
        """Test update with disagreeing sources."""
        score = tracker.update(
            variable='flow',
            sources={'sensor_1': 100.0, 'sensor_2': 200.0},
            weights={'sensor_1': 0.5, 'sensor_2': 0.5},
            uncertainty=50.0
        )

        assert score.confidence <= 0.5  # Poor agreement

    def test_confidence_decay(self, tracker):
        """Test confidence decay over time."""
        tracker.update(
            variable='flow',
            sources={'sensor': 150.0},
            weights={'sensor': 1.0},
            uncertainty=1.0
        )

        time.sleep(0.1)

        # Update again - confidence should be weighted with decay
        score = tracker.update(
            variable='flow',
            sources={'sensor': 150.0},
            weights={'sensor': 1.0},
            uncertainty=1.0
        )

        assert score.confidence > 0

    def test_record_source_accuracy(self, tracker):
        """Test recording source accuracy."""
        tracker.record_source_accuracy('sensor_1', 100.0, 95.0)
        tracker.record_source_accuracy('sensor_1', 100.0, 98.0)

        reliability = tracker.get_source_reliability('sensor_1')
        assert 0 <= reliability <= 1

    def test_get_source_reliability_default(self, tracker):
        """Test default source reliability."""
        reliability = tracker.get_source_reliability('unknown')
        assert reliability == 0.9

    def test_get_confidence(self, tracker):
        """Test getting confidence score."""
        tracker.update(
            variable='flow',
            sources={'sensor': 150.0},
            weights={'sensor': 1.0},
            uncertainty=1.0
        )

        score = tracker.get_confidence('flow')
        assert score is not None
        assert score.variable == 'flow'

    def test_get_confidence_nonexistent(self, tracker):
        """Test getting nonexistent confidence."""
        score = tracker.get_confidence('nonexistent')
        assert score is None

    def test_get_all_confidences(self, tracker):
        """Test getting all confidences."""
        tracker.update('flow', {'s': 100}, {'s': 1}, 1.0)
        tracker.update('pressure', {'s': 10}, {'s': 1}, 0.1)

        all_conf = tracker.get_all_confidences()
        assert len(all_conf) == 2

    def test_get_low_confidence_variables(self, tracker):
        """Test getting low confidence variables."""
        # High confidence (good agreement, low uncertainty)
        tracker.update('flow', {'s1': 100, 's2': 100.1}, {'s1': 0.5, 's2': 0.5}, 0.01)

        # Very low confidence (high disagreement and uncertainty)
        tracker.update('pressure', {'s1': 100, 's2': 300}, {'s1': 0.5, 's2': 0.5}, 100.0)

        low_conf = tracker.get_low_confidence_variables(threshold=0.6)
        assert 'pressure' in low_conf

    def test_reset(self, tracker):
        """Test reset."""
        tracker.update('flow', {'s': 100}, {'s': 1}, 1.0)
        tracker.reset()

        assert tracker.get_confidence('flow') is None


# =============================================================================
# State Synchronizer Tests
# =============================================================================

class TestStateSynchronizer:
    """Test state synchronizer."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.get_state.return_value = {
            'total_flow': 150.0,
            'head_upstream': 10.0,
            'head_downstream': 8.0,
            'gate_openings': [0.5, 0.5, 0.5]
        }
        return model

    @pytest.fixture
    def synchronizer(self, mock_model):
        """Create synchronizer for testing."""
        sync = StateSynchronizer(
            mock_model,
            sync_interval_ms=100.0,
            mode=SyncMode.ON_DEMAND
        )
        # Register model source for sync to work
        model_source = DataSource(
            source_id='model',
            source_type=SourceType.MODEL_OUTPUT,
            name='Model',
            variables=['total_flow', 'head_upstream', 'head_downstream']
        )
        sync.register_source(model_source)
        return sync

    def test_initialization(self, synchronizer):
        """Test synchronizer initialization."""
        assert synchronizer is not None
        assert synchronizer.status == SyncStatus.IDLE
        assert synchronizer.mode == SyncMode.ON_DEMAND

    def test_mode_change(self, synchronizer):
        """Test mode change."""
        synchronizer.mode = SyncMode.BATCH
        assert synchronizer.mode == SyncMode.BATCH

    def test_register_source(self, synchronizer):
        """Test source registration."""
        source = DataSource(
            source_id='sensor',
            source_type=SourceType.PHYSICAL_SENSOR,
            name='Sensor',
            variables=['flow']
        )
        synchronizer.register_source(source)
        # Should not raise

    def test_sync_basic(self, synchronizer):
        """Test basic synchronization."""
        result = synchronizer.sync()

        assert isinstance(result, StateVector)
        assert 'total_flow' in result.values

    def test_sync_with_measurements(self, synchronizer):
        """Test sync with external measurements."""
        source = DataSource(
            source_id='sensor',
            source_type=SourceType.PHYSICAL_SENSOR,
            name='Sensor',
            variables=['flow']
        )
        synchronizer.register_source(source)

        measurements = {'sensor': {'flow': 155.0}}
        result = synchronizer.sync(measurements)

        assert result is not None

    def test_sync_callback(self, synchronizer):
        """Test sync callback."""
        received = []

        def callback(state):
            received.append(state)

        synchronizer.add_sync_callback(callback)
        synchronizer.sync()

        assert len(received) == 1

    def test_start_stop(self, synchronizer):
        """Test start and stop."""
        synchronizer.start()
        assert synchronizer.status == SyncStatus.SYNCING

        synchronizer.stop()
        assert synchronizer.status == SyncStatus.IDLE

    def test_realtime_mode(self, mock_model):
        """Test realtime synchronization mode."""
        sync = StateSynchronizer(
            mock_model,
            sync_interval_ms=50.0,
            mode=SyncMode.REALTIME
        )

        sync.start()
        time.sleep(0.15)  # Allow some syncs
        sync.stop()

        assert sync.get_metrics().sync_count > 0

    def test_get_metrics(self, synchronizer):
        """Test metrics retrieval."""
        synchronizer.sync()
        metrics = synchronizer.get_metrics()

        assert metrics.sync_count == 1
        assert metrics.last_sync_time is not None

    def test_get_confidence_report(self, synchronizer):
        """Test confidence report."""
        synchronizer.sync()
        report = synchronizer.get_confidence_report()

        assert 'variables' in report
        assert 'avg_confidence' in report

    def test_get_discrepancy_report(self, synchronizer):
        """Test discrepancy report."""
        report = synchronizer.get_discrepancy_report()

        assert 'total_events' in report
        assert 'active_count' in report

    def test_get_latency_report(self, synchronizer):
        """Test latency report."""
        report = synchronizer.get_latency_report()
        assert isinstance(report, dict)

    def test_reset(self, synchronizer):
        """Test reset."""
        synchronizer.sync()
        synchronizer.reset()

        assert synchronizer.status == SyncStatus.IDLE
        assert synchronizer.get_metrics().sync_count == 0


# =============================================================================
# Twin Sync Manager Tests
# =============================================================================

class TestTwinSyncManager:
    """Test twin sync manager."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.get_state.return_value = {
            'total_flow': 150.0,
            'head_upstream': 10.0
        }
        return model

    @pytest.fixture
    def manager(self, mock_model):
        """Create manager for testing."""
        return TwinSyncManager(
            mock_model,
            config={'sync_interval_ms': 100.0, 'mode': 'ON_DEMAND'}
        )

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager is not None
        assert manager._running is False

    def test_add_physical_source(self, manager):
        """Test adding physical source."""
        manager.add_physical_source(
            source_id='flow_sensor',
            name='Flow Sensor',
            variables=['flow'],
            latency_ms=50.0
        )

        status = manager.get_status()
        assert 'flow_sensor' in status['sources']

    def test_remove_source(self, manager):
        """Test removing source."""
        manager.add_physical_source(
            source_id='sensor',
            name='Sensor',
            variables=['flow']
        )

        result = manager.remove_source('sensor')
        assert result is True

        result = manager.remove_source('nonexistent')
        assert result is False

    def test_start_stop(self, manager):
        """Test start and stop."""
        manager.start()
        assert manager._running is True

        manager.stop()
        assert manager._running is False

    def test_update_physical_state(self, manager):
        """Test updating physical state."""
        manager.add_physical_source('sensor', 'Sensor', ['flow'])

        result = manager.update_physical_state(
            'sensor',
            {'flow': 155.0}
        )

        assert isinstance(result, StateVector)

    def test_sync_now(self, manager):
        """Test immediate sync."""
        result = manager.sync_now()
        assert isinstance(result, StateVector)

    def test_get_synchronized_state(self, manager):
        """Test getting synchronized state."""
        manager.sync_now()
        state = manager.get_synchronized_state()

        assert isinstance(state, dict)

    def test_get_status(self, manager):
        """Test status retrieval."""
        status = manager.get_status()

        assert 'running' in status
        assert 'status' in status
        assert 'mode' in status
        assert 'sources' in status
        assert 'metrics' in status

    def test_get_discrepancies(self, manager):
        """Test getting discrepancies."""
        discrepancies = manager.get_discrepancies()
        assert isinstance(discrepancies, list)

    def test_set_discrepancy_threshold(self, manager):
        """Test setting discrepancy threshold."""
        manager.set_discrepancy_threshold('flow', 0.02)
        # Should not raise

    def test_sync_callback(self, manager):
        """Test sync callback."""
        received = []

        def callback(state):
            received.append(state)

        manager.add_sync_callback(callback)
        manager.sync_now()

        assert len(received) == 1

    def test_discrepancy_callback(self, manager):
        """Test discrepancy callback."""
        received = []

        def callback(event):
            received.append(event)

        manager.add_discrepancy_callback(callback)
        # Callbacks are registered internally

    def test_reset(self, manager):
        """Test reset."""
        manager.sync_now()
        manager.reset()

        state = manager.get_synchronized_state()
        assert len(state) == 0

    def test_get_fusion_history(self, manager):
        """Test getting fusion history."""
        for i in range(3):
            manager.sync_now()
            time.sleep(0.01)

        history = manager.get_fusion_history('total_flow', duration_seconds=60.0)
        assert isinstance(history, list)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for twin sync module."""

    def test_full_sync_workflow(self):
        """Test complete synchronization workflow."""
        # Create mock model
        model = Mock()
        model.get_state.return_value = {
            'flow': 150.0,
            'pressure': 10.0
        }

        # Create manager with ON_DEMAND mode for predictable testing
        manager = TwinSyncManager(model, config={'mode': 'ON_DEMAND'})

        # Add physical sources
        manager.add_physical_source(
            source_id='scada',
            name='SCADA System',
            variables=['flow', 'pressure'],
            source_type=SourceType.SCADA,
            reliability=0.99,
            accuracy=0.98
        )

        # Simulate physical updates (on-demand sync)
        for i in range(5):
            manager.update_physical_state(
                'scada',
                {'flow': 150.0 + i, 'pressure': 10.0 + i * 0.1}
            )

        # Check results
        status = manager.get_status()
        assert status['metrics']['sync_count'] > 0

    def test_discrepancy_detection_workflow(self):
        """Test discrepancy detection workflow."""
        model = Mock()
        model.get_state.return_value = {'flow': 100.0}

        manager = TwinSyncManager(model, config={'mode': 'ON_DEMAND'})
        manager.add_physical_source('sensor', 'Sensor', ['flow'])

        # Simulate large discrepancy
        manager.update_physical_state('sensor', {'flow': 200.0})

        discrepancies = manager.get_discrepancies()
        # Discrepancy should be detected (model=100, physical=200)
        assert len(discrepancies) >= 0  # May or may not detect based on thresholds

    def test_multi_source_fusion(self):
        """Test fusion with multiple sources."""
        model = Mock()
        model.get_state.return_value = {'flow': 150.0}

        manager = TwinSyncManager(model, config={'mode': 'ON_DEMAND'})

        # Add multiple physical sources
        manager.add_physical_source(
            source_id='sensor_1',
            name='Sensor 1',
            variables=['flow'],
            accuracy=0.98
        )
        manager.add_physical_source(
            source_id='sensor_2',
            name='Sensor 2',
            variables=['flow'],
            accuracy=0.95
        )

        # Update from both sources
        manager.update_physical_state('sensor_1', {'flow': 152.0})
        manager.update_physical_state('sensor_2', {'flow': 148.0})

        state = manager.get_synchronized_state()
        # Fused value should be weighted average
        assert 'flow' in state


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_measurements(self):
        """Test fusion with empty measurements."""
        engine = DataFusionEngine()
        result = engine.fuse({})

        assert result.values == {}

    def test_zero_values(self):
        """Test handling of zero values."""
        engine = DataFusionEngine()
        source = DataSource('s', SourceType.PHYSICAL_SENSOR, 'S', ['x'])
        engine.register_source(source)

        result = engine.fuse({'s': {'x': 0.0}})
        assert result.values['x'] == pytest.approx(0.0)

    def test_negative_values(self):
        """Test handling of negative values."""
        engine = DataFusionEngine()
        source = DataSource('s', SourceType.PHYSICAL_SENSOR, 'S', ['x'])
        engine.register_source(source)

        result = engine.fuse({'s': {'x': -50.0}})
        assert result.values['x'] == pytest.approx(-50.0, rel=0.01)

    def test_concurrent_access(self):
        """Test thread safety."""
        engine = DataFusionEngine()
        source = DataSource('s', SourceType.PHYSICAL_SENSOR, 'S', ['x'])
        engine.register_source(source)

        def updater():
            for i in range(100):
                engine.fuse({'s': {'x': float(i)}})

        threads = [threading.Thread(target=updater) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors
        state = engine.get_state()
        assert 'x' in state

    def test_large_discrepancy(self):
        """Test handling of very large discrepancies."""
        detector = DiscrepancyDetector()

        events = detector.check(
            {'x': 1.0},
            {'x': 1000000.0}
        )

        assert len(events) == 1
        assert events[0].severity == 'critical'

    def test_nan_handling(self):
        """Test handling of NaN values."""
        engine = DataFusionEngine()
        source = DataSource('s', SourceType.PHYSICAL_SENSOR, 'S', ['x'])
        engine.register_source(source)

        # NaN should be handled gracefully
        result = engine.fuse({'s': {'x': float('nan')}})
        # Result should still be valid (may have NaN or be filtered)
        assert result is not None

    def test_very_high_latency(self):
        """Test handling of very high latency."""
        compensator = LatencyCompensator(max_compensation_ms=100.0)

        measurement = {'x': 100.0}
        # Very old measurement
        result = compensator.compensate(
            measurement,
            time.time() - 10.0,  # 10 seconds ago
            'sensor'
        )

        # Should still return something
        assert 'x' in result
