# -*- coding: utf-8 -*-
"""
Tests for Anomaly Detection Module.

测试异常检测模块。
"""

import pytest
import time
from datetime import datetime

from src.data.anomaly import (
    AnomalyDetector, AnomalyType, Anomaly, Severity,
    ThresholdRule, RateRule
)


# =============================================================================
# Anomaly Tests
# =============================================================================

class TestAnomaly:
    """Tests for Anomaly dataclass."""

    def test_creation(self):
        """Test anomaly creation."""
        anomaly = Anomaly(
            anomaly_id="test_001",
            timestamp=1000.0,
            anomaly_type=AnomalyType.THRESHOLD_HIGH,
            severity=Severity.WARNING,
            series_name="flow",
            value=500.0,
            message="Flow too high"
        )

        assert anomaly.anomaly_id == "test_001"
        assert anomaly.anomaly_type == AnomalyType.THRESHOLD_HIGH
        assert anomaly.severity == Severity.WARNING
        assert not anomaly.acknowledged

    def test_acknowledge(self):
        """Test anomaly acknowledgment."""
        anomaly = Anomaly(
            anomaly_id="test",
            timestamp=1000.0,
            anomaly_type=AnomalyType.THRESHOLD_HIGH,
            severity=Severity.INFO,
            series_name="test",
            value=0.0,
            message="Test"
        )

        assert not anomaly.acknowledged
        anomaly.acknowledged = True
        assert anomaly.acknowledged

    def test_to_dict(self):
        """Test anomaly serialization."""
        anomaly = Anomaly(
            anomaly_id="test",
            timestamp=1000.0,
            anomaly_type=AnomalyType.SPIKE,
            severity=Severity.CRITICAL,
            series_name="flow",
            value=100.0,
            message="Test spike"
        )

        d = anomaly.to_dict()
        assert d['anomaly_id'] == "test"
        assert d['type'] == "SPIKE"
        assert d['severity'] == "CRITICAL"


class TestSeverity:
    """Tests for Severity enum."""

    def test_severity_values(self):
        """Test severity enum values."""
        assert Severity.INFO.value < Severity.WARNING.value
        assert Severity.WARNING.value < Severity.CRITICAL.value
        assert Severity.CRITICAL.value < Severity.EMERGENCY.value


class TestAnomalyType:
    """Tests for AnomalyType enum."""

    def test_anomaly_types(self):
        """Test anomaly type existence."""
        assert AnomalyType.THRESHOLD_HIGH is not None
        assert AnomalyType.THRESHOLD_LOW is not None
        assert AnomalyType.RATE_OF_CHANGE is not None
        assert AnomalyType.STATISTICAL is not None
        assert AnomalyType.SPIKE is not None
        assert AnomalyType.DROP is not None


# =============================================================================
# ThresholdRule Tests
# =============================================================================

class TestThresholdRule:
    """Tests for ThresholdRule."""

    def test_creation(self):
        """Test rule creation."""
        rule = ThresholdRule(
            name="flow_high",
            series_name="flow",
            low=0,
            high=500,
        )

        assert rule.name == "flow_high"
        assert rule.series_name == "flow"
        assert rule.low == 0
        assert rule.high == 500
        assert rule.enabled

    def test_with_emergency_limits(self):
        """Test rule with emergency limits."""
        rule = ThresholdRule(
            name="flow_check",
            series_name="flow",
            low=50,
            high=450,
            low_low=0,
            high_high=500
        )

        assert rule.low_low == 0
        assert rule.high_high == 500


# =============================================================================
# RateRule Tests
# =============================================================================

class TestRateRule:
    """Tests for RateRule."""

    def test_creation(self):
        """Test rule creation."""
        rule = RateRule(
            name="vib_rate",
            series_name="vibration",
            max_rate=10.0,
            window=5.0,
        )

        assert rule.name == "vib_rate"
        assert rule.max_rate == 10.0
        assert rule.window == 5.0
        assert rule.enabled


# =============================================================================
# AnomalyDetector Tests
# =============================================================================

class TestAnomalyDetector:
    """Tests for AnomalyDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = AnomalyDetector()
        assert len(detector._threshold_rules) == 0
        assert len(detector._rate_rules) == 0

    def test_add_threshold_rule(self):
        """Test adding threshold rule."""
        detector = AnomalyDetector()

        rule = ThresholdRule(
            name="flow_high",
            series_name="flow",
            low=0,
            high=500,
        )
        detector.add_threshold_rule(rule)

        assert "flow_high" in detector._threshold_rules

    def test_add_rate_rule(self):
        """Test adding rate rule."""
        detector = AnomalyDetector()

        rule = RateRule(
            name="vib_rate",
            series_name="vibration",
            max_rate=10.0,
            window=5.0,
        )
        detector.add_rate_rule(rule)

        assert "vib_rate" in detector._rate_rules

    def test_remove_rule(self):
        """Test removing rule."""
        detector = AnomalyDetector()

        rule = ThresholdRule(
            name="test",
            series_name="flow",
            high=500,
        )
        detector.add_threshold_rule(rule)
        detector.remove_rule("test")

        assert "test" not in detector._threshold_rules

    def test_detect_threshold_high(self):
        """Test threshold high detection."""
        detector = AnomalyDetector()

        detector.add_threshold_rule(ThresholdRule(
            name="flow_check",
            series_name="flow",
            high=100,
        ))

        # Normal value
        anomalies = detector.detect("flow", 50.0)
        assert len(anomalies) == 0

        # High value
        anomalies = detector.detect("flow", 150.0)
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == AnomalyType.THRESHOLD_HIGH

    def test_detect_threshold_low(self):
        """Test threshold low detection."""
        detector = AnomalyDetector()

        detector.add_threshold_rule(ThresholdRule(
            name="flow_check",
            series_name="flow",
            low=10,
        ))

        # Normal value
        anomalies = detector.detect("flow", 50.0)
        assert len(anomalies) == 0

        # Low value
        anomalies = detector.detect("flow", 5.0)
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == AnomalyType.THRESHOLD_LOW

    def test_detect_threshold_emergency(self):
        """Test emergency threshold detection."""
        detector = AnomalyDetector()

        detector.add_threshold_rule(ThresholdRule(
            name="flow_check",
            series_name="flow",
            high=400,
            high_high=500,
        ))

        # Emergency high value
        anomalies = detector.detect("flow", 550.0)
        assert len(anomalies) == 1
        assert anomalies[0].severity == Severity.EMERGENCY

    def test_detect_rate_of_change(self):
        """Test rate of change detection."""
        detector = AnomalyDetector()

        detector.add_rate_rule(RateRule(
            name="rate_check",
            series_name="sensor",
            max_rate=10.0,
            window=5.0,
        ))

        base_time = time.time()

        # Normal rate
        detector.detect("sensor", 100.0, base_time)
        anomalies = detector.detect("sensor", 105.0, base_time + 1)
        assert len(anomalies) == 0

        # Rapid change
        anomalies = detector.detect("sensor", 200.0, base_time + 2)
        assert len(anomalies) >= 1
        assert any(a.anomaly_type in [AnomalyType.SPIKE, AnomalyType.DROP] for a in anomalies)

    def test_configure_statistical(self):
        """Test statistical detection configuration."""
        detector = AnomalyDetector()

        detector.configure_statistical(
            series_name="flow",
            mean=100.0,
            std=10.0,
            n_sigma=3.0
        )

        assert "flow" in detector._stats
        assert detector._stats["flow"]["mean"] == 100.0

    def test_detect_statistical_anomaly(self):
        """Test statistical anomaly detection."""
        detector = AnomalyDetector()

        detector.configure_statistical(
            series_name="flow",
            mean=100.0,
            std=10.0,
            n_sigma=2.0
        )

        # Normal value
        anomalies = detector.detect("flow", 105.0)
        assert len(anomalies) == 0

        # Statistical anomaly (beyond 2 sigma)
        anomalies = detector.detect("flow", 150.0)
        assert any(a.anomaly_type == AnomalyType.STATISTICAL for a in anomalies)

    def test_detect_pattern_flatline(self):
        """Test flatline pattern detection."""
        detector = AnomalyDetector()

        # Send same value multiple times
        for i in range(15):
            anomalies = detector.detect("sensor", 50.0)

        # Should detect flatline
        assert any(a.anomaly_type == AnomalyType.FLATLINE for a in anomalies)

    def test_acknowledge_anomaly(self):
        """Test acknowledging anomaly."""
        detector = AnomalyDetector()

        detector.add_threshold_rule(ThresholdRule(
            name="test",
            series_name="flow",
            high=100,
        ))

        anomalies = detector.detect("flow", 150.0)
        assert len(anomalies) > 0

        anomaly_id = anomalies[0].anomaly_id
        success = detector.acknowledge_anomaly(anomaly_id)
        assert success

        # Check it's acknowledged
        all_anomalies = detector.get_anomalies()
        for a in all_anomalies:
            if a.anomaly_id == anomaly_id:
                assert a.acknowledged

    def test_get_anomalies_filtered(self):
        """Test getting filtered anomalies."""
        detector = AnomalyDetector()

        detector.add_threshold_rule(ThresholdRule(
            name="flow_check",
            series_name="flow",
            high=100,
        ))
        detector.add_threshold_rule(ThresholdRule(
            name="temp_check",
            series_name="temp",
            high=50,
        ))

        detector.detect("flow", 150.0)
        detector.detect("temp", 60.0)

        # Filter by series
        flow_anomalies = detector.get_anomalies(series_name="flow")
        assert all(a.series_name == "flow" for a in flow_anomalies)

    def test_get_anomalies_by_severity(self):
        """Test getting anomalies by severity."""
        detector = AnomalyDetector()

        detector.add_threshold_rule(ThresholdRule(
            name="flow_check",
            series_name="flow",
            high=100,
            high_high=200,
        ))

        # Generate warning (high)
        detector.detect("flow", 150.0)
        # Generate emergency (high_high)
        detector.detect("flow", 250.0)

        # Get only critical and above
        critical_anomalies = detector.get_anomalies(severity=Severity.CRITICAL)
        assert all(a.severity.value >= Severity.CRITICAL.value for a in critical_anomalies)

    def test_get_unacknowledged_only(self):
        """Test getting only unacknowledged anomalies."""
        detector = AnomalyDetector()

        detector.add_threshold_rule(ThresholdRule(
            name="test",
            series_name="flow",
            high=100,
        ))

        anomalies = detector.detect("flow", 150.0)
        anomaly_id = anomalies[0].anomaly_id
        detector.acknowledge_anomaly(anomaly_id)

        # Generate another
        detector.detect("flow", 160.0)

        unack = detector.get_anomalies(unacknowledged_only=True)
        assert all(not a.acknowledged for a in unack)

    def test_get_active_alarms(self):
        """Test getting active alarm states."""
        detector = AnomalyDetector()

        detector.add_threshold_rule(ThresholdRule(
            name="flow_check",
            series_name="flow",
            high=100,
        ))

        detector.detect("flow", 150.0)

        alarms = detector.get_active_alarms()
        assert any(alarms.values())

    def test_reset(self):
        """Test detector reset."""
        detector = AnomalyDetector()

        detector.add_threshold_rule(ThresholdRule(
            name="test",
            series_name="flow",
            high=100,
        ))

        detector.detect("flow", 50.0)
        detector.detect("flow", 60.0)

        detector.reset()
        assert len(detector._history) == 0
        assert len(detector._alarm_states) == 0

    def test_get_stats(self):
        """Test getting statistics."""
        detector = AnomalyDetector()

        detector.add_threshold_rule(ThresholdRule(
            name="test",
            series_name="flow",
            high=100,
        ))

        detector.detect("flow", 150.0)
        detector.detect("flow", 200.0)

        stats = detector.get_stats()
        assert 'total_anomalies' in stats
        assert 'by_severity' in stats
        assert 'by_type' in stats
        assert stats['total_anomalies'] >= 1

    def test_callback(self):
        """Test anomaly callback."""
        detector = AnomalyDetector()

        received = []
        detector.add_callback(lambda a: received.append(a))

        detector.add_threshold_rule(ThresholdRule(
            name="test",
            series_name="flow",
            high=100,
        ))

        detector.detect("flow", 150.0)
        assert len(received) > 0

    def test_detect_batch(self):
        """Test batch detection."""
        detector = AnomalyDetector()

        detector.add_threshold_rule(ThresholdRule(
            name="test",
            series_name="flow",
            high=100,
        ))

        base_time = time.time()
        values = [
            (base_time, 50.0),
            (base_time + 1, 150.0),  # Anomaly
            (base_time + 2, 80.0),
            (base_time + 3, 200.0),  # Anomaly
        ]

        anomalies = detector.detect_batch("flow", values)
        assert len(anomalies) >= 1

    def test_deadband(self):
        """Test deadband functionality."""
        detector = AnomalyDetector()

        detector.add_threshold_rule(ThresholdRule(
            name="test",
            series_name="flow",
            high=100,
            deadband=5.0,
        ))

        # Trigger alarm
        anomalies = detector.detect("flow", 110.0)
        assert len(anomalies) == 1

        # Value still above threshold but alarm already active
        anomalies = detector.detect("flow", 105.0)
        assert len(anomalies) == 0  # No new alarm

        # Value drops below threshold but within deadband
        detector.detect("flow", 98.0)
        # Alarm should still be active due to deadband

        # Value drops below deadband
        detector.detect("flow", 90.0)
        # Now alarm should clear

        # Going back above threshold should trigger new alarm
        anomalies = detector.detect("flow", 110.0)
        assert len(anomalies) == 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestAnomalyIntegration:
    """Integration tests for anomaly detection."""

    def test_multi_rule_detection(self):
        """Test detection with multiple rule types."""
        detector = AnomalyDetector()

        # Add threshold rule
        detector.add_threshold_rule(ThresholdRule(
            name="threshold",
            series_name="sensor",
            high=100,
        ))

        # Add rate rule
        detector.add_rate_rule(RateRule(
            name="rate",
            series_name="sensor",
            max_rate=50,
            window=5,
        ))

        base_time = time.time()

        # Normal data
        for i in range(10):
            detector.detect("sensor", 50.0, base_time + i)

        # Spike (should trigger both rules)
        anomalies = detector.detect("sensor", 200.0, base_time + 10)

        # Should detect at least one anomaly
        assert len(anomalies) >= 1

    def test_continuous_monitoring(self):
        """Test continuous monitoring scenario."""
        detector = AnomalyDetector()

        detector.add_threshold_rule(ThresholdRule(
            name="flow_limit",
            series_name="flow",
            high=500,
        ))

        detector.configure_statistical(
            series_name="flow",
            mean=300,
            std=50,
            n_sigma=3.0
        )

        base_time = time.time()

        # Simulate continuous monitoring
        total_anomalies = 0
        for i in range(100):
            # Simulate varying values
            import random
            value = 300 + random.gauss(0, 30)

            # Occasionally inject high value
            if i % 20 == 0:
                value = 600

            anomalies = detector.detect("flow", value, base_time + i)
            total_anomalies += len(anomalies)

        # Should have detected some anomalies
        stats = detector.get_stats()
        assert stats['total_anomalies'] > 0

    def test_multi_series_monitoring(self):
        """Test monitoring multiple series."""
        detector = AnomalyDetector()

        # Rules for different series
        detector.add_threshold_rule(ThresholdRule(
            name="flow_check",
            series_name="flow",
            high=500,
        ))
        detector.add_threshold_rule(ThresholdRule(
            name="temp_check",
            series_name="temperature",
            high=80,
        ))
        detector.add_threshold_rule(ThresholdRule(
            name="pressure_check",
            series_name="pressure",
            high=10,
        ))

        base_time = time.time()

        # Simulate monitoring multiple series
        detector.detect("flow", 400, base_time)
        detector.detect("temperature", 85, base_time)  # Anomaly
        detector.detect("pressure", 8, base_time)

        flow_anomalies = detector.get_anomalies(series_name="flow")
        temp_anomalies = detector.get_anomalies(series_name="temperature")

        assert len(flow_anomalies) == 0
        assert len(temp_anomalies) > 0
