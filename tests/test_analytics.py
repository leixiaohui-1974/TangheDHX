"""
Tests for Analytics and Reporting Module.

Tests cover:
- Operation statistics
- KPI tracking
- Alarm event management
- Report generation
- Performance benchmarking
- Analytics engine integration
"""

import pytest
import json
from datetime import datetime, timedelta

from src.analytics.reporting import (
    # Enums
    KPICategory,
    AlarmSeverity,
    AlarmCategory,
    ReportType,
    ReportFormat,
    # Data classes
    KPIDefinition,
    KPIValue,
    AlarmEvent,
    StatisticsSummary,
    PerformanceMetrics,
    # Main classes
    OperationStatistics,
    KPITracker,
    AlarmEventManager,
    PerformanceBenchmark,
    ReportGenerator,
    AnalyticsEngine
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def operation_stats():
    """Create operation statistics tracker."""
    return OperationStatistics(history_hours=24)


@pytest.fixture
def kpi_tracker():
    """Create KPI tracker."""
    return KPITracker()


@pytest.fixture
def alarm_manager():
    """Create alarm event manager."""
    return AlarmEventManager(max_events=100)


@pytest.fixture
def benchmark():
    """Create performance benchmark."""
    return PerformanceBenchmark()


@pytest.fixture
def analytics_engine():
    """Create analytics engine."""
    return AnalyticsEngine(history_hours=24)


# =============================================================================
# Test Enums
# =============================================================================

class TestEnums:
    """Test enum definitions."""

    def test_kpi_category_values(self):
        """Test KPI category values."""
        assert KPICategory.HYDRAULIC.value == "hydraulic"
        assert KPICategory.MECHANICAL.value == "mechanical"
        assert KPICategory.CONTROL.value == "control"
        assert KPICategory.SAFETY.value == "safety"
        assert KPICategory.EFFICIENCY.value == "efficiency"
        assert KPICategory.MAINTENANCE.value == "maintenance"

    def test_alarm_severity_ordering(self):
        """Test alarm severity ordering."""
        assert AlarmSeverity.INFO.value < AlarmSeverity.WARNING.value
        assert AlarmSeverity.WARNING.value < AlarmSeverity.ALARM.value
        assert AlarmSeverity.ALARM.value < AlarmSeverity.CRITICAL.value
        assert AlarmSeverity.CRITICAL.value < AlarmSeverity.EMERGENCY.value

    def test_alarm_category_values(self):
        """Test alarm category values."""
        assert AlarmCategory.FLOW.value == "flow"
        assert AlarmCategory.VIBRATION.value == "vibration"
        assert AlarmCategory.GATE.value == "gate"

    def test_report_type_values(self):
        """Test report type values."""
        assert ReportType.DAILY.value == "daily"
        assert ReportType.WEEKLY.value == "weekly"
        assert ReportType.MONTHLY.value == "monthly"


# =============================================================================
# Test Data Classes
# =============================================================================

class TestKPIDefinition:
    """Test KPIDefinition dataclass."""

    def test_creation(self):
        """Test KPI definition creation."""
        kpi = KPIDefinition(
            name="test_kpi",
            category=KPICategory.CONTROL,
            unit="m³/s",
            description="Test KPI",
            target=100.0,
            warning_high=120.0,
            critical_high=150.0
        )
        assert kpi.name == "test_kpi"
        assert kpi.category == KPICategory.CONTROL
        assert kpi.target == 100.0

    def test_default_values(self):
        """Test default values."""
        kpi = KPIDefinition(
            name="simple",
            category=KPICategory.HYDRAULIC,
            unit="%",
            description="Simple KPI",
            target=50.0
        )
        assert kpi.warning_low is None
        assert kpi.aggregation == "mean"


class TestAlarmEvent:
    """Test AlarmEvent dataclass."""

    def test_creation(self):
        """Test alarm event creation."""
        event = AlarmEvent(
            event_id="ALM-001",
            timestamp=datetime.now(),
            severity=AlarmSeverity.WARNING,
            category=AlarmCategory.FLOW,
            source="flow_sensor",
            message="Flow too high"
        )
        assert event.event_id == "ALM-001"
        assert event.severity == AlarmSeverity.WARNING
        assert not event.acknowledged
        assert not event.cleared


# =============================================================================
# Test Operation Statistics
# =============================================================================

class TestOperationStatistics:
    """Test OperationStatistics class."""

    def test_initialization(self, operation_stats):
        """Test initialization."""
        assert operation_stats.history_hours == 24

    def test_record_single(self, operation_stats):
        """Test recording single metric."""
        operation_stats.record("test_metric", 42.0)
        summary = operation_stats.get_summary("test_metric")
        assert summary is not None
        assert summary.mean == 42.0

    def test_record_batch(self, operation_stats):
        """Test recording batch metrics."""
        operation_stats.record_batch({
            "metric_a": 10.0,
            "metric_b": 20.0,
            "metric_c": 30.0
        })
        assert operation_stats.get_summary("metric_a") is not None
        assert operation_stats.get_summary("metric_b") is not None
        assert operation_stats.get_summary("metric_c") is not None

    def test_summary_statistics(self, operation_stats):
        """Test summary statistics calculation."""
        for i in range(100):
            operation_stats.record("data", float(i))

        summary = operation_stats.get_summary("data")
        assert summary.count == 100
        assert summary.min == 0.0
        assert summary.max == 99.0
        assert 48 < summary.mean < 51  # Mean should be around 49.5

    def test_get_all_summaries(self, operation_stats):
        """Test getting all summaries."""
        operation_stats.record("a", 1.0)
        operation_stats.record("b", 2.0)
        summaries = operation_stats.get_all_summaries()
        assert "a" in summaries
        assert "b" in summaries

    def test_uptime_calculation(self, operation_stats):
        """Test uptime percentage calculation."""
        uptime = operation_stats.get_uptime_pct()
        assert 0 <= uptime <= 100

    def test_downtime_recording(self, operation_stats):
        """Test downtime recording."""
        operation_stats.record_downtime(1.0)
        assert operation_stats._downtime_hours == 1.0


# =============================================================================
# Test KPI Tracker
# =============================================================================

class TestKPITracker:
    """Test KPITracker class."""

    def test_initialization(self, kpi_tracker):
        """Test initialization with default KPIs."""
        assert len(kpi_tracker._definitions) > 0
        assert "total_flow" in kpi_tracker._definitions

    def test_define_custom_kpi(self, kpi_tracker):
        """Test defining custom KPI."""
        kpi_tracker.define_kpi(KPIDefinition(
            name="custom_kpi",
            category=KPICategory.CONTROL,
            unit="units",
            description="Custom KPI",
            target=50.0
        ))
        assert "custom_kpi" in kpi_tracker._definitions

    def test_update_kpi(self, kpi_tracker):
        """Test updating KPI value."""
        result = kpi_tracker.update("total_flow", 100.0)
        assert result is not None
        assert result.value == 100.0
        assert result.status == "normal"

    def test_kpi_status_warning(self, kpi_tracker):
        """Test warning status detection."""
        # total_flow has warning_high=120
        result = kpi_tracker.update("total_flow", 125.0)
        assert result.status == "warning"

    def test_kpi_status_critical(self, kpi_tracker):
        """Test critical status detection."""
        # total_flow has critical_high=150
        result = kpi_tracker.update("total_flow", 160.0)
        assert result.status == "critical"

    def test_get_current(self, kpi_tracker):
        """Test getting current value."""
        kpi_tracker.update("total_flow", 95.0)
        current = kpi_tracker.get_current("total_flow")
        assert current is not None
        assert current.value == 95.0

    def test_get_all_current(self, kpi_tracker):
        """Test getting all current values."""
        kpi_tracker.update("total_flow", 100.0)
        kpi_tracker.update("max_vibration", 1.0)
        all_current = kpi_tracker.get_all_current()
        assert "total_flow" in all_current
        assert "max_vibration" in all_current

    def test_get_by_category(self, kpi_tracker):
        """Test getting KPIs by category."""
        kpi_tracker.update("total_flow", 100.0)
        hydraulic = kpi_tracker.get_by_category(KPICategory.HYDRAULIC)
        assert len(hydraulic) > 0

    def test_get_alerts(self, kpi_tracker):
        """Test getting KPI alerts."""
        kpi_tracker.update("total_flow", 160.0)  # Critical
        alerts = kpi_tracker.get_alerts()
        assert len(alerts) > 0
        assert any(a.name == "total_flow" for a in alerts)

    def test_trend_calculation(self, kpi_tracker):
        """Test trend calculation."""
        for i in range(20):
            kpi_tracker.update("total_flow", 100.0 + i)
        current = kpi_tracker.get_current("total_flow")
        assert current.trend == "up"


# =============================================================================
# Test Alarm Event Manager
# =============================================================================

class TestAlarmEventManager:
    """Test AlarmEventManager class."""

    def test_initialization(self, alarm_manager):
        """Test initialization."""
        assert len(alarm_manager._active_alarms) == 0

    def test_raise_alarm(self, alarm_manager):
        """Test raising alarm."""
        event = alarm_manager.raise_alarm(
            severity=AlarmSeverity.WARNING,
            category=AlarmCategory.FLOW,
            source="test_source",
            message="Test alarm"
        )
        assert event.event_id.startswith("ALM-")
        assert event.severity == AlarmSeverity.WARNING
        assert len(alarm_manager._active_alarms) == 1

    def test_acknowledge_alarm(self, alarm_manager):
        """Test acknowledging alarm."""
        event = alarm_manager.raise_alarm(
            severity=AlarmSeverity.ALARM,
            category=AlarmCategory.GATE,
            source="gate_1",
            message="Gate stuck"
        )
        result = alarm_manager.acknowledge(event.event_id, "operator1")
        assert result is True
        assert event.acknowledged is True
        assert event.acknowledged_by == "operator1"

    def test_clear_alarm(self, alarm_manager):
        """Test clearing alarm."""
        event = alarm_manager.raise_alarm(
            severity=AlarmSeverity.WARNING,
            category=AlarmCategory.SENSOR,
            source="sensor_1",
            message="Sensor drift"
        )
        result = alarm_manager.clear(event.event_id)
        assert result is True
        assert event.cleared is True
        assert len(alarm_manager._active_alarms) == 0

    def test_get_active(self, alarm_manager):
        """Test getting active alarms."""
        alarm_manager.raise_alarm(
            severity=AlarmSeverity.WARNING,
            category=AlarmCategory.FLOW,
            source="s1",
            message="m1"
        )
        alarm_manager.raise_alarm(
            severity=AlarmSeverity.ALARM,
            category=AlarmCategory.GATE,
            source="s2",
            message="m2"
        )
        active = alarm_manager.get_active()
        assert len(active) == 2

    def test_get_unacknowledged(self, alarm_manager):
        """Test getting unacknowledged alarms."""
        e1 = alarm_manager.raise_alarm(
            severity=AlarmSeverity.WARNING,
            category=AlarmCategory.FLOW,
            source="s1",
            message="m1"
        )
        alarm_manager.raise_alarm(
            severity=AlarmSeverity.ALARM,
            category=AlarmCategory.GATE,
            source="s2",
            message="m2"
        )
        alarm_manager.acknowledge(e1.event_id, "op1")

        unack = alarm_manager.get_unacknowledged()
        assert len(unack) == 1

    def test_get_by_severity(self, alarm_manager):
        """Test getting alarms by severity."""
        alarm_manager.raise_alarm(
            severity=AlarmSeverity.WARNING,
            category=AlarmCategory.FLOW,
            source="s1",
            message="m1"
        )
        alarm_manager.raise_alarm(
            severity=AlarmSeverity.CRITICAL,
            category=AlarmCategory.GATE,
            source="s2",
            message="m2"
        )

        warnings = alarm_manager.get_by_severity(AlarmSeverity.WARNING)
        assert len(warnings) == 1

    def test_get_statistics(self, alarm_manager):
        """Test getting alarm statistics."""
        alarm_manager.raise_alarm(
            severity=AlarmSeverity.WARNING,
            category=AlarmCategory.FLOW,
            source="s1",
            message="m1"
        )
        stats = alarm_manager.get_statistics(24)

        assert stats['total_events'] == 1
        assert stats['active_count'] == 1
        assert 'by_severity' in stats
        assert 'by_category' in stats


# =============================================================================
# Test Performance Benchmark
# =============================================================================

class TestPerformanceBenchmark:
    """Test PerformanceBenchmark class."""

    def test_initialization(self, benchmark):
        """Test initialization."""
        assert benchmark._operating_hours == 0.0
        assert benchmark._downtime_hours == 0.0

    def test_record_operation(self, benchmark):
        """Test recording operation."""
        benchmark.record_operation(hours=1.0, flow_achieved=95.0, target_flow=100.0)
        assert benchmark._operating_hours == 1.0
        assert len(benchmark._flow_achievements) == 1

    def test_record_downtime(self, benchmark):
        """Test recording downtime."""
        benchmark.record_downtime(2.0)
        assert benchmark._downtime_hours == 2.0

    def test_record_failure(self, benchmark):
        """Test recording failure."""
        benchmark.record_failure("Test failure", repair_hours=4.0)
        assert len(benchmark._failures) == 1
        assert len(benchmark._repairs) == 1

    def test_get_metrics(self, benchmark):
        """Test getting metrics."""
        benchmark.record_operation(hours=10.0, flow_achieved=95.0, target_flow=100.0)
        benchmark.record_downtime(1.0)

        metrics = benchmark.get_metrics()
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.uptime_pct < 100
        assert metrics.flow_achievement_pct > 0


# =============================================================================
# Test Report Generator
# =============================================================================

class TestReportGenerator:
    """Test ReportGenerator class."""

    @pytest.fixture
    def report_generator(self, operation_stats, kpi_tracker, alarm_manager, benchmark):
        """Create report generator."""
        return ReportGenerator(operation_stats, kpi_tracker, alarm_manager, benchmark)

    def test_initialization(self, report_generator):
        """Test initialization."""
        assert report_generator.statistics is not None
        assert report_generator.kpi_tracker is not None

    def test_generate_json_report(self, report_generator, kpi_tracker):
        """Test JSON report generation."""
        kpi_tracker.update("total_flow", 100.0)
        report = report_generator.generate(ReportType.DAILY, ReportFormat.JSON)

        data = json.loads(report)
        assert 'report_type' in data
        assert data['report_type'] == 'daily'
        assert 'performance' in data
        assert 'kpis' in data

    def test_generate_text_report(self, report_generator, kpi_tracker):
        """Test text report generation."""
        kpi_tracker.update("total_flow", 100.0)
        report = report_generator.generate(ReportType.DAILY, ReportFormat.TEXT)

        assert "DAILY REPORT" in report
        assert "PERFORMANCE METRICS" in report
        assert "KEY PERFORMANCE INDICATORS" in report

    def test_generate_html_report(self, report_generator, kpi_tracker):
        """Test HTML report generation."""
        kpi_tracker.update("total_flow", 100.0)
        report = report_generator.generate(ReportType.DAILY, ReportFormat.HTML)

        assert "<!DOCTYPE html>" in report
        assert "<table>" in report
        assert "Performance Metrics" in report

    def test_weekly_report(self, report_generator):
        """Test weekly report period."""
        report = report_generator.generate(ReportType.WEEKLY, ReportFormat.JSON)
        data = json.loads(report)
        # Weekly report should cover ~168 hours
        assert data['period']['hours'] >= 160


# =============================================================================
# Test Analytics Engine
# =============================================================================

class TestAnalyticsEngine:
    """Test AnalyticsEngine class."""

    def test_initialization(self, analytics_engine):
        """Test initialization."""
        assert analytics_engine.statistics is not None
        assert analytics_engine.kpi_tracker is not None
        assert analytics_engine.alarm_manager is not None
        assert analytics_engine.benchmark is not None

    def test_update(self, analytics_engine):
        """Test update with data."""
        result = analytics_engine.update({
            'total_flow': 100.0,
            'max_vibration': 1.5,
            'target_flow': 100.0
        })

        assert 'timestamp' in result
        assert 'kpis' in result
        assert 'total_flow' in result['kpis']

    def test_update_triggers_alarm_on_critical(self, analytics_engine):
        """Test that critical KPI triggers alarm."""
        analytics_engine.update({
            'total_flow': 200.0  # Way above critical threshold
        })

        active = analytics_engine.alarm_manager.get_active()
        assert len(active) > 0

    def test_generate_report(self, analytics_engine):
        """Test report generation."""
        analytics_engine.update({'total_flow': 100.0})
        report = analytics_engine.generate_report(ReportType.DAILY)

        data = json.loads(report)
        assert 'report_type' in data

    def test_get_dashboard_data(self, analytics_engine):
        """Test dashboard data retrieval."""
        analytics_engine.update({'total_flow': 100.0, 'max_vibration': 1.0})
        dashboard = analytics_engine.get_dashboard_data()

        assert 'timestamp' in dashboard
        assert 'system_health' in dashboard
        assert 'kpis' in dashboard
        assert 'alarms' in dashboard

    def test_get_status(self, analytics_engine):
        """Test status retrieval."""
        analytics_engine.update({'total_flow': 100.0})
        status = analytics_engine.get_status()

        assert 'kpi_count' in status
        assert 'total_alarms' in status
        assert 'uptime_pct' in status


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for analytics module."""

    def test_full_workflow(self, analytics_engine):
        """Test complete analytics workflow."""
        # Simulate operation over time
        for i in range(100):
            flow = 95.0 + (i % 20)  # Varying flow
            vibration = 0.5 + (i % 10) * 0.1

            analytics_engine.update({
                'total_flow': flow,
                'max_vibration': vibration,
                'target_flow': 100.0,
                'flow_error': abs(flow - 100.0)
            })

        # Generate reports
        json_report = analytics_engine.generate_report(ReportType.DAILY, ReportFormat.JSON)
        text_report = analytics_engine.generate_report(ReportType.DAILY, ReportFormat.TEXT)

        # Verify reports contain data
        json_data = json.loads(json_report)
        assert json_data['kpis']

        # Check dashboard
        dashboard = analytics_engine.get_dashboard_data()
        assert dashboard['kpis']

    def test_alarm_lifecycle(self, analytics_engine):
        """Test complete alarm lifecycle."""
        # Trigger alarm with critical value
        analytics_engine.update({'total_flow': 200.0})

        # Get active alarms
        active = analytics_engine.alarm_manager.get_active()
        assert len(active) > 0

        # Acknowledge alarm
        event_id = active[0].event_id
        analytics_engine.alarm_manager.acknowledge(event_id, "test_operator")

        # Verify acknowledged
        unack = analytics_engine.alarm_manager.get_unacknowledged()
        assert len(unack) == 0

        # Clear alarm
        analytics_engine.alarm_manager.clear(event_id)
        assert len(analytics_engine.alarm_manager.get_active()) == 0

    def test_kpi_trend_detection(self, analytics_engine):
        """Test KPI trend detection over time."""
        # Create increasing trend
        for i in range(30):
            analytics_engine.update({'total_flow': 90.0 + i})

        current = analytics_engine.kpi_tracker.get_current('total_flow')
        assert current.trend == "up"

        # Create decreasing trend
        for i in range(30):
            analytics_engine.update({'max_vibration': 3.0 - i * 0.05})

        current = analytics_engine.kpi_tracker.get_current('max_vibration')
        assert current.trend == "down"


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_statistics(self, operation_stats):
        """Test getting summary for non-existent metric."""
        summary = operation_stats.get_summary("nonexistent")
        assert summary is None

    def test_unknown_kpi_update(self, kpi_tracker):
        """Test updating unknown KPI."""
        result = kpi_tracker.update("unknown_kpi", 100.0)
        assert result is None

    def test_clear_nonexistent_alarm(self, alarm_manager):
        """Test clearing non-existent alarm."""
        result = alarm_manager.clear("NONEXISTENT")
        assert result is False

    def test_acknowledge_nonexistent_alarm(self, alarm_manager):
        """Test acknowledging non-existent alarm."""
        result = alarm_manager.acknowledge("NONEXISTENT", "operator")
        assert result is False

    def test_zero_target_kpi(self, kpi_tracker):
        """Test KPI with zero target (division handling)."""
        kpi_tracker.define_kpi(KPIDefinition(
            name="zero_target",
            category=KPICategory.CONTROL,
            unit="units",
            description="Zero target KPI",
            target=0.0
        ))
        result = kpi_tracker.update("zero_target", 10.0)
        assert result is not None
        assert result.deviation_pct == 0.0  # Should handle gracefully

    def test_negative_values(self, operation_stats):
        """Test handling negative values."""
        operation_stats.record("negative_test", -50.0)
        summary = operation_stats.get_summary("negative_test")
        assert summary.mean == -50.0

    def test_large_data_volume(self, operation_stats):
        """Test handling large data volume."""
        for i in range(10000):
            operation_stats.record("volume_test", float(i))

        summary = operation_stats.get_summary("volume_test")
        assert summary is not None
        assert summary.count > 0
