"""
Advanced Analytics and Reporting System.

Provides comprehensive analytics capabilities:
- Operation statistics tracking
- KPI monitoring and trending
- Alarm event management
- Automated report generation
- Performance benchmarking
"""

import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class KPICategory(Enum):
    """KPI categories for classification."""
    HYDRAULIC = "hydraulic"
    MECHANICAL = "mechanical"
    CONTROL = "control"
    SAFETY = "safety"
    EFFICIENCY = "efficiency"
    MAINTENANCE = "maintenance"


class AlarmSeverity(Enum):
    """Alarm severity levels."""
    INFO = 1
    WARNING = 2
    ALARM = 3
    CRITICAL = 4
    EMERGENCY = 5


class AlarmCategory(Enum):
    """Alarm categories."""
    FLOW = "flow"
    PRESSURE = "pressure"
    VIBRATION = "vibration"
    GATE = "gate"
    SENSOR = "sensor"
    CONTROL = "control"
    SYSTEM = "system"
    MAINTENANCE = "maintenance"


class ReportType(Enum):
    """Report types."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    INCIDENT = "incident"
    MAINTENANCE = "maintenance"
    PERFORMANCE = "performance"
    CUSTOM = "custom"


class ReportFormat(Enum):
    """Report output formats."""
    JSON = "json"
    HTML = "html"
    TEXT = "text"
    CSV = "csv"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class KPIDefinition:
    """Definition of a Key Performance Indicator."""
    name: str
    category: KPICategory
    unit: str
    description: str
    target: float
    warning_low: Optional[float] = None
    warning_high: Optional[float] = None
    critical_low: Optional[float] = None
    critical_high: Optional[float] = None
    aggregation: str = "mean"  # mean, sum, max, min, last


@dataclass
class KPIValue:
    """Current KPI value with status."""
    name: str
    value: float
    target: float
    deviation_pct: float
    status: str  # "normal", "warning", "critical"
    trend: str  # "up", "down", "stable"
    timestamp: datetime


@dataclass
class AlarmEvent:
    """Alarm event record."""
    event_id: str
    timestamp: datetime
    severity: AlarmSeverity
    category: AlarmCategory
    source: str
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    cleared: bool = False
    cleared_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StatisticsSummary:
    """Summary statistics for a metric."""
    name: str
    count: int
    mean: float
    std: float
    min: float
    max: float
    p25: float
    p50: float
    p75: float
    p95: float
    trend: float  # Slope of linear fit


@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    uptime_pct: float
    availability_pct: float
    flow_achievement_pct: float
    energy_efficiency: float
    alarm_rate: float  # Alarms per hour
    mtbf_hours: float  # Mean time between failures
    mttr_hours: float  # Mean time to repair


# =============================================================================
# Operation Statistics
# =============================================================================

class OperationStatistics:
    """
    Tracks and analyzes operational statistics.
    """

    def __init__(self, history_hours: int = 720):  # 30 days default
        """
        Initialize operation statistics tracker.

        Args:
            history_hours: Hours of history to maintain
        """
        self.history_hours = history_hours
        self._metrics: Dict[str, deque] = {}
        self._start_time: datetime = datetime.now()
        self._total_operating_hours: float = 0.0
        self._downtime_hours: float = 0.0

        logger.info("OperationStatistics initialized with %d hours history", history_hours)

    def record(self, metric_name: str, value: float, timestamp: Optional[datetime] = None) -> None:
        """Record a metric value."""
        if timestamp is None:
            timestamp = datetime.now()

        if metric_name not in self._metrics:
            # Estimate max entries based on 1-minute intervals
            max_entries = self.history_hours * 60
            self._metrics[metric_name] = deque(maxlen=max_entries)

        self._metrics[metric_name].append({
            'value': value,
            'timestamp': timestamp
        })

    def record_batch(self, data: Dict[str, float], timestamp: Optional[datetime] = None) -> None:
        """Record multiple metrics at once."""
        for name, value in data.items():
            self.record(name, value, timestamp)

    def get_summary(self, metric_name: str, hours: Optional[int] = None) -> Optional[StatisticsSummary]:
        """Get statistical summary for a metric."""
        if metric_name not in self._metrics:
            return None

        data = self._get_recent_data(metric_name, hours)
        if not data:
            return None

        values = np.array([d['value'] for d in data])

        # Calculate trend
        if len(values) > 2:
            x = np.arange(len(values))
            try:
                slope, _ = np.polyfit(x, values, 1)
            except (np.linalg.LinAlgError, ValueError):
                slope = 0.0
        else:
            slope = 0.0

        return StatisticsSummary(
            name=metric_name,
            count=len(values),
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            min=float(np.min(values)),
            max=float(np.max(values)),
            p25=float(np.percentile(values, 25)),
            p50=float(np.percentile(values, 50)),
            p75=float(np.percentile(values, 75)),
            p95=float(np.percentile(values, 95)),
            trend=float(slope)
        )

    def _get_recent_data(self, metric_name: str, hours: Optional[int] = None) -> List[Dict]:
        """Get recent data points for a metric."""
        if metric_name not in self._metrics:
            return []

        data = list(self._metrics[metric_name])
        if hours is None:
            return data

        cutoff = datetime.now() - timedelta(hours=hours)
        return [d for d in data if d['timestamp'] >= cutoff]

    def get_all_summaries(self, hours: Optional[int] = None) -> Dict[str, StatisticsSummary]:
        """Get summaries for all tracked metrics."""
        summaries = {}
        for name in self._metrics:
            summary = self.get_summary(name, hours)
            if summary:
                summaries[name] = summary
        return summaries

    def record_downtime(self, hours: float) -> None:
        """Record downtime hours."""
        self._downtime_hours += hours

    def get_uptime_pct(self) -> float:
        """Calculate uptime percentage."""
        total_hours = (datetime.now() - self._start_time).total_seconds() / 3600
        if total_hours <= 0:
            return 100.0
        return max(0.0, min(100.0, (total_hours - self._downtime_hours) / total_hours * 100))


# =============================================================================
# KPI Tracker
# =============================================================================

class KPITracker:
    """
    Tracks Key Performance Indicators with targets and thresholds.
    """

    def __init__(self):
        """Initialize KPI tracker."""
        self._definitions: Dict[str, KPIDefinition] = {}
        self._current_values: Dict[str, deque] = {}
        self._history_length = 1000

        # Initialize default KPIs
        self._init_default_kpis()

        logger.info("KPITracker initialized with %d default KPIs", len(self._definitions))

    def _init_default_kpis(self) -> None:
        """Initialize default KPI definitions."""
        default_kpis = [
            KPIDefinition(
                name="total_flow",
                category=KPICategory.HYDRAULIC,
                unit="m³/s",
                description="Total flow through all gates",
                target=100.0,
                warning_low=80.0,
                warning_high=120.0,
                critical_low=50.0,
                critical_high=150.0
            ),
            KPIDefinition(
                name="flow_error",
                category=KPICategory.CONTROL,
                unit="%",
                description="Flow tracking error percentage",
                target=0.0,
                warning_high=10.0,
                critical_high=20.0
            ),
            KPIDefinition(
                name="max_vibration",
                category=KPICategory.MECHANICAL,
                unit="m/s²",
                description="Maximum gate vibration",
                target=0.5,
                warning_high=2.0,
                critical_high=3.5
            ),
            KPIDefinition(
                name="control_response_time",
                category=KPICategory.CONTROL,
                unit="s",
                description="Control loop response time",
                target=1.0,
                warning_high=3.0,
                critical_high=5.0
            ),
            KPIDefinition(
                name="system_availability",
                category=KPICategory.EFFICIENCY,
                unit="%",
                description="System availability",
                target=99.0,
                warning_low=95.0,
                critical_low=90.0
            ),
            KPIDefinition(
                name="energy_efficiency",
                category=KPICategory.EFFICIENCY,
                unit="%",
                description="Energy efficiency index",
                target=90.0,
                warning_low=80.0,
                critical_low=70.0
            ),
            KPIDefinition(
                name="maintenance_compliance",
                category=KPICategory.MAINTENANCE,
                unit="%",
                description="Maintenance schedule compliance",
                target=100.0,
                warning_low=90.0,
                critical_low=80.0
            )
        ]

        for kpi in default_kpis:
            self.define_kpi(kpi)

    def define_kpi(self, definition: KPIDefinition) -> None:
        """Define or update a KPI."""
        self._definitions[definition.name] = definition
        if definition.name not in self._current_values:
            self._current_values[definition.name] = deque(maxlen=self._history_length)

    def update(self, name: str, value: float, timestamp: Optional[datetime] = None) -> Optional[KPIValue]:
        """Update a KPI value."""
        if name not in self._definitions:
            logger.warning("Unknown KPI: %s", name)
            return None

        if timestamp is None:
            timestamp = datetime.now()

        definition = self._definitions[name]

        # Store value
        self._current_values[name].append({
            'value': value,
            'timestamp': timestamp
        })

        # Calculate status
        status = self._calculate_status(value, definition)

        # Calculate trend
        trend = self._calculate_trend(name)

        # Calculate deviation
        deviation_pct = ((value - definition.target) / definition.target * 100
                        if definition.target != 0 else 0.0)

        return KPIValue(
            name=name,
            value=value,
            target=definition.target,
            deviation_pct=deviation_pct,
            status=status,
            trend=trend,
            timestamp=timestamp
        )

    def _calculate_status(self, value: float, definition: KPIDefinition) -> str:
        """Calculate KPI status based on thresholds."""
        # Check critical thresholds
        if definition.critical_low is not None and value < definition.critical_low:
            return "critical"
        if definition.critical_high is not None and value > definition.critical_high:
            return "critical"

        # Check warning thresholds
        if definition.warning_low is not None and value < definition.warning_low:
            return "warning"
        if definition.warning_high is not None and value > definition.warning_high:
            return "warning"

        return "normal"

    def _calculate_trend(self, name: str) -> str:
        """Calculate trend direction."""
        if name not in self._current_values:
            return "stable"

        data = list(self._current_values[name])
        if len(data) < 5:
            return "stable"

        recent_values = [d['value'] for d in data[-10:]]
        try:
            slope, _ = np.polyfit(range(len(recent_values)), recent_values, 1)
            if slope > 0.01:
                return "up"
            elif slope < -0.01:
                return "down"
        except (np.linalg.LinAlgError, ValueError):
            pass

        return "stable"

    def get_current(self, name: str) -> Optional[KPIValue]:
        """Get current KPI value."""
        if name not in self._current_values or not self._current_values[name]:
            return None

        latest = self._current_values[name][-1]
        definition = self._definitions[name]

        status = self._calculate_status(latest['value'], definition)
        trend = self._calculate_trend(name)
        deviation_pct = ((latest['value'] - definition.target) / definition.target * 100
                        if definition.target != 0 else 0.0)

        return KPIValue(
            name=name,
            value=latest['value'],
            target=definition.target,
            deviation_pct=deviation_pct,
            status=status,
            trend=trend,
            timestamp=latest['timestamp']
        )

    def get_all_current(self) -> Dict[str, KPIValue]:
        """Get all current KPI values."""
        results = {}
        for name in self._definitions:
            value = self.get_current(name)
            if value:
                results[name] = value
        return results

    def get_by_category(self, category: KPICategory) -> Dict[str, KPIValue]:
        """Get KPIs by category."""
        results = {}
        for name, definition in self._definitions.items():
            if definition.category == category:
                value = self.get_current(name)
                if value:
                    results[name] = value
        return results

    def get_alerts(self) -> List[KPIValue]:
        """Get KPIs that are in warning or critical status."""
        alerts = []
        for name in self._definitions:
            value = self.get_current(name)
            if value and value.status in ["warning", "critical"]:
                alerts.append(value)
        return alerts


# =============================================================================
# Alarm Event Manager
# =============================================================================

class AlarmEventManager:
    """
    Manages alarm events with acknowledgment and analysis.
    """

    def __init__(self, max_events: int = 10000):
        """
        Initialize alarm event manager.

        Args:
            max_events: Maximum events to keep in history
        """
        self._events: deque = deque(maxlen=max_events)
        self._active_alarms: Dict[str, AlarmEvent] = {}
        self._event_counter: int = 0

        # Statistics
        self._total_events: int = 0
        self._events_by_severity: Dict[AlarmSeverity, int] = {s: 0 for s in AlarmSeverity}
        self._events_by_category: Dict[AlarmCategory, int] = {c: 0 for c in AlarmCategory}

        logger.info("AlarmEventManager initialized")

    def raise_alarm(
        self,
        severity: AlarmSeverity,
        category: AlarmCategory,
        source: str,
        message: str,
        value: Optional[float] = None,
        threshold: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AlarmEvent:
        """Raise a new alarm event."""
        self._event_counter += 1
        event_id = f"ALM-{datetime.now().strftime('%Y%m%d')}-{self._event_counter:06d}"

        event = AlarmEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            source=source,
            message=message,
            value=value,
            threshold=threshold,
            metadata=metadata or {}
        )

        self._events.append(event)
        self._active_alarms[event_id] = event

        # Update statistics
        self._total_events += 1
        self._events_by_severity[severity] += 1
        self._events_by_category[category] += 1

        logger.warning(
            "Alarm raised: %s [%s] %s - %s",
            event_id, severity.name, source, message
        )

        return event

    def acknowledge(self, event_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alarm."""
        if event_id not in self._active_alarms:
            return False

        event = self._active_alarms[event_id]
        event.acknowledged = True
        event.acknowledged_by = acknowledged_by
        event.acknowledged_at = datetime.now()

        logger.info("Alarm acknowledged: %s by %s", event_id, acknowledged_by)
        return True

    def clear(self, event_id: str) -> bool:
        """Clear an alarm."""
        if event_id not in self._active_alarms:
            return False

        event = self._active_alarms[event_id]
        event.cleared = True
        event.cleared_at = datetime.now()
        event.duration_seconds = (event.cleared_at - event.timestamp).total_seconds()

        del self._active_alarms[event_id]

        logger.info("Alarm cleared: %s (duration: %.1fs)", event_id, event.duration_seconds)
        return True

    def get_active(self) -> List[AlarmEvent]:
        """Get all active (uncleared) alarms."""
        return list(self._active_alarms.values())

    def get_unacknowledged(self) -> List[AlarmEvent]:
        """Get unacknowledged active alarms."""
        return [e for e in self._active_alarms.values() if not e.acknowledged]

    def get_by_severity(self, severity: AlarmSeverity) -> List[AlarmEvent]:
        """Get alarms by severity."""
        return [e for e in self._events if e.severity == severity]

    def get_by_category(self, category: AlarmCategory) -> List[AlarmEvent]:
        """Get alarms by category."""
        return [e for e in self._events if e.category == category]

    def get_recent(self, hours: int = 24) -> List[AlarmEvent]:
        """Get recent alarms."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [e for e in self._events if e.timestamp >= cutoff]

    def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get alarm statistics."""
        recent = self.get_recent(hours)

        return {
            'total_events': self._total_events,
            'active_count': len(self._active_alarms),
            'unacknowledged_count': len(self.get_unacknowledged()),
            'recent_count': len(recent),
            'by_severity': {s.name: self._events_by_severity[s] for s in AlarmSeverity},
            'by_category': {c.name: self._events_by_category[c] for c in AlarmCategory},
            'alarm_rate_per_hour': len(recent) / hours if hours > 0 else 0
        }


# =============================================================================
# Performance Benchmark
# =============================================================================

class PerformanceBenchmark:
    """
    Calculates and tracks system performance metrics.
    """

    def __init__(self):
        """Initialize performance benchmark."""
        self._start_time: datetime = datetime.now()
        self._operating_hours: float = 0.0
        self._downtime_hours: float = 0.0
        self._failures: List[Dict] = []
        self._repairs: List[Dict] = []
        self._flow_achievements: deque = deque(maxlen=10000)
        self._energy_readings: deque = deque(maxlen=10000)

        logger.info("PerformanceBenchmark initialized")

    def record_operation(self, hours: float, flow_achieved: float, target_flow: float) -> None:
        """Record operation data."""
        self._operating_hours += hours
        achievement = (flow_achieved / target_flow * 100) if target_flow > 0 else 0
        self._flow_achievements.append({
            'timestamp': datetime.now(),
            'achievement': achievement
        })

    def record_downtime(self, hours: float) -> None:
        """Record downtime."""
        self._downtime_hours += hours

    def record_failure(self, description: str, repair_hours: float) -> None:
        """Record a failure event."""
        self._failures.append({
            'timestamp': datetime.now(),
            'description': description
        })
        self._repairs.append({
            'timestamp': datetime.now(),
            'hours': repair_hours
        })

    def record_energy(self, consumption: float, output: float) -> None:
        """Record energy data."""
        efficiency = (output / consumption * 100) if consumption > 0 else 0
        self._energy_readings.append({
            'timestamp': datetime.now(),
            'efficiency': efficiency
        })

    def get_metrics(self) -> PerformanceMetrics:
        """Calculate current performance metrics."""
        total_hours = self._operating_hours + self._downtime_hours

        # Uptime and availability
        uptime_pct = (self._operating_hours / total_hours * 100) if total_hours > 0 else 100
        availability_pct = uptime_pct  # Simplified

        # Flow achievement
        if self._flow_achievements:
            flow_achievement = np.mean([a['achievement'] for a in self._flow_achievements])
        else:
            flow_achievement = 100.0

        # Energy efficiency
        if self._energy_readings:
            energy_efficiency = np.mean([e['efficiency'] for e in self._energy_readings])
        else:
            energy_efficiency = 90.0

        # MTBF/MTTR
        if len(self._failures) > 1:
            mtbf = self._operating_hours / len(self._failures)
        else:
            mtbf = self._operating_hours if self._operating_hours > 0 else 1000.0

        if self._repairs:
            mttr = np.mean([r['hours'] for r in self._repairs])
        else:
            mttr = 0.0

        return PerformanceMetrics(
            uptime_pct=uptime_pct,
            availability_pct=availability_pct,
            flow_achievement_pct=flow_achievement,
            energy_efficiency=energy_efficiency,
            alarm_rate=0.0,  # Set by alarm manager
            mtbf_hours=mtbf,
            mttr_hours=mttr
        )


# =============================================================================
# Report Generator
# =============================================================================

class ReportGenerator:
    """
    Generates formatted reports from analytics data.
    """

    def __init__(
        self,
        statistics: OperationStatistics,
        kpi_tracker: KPITracker,
        alarm_manager: AlarmEventManager,
        benchmark: PerformanceBenchmark
    ):
        """
        Initialize report generator.

        Args:
            statistics: Operation statistics tracker
            kpi_tracker: KPI tracker
            alarm_manager: Alarm event manager
            benchmark: Performance benchmark
        """
        self.statistics = statistics
        self.kpi_tracker = kpi_tracker
        self.alarm_manager = alarm_manager
        self.benchmark = benchmark

        logger.info("ReportGenerator initialized")

    def generate(
        self,
        report_type: ReportType,
        format: ReportFormat = ReportFormat.JSON,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> str:
        """Generate a report."""
        if end_time is None:
            end_time = datetime.now()

        if start_time is None:
            if report_type == ReportType.DAILY:
                start_time = end_time - timedelta(days=1)
            elif report_type == ReportType.WEEKLY:
                start_time = end_time - timedelta(weeks=1)
            elif report_type == ReportType.MONTHLY:
                start_time = end_time - timedelta(days=30)
            else:
                start_time = end_time - timedelta(days=1)

        hours = (end_time - start_time).total_seconds() / 3600

        # Gather report data
        report_data = self._gather_report_data(report_type, hours, start_time, end_time)

        # Format output
        if format == ReportFormat.JSON:
            return self._format_json(report_data)
        elif format == ReportFormat.TEXT:
            return self._format_text(report_data)
        elif format == ReportFormat.HTML:
            return self._format_html(report_data)
        else:
            return self._format_json(report_data)

    def _gather_report_data(
        self,
        report_type: ReportType,
        hours: float,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Gather data for report."""
        # Performance metrics
        perf_metrics = self.benchmark.get_metrics()

        # KPI summary
        kpi_values = self.kpi_tracker.get_all_current()
        kpi_alerts = self.kpi_tracker.get_alerts()

        # Alarm statistics
        alarm_stats = self.alarm_manager.get_statistics(int(hours))

        # Operation statistics
        op_summaries = self.statistics.get_all_summaries(int(hours))

        return {
            'report_type': report_type.value,
            'generated_at': datetime.now().isoformat(),
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'hours': hours
            },
            'performance': {
                'uptime_pct': perf_metrics.uptime_pct,
                'availability_pct': perf_metrics.availability_pct,
                'flow_achievement_pct': perf_metrics.flow_achievement_pct,
                'energy_efficiency': perf_metrics.energy_efficiency,
                'mtbf_hours': perf_metrics.mtbf_hours,
                'mttr_hours': perf_metrics.mttr_hours
            },
            'kpis': {
                name: {
                    'value': v.value,
                    'target': v.target,
                    'deviation_pct': v.deviation_pct,
                    'status': v.status,
                    'trend': v.trend
                }
                for name, v in kpi_values.items()
            },
            'kpi_alerts': [
                {'name': a.name, 'value': a.value, 'status': a.status}
                for a in kpi_alerts
            ],
            'alarms': alarm_stats,
            'statistics': {
                name: {
                    'mean': s.mean,
                    'std': s.std,
                    'min': s.min,
                    'max': s.max,
                    'trend': s.trend
                }
                for name, s in op_summaries.items()
            }
        }

    def _format_json(self, data: Dict[str, Any]) -> str:
        """Format report as JSON."""
        return json.dumps(data, indent=2, default=str)

    def _format_text(self, data: Dict[str, Any]) -> str:
        """Format report as plain text."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"  {data['report_type'].upper()} REPORT")
        lines.append(f"  Generated: {data['generated_at']}")
        lines.append("=" * 60)
        lines.append("")

        # Period
        lines.append(f"Period: {data['period']['start']} to {data['period']['end']}")
        lines.append(f"Duration: {data['period']['hours']:.1f} hours")
        lines.append("")

        # Performance
        lines.append("-" * 40)
        lines.append("PERFORMANCE METRICS")
        lines.append("-" * 40)
        perf = data['performance']
        lines.append(f"  Uptime:            {perf['uptime_pct']:.1f}%")
        lines.append(f"  Availability:      {perf['availability_pct']:.1f}%")
        lines.append(f"  Flow Achievement:  {perf['flow_achievement_pct']:.1f}%")
        lines.append(f"  Energy Efficiency: {perf['energy_efficiency']:.1f}%")
        lines.append(f"  MTBF:              {perf['mtbf_hours']:.1f} hours")
        lines.append(f"  MTTR:              {perf['mttr_hours']:.1f} hours")
        lines.append("")

        # KPIs
        lines.append("-" * 40)
        lines.append("KEY PERFORMANCE INDICATORS")
        lines.append("-" * 40)
        for name, kpi in data['kpis'].items():
            status_marker = "✓" if kpi['status'] == 'normal' else "⚠" if kpi['status'] == 'warning' else "✗"
            lines.append(f"  {status_marker} {name}: {kpi['value']:.2f} (target: {kpi['target']:.2f})")
        lines.append("")

        # Alarms
        lines.append("-" * 40)
        lines.append("ALARM SUMMARY")
        lines.append("-" * 40)
        alarms = data['alarms']
        lines.append(f"  Total Events:      {alarms['total_events']}")
        lines.append(f"  Active Alarms:     {alarms['active_count']}")
        lines.append(f"  Unacknowledged:    {alarms['unacknowledged_count']}")
        lines.append(f"  Alarm Rate:        {alarms['alarm_rate_per_hour']:.2f}/hour")
        lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    def _format_html(self, data: Dict[str, Any]) -> str:
        """Format report as HTML."""
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html><head>")
        html.append("<title>System Report</title>")
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        html.append("h1 { color: #333; }")
        html.append("h2 { color: #666; border-bottom: 1px solid #ccc; }")
        html.append("table { border-collapse: collapse; width: 100%; margin: 10px 0; }")
        html.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("th { background-color: #4CAF50; color: white; }")
        html.append(".normal { color: green; }")
        html.append(".warning { color: orange; }")
        html.append(".critical { color: red; }")
        html.append("</style>")
        html.append("</head><body>")

        html.append(f"<h1>{data['report_type'].upper()} Report</h1>")
        html.append(f"<p>Generated: {data['generated_at']}</p>")
        html.append(f"<p>Period: {data['period']['start']} to {data['period']['end']}</p>")

        # Performance
        html.append("<h2>Performance Metrics</h2>")
        html.append("<table>")
        html.append("<tr><th>Metric</th><th>Value</th></tr>")
        perf = data['performance']
        html.append(f"<tr><td>Uptime</td><td>{perf['uptime_pct']:.1f}%</td></tr>")
        html.append(f"<tr><td>Availability</td><td>{perf['availability_pct']:.1f}%</td></tr>")
        html.append(f"<tr><td>Flow Achievement</td><td>{perf['flow_achievement_pct']:.1f}%</td></tr>")
        html.append(f"<tr><td>Energy Efficiency</td><td>{perf['energy_efficiency']:.1f}%</td></tr>")
        html.append("</table>")

        # KPIs
        html.append("<h2>Key Performance Indicators</h2>")
        html.append("<table>")
        html.append("<tr><th>KPI</th><th>Value</th><th>Target</th><th>Status</th></tr>")
        for name, kpi in data['kpis'].items():
            html.append(f"<tr><td>{name}</td><td>{kpi['value']:.2f}</td>")
            html.append(f"<td>{kpi['target']:.2f}</td>")
            html.append(f"<td class='{kpi['status']}'>{kpi['status']}</td></tr>")
        html.append("</table>")

        html.append("</body></html>")

        return "\n".join(html)


# =============================================================================
# Analytics Engine (Main Interface)
# =============================================================================

class AnalyticsEngine:
    """
    Integrated analytics engine for the Tanghe Inverted Siphon.

    Provides unified access to all analytics capabilities.
    """

    def __init__(self, history_hours: int = 720):
        """
        Initialize analytics engine.

        Args:
            history_hours: Hours of history to maintain
        """
        self.statistics = OperationStatistics(history_hours)
        self.kpi_tracker = KPITracker()
        self.alarm_manager = AlarmEventManager()
        self.benchmark = PerformanceBenchmark()
        self.report_generator = ReportGenerator(
            self.statistics,
            self.kpi_tracker,
            self.alarm_manager,
            self.benchmark
        )

        logger.info("AnalyticsEngine initialized")

    def update(self, data: Dict[str, Any], timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Update analytics with new data.

        Args:
            data: Dictionary of metric values
            timestamp: Optional timestamp

        Returns:
            Update result with KPI statuses and any alerts
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Record statistics
        self.statistics.record_batch(data, timestamp)

        # Update KPIs
        kpi_results = {}
        for name, value in data.items():
            kpi_value = self.kpi_tracker.update(name, value, timestamp)
            if kpi_value:
                kpi_results[name] = {
                    'value': kpi_value.value,
                    'status': kpi_value.status,
                    'trend': kpi_value.trend
                }

                # Raise alarm for critical KPIs
                if kpi_value.status == "critical":
                    self.alarm_manager.raise_alarm(
                        severity=AlarmSeverity.CRITICAL,
                        category=AlarmCategory.CONTROL,
                        source=name,
                        message=f"KPI {name} at critical level: {kpi_value.value:.2f}",
                        value=kpi_value.value,
                        threshold=self.kpi_tracker._definitions[name].critical_high or
                                  self.kpi_tracker._definitions[name].critical_low
                    )

        # Update benchmark
        if 'total_flow' in data and 'target_flow' in data:
            self.benchmark.record_operation(
                hours=0.1,  # Assume 6-minute intervals
                flow_achieved=data['total_flow'],
                target_flow=data['target_flow']
            )

        return {
            'timestamp': timestamp.isoformat(),
            'kpis': kpi_results,
            'active_alarms': len(self.alarm_manager.get_active()),
            'alerts': [
                {'name': a.name, 'status': a.status}
                for a in self.kpi_tracker.get_alerts()
            ]
        }

    def generate_report(
        self,
        report_type: ReportType = ReportType.DAILY,
        format: ReportFormat = ReportFormat.JSON
    ) -> str:
        """Generate a report."""
        return self.report_generator.generate(report_type, format)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard display."""
        kpis = self.kpi_tracker.get_all_current()
        alarms = self.alarm_manager.get_statistics(24)
        metrics = self.benchmark.get_metrics()

        return {
            'timestamp': datetime.now().isoformat(),
            'system_health': {
                'uptime': metrics.uptime_pct,
                'availability': metrics.availability_pct,
                'performance': metrics.flow_achievement_pct
            },
            'kpis': {
                name: {
                    'value': v.value,
                    'target': v.target,
                    'status': v.status,
                    'trend': v.trend
                }
                for name, v in kpis.items()
            },
            'alarms': {
                'active': alarms['active_count'],
                'unacknowledged': alarms['unacknowledged_count'],
                'rate': alarms['alarm_rate_per_hour']
            },
            'alerts': [
                {'name': a.name, 'status': a.status, 'value': a.value}
                for a in self.kpi_tracker.get_alerts()
            ]
        }

    def get_status(self) -> Dict[str, Any]:
        """Get analytics engine status."""
        return {
            'statistics_metrics': len(self.statistics._metrics),
            'kpi_count': len(self.kpi_tracker._definitions),
            'total_alarms': self.alarm_manager._total_events,
            'active_alarms': len(self.alarm_manager.get_active()),
            'uptime_pct': self.statistics.get_uptime_pct()
        }
