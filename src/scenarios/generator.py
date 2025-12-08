"""
Simulation Scenario Generator.

Provides automated scenario generation for testing and validation
of the Tanghe Inverted Siphon digital twin system.
"""

import logging
import uuid
import random
import copy
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class ScenarioCategory(Enum):
    """Categories of simulation scenarios."""
    NORMAL = "normal"
    FLOOD = "flood"
    DROUGHT = "drought"
    EQUIPMENT_FAILURE = "equipment_failure"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"
    CALIBRATION = "calibration"
    STRESS_TEST = "stress_test"
    OPTIMIZATION = "optimization"
    CUSTOM = "custom"


class ScenarioSeverity(Enum):
    """Severity levels for scenarios."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ParameterDistribution(Enum):
    """Types of parameter distributions."""
    UNIFORM = "uniform"
    NORMAL = "normal"
    TRIANGULAR = "triangular"
    EXPONENTIAL = "exponential"
    CONSTANT = "constant"


class EventTrigger(Enum):
    """Types of event triggers."""
    TIME = "time"
    CONDITION = "condition"
    RANDOM = "random"
    PERIODIC = "periodic"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ParameterRange:
    """
    Defines a range for scenario parameters.

    Attributes:
        name: Parameter name
        min_value: Minimum value
        max_value: Maximum value
        default_value: Default value
        distribution: Distribution type for randomization
        unit: Unit of measurement
    """
    name: str
    min_value: float
    max_value: float
    default_value: Optional[float] = None
    distribution: ParameterDistribution = ParameterDistribution.UNIFORM
    unit: str = ""

    def __post_init__(self):
        if self.default_value is None:
            self.default_value = (self.min_value + self.max_value) / 2

    def sample(self, rng: Optional[np.random.Generator] = None) -> float:
        """Sample a value from the distribution."""
        if rng is None:
            rng = np.random.default_rng()

        if self.distribution == ParameterDistribution.CONSTANT:
            return self.default_value
        elif self.distribution == ParameterDistribution.UNIFORM:
            return rng.uniform(self.min_value, self.max_value)
        elif self.distribution == ParameterDistribution.NORMAL:
            mean = self.default_value
            std = (self.max_value - self.min_value) / 6  # 99.7% within range
            value = rng.normal(mean, std)
            return np.clip(value, self.min_value, self.max_value)
        elif self.distribution == ParameterDistribution.TRIANGULAR:
            return rng.triangular(self.min_value, self.default_value, self.max_value)
        elif self.distribution == ParameterDistribution.EXPONENTIAL:
            scale = (self.max_value - self.min_value) / 3
            value = self.min_value + rng.exponential(scale)
            return min(value, self.max_value)
        else:
            return self.default_value


@dataclass
class ScenarioEvent:
    """
    An event that occurs during a scenario.

    Attributes:
        event_id: Unique event identifier
        name: Event name
        trigger_type: How the event is triggered
        trigger_time: Time offset for time-triggered events (seconds)
        trigger_condition: Condition for condition-triggered events
        action: Action to perform
        parameters: Event parameters
        duration: Event duration (seconds)
        repeat: Whether the event repeats
        repeat_interval: Interval for repeating events
    """
    event_id: str
    name: str
    trigger_type: EventTrigger
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    trigger_time: float = 0.0
    trigger_condition: Optional[str] = None
    duration: float = 0.0
    repeat: bool = False
    repeat_interval: float = 0.0

    @classmethod
    def create(
        cls,
        name: str,
        action: str,
        trigger_time: float = 0.0,
        **kwargs
    ) -> 'ScenarioEvent':
        """Create a time-triggered event."""
        return cls(
            event_id=str(uuid.uuid4())[:8],
            name=name,
            trigger_type=EventTrigger.TIME,
            trigger_time=trigger_time,
            action=action,
            **kwargs
        )


@dataclass
class ScenarioTimeline:
    """
    Timeline of events for a scenario.

    Attributes:
        events: List of events in the timeline
        total_duration: Total scenario duration (seconds)
    """
    events: List[ScenarioEvent] = field(default_factory=list)
    total_duration: float = 3600.0  # 1 hour default

    def add_event(self, event: ScenarioEvent) -> None:
        """Add an event to the timeline."""
        self.events.append(event)
        self.events.sort(key=lambda e: e.trigger_time)

    def get_events_at_time(self, time: float, tolerance: float = 0.1) -> List[ScenarioEvent]:
        """Get events that should trigger at a given time."""
        triggered = []
        for event in self.events:
            if event.trigger_type == EventTrigger.TIME:
                if abs(event.trigger_time - time) <= tolerance:
                    triggered.append(event)
            elif event.trigger_type == EventTrigger.PERIODIC:
                if event.repeat_interval > 0:
                    elapsed = time - event.trigger_time
                    if elapsed >= 0 and elapsed % event.repeat_interval <= tolerance:
                        triggered.append(event)
        return triggered

    def get_active_events(self, time: float) -> List[ScenarioEvent]:
        """Get events that are currently active (started but not ended)."""
        active = []
        for event in self.events:
            start = event.trigger_time
            end = start + event.duration
            if start <= time < end:
                active.append(event)
        return active


@dataclass
class ScenarioTemplate:
    """
    Template for generating scenarios.

    Attributes:
        template_id: Unique template identifier
        name: Template name
        category: Scenario category
        severity: Default severity level
        description: Template description
        parameters: Parameter ranges for the scenario
        base_timeline: Base timeline of events
        variations: Possible variations
    """
    template_id: str
    name: str
    category: ScenarioCategory
    severity: ScenarioSeverity = ScenarioSeverity.MEDIUM
    description: str = ""
    parameters: Dict[str, ParameterRange] = field(default_factory=dict)
    base_timeline: ScenarioTimeline = field(default_factory=ScenarioTimeline)
    variations: List[str] = field(default_factory=list)

    def add_parameter(self, param: ParameterRange) -> None:
        """Add a parameter range."""
        self.parameters[param.name] = param

    def generate_parameters(
        self,
        rng: Optional[np.random.Generator] = None,
        overrides: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Generate random parameters from ranges."""
        params = {}
        for name, param_range in self.parameters.items():
            if overrides and name in overrides:
                params[name] = overrides[name]
            else:
                params[name] = param_range.sample(rng)
        return params


@dataclass
class ScenarioResult:
    """
    Result of scenario execution.

    Attributes:
        scenario_id: Scenario identifier
        template_id: Source template identifier
        start_time: Start timestamp
        end_time: End timestamp
        parameters: Parameters used
        metrics: Collected metrics
        events_executed: Events that were executed
        success: Whether scenario completed successfully
        errors: Any errors encountered
    """
    scenario_id: str
    template_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    parameters: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    events_executed: List[str] = field(default_factory=list)
    success: bool = True
    errors: List[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Get scenario duration in seconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

    def add_metric(self, name: str, value: float) -> None:
        """Add a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get_metric_summary(self, name: str) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return {}

        values = np.array(self.metrics[name])
        return {
            'count': len(values),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'scenario_id': self.scenario_id,
            'template_id': self.template_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'parameters': self.parameters,
            'metrics_summary': {
                name: self.get_metric_summary(name)
                for name in self.metrics
            },
            'events_executed': self.events_executed,
            'success': self.success,
            'errors': self.errors
        }


# =============================================================================
# Scenario Library
# =============================================================================

class ScenarioLibrary:
    """
    Library of predefined scenario templates.
    """

    def __init__(self):
        """Initialize scenario library."""
        self._templates: Dict[str, ScenarioTemplate] = {}
        self._load_default_templates()
        logger.info("ScenarioLibrary initialized with %d templates", len(self._templates))

    def _load_default_templates(self) -> None:
        """Load default scenario templates."""
        # Normal operation template
        self._add_normal_operation()

        # Flood scenarios
        self._add_flood_scenarios()

        # Drought scenarios
        self._add_drought_scenarios()

        # Equipment failure scenarios
        self._add_equipment_failure_scenarios()

        # Emergency scenarios
        self._add_emergency_scenarios()

        # Maintenance scenarios
        self._add_maintenance_scenarios()

        # Stress test scenarios
        self._add_stress_test_scenarios()

    def _add_normal_operation(self) -> None:
        """Add normal operation template."""
        template = ScenarioTemplate(
            template_id="normal_operation",
            name="Normal Operation",
            category=ScenarioCategory.NORMAL,
            severity=ScenarioSeverity.LOW,
            description="Standard operating conditions with typical flow variations"
        )

        template.add_parameter(ParameterRange(
            name="target_flow",
            min_value=100.0,
            max_value=200.0,
            default_value=150.0,
            distribution=ParameterDistribution.NORMAL,
            unit="m³/s"
        ))
        template.add_parameter(ParameterRange(
            name="head_upstream",
            min_value=28.0,
            max_value=32.0,
            default_value=30.0,
            distribution=ParameterDistribution.NORMAL,
            unit="m"
        ))
        template.add_parameter(ParameterRange(
            name="duration",
            min_value=1800.0,
            max_value=7200.0,
            default_value=3600.0,
            distribution=ParameterDistribution.UNIFORM,
            unit="s"
        ))

        self._templates[template.template_id] = template

    def _add_flood_scenarios(self) -> None:
        """Add flood scenario templates."""
        # Minor flood
        minor = ScenarioTemplate(
            template_id="flood_minor",
            name="Minor Flood Event",
            category=ScenarioCategory.FLOOD,
            severity=ScenarioSeverity.MEDIUM,
            description="Minor flood with moderate flow increase"
        )
        minor.add_parameter(ParameterRange(
            name="peak_flow",
            min_value=250.0,
            max_value=350.0,
            default_value=300.0,
            unit="m³/s"
        ))
        minor.add_parameter(ParameterRange(
            name="rise_time",
            min_value=300.0,
            max_value=900.0,
            default_value=600.0,
            unit="s"
        ))
        minor.add_parameter(ParameterRange(
            name="peak_duration",
            min_value=600.0,
            max_value=1800.0,
            default_value=1200.0,
            unit="s"
        ))

        timeline = ScenarioTimeline(total_duration=3600.0)
        timeline.add_event(ScenarioEvent.create(
            name="flood_start",
            action="increase_upstream_head",
            trigger_time=0.0,
            parameters={'rate': 0.01}
        ))
        timeline.add_event(ScenarioEvent.create(
            name="flood_peak",
            action="maintain_high_flow",
            trigger_time=600.0,
            duration=1200.0
        ))
        timeline.add_event(ScenarioEvent.create(
            name="flood_recede",
            action="decrease_upstream_head",
            trigger_time=1800.0,
            parameters={'rate': 0.005}
        ))
        minor.base_timeline = timeline

        self._templates[minor.template_id] = minor

        # Major flood
        major = ScenarioTemplate(
            template_id="flood_major",
            name="Major Flood Event",
            category=ScenarioCategory.FLOOD,
            severity=ScenarioSeverity.HIGH,
            description="Major flood with significant flow increase"
        )
        major.add_parameter(ParameterRange(
            name="peak_flow",
            min_value=350.0,
            max_value=450.0,
            default_value=400.0,
            unit="m³/s"
        ))
        major.add_parameter(ParameterRange(
            name="rise_time",
            min_value=180.0,
            max_value=600.0,
            default_value=300.0,
            unit="s"
        ))

        self._templates[major.template_id] = major

        # Flash flood
        flash = ScenarioTemplate(
            template_id="flood_flash",
            name="Flash Flood",
            category=ScenarioCategory.FLOOD,
            severity=ScenarioSeverity.CRITICAL,
            description="Rapid onset flash flood requiring emergency response"
        )
        flash.add_parameter(ParameterRange(
            name="peak_flow",
            min_value=400.0,
            max_value=500.0,
            default_value=450.0,
            unit="m³/s"
        ))
        flash.add_parameter(ParameterRange(
            name="rise_time",
            min_value=60.0,
            max_value=180.0,
            default_value=120.0,
            unit="s"
        ))

        self._templates[flash.template_id] = flash

    def _add_drought_scenarios(self) -> None:
        """Add drought scenario templates."""
        # Low flow
        low_flow = ScenarioTemplate(
            template_id="drought_low_flow",
            name="Low Flow Conditions",
            category=ScenarioCategory.DROUGHT,
            severity=ScenarioSeverity.MEDIUM,
            description="Extended period of below-normal flow"
        )
        low_flow.add_parameter(ParameterRange(
            name="min_flow",
            min_value=30.0,
            max_value=70.0,
            default_value=50.0,
            unit="m³/s"
        ))
        low_flow.add_parameter(ParameterRange(
            name="duration",
            min_value=3600.0,
            max_value=14400.0,
            default_value=7200.0,
            unit="s"
        ))

        self._templates[low_flow.template_id] = low_flow

        # Severe drought
        severe = ScenarioTemplate(
            template_id="drought_severe",
            name="Severe Drought",
            category=ScenarioCategory.DROUGHT,
            severity=ScenarioSeverity.HIGH,
            description="Severe drought with minimum flow requirements"
        )
        severe.add_parameter(ParameterRange(
            name="min_flow",
            min_value=10.0,
            max_value=30.0,
            default_value=20.0,
            unit="m³/s"
        ))

        self._templates[severe.template_id] = severe

    def _add_equipment_failure_scenarios(self) -> None:
        """Add equipment failure scenario templates."""
        # Gate stuck
        gate_stuck = ScenarioTemplate(
            template_id="failure_gate_stuck",
            name="Gate Stuck Failure",
            category=ScenarioCategory.EQUIPMENT_FAILURE,
            severity=ScenarioSeverity.HIGH,
            description="Single gate becomes stuck in current position"
        )
        gate_stuck.add_parameter(ParameterRange(
            name="gate_index",
            min_value=0,
            max_value=2,
            default_value=0,
            distribution=ParameterDistribution.UNIFORM
        ))
        gate_stuck.add_parameter(ParameterRange(
            name="failure_time",
            min_value=60.0,
            max_value=600.0,
            default_value=300.0,
            unit="s"
        ))
        gate_stuck.add_parameter(ParameterRange(
            name="repair_time",
            min_value=300.0,
            max_value=1800.0,
            default_value=900.0,
            unit="s"
        ))

        timeline = ScenarioTimeline(total_duration=3600.0)
        timeline.add_event(ScenarioEvent.create(
            name="gate_failure",
            action="inject_fault",
            trigger_time=300.0,
            parameters={'fault_type': 'stuck'}
        ))
        timeline.add_event(ScenarioEvent.create(
            name="gate_repair",
            action="clear_fault",
            trigger_time=1200.0
        ))
        gate_stuck.base_timeline = timeline

        self._templates[gate_stuck.template_id] = gate_stuck

        # Sensor failure
        sensor_fail = ScenarioTemplate(
            template_id="failure_sensor",
            name="Sensor Failure",
            category=ScenarioCategory.EQUIPMENT_FAILURE,
            severity=ScenarioSeverity.MEDIUM,
            description="Sensor provides incorrect readings"
        )
        sensor_fail.add_parameter(ParameterRange(
            name="sensor_type",
            min_value=0,
            max_value=3,
            default_value=0
        ))
        sensor_fail.add_parameter(ParameterRange(
            name="error_magnitude",
            min_value=0.1,
            max_value=0.5,
            default_value=0.2
        ))

        self._templates[sensor_fail.template_id] = sensor_fail

        # Multiple failures
        multi_fail = ScenarioTemplate(
            template_id="failure_multiple",
            name="Multiple Equipment Failures",
            category=ScenarioCategory.EQUIPMENT_FAILURE,
            severity=ScenarioSeverity.CRITICAL,
            description="Multiple simultaneous equipment failures"
        )
        multi_fail.add_parameter(ParameterRange(
            name="num_failures",
            min_value=2,
            max_value=3,
            default_value=2
        ))

        self._templates[multi_fail.template_id] = multi_fail

    def _add_emergency_scenarios(self) -> None:
        """Add emergency scenario templates."""
        # Emergency shutdown
        shutdown = ScenarioTemplate(
            template_id="emergency_shutdown",
            name="Emergency Shutdown",
            category=ScenarioCategory.EMERGENCY,
            severity=ScenarioSeverity.CRITICAL,
            description="Emergency shutdown procedure test"
        )
        shutdown.add_parameter(ParameterRange(
            name="shutdown_rate",
            min_value=0.1,
            max_value=0.3,
            default_value=0.2,
            unit="1/s"
        ))

        timeline = ScenarioTimeline(total_duration=600.0)
        timeline.add_event(ScenarioEvent.create(
            name="trigger_emergency",
            action="emergency_stop",
            trigger_time=60.0
        ))
        timeline.add_event(ScenarioEvent.create(
            name="recovery",
            action="reset_emergency",
            trigger_time=300.0
        ))
        shutdown.base_timeline = timeline

        self._templates[shutdown.template_id] = shutdown

        # Rapid flow change
        rapid = ScenarioTemplate(
            template_id="emergency_rapid_change",
            name="Rapid Flow Change",
            category=ScenarioCategory.EMERGENCY,
            severity=ScenarioSeverity.HIGH,
            description="Sudden large change in flow requirements"
        )
        rapid.add_parameter(ParameterRange(
            name="flow_change",
            min_value=100.0,
            max_value=200.0,
            default_value=150.0,
            unit="m³/s"
        ))
        rapid.add_parameter(ParameterRange(
            name="change_time",
            min_value=30.0,
            max_value=120.0,
            default_value=60.0,
            unit="s"
        ))

        self._templates[rapid.template_id] = rapid

    def _add_maintenance_scenarios(self) -> None:
        """Add maintenance scenario templates."""
        # Scheduled maintenance
        scheduled = ScenarioTemplate(
            template_id="maintenance_scheduled",
            name="Scheduled Maintenance",
            category=ScenarioCategory.MAINTENANCE,
            severity=ScenarioSeverity.LOW,
            description="Planned gate maintenance with reduced capacity"
        )
        scheduled.add_parameter(ParameterRange(
            name="gate_index",
            min_value=0,
            max_value=2,
            default_value=0
        ))
        scheduled.add_parameter(ParameterRange(
            name="maintenance_duration",
            min_value=1800.0,
            max_value=7200.0,
            default_value=3600.0,
            unit="s"
        ))

        self._templates[scheduled.template_id] = scheduled

        # Calibration
        calibration = ScenarioTemplate(
            template_id="maintenance_calibration",
            name="Sensor Calibration",
            category=ScenarioCategory.CALIBRATION,
            severity=ScenarioSeverity.LOW,
            description="Sensor calibration procedure"
        )
        calibration.add_parameter(ParameterRange(
            name="calibration_points",
            min_value=3,
            max_value=10,
            default_value=5
        ))

        self._templates[calibration.template_id] = calibration

    def _add_stress_test_scenarios(self) -> None:
        """Add stress test scenario templates."""
        # High frequency changes
        high_freq = ScenarioTemplate(
            template_id="stress_high_frequency",
            name="High Frequency Flow Changes",
            category=ScenarioCategory.STRESS_TEST,
            severity=ScenarioSeverity.MEDIUM,
            description="Rapid oscillating flow setpoint changes"
        )
        high_freq.add_parameter(ParameterRange(
            name="oscillation_period",
            min_value=30.0,
            max_value=120.0,
            default_value=60.0,
            unit="s"
        ))
        high_freq.add_parameter(ParameterRange(
            name="amplitude",
            min_value=20.0,
            max_value=50.0,
            default_value=30.0,
            unit="m³/s"
        ))

        timeline = ScenarioTimeline(total_duration=1800.0)
        timeline.add_event(ScenarioEvent.create(
            name="oscillation",
            action="oscillate_target",
            trigger_time=0.0,
            repeat=True,
            repeat_interval=60.0
        ))
        high_freq.base_timeline = timeline

        self._templates[high_freq.template_id] = high_freq

        # Extreme conditions
        extreme = ScenarioTemplate(
            template_id="stress_extreme",
            name="Extreme Operating Conditions",
            category=ScenarioCategory.STRESS_TEST,
            severity=ScenarioSeverity.HIGH,
            description="System operation at design limits"
        )
        extreme.add_parameter(ParameterRange(
            name="max_flow",
            min_value=400.0,
            max_value=500.0,
            default_value=450.0,
            unit="m³/s"
        ))
        extreme.add_parameter(ParameterRange(
            name="max_head_diff",
            min_value=8.0,
            max_value=12.0,
            default_value=10.0,
            unit="m"
        ))

        self._templates[extreme.template_id] = extreme

        # Long duration
        long_run = ScenarioTemplate(
            template_id="stress_long_duration",
            name="Long Duration Test",
            category=ScenarioCategory.STRESS_TEST,
            severity=ScenarioSeverity.MEDIUM,
            description="Extended operation stability test"
        )
        long_run.add_parameter(ParameterRange(
            name="duration",
            min_value=28800.0,  # 8 hours
            max_value=86400.0,  # 24 hours
            default_value=43200.0,  # 12 hours
            unit="s"
        ))

        self._templates[long_run.template_id] = long_run

    def get_template(self, template_id: str) -> Optional[ScenarioTemplate]:
        """Get a template by ID."""
        return self._templates.get(template_id)

    def list_templates(
        self,
        category: Optional[ScenarioCategory] = None,
        severity: Optional[ScenarioSeverity] = None
    ) -> List[ScenarioTemplate]:
        """List available templates with optional filtering."""
        templates = list(self._templates.values())

        if category:
            templates = [t for t in templates if t.category == category]
        if severity:
            templates = [t for t in templates if t.severity == severity]

        return templates

    def add_template(self, template: ScenarioTemplate) -> None:
        """Add a custom template."""
        self._templates[template.template_id] = template
        logger.info("Added template: %s", template.name)

    def get_categories(self) -> List[str]:
        """Get list of available categories."""
        return list(set(t.category.value for t in self._templates.values()))

    def get_summary(self) -> Dict[str, Any]:
        """Get library summary."""
        by_category = {}
        by_severity = {}

        for template in self._templates.values():
            cat = template.category.value
            sev = template.severity.name

            by_category[cat] = by_category.get(cat, 0) + 1
            by_severity[sev] = by_severity.get(sev, 0) + 1

        return {
            'total_templates': len(self._templates),
            'by_category': by_category,
            'by_severity': by_severity,
            'template_ids': list(self._templates.keys())
        }


# =============================================================================
# Scenario Generator
# =============================================================================

class ScenarioGenerator:
    """
    Generates scenarios from templates with parameter randomization.
    """

    def __init__(self, library: Optional[ScenarioLibrary] = None, seed: Optional[int] = None):
        """
        Initialize scenario generator.

        Args:
            library: Scenario library (creates default if None)
            seed: Random seed for reproducibility
        """
        self.library = library or ScenarioLibrary()
        self._rng = np.random.default_rng(seed)
        self._generated_count = 0

        logger.info("ScenarioGenerator initialized")

    def generate(
        self,
        template_id: str,
        parameter_overrides: Optional[Dict[str, float]] = None,
        timeline_modifier: Optional[Callable[[ScenarioTimeline], ScenarioTimeline]] = None
    ) -> Tuple[Dict[str, float], ScenarioTimeline]:
        """
        Generate a scenario from a template.

        Args:
            template_id: Template to use
            parameter_overrides: Specific parameter values to use
            timeline_modifier: Function to modify the timeline

        Returns:
            Tuple of (parameters, timeline)
        """
        template = self.library.get_template(template_id)
        if template is None:
            raise ValueError(f"Template not found: {template_id}")

        # Generate parameters
        params = template.generate_parameters(self._rng, parameter_overrides)

        # Copy and optionally modify timeline
        timeline = copy.deepcopy(template.base_timeline)
        if timeline_modifier:
            timeline = timeline_modifier(timeline)

        self._generated_count += 1
        logger.debug("Generated scenario from template %s", template_id)

        return params, timeline

    def generate_random(
        self,
        category: Optional[ScenarioCategory] = None,
        min_severity: ScenarioSeverity = ScenarioSeverity.LOW,
        max_severity: ScenarioSeverity = ScenarioSeverity.CRITICAL
    ) -> Tuple[str, Dict[str, float], ScenarioTimeline]:
        """
        Generate a random scenario.

        Args:
            category: Optional category filter
            min_severity: Minimum severity level
            max_severity: Maximum severity level

        Returns:
            Tuple of (template_id, parameters, timeline)
        """
        templates = self.library.list_templates(category)

        # Filter by severity
        templates = [
            t for t in templates
            if min_severity.value <= t.severity.value <= max_severity.value
        ]

        if not templates:
            raise ValueError("No matching templates found")

        template = self._rng.choice(templates)
        params, timeline = self.generate(template.template_id)

        return template.template_id, params, timeline

    def generate_sequence(
        self,
        template_ids: List[str],
        gap_between: float = 300.0
    ) -> Tuple[Dict[str, Dict[str, float]], ScenarioTimeline]:
        """
        Generate a sequence of scenarios.

        Args:
            template_ids: List of templates to chain
            gap_between: Time gap between scenarios (seconds)

        Returns:
            Tuple of (all_parameters, combined_timeline)
        """
        all_params = {}
        combined_timeline = ScenarioTimeline()

        current_offset = 0.0

        for i, template_id in enumerate(template_ids):
            params, timeline = self.generate(template_id)
            all_params[f"{template_id}_{i}"] = params

            # Add events with time offset
            for event in timeline.events:
                new_event = copy.deepcopy(event)
                new_event.trigger_time += current_offset
                combined_timeline.add_event(new_event)

            current_offset += timeline.total_duration + gap_between

        combined_timeline.total_duration = current_offset - gap_between

        return all_params, combined_timeline

    def generate_monte_carlo(
        self,
        template_id: str,
        n_samples: int = 100
    ) -> List[Tuple[Dict[str, float], ScenarioTimeline]]:
        """
        Generate multiple scenario variations using Monte Carlo sampling.

        Args:
            template_id: Template to use
            n_samples: Number of samples to generate

        Returns:
            List of (parameters, timeline) tuples
        """
        scenarios = []
        for _ in range(n_samples):
            params, timeline = self.generate(template_id)
            scenarios.append((params, timeline))

        logger.info("Generated %d Monte Carlo samples for %s", n_samples, template_id)
        return scenarios

    def get_statistics(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            'generated_count': self._generated_count,
            'library_summary': self.library.get_summary()
        }


# =============================================================================
# Scenario Executor
# =============================================================================

class ScenarioExecutor:
    """
    Executes scenarios on the simulation model.
    """

    def __init__(self, model: Any):
        """
        Initialize scenario executor.

        Args:
            model: Physics model to execute scenarios on
        """
        self.model = model
        self._current_scenario: Optional[ScenarioResult] = None
        self._action_handlers: Dict[str, Callable] = {}
        self._register_default_handlers()

        logger.info("ScenarioExecutor initialized")

    def _register_default_handlers(self) -> None:
        """Register default action handlers."""
        self._action_handlers = {
            'set_target_flow': self._handle_set_target,
            'increase_upstream_head': self._handle_increase_head,
            'decrease_upstream_head': self._handle_decrease_head,
            'maintain_high_flow': self._handle_maintain,
            'inject_fault': self._handle_inject_fault,
            'clear_fault': self._handle_clear_fault,
            'emergency_stop': self._handle_emergency,
            'reset_emergency': self._handle_reset_emergency,
            'oscillate_target': self._handle_oscillate,
        }

    def register_handler(self, action: str, handler: Callable) -> None:
        """Register a custom action handler."""
        self._action_handlers[action] = handler

    def _handle_set_target(self, event: ScenarioEvent, time: float) -> None:
        """Handle set target flow action."""
        target = event.parameters.get('target', 150.0)
        logger.debug("Setting target flow to %.1f at t=%.1f", target, time)

    def _handle_increase_head(self, event: ScenarioEvent, time: float) -> None:
        """Handle increase head action."""
        rate = event.parameters.get('rate', 0.01)
        logger.debug("Increasing head at rate %.3f", rate)

    def _handle_decrease_head(self, event: ScenarioEvent, time: float) -> None:
        """Handle decrease head action."""
        rate = event.parameters.get('rate', 0.005)
        logger.debug("Decreasing head at rate %.3f", rate)

    def _handle_maintain(self, event: ScenarioEvent, time: float) -> None:
        """Handle maintain action."""
        logger.debug("Maintaining current state")

    def _handle_inject_fault(self, event: ScenarioEvent, time: float) -> None:
        """Handle fault injection."""
        gate = int(event.parameters.get('gate_index', 0))
        fault_type = event.parameters.get('fault_type', 'stuck')
        if hasattr(self.model, 'inject_fault'):
            self.model.inject_fault(gate, fault_type)
        logger.info("Injected %s fault on gate %d", fault_type, gate)

    def _handle_clear_fault(self, event: ScenarioEvent, time: float) -> None:
        """Handle fault clearing."""
        gate = int(event.parameters.get('gate_index', 0))
        if hasattr(self.model, 'inject_fault'):
            self.model.inject_fault(gate, 'clear')
        logger.info("Cleared fault on gate %d", gate)

    def _handle_emergency(self, event: ScenarioEvent, time: float) -> None:
        """Handle emergency stop."""
        logger.warning("EMERGENCY STOP triggered at t=%.1f", time)

    def _handle_reset_emergency(self, event: ScenarioEvent, time: float) -> None:
        """Handle emergency reset."""
        logger.info("Emergency reset at t=%.1f", time)

    def _handle_oscillate(self, event: ScenarioEvent, time: float) -> None:
        """Handle oscillation action."""
        period = event.parameters.get('period', 60.0)
        amplitude = event.parameters.get('amplitude', 30.0)
        logger.debug("Oscillating with period %.1f and amplitude %.1f", period, amplitude)

    def start(
        self,
        template_id: str,
        parameters: Dict[str, float],
        timeline: ScenarioTimeline
    ) -> ScenarioResult:
        """
        Start a scenario.

        Args:
            template_id: Template used
            parameters: Scenario parameters
            timeline: Event timeline

        Returns:
            ScenarioResult for tracking
        """
        self._current_scenario = ScenarioResult(
            scenario_id=str(uuid.uuid4())[:8],
            template_id=template_id,
            start_time=datetime.now(),
            parameters=parameters
        )

        logger.info("Started scenario %s from template %s",
                   self._current_scenario.scenario_id, template_id)

        return self._current_scenario

    def step(self, time: float, timeline: ScenarioTimeline) -> List[ScenarioEvent]:
        """
        Execute scenario step.

        Args:
            time: Current simulation time
            timeline: Scenario timeline

        Returns:
            List of executed events
        """
        if self._current_scenario is None:
            return []

        # Get events to execute
        events = timeline.get_events_at_time(time)

        for event in events:
            self._execute_event(event, time)

        # Collect metrics
        if hasattr(self.model, 'get_state'):
            state = self.model.get_state()
            self._current_scenario.add_metric('total_flow', state.get('total_flow', 0))

            vibrations = state.get('vibrations', [])
            if vibrations:
                self._current_scenario.add_metric('max_vibration', max(vibrations))

        return events

    def _execute_event(self, event: ScenarioEvent, time: float) -> None:
        """Execute a single event."""
        handler = self._action_handlers.get(event.action)

        if handler:
            try:
                handler(event, time)
                if self._current_scenario:
                    self._current_scenario.events_executed.append(event.event_id)
            except Exception as e:
                logger.error("Error executing event %s: %s", event.name, e)
                if self._current_scenario:
                    self._current_scenario.errors.append(f"Event {event.name}: {e}")
        else:
            logger.warning("No handler for action: %s", event.action)

    def stop(self) -> Optional[ScenarioResult]:
        """
        Stop current scenario.

        Returns:
            Final scenario result
        """
        if self._current_scenario is None:
            return None

        self._current_scenario.end_time = datetime.now()
        result = self._current_scenario
        self._current_scenario = None

        logger.info("Scenario %s completed in %.1f seconds",
                   result.scenario_id, result.duration_seconds)

        return result

    def get_current_scenario(self) -> Optional[ScenarioResult]:
        """Get current scenario result."""
        return self._current_scenario


# =============================================================================
# Automated Test Runner
# =============================================================================

class AutomatedTestRunner:
    """
    Runs automated test scenarios.
    """

    def __init__(
        self,
        model: Any,
        generator: Optional[ScenarioGenerator] = None
    ):
        """
        Initialize test runner.

        Args:
            model: Physics model
            generator: Scenario generator
        """
        self.model = model
        self.generator = generator or ScenarioGenerator()
        self.executor = ScenarioExecutor(model)
        self._results: List[ScenarioResult] = []

        logger.info("AutomatedTestRunner initialized")

    def run_scenario(
        self,
        template_id: str,
        dt: float = 0.1,
        real_time: bool = False,
        parameter_overrides: Optional[Dict[str, float]] = None
    ) -> ScenarioResult:
        """
        Run a single scenario.

        Args:
            template_id: Template to run
            dt: Time step (seconds)
            real_time: Whether to run in real-time
            parameter_overrides: Parameter overrides

        Returns:
            Scenario result
        """
        # Generate scenario
        params, timeline = self.generator.generate(template_id, parameter_overrides)

        # Start scenario
        result = self.executor.start(template_id, params, timeline)

        # Run simulation
        time = 0.0
        while time < timeline.total_duration:
            # Execute scenario step
            self.executor.step(time, timeline)

            # Step physics model if available
            if hasattr(self.model, 'step'):
                self.model.step(self.model.gate_openings if hasattr(self.model, 'gate_openings')
                               else [1.0, 1.0, 1.0], dt)

            time += dt

            if real_time:
                import time as time_module
                time_module.sleep(dt)

        # Stop and record result
        final_result = self.executor.stop()
        if final_result:
            self._results.append(final_result)

        return final_result

    def run_batch(
        self,
        template_ids: List[str],
        dt: float = 0.1
    ) -> List[ScenarioResult]:
        """
        Run multiple scenarios in batch.

        Args:
            template_ids: Templates to run
            dt: Time step

        Returns:
            List of results
        """
        results = []
        for template_id in template_ids:
            try:
                result = self.run_scenario(template_id, dt)
                results.append(result)
            except Exception as e:
                logger.error("Failed to run scenario %s: %s", template_id, e)

        return results

    def run_category(
        self,
        category: ScenarioCategory,
        dt: float = 0.1
    ) -> List[ScenarioResult]:
        """
        Run all scenarios in a category.

        Args:
            category: Category to run
            dt: Time step

        Returns:
            List of results
        """
        templates = self.generator.library.list_templates(category)
        template_ids = [t.template_id for t in templates]
        return self.run_batch(template_ids, dt)

    def run_monte_carlo(
        self,
        template_id: str,
        n_samples: int = 100,
        dt: float = 0.1
    ) -> List[ScenarioResult]:
        """
        Run Monte Carlo test suite.

        Args:
            template_id: Template to test
            n_samples: Number of samples
            dt: Time step

        Returns:
            List of results
        """
        scenarios = self.generator.generate_monte_carlo(template_id, n_samples)

        results = []
        for params, timeline in scenarios:
            result = self.executor.start(template_id, params, timeline)

            time = 0.0
            while time < timeline.total_duration:
                self.executor.step(time, timeline)
                if hasattr(self.model, 'step'):
                    self.model.step(self.model.gate_openings if hasattr(self.model, 'gate_openings')
                                   else [1.0, 1.0, 1.0], dt)
                time += dt

            final = self.executor.stop()
            if final:
                results.append(final)
                self._results.append(final)

        return results

    def get_all_results(self) -> List[ScenarioResult]:
        """Get all test results."""
        return self._results

    def get_summary(self) -> Dict[str, Any]:
        """Get test run summary."""
        if not self._results:
            return {'total_runs': 0}

        successful = sum(1 for r in self._results if r.success)
        failed = len(self._results) - successful

        durations = [r.duration_seconds for r in self._results]

        by_template = {}
        for r in self._results:
            if r.template_id not in by_template:
                by_template[r.template_id] = {'success': 0, 'failed': 0}
            if r.success:
                by_template[r.template_id]['success'] += 1
            else:
                by_template[r.template_id]['failed'] += 1

        return {
            'total_runs': len(self._results),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(self._results) * 100,
            'total_duration_seconds': sum(durations),
            'mean_duration_seconds': np.mean(durations),
            'by_template': by_template
        }

    def clear_results(self) -> None:
        """Clear all results."""
        self._results = []
