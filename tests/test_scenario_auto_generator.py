"""
Tests for Scenario Auto Generator Module.

Tests cover:
- Parameter ranges
- Scenario templates
- Scenario library
- Scenario generator
- Scenario executor
- Automated test runner
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from src.scenarios.generator import (
    # Enums
    ScenarioCategory,
    ScenarioSeverity,
    ParameterDistribution,
    EventTrigger,
    # Data classes
    ParameterRange,
    ScenarioEvent,
    ScenarioTimeline,
    ScenarioTemplate,
    ScenarioResult,
    # Classes
    ScenarioLibrary,
    ScenarioGenerator,
    ScenarioExecutor,
    AutomatedTestRunner
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def parameter_range():
    """Create a basic parameter range."""
    return ParameterRange(
        name="test_param",
        min_value=0.0,
        max_value=100.0,
        default_value=50.0,
        distribution=ParameterDistribution.UNIFORM,
        unit="m"
    )


@pytest.fixture
def scenario_event():
    """Create a basic scenario event."""
    return ScenarioEvent.create(
        name="test_event",
        action="test_action",
        trigger_time=100.0,
        parameters={'key': 'value'}
    )


@pytest.fixture
def scenario_timeline():
    """Create a scenario timeline."""
    timeline = ScenarioTimeline(total_duration=3600.0)
    timeline.add_event(ScenarioEvent.create("event1", "action1", trigger_time=0.0))
    timeline.add_event(ScenarioEvent.create("event2", "action2", trigger_time=600.0, duration=300.0))
    timeline.add_event(ScenarioEvent.create("event3", "action3", trigger_time=1200.0))
    return timeline


@pytest.fixture
def scenario_template():
    """Create a scenario template."""
    template = ScenarioTemplate(
        template_id="test_template",
        name="Test Template",
        category=ScenarioCategory.NORMAL,
        severity=ScenarioSeverity.MEDIUM,
        description="A test template"
    )
    template.add_parameter(ParameterRange(
        name="param1",
        min_value=0.0,
        max_value=100.0,
        default_value=50.0
    ))
    template.add_parameter(ParameterRange(
        name="param2",
        min_value=1.0,
        max_value=10.0,
        default_value=5.0
    ))
    return template


@pytest.fixture
def scenario_library():
    """Create a scenario library."""
    return ScenarioLibrary()


@pytest.fixture
def scenario_generator(scenario_library):
    """Create a scenario generator."""
    return ScenarioGenerator(scenario_library, seed=42)


@pytest.fixture
def mock_model():
    """Create a mock physics model."""
    model = Mock()
    model.gate_openings = np.array([1.0, 1.0, 1.0])
    model.get_state.return_value = {
        'total_flow': 150.0,
        'flows': [50.0, 50.0, 50.0],
        'vibrations': [5.0, 4.0, 3.0],
        'head_upstream': 30.0,
        'head_downstream': 25.0
    }
    model.inject_fault = Mock()
    model.step = Mock()
    return model


@pytest.fixture
def scenario_executor(mock_model):
    """Create a scenario executor."""
    return ScenarioExecutor(mock_model)


@pytest.fixture
def test_runner(mock_model, scenario_generator):
    """Create an automated test runner."""
    return AutomatedTestRunner(mock_model, scenario_generator)


# =============================================================================
# Test Enums
# =============================================================================

class TestEnums:
    """Test enum definitions."""

    def test_scenario_category_values(self):
        """Test scenario category values."""
        assert ScenarioCategory.NORMAL.value == "normal"
        assert ScenarioCategory.FLOOD.value == "flood"
        assert ScenarioCategory.DROUGHT.value == "drought"
        assert ScenarioCategory.EQUIPMENT_FAILURE.value == "equipment_failure"
        assert ScenarioCategory.EMERGENCY.value == "emergency"

    def test_scenario_severity_values(self):
        """Test scenario severity ordering."""
        assert ScenarioSeverity.LOW.value < ScenarioSeverity.MEDIUM.value
        assert ScenarioSeverity.MEDIUM.value < ScenarioSeverity.HIGH.value
        assert ScenarioSeverity.HIGH.value < ScenarioSeverity.CRITICAL.value

    def test_parameter_distribution_values(self):
        """Test parameter distribution values."""
        assert ParameterDistribution.UNIFORM.value == "uniform"
        assert ParameterDistribution.NORMAL.value == "normal"
        assert ParameterDistribution.TRIANGULAR.value == "triangular"

    def test_event_trigger_values(self):
        """Test event trigger values."""
        assert EventTrigger.TIME.value == "time"
        assert EventTrigger.CONDITION.value == "condition"
        assert EventTrigger.RANDOM.value == "random"
        assert EventTrigger.PERIODIC.value == "periodic"


# =============================================================================
# Test Parameter Range
# =============================================================================

class TestParameterRange:
    """Test ParameterRange class."""

    def test_initialization(self, parameter_range):
        """Test parameter range initialization."""
        assert parameter_range.name == "test_param"
        assert parameter_range.min_value == 0.0
        assert parameter_range.max_value == 100.0
        assert parameter_range.default_value == 50.0
        assert parameter_range.unit == "m"

    def test_default_value_calculation(self):
        """Test default value is calculated if not provided."""
        param = ParameterRange(name="test", min_value=0.0, max_value=100.0)
        assert param.default_value == 50.0

    def test_sample_uniform(self):
        """Test uniform distribution sampling."""
        param = ParameterRange(
            name="test",
            min_value=0.0,
            max_value=100.0,
            distribution=ParameterDistribution.UNIFORM
        )
        rng = np.random.default_rng(42)
        samples = [param.sample(rng) for _ in range(100)]
        assert all(0.0 <= s <= 100.0 for s in samples)

    def test_sample_normal(self):
        """Test normal distribution sampling."""
        param = ParameterRange(
            name="test",
            min_value=0.0,
            max_value=100.0,
            default_value=50.0,
            distribution=ParameterDistribution.NORMAL
        )
        rng = np.random.default_rng(42)
        samples = [param.sample(rng) for _ in range(100)]
        # Should be clipped to range
        assert all(0.0 <= s <= 100.0 for s in samples)
        # Mean should be near default
        assert 40.0 <= np.mean(samples) <= 60.0

    def test_sample_triangular(self):
        """Test triangular distribution sampling."""
        param = ParameterRange(
            name="test",
            min_value=0.0,
            max_value=100.0,
            default_value=50.0,
            distribution=ParameterDistribution.TRIANGULAR
        )
        rng = np.random.default_rng(42)
        samples = [param.sample(rng) for _ in range(100)]
        assert all(0.0 <= s <= 100.0 for s in samples)

    def test_sample_constant(self):
        """Test constant distribution."""
        param = ParameterRange(
            name="test",
            min_value=0.0,
            max_value=100.0,
            default_value=42.0,
            distribution=ParameterDistribution.CONSTANT
        )
        samples = [param.sample() for _ in range(10)]
        assert all(s == 42.0 for s in samples)

    def test_sample_exponential(self):
        """Test exponential distribution sampling."""
        param = ParameterRange(
            name="test",
            min_value=0.0,
            max_value=100.0,
            distribution=ParameterDistribution.EXPONENTIAL
        )
        rng = np.random.default_rng(42)
        samples = [param.sample(rng) for _ in range(100)]
        assert all(0.0 <= s <= 100.0 for s in samples)


# =============================================================================
# Test Scenario Event
# =============================================================================

class TestScenarioEvent:
    """Test ScenarioEvent class."""

    def test_creation(self, scenario_event):
        """Test event creation."""
        assert scenario_event.name == "test_event"
        assert scenario_event.action == "test_action"
        assert scenario_event.trigger_time == 100.0
        assert scenario_event.parameters == {'key': 'value'}

    def test_create_factory(self):
        """Test create factory method."""
        event = ScenarioEvent.create(
            name="factory_event",
            action="factory_action",
            trigger_time=200.0,
            duration=50.0
        )
        assert event.name == "factory_event"
        assert event.trigger_type == EventTrigger.TIME
        assert event.trigger_time == 200.0
        assert event.duration == 50.0
        assert event.event_id is not None


# =============================================================================
# Test Scenario Timeline
# =============================================================================

class TestScenarioTimeline:
    """Test ScenarioTimeline class."""

    def test_initialization(self):
        """Test timeline initialization."""
        timeline = ScenarioTimeline(total_duration=7200.0)
        assert timeline.total_duration == 7200.0
        assert len(timeline.events) == 0

    def test_add_event(self, scenario_timeline):
        """Test adding events."""
        assert len(scenario_timeline.events) == 3

    def test_events_sorted(self, scenario_timeline):
        """Test events are sorted by time."""
        times = [e.trigger_time for e in scenario_timeline.events]
        assert times == sorted(times)

    def test_get_events_at_time(self, scenario_timeline):
        """Test getting events at a specific time."""
        events = scenario_timeline.get_events_at_time(0.0)
        assert len(events) == 1
        assert events[0].name == "event1"

        events = scenario_timeline.get_events_at_time(600.0)
        assert len(events) == 1
        assert events[0].name == "event2"

    def test_get_events_with_tolerance(self, scenario_timeline):
        """Test getting events with time tolerance."""
        events = scenario_timeline.get_events_at_time(0.05, tolerance=0.1)
        assert len(events) == 1

    def test_get_active_events(self, scenario_timeline):
        """Test getting active events."""
        # event2 has duration 300s starting at 600s
        active = scenario_timeline.get_active_events(700.0)
        assert len(active) == 1
        assert active[0].name == "event2"

        active = scenario_timeline.get_active_events(950.0)
        assert len(active) == 0  # event2 ended at 900

    def test_periodic_events(self):
        """Test periodic event triggering."""
        timeline = ScenarioTimeline()
        periodic = ScenarioEvent(
            event_id="periodic",
            name="periodic_event",
            trigger_type=EventTrigger.PERIODIC,
            action="periodic_action",
            trigger_time=0.0,
            repeat=True,
            repeat_interval=60.0
        )
        timeline.add_event(periodic)

        # Should trigger at 0, 60, 120, etc.
        assert len(timeline.get_events_at_time(0.0)) == 1
        assert len(timeline.get_events_at_time(60.0)) == 1
        assert len(timeline.get_events_at_time(120.0)) == 1
        assert len(timeline.get_events_at_time(30.0)) == 0


# =============================================================================
# Test Scenario Template
# =============================================================================

class TestScenarioTemplate:
    """Test ScenarioTemplate class."""

    def test_initialization(self, scenario_template):
        """Test template initialization."""
        assert scenario_template.template_id == "test_template"
        assert scenario_template.name == "Test Template"
        assert scenario_template.category == ScenarioCategory.NORMAL
        assert scenario_template.severity == ScenarioSeverity.MEDIUM

    def test_add_parameter(self, scenario_template):
        """Test adding parameters."""
        assert "param1" in scenario_template.parameters
        assert "param2" in scenario_template.parameters

    def test_generate_parameters(self, scenario_template):
        """Test parameter generation."""
        params = scenario_template.generate_parameters()
        assert "param1" in params
        assert "param2" in params
        assert 0.0 <= params["param1"] <= 100.0
        assert 1.0 <= params["param2"] <= 10.0

    def test_generate_parameters_with_overrides(self, scenario_template):
        """Test parameter generation with overrides."""
        params = scenario_template.generate_parameters(
            overrides={"param1": 75.0}
        )
        assert params["param1"] == 75.0
        # param2 should still be random
        assert 1.0 <= params["param2"] <= 10.0

    def test_reproducible_parameters(self, scenario_template):
        """Test reproducible parameter generation with seed."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        params1 = scenario_template.generate_parameters(rng1)
        params2 = scenario_template.generate_parameters(rng2)

        assert params1 == params2


# =============================================================================
# Test Scenario Result
# =============================================================================

class TestScenarioResult:
    """Test ScenarioResult class."""

    def test_initialization(self):
        """Test result initialization."""
        result = ScenarioResult(
            scenario_id="test_001",
            template_id="test_template",
            start_time=datetime.now()
        )
        assert result.scenario_id == "test_001"
        assert result.success is True
        assert len(result.errors) == 0

    def test_duration_calculation(self):
        """Test duration calculation."""
        start = datetime.now()
        end = start + timedelta(seconds=100)
        result = ScenarioResult(
            scenario_id="test",
            template_id="test",
            start_time=start,
            end_time=end
        )
        assert result.duration_seconds == 100.0

    def test_add_metric(self):
        """Test adding metrics."""
        result = ScenarioResult(
            scenario_id="test",
            template_id="test",
            start_time=datetime.now()
        )
        result.add_metric("flow", 100.0)
        result.add_metric("flow", 110.0)
        result.add_metric("flow", 105.0)

        assert len(result.metrics["flow"]) == 3

    def test_get_metric_summary(self):
        """Test metric summary calculation."""
        result = ScenarioResult(
            scenario_id="test",
            template_id="test",
            start_time=datetime.now()
        )
        for v in [100.0, 110.0, 90.0, 105.0, 95.0]:
            result.add_metric("flow", v)

        summary = result.get_metric_summary("flow")
        assert summary['count'] == 5
        assert summary['mean'] == 100.0
        assert summary['min'] == 90.0
        assert summary['max'] == 110.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ScenarioResult(
            scenario_id="test",
            template_id="test_template",
            start_time=datetime.now(),
            parameters={'param1': 50.0}
        )
        result.add_metric("flow", 100.0)

        d = result.to_dict()
        assert d['scenario_id'] == "test"
        assert d['template_id'] == "test_template"
        assert 'metrics_summary' in d


# =============================================================================
# Test Scenario Library
# =============================================================================

class TestScenarioLibrary:
    """Test ScenarioLibrary class."""

    def test_initialization(self, scenario_library):
        """Test library initialization."""
        assert len(scenario_library._templates) > 0

    def test_default_templates_loaded(self, scenario_library):
        """Test default templates are loaded."""
        # Check for expected templates
        assert scenario_library.get_template("normal_operation") is not None
        assert scenario_library.get_template("flood_minor") is not None
        assert scenario_library.get_template("flood_major") is not None
        assert scenario_library.get_template("drought_low_flow") is not None
        assert scenario_library.get_template("failure_gate_stuck") is not None
        assert scenario_library.get_template("emergency_shutdown") is not None

    def test_get_template(self, scenario_library):
        """Test getting a template."""
        template = scenario_library.get_template("normal_operation")
        assert template is not None
        assert template.category == ScenarioCategory.NORMAL

    def test_get_nonexistent_template(self, scenario_library):
        """Test getting a nonexistent template."""
        template = scenario_library.get_template("nonexistent")
        assert template is None

    def test_list_templates(self, scenario_library):
        """Test listing templates."""
        templates = scenario_library.list_templates()
        assert len(templates) > 0

    def test_list_templates_by_category(self, scenario_library):
        """Test listing templates by category."""
        flood_templates = scenario_library.list_templates(category=ScenarioCategory.FLOOD)
        assert len(flood_templates) > 0
        assert all(t.category == ScenarioCategory.FLOOD for t in flood_templates)

    def test_list_templates_by_severity(self, scenario_library):
        """Test listing templates by severity."""
        critical_templates = scenario_library.list_templates(severity=ScenarioSeverity.CRITICAL)
        assert len(critical_templates) > 0
        assert all(t.severity == ScenarioSeverity.CRITICAL for t in critical_templates)

    def test_add_template(self, scenario_library):
        """Test adding a custom template."""
        custom = ScenarioTemplate(
            template_id="custom_test",
            name="Custom Test",
            category=ScenarioCategory.CUSTOM,
            severity=ScenarioSeverity.LOW
        )
        scenario_library.add_template(custom)

        assert scenario_library.get_template("custom_test") is not None

    def test_get_categories(self, scenario_library):
        """Test getting categories."""
        categories = scenario_library.get_categories()
        assert "normal" in categories
        assert "flood" in categories

    def test_get_summary(self, scenario_library):
        """Test getting library summary."""
        summary = scenario_library.get_summary()
        assert summary['total_templates'] > 0
        assert 'by_category' in summary
        assert 'by_severity' in summary
        assert 'template_ids' in summary


# =============================================================================
# Test Scenario Generator
# =============================================================================

class TestScenarioGenerator:
    """Test ScenarioGenerator class."""

    def test_initialization(self, scenario_generator):
        """Test generator initialization."""
        assert scenario_generator.library is not None
        assert scenario_generator._generated_count == 0

    def test_generate_scenario(self, scenario_generator):
        """Test generating a scenario."""
        params, timeline = scenario_generator.generate("normal_operation")

        assert "target_flow" in params
        assert "head_upstream" in params
        assert timeline is not None

    def test_generate_with_overrides(self, scenario_generator):
        """Test generating with parameter overrides."""
        params, timeline = scenario_generator.generate(
            "normal_operation",
            parameter_overrides={"target_flow": 175.0}
        )

        assert params["target_flow"] == 175.0

    def test_generate_nonexistent_template(self, scenario_generator):
        """Test generating from nonexistent template."""
        with pytest.raises(ValueError):
            scenario_generator.generate("nonexistent")

    def test_generate_random(self, scenario_generator):
        """Test generating a random scenario."""
        template_id, params, timeline = scenario_generator.generate_random()

        assert template_id is not None
        assert params is not None
        assert timeline is not None

    def test_generate_random_by_category(self, scenario_generator):
        """Test generating random scenario by category."""
        template_id, params, timeline = scenario_generator.generate_random(
            category=ScenarioCategory.FLOOD
        )

        template = scenario_generator.library.get_template(template_id)
        assert template.category == ScenarioCategory.FLOOD

    def test_generate_random_by_severity(self, scenario_generator):
        """Test generating random scenario by severity range."""
        template_id, params, timeline = scenario_generator.generate_random(
            min_severity=ScenarioSeverity.HIGH,
            max_severity=ScenarioSeverity.CRITICAL
        )

        template = scenario_generator.library.get_template(template_id)
        assert template.severity.value >= ScenarioSeverity.HIGH.value

    def test_generate_sequence(self, scenario_generator):
        """Test generating a sequence of scenarios."""
        all_params, timeline = scenario_generator.generate_sequence(
            ["normal_operation", "flood_minor"],
            gap_between=100.0
        )

        assert len(all_params) == 2
        assert "normal_operation_0" in all_params
        assert "flood_minor_1" in all_params
        assert timeline.total_duration > 0

    def test_generate_monte_carlo(self, scenario_generator):
        """Test Monte Carlo scenario generation."""
        scenarios = scenario_generator.generate_monte_carlo(
            "normal_operation",
            n_samples=10
        )

        assert len(scenarios) == 10
        for params, timeline in scenarios:
            assert "target_flow" in params

    def test_statistics(self, scenario_generator):
        """Test generator statistics."""
        scenario_generator.generate("normal_operation")
        scenario_generator.generate("flood_minor")

        stats = scenario_generator.get_statistics()
        assert stats['generated_count'] == 2


# =============================================================================
# Test Scenario Executor
# =============================================================================

class TestScenarioExecutor:
    """Test ScenarioExecutor class."""

    def test_initialization(self, scenario_executor):
        """Test executor initialization."""
        assert scenario_executor.model is not None
        assert len(scenario_executor._action_handlers) > 0

    def test_register_handler(self, scenario_executor):
        """Test registering a custom handler."""
        handler = Mock()
        scenario_executor.register_handler("custom_action", handler)

        assert "custom_action" in scenario_executor._action_handlers

    def test_start_scenario(self, scenario_executor):
        """Test starting a scenario."""
        result = scenario_executor.start(
            "test_template",
            {"param1": 50.0},
            ScenarioTimeline()
        )

        assert result.scenario_id is not None
        assert result.template_id == "test_template"
        assert result.parameters["param1"] == 50.0

    def test_step_scenario(self, scenario_executor):
        """Test stepping through a scenario."""
        timeline = ScenarioTimeline()
        timeline.add_event(ScenarioEvent.create(
            "test_event",
            "inject_fault",
            trigger_time=0.0,
            parameters={'gate_index': 0, 'fault_type': 'stuck'}
        ))

        scenario_executor.start("test", {}, timeline)
        events = scenario_executor.step(0.0, timeline)

        assert len(events) == 1
        scenario_executor.model.inject_fault.assert_called()

    def test_stop_scenario(self, scenario_executor):
        """Test stopping a scenario."""
        scenario_executor.start("test", {}, ScenarioTimeline())
        result = scenario_executor.stop()

        assert result is not None
        assert result.end_time is not None

    def test_get_current_scenario(self, scenario_executor):
        """Test getting current scenario."""
        assert scenario_executor.get_current_scenario() is None

        scenario_executor.start("test", {}, ScenarioTimeline())
        assert scenario_executor.get_current_scenario() is not None

    def test_metrics_collection(self, scenario_executor):
        """Test metrics are collected during step."""
        timeline = ScenarioTimeline(total_duration=100.0)
        scenario_executor.start("test", {}, timeline)

        # Step multiple times
        for t in [0.0, 10.0, 20.0]:
            scenario_executor.step(t, timeline)

        result = scenario_executor.stop()
        assert "total_flow" in result.metrics
        assert len(result.metrics["total_flow"]) == 3


# =============================================================================
# Test Automated Test Runner
# =============================================================================

class TestAutomatedTestRunner:
    """Test AutomatedTestRunner class."""

    def test_initialization(self, test_runner):
        """Test runner initialization."""
        assert test_runner.model is not None
        assert test_runner.generator is not None
        assert test_runner.executor is not None

    def test_run_scenario(self, test_runner):
        """Test running a single scenario."""
        # Use a template with short duration
        result = test_runner.run_scenario(
            "normal_operation",
            dt=1.0,
            parameter_overrides={"duration": 10.0}
        )

        assert result is not None
        assert result.success is True

    def test_run_batch(self, test_runner):
        """Test running batch scenarios."""
        results = test_runner.run_batch(
            ["normal_operation", "normal_operation"],
            dt=1.0
        )

        assert len(results) == 2

    def test_run_category(self, test_runner):
        """Test running all scenarios in a category."""
        # This would run all normal category scenarios
        results = test_runner.run_category(ScenarioCategory.NORMAL, dt=1.0)
        assert len(results) > 0

    def test_run_monte_carlo(self, test_runner):
        """Test Monte Carlo test run."""
        results = test_runner.run_monte_carlo(
            "normal_operation",
            n_samples=3,
            dt=1.0
        )

        assert len(results) == 3

    def test_get_all_results(self, test_runner):
        """Test getting all results."""
        test_runner.run_scenario("normal_operation", dt=1.0,
                                parameter_overrides={"duration": 5.0})
        results = test_runner.get_all_results()
        assert len(results) >= 1

    def test_get_summary(self, test_runner):
        """Test getting summary."""
        test_runner.run_scenario("normal_operation", dt=1.0,
                                parameter_overrides={"duration": 5.0})
        summary = test_runner.get_summary()

        assert summary['total_runs'] >= 1
        assert 'success_rate' in summary

    def test_clear_results(self, test_runner):
        """Test clearing results."""
        test_runner.run_scenario("normal_operation", dt=1.0,
                                parameter_overrides={"duration": 5.0})
        test_runner.clear_results()

        assert len(test_runner.get_all_results()) == 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for scenario generator module."""

    def test_full_workflow(self, mock_model):
        """Test complete scenario generation workflow."""
        # Create components
        library = ScenarioLibrary()
        generator = ScenarioGenerator(library, seed=42)
        executor = ScenarioExecutor(mock_model)

        # Generate scenario
        params, timeline = generator.generate("flood_minor")

        # Start execution
        result = executor.start("flood_minor", params, timeline)

        # Step through scenario
        for t in range(0, int(timeline.total_duration), 100):
            executor.step(float(t), timeline)

        # Stop and verify
        final = executor.stop()
        assert final is not None
        assert final.success is True
        assert len(final.metrics) > 0

    def test_sequence_execution(self, mock_model):
        """Test executing a sequence of scenarios."""
        generator = ScenarioGenerator(seed=42)
        executor = ScenarioExecutor(mock_model)

        # Generate sequence
        all_params, timeline = generator.generate_sequence(
            ["normal_operation", "flood_minor"],
            gap_between=100.0
        )

        # Execute
        result = executor.start("sequence", {}, timeline)

        time = 0.0
        while time < timeline.total_duration:
            executor.step(time, timeline)
            time += 100.0

        final = executor.stop()
        assert final is not None

    def test_reproducibility(self):
        """Test scenario generation reproducibility."""
        gen1 = ScenarioGenerator(seed=12345)
        gen2 = ScenarioGenerator(seed=12345)

        params1, _ = gen1.generate("normal_operation")
        params2, _ = gen2.generate("normal_operation")

        assert params1 == params2


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_timeline(self, scenario_executor):
        """Test with empty timeline."""
        timeline = ScenarioTimeline()
        scenario_executor.start("test", {}, timeline)
        events = scenario_executor.step(0.0, timeline)
        assert len(events) == 0

    def test_unknown_action(self, scenario_executor):
        """Test handling unknown action."""
        timeline = ScenarioTimeline()
        timeline.add_event(ScenarioEvent.create(
            "unknown",
            "unknown_action",
            trigger_time=0.0
        ))

        scenario_executor.start("test", {}, timeline)
        # Should not raise, just log warning
        events = scenario_executor.step(0.0, timeline)
        assert len(events) == 1

    def test_no_matching_templates(self, scenario_generator):
        """Test when no templates match filter."""
        # Create impossible filter
        with pytest.raises(ValueError):
            scenario_generator.generate_random(
                category=ScenarioCategory.CUSTOM,
                min_severity=ScenarioSeverity.CRITICAL
            )

    def test_metric_summary_empty(self):
        """Test metric summary for nonexistent metric."""
        result = ScenarioResult(
            scenario_id="test",
            template_id="test",
            start_time=datetime.now()
        )
        summary = result.get_metric_summary("nonexistent")
        assert summary == {}

    def test_stop_without_start(self, scenario_executor):
        """Test stopping without starting."""
        result = scenario_executor.stop()
        assert result is None

    def test_multiple_events_same_time(self, scenario_executor):
        """Test multiple events at the same time."""
        timeline = ScenarioTimeline()
        timeline.add_event(ScenarioEvent.create("e1", "inject_fault", trigger_time=0.0,
                                                parameters={'gate_index': 0, 'fault_type': 'stuck'}))
        timeline.add_event(ScenarioEvent.create("e2", "inject_fault", trigger_time=0.0,
                                                parameters={'gate_index': 1, 'fault_type': 'stuck'}))

        scenario_executor.start("test", {}, timeline)
        events = scenario_executor.step(0.0, timeline)
        assert len(events) == 2
