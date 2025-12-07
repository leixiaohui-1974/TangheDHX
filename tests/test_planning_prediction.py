# -*- coding: utf-8 -*-
"""
Tests for Planning and Prediction Module.
"""

import unittest
import numpy as np

from src.control.planning_prediction import (
    PlanningContext,
    PlanningContextGenerator,
    DispatchPlan,
    MaintenancePlan,
    InspectionPlan,
    WeatherForecast,
    InflowPrediction,
    WaterLevelPrediction,
    PredictionAwareDetector,
    TimeOfDay,
    Season,
    WeekDay,
    WeatherType,
    PlanType,
    PlanPriority,
    OperationEvent,
)
from src.control.scenario_generator import (
    ExtendedScenarioGenerator,
    PlanningAwareScenarioGenerator,
    SeasonalScenarioGenerator,
    PredictionScenarioGenerator,
    DispatchScheduleGenerator,
    ScenarioSpec,
)
from src.simulation.physics import TangheSiphonModel
from src.config import reset_config


class TestWeatherForecast(unittest.TestCase):
    """Test weather forecast functionality."""

    def test_weather_creation(self):
        """Test creating weather forecast."""
        weather = WeatherForecast(
            start_time=0.0,
            end_time=3600.0,
            weather_type=WeatherType.CLEAR,
            temperature=25.0,
            humidity=65.0,
        )
        self.assertEqual(weather.weather_type, WeatherType.CLEAR)
        self.assertEqual(weather.temperature, 25.0)

    def test_inflow_effect_clear(self):
        """Test inflow effect for clear weather."""
        weather = WeatherForecast(
            start_time=0.0,
            end_time=3600.0,
            weather_type=WeatherType.CLEAR,
        )
        self.assertEqual(weather.affects_inflow(), 1.0)

    def test_inflow_effect_storm(self):
        """Test inflow effect for storm weather."""
        weather = WeatherForecast(
            start_time=0.0,
            end_time=3600.0,
            weather_type=WeatherType.STORM,
            precipitation=50.0,
        )
        effect = weather.affects_inflow()
        self.assertGreater(effect, 1.5)

    def test_inflow_effect_moderate_rain(self):
        """Test inflow effect for moderate rain."""
        weather = WeatherForecast(
            start_time=0.0,
            end_time=3600.0,
            weather_type=WeatherType.MODERATE_RAIN,
            precipitation=10.0,
        )
        effect = weather.affects_inflow()
        self.assertGreater(effect, 1.0)
        self.assertLess(effect, 1.5)


class TestInflowPrediction(unittest.TestCase):
    """Test inflow prediction functionality."""

    def test_prediction_creation(self):
        """Test creating inflow prediction."""
        pred = InflowPrediction(
            time=0.0,
            expected_flow=100.0,
            lower_bound=90.0,
            upper_bound=110.0,
        )
        self.assertEqual(pred.expected_flow, 100.0)
        self.assertEqual(pred.get_range(), (90.0, 110.0))

    def test_contains(self):
        """Test value containment check."""
        pred = InflowPrediction(
            time=0.0,
            expected_flow=100.0,
            lower_bound=90.0,
            upper_bound=110.0,
        )
        self.assertTrue(pred.contains(100.0))
        self.assertTrue(pred.contains(95.0))
        self.assertFalse(pred.contains(80.0))
        self.assertFalse(pred.contains(120.0))


class TestWaterLevelPrediction(unittest.TestCase):
    """Test water level prediction functionality."""

    def test_head_difference(self):
        """Test head difference calculation."""
        pred = WaterLevelPrediction(
            time=0.0,
            upstream_level=10.0,
            downstream_level=8.0,
        )
        self.assertEqual(pred.get_head_difference(), 2.0)

    def test_head_range(self):
        """Test head range calculation."""
        pred = WaterLevelPrediction(
            time=0.0,
            upstream_level=10.0,
            downstream_level=8.0,
            upstream_uncertainty=0.3,
            downstream_uncertainty=0.2,
        )
        low, high = pred.get_head_range()
        self.assertLess(low, 2.0)
        self.assertGreater(high, 2.0)


class TestDispatchPlan(unittest.TestCase):
    """Test dispatch plan functionality."""

    def test_plan_creation(self):
        """Test creating dispatch plan."""
        plan = DispatchPlan(
            plan_id="DP001",
            name="Test Plan",
            start_time=0.0,
            end_time=3600.0,
            flow_schedule=[(0.0, 100.0), (1800.0, 150.0)],
        )
        self.assertEqual(plan.plan_id, "DP001")
        self.assertTrue(plan.is_active_at(1000.0))
        self.assertFalse(plan.is_active_at(5000.0))

    def test_target_flow_interpolation(self):
        """Test target flow interpolation."""
        plan = DispatchPlan(
            plan_id="DP001",
            name="Test Plan",
            start_time=0.0,
            end_time=3600.0,
            flow_schedule=[(0.0, 100.0), (100.0, 200.0)],
        )
        # At start
        self.assertEqual(plan.get_target_flow_at(0.0), 100.0)
        # At end
        self.assertAlmostEqual(plan.get_target_flow_at(100.0), 200.0, places=1)
        # Midpoint
        self.assertAlmostEqual(plan.get_target_flow_at(50.0), 150.0, places=1)


class TestMaintenancePlan(unittest.TestCase):
    """Test maintenance plan functionality."""

    def test_plan_creation(self):
        """Test creating maintenance plan."""
        plan = MaintenancePlan(
            plan_id="MP001",
            name="Gate Maintenance",
            start_time=600.0,
            end_time=3600.0,
            affected_gates=[1],
            gate_availability={1: False},
            reduced_capacity=0.67,
        )
        self.assertEqual(plan.plan_id, "MP001")
        self.assertFalse(plan.is_gate_available(1))
        self.assertTrue(plan.is_gate_available(0))

    def test_available_gates(self):
        """Test getting available gates."""
        plan = MaintenancePlan(
            plan_id="MP001",
            name="Gate Maintenance",
            start_time=600.0,
            end_time=3600.0,
            affected_gates=[0, 2],
            gate_availability={0: False, 2: False},
        )
        available = plan.get_available_gates(3)
        self.assertEqual(available, [1])


class TestPlanningContext(unittest.TestCase):
    """Test planning context functionality."""

    def test_empty_context(self):
        """Test empty planning context."""
        ctx = PlanningContext()
        self.assertEqual(ctx.simulation_time, 0.0)
        self.assertIsNone(ctx.get_active_dispatch_plan())
        self.assertEqual(ctx.get_active_maintenance(), [])

    def test_context_with_dispatch(self):
        """Test context with dispatch plan."""
        ctx = PlanningContext(simulation_time=500.0)
        ctx.dispatch_plans.append(DispatchPlan(
            plan_id="DP001",
            name="Test",
            start_time=0.0,
            end_time=3600.0,
            flow_schedule=[(0.0, 100.0)],
        ))
        self.assertIsNotNone(ctx.get_active_dispatch_plan())

    def test_expected_inflow(self):
        """Test expected inflow calculation."""
        ctx = PlanningContext(
            simulation_time=0.0,
            season=Season.SUMMER,
        )
        # Add inflow predictions
        ctx.inflow_predictions.append(InflowPrediction(
            time=10.0,
            expected_flow=100.0,
            lower_bound=90.0,
            upper_bound=110.0,
        ))
        expected = ctx.get_expected_inflow()
        self.assertGreater(expected, 100.0)  # Summer multiplier

    def test_operational_constraints(self):
        """Test getting operational constraints."""
        ctx = PlanningContext(simulation_time=100.0)
        ctx.dispatch_plans.append(DispatchPlan(
            plan_id="DP001",
            name="Test",
            start_time=0.0,
            end_time=3600.0,
            max_flow=150.0,
            min_flow=30.0,
        ))
        constraints = ctx.get_operational_constraints()
        self.assertEqual(constraints['max_flow'], 150.0)
        self.assertEqual(constraints['min_flow'], 30.0)

    def test_scenario_hints(self):
        """Test scenario hints generation."""
        ctx = PlanningContext(
            simulation_time=0.0,
            flood_warning=True,
        )
        hints = ctx.get_scenario_hints()
        self.assertIn('FLOOD_CONDITION', hints['possible_scenarios'])
        self.assertIn('flood_warning', hints['risk_factors'])

    def test_to_dict(self):
        """Test serialization to dictionary."""
        ctx = PlanningContext(
            simulation_time=100.0,
            season=Season.SUMMER,
            time_of_day=TimeOfDay.MORNING,
        )
        d = ctx.to_dict()
        self.assertEqual(d['time'], 100.0)
        self.assertEqual(d['season'], 'SUMMER')
        self.assertEqual(d['time_of_day'], 'MORNING')


class TestPlanningContextGenerator(unittest.TestCase):
    """Test planning context generator."""

    def setUp(self):
        self.gen = PlanningContextGenerator(seed=42)

    def test_generate_normal_day(self):
        """Test generating normal day context."""
        ctx = self.gen.generate_normal_day(0.0)
        self.assertIsInstance(ctx, PlanningContext)
        self.assertGreater(len(ctx.dispatch_plans), 0)
        self.assertGreater(len(ctx.weather_forecasts), 0)

    def test_generate_flood_scenario(self):
        """Test generating flood scenario context."""
        ctx = self.gen.generate_flood_scenario(0.0)
        self.assertTrue(ctx.flood_warning)
        self.assertEqual(ctx.season, Season.SUMMER)

    def test_generate_drought_scenario(self):
        """Test generating drought scenario context."""
        ctx = self.gen.generate_drought_scenario(0.0)
        self.assertTrue(ctx.drought_warning)
        self.assertEqual(ctx.season, Season.WINTER)

    def test_generate_maintenance_scenario(self):
        """Test generating maintenance scenario context."""
        ctx = self.gen.generate_maintenance_scenario(0.0, gate_to_maintain=1)
        self.assertGreater(len(ctx.maintenance_plans), 0)
        self.assertIn(1, ctx.maintenance_plans[0].affected_gates)

    def test_generate_random_context(self):
        """Test generating random context."""
        ctx = self.gen.generate_random_context(0.0)
        self.assertIsInstance(ctx, PlanningContext)


class TestPredictionAwareDetector(unittest.TestCase):
    """Test prediction-aware scenario detector."""

    def setUp(self):
        reset_config()
        self.model = TangheSiphonModel()
        self.detector = PredictionAwareDetector(self.model)

    def tearDown(self):
        reset_config()

    def test_detector_creation(self):
        """Test creating detector."""
        self.assertIsNotNone(self.detector)
        self.assertIsNone(self.detector._context)

    def test_set_context(self):
        """Test setting planning context."""
        ctx = PlanningContext(flood_warning=True)
        self.detector.set_context(ctx)
        self.assertEqual(self.detector._context, ctx)

    def test_get_risk_assessment(self):
        """Test risk assessment."""
        ctx = PlanningContext(flood_warning=True, season=Season.SUMMER)
        self.detector.set_context(ctx)
        risks = self.detector.get_risk_assessment()
        self.assertGreater(risks['FLOOD_CONDITION'], 0.5)


class TestPlanningAwareScenarioGenerator(unittest.TestCase):
    """Test planning-aware scenario generator."""

    def setUp(self):
        self.gen = PlanningAwareScenarioGenerator(seed=42)

    def test_generate(self):
        """Test generating scenarios."""
        scenarios = list(self.gen.generate())
        self.assertGreater(len(scenarios), 100)

    def test_scenario_structure(self):
        """Test scenario has correct structure."""
        scenario = next(self.gen.generate())
        self.assertIsInstance(scenario, ScenarioSpec)
        self.assertEqual(scenario.category, "planning")
        self.assertIn('context_type', scenario.flow_params)

    def test_count(self):
        """Test scenario count."""
        count = self.gen.count()
        self.assertGreater(count, 200)


class TestSeasonalScenarioGenerator(unittest.TestCase):
    """Test seasonal scenario generator."""

    def setUp(self):
        self.gen = SeasonalScenarioGenerator(seed=42)

    def test_generate(self):
        """Test generating scenarios."""
        scenarios = list(self.gen.generate())
        self.assertGreater(len(scenarios), 50)

    def test_seasonal_variations(self):
        """Test different seasons are generated."""
        scenarios = list(self.gen.generate())
        seasons = set(s.flow_params.get('season') for s in scenarios)
        self.assertGreater(len(seasons), 1)

    def test_flood_drought_scenarios(self):
        """Test flood and drought scenarios are generated."""
        scenarios = list(self.gen.generate())
        has_flood = any(s.flow_params.get('is_flood') for s in scenarios)
        has_drought = any(s.flow_params.get('is_drought') for s in scenarios)
        self.assertTrue(has_flood)
        self.assertTrue(has_drought)


class TestPredictionScenarioGenerator(unittest.TestCase):
    """Test prediction scenario generator."""

    def setUp(self):
        self.gen = PredictionScenarioGenerator(seed=42)

    def test_generate(self):
        """Test generating scenarios."""
        scenarios = list(self.gen.generate())
        self.assertGreater(len(scenarios), 100)

    def test_accuracy_types(self):
        """Test different accuracy types are generated."""
        scenarios = list(self.gen.generate())
        accuracy_types = set(s.flow_params.get('accuracy_type') for s in scenarios)
        self.assertIn('accurate', accuracy_types)
        self.assertIn('under_15', accuracy_types)
        self.assertIn('sudden_change', accuracy_types)

    def test_count(self):
        """Test scenario count."""
        count = self.gen.count()
        self.assertEqual(count, 8 * 5 * 3)  # accuracies * flows * variations


class TestDispatchScheduleGenerator(unittest.TestCase):
    """Test dispatch schedule generator."""

    def setUp(self):
        self.gen = DispatchScheduleGenerator(seed=42)

    def test_generate(self):
        """Test generating scenarios."""
        scenarios = list(self.gen.generate())
        self.assertEqual(len(scenarios), 5 * 4)  # patterns * variations

    def test_patterns(self):
        """Test different patterns are generated."""
        scenarios = list(self.gen.generate())
        patterns = set(s.flow_params.get('pattern') for s in scenarios)
        self.assertIn('weekday_normal', patterns)
        self.assertIn('peak_demand', patterns)

    def test_events(self):
        """Test scenarios have events."""
        scenarios = list(self.gen.generate())
        for scenario in scenarios:
            self.assertGreater(len(scenario.events), 0)


class TestExtendedScenarioGenerator(unittest.TestCase):
    """Test extended scenario generator."""

    def setUp(self):
        self.gen = ExtendedScenarioGenerator(seed=42)

    def test_has_planning_generators(self):
        """Test extended generator has planning generators."""
        self.assertIn('planning', self.gen.generators)
        self.assertIn('seasonal', self.gen.generators)
        self.assertIn('prediction', self.gen.generators)
        self.assertIn('dispatch', self.gen.generators)

    def test_count_total(self):
        """Test total count includes planning scenarios."""
        counts = self.gen.count_total()
        self.assertIn('planning', counts)
        self.assertIn('seasonal', counts)
        self.assertIn('prediction', counts)
        self.assertIn('dispatch', counts)
        self.assertGreater(counts['total'], 7000)

    def test_generate_by_category(self):
        """Test generating by planning category."""
        planning = list(self.gen.generate_by_category('planning'))
        self.assertGreater(len(planning), 100)
        for s in planning:
            self.assertEqual(s.category, 'planning')


class TestIntegration(unittest.TestCase):
    """Integration tests for planning and prediction."""

    def setUp(self):
        reset_config()
        self.model = TangheSiphonModel()

    def tearDown(self):
        reset_config()

    def test_full_scenario_with_planning(self):
        """Test running scenario with planning context."""
        from src.control.integrated_controller import IntegratedController
        from src.control.scenario_generator import ScenarioTestRunner

        controller = IntegratedController(self.model)
        runner = ScenarioTestRunner(self.model, controller)

        # Create planning-based scenario
        gen = PlanningAwareScenarioGenerator(seed=42)
        scenario = next(gen.generate())

        result = runner.run_scenario(scenario, dt=0.1)
        self.assertIsNotNone(result)
        self.assertIn('max_flow_error', result.metrics)

    def test_detector_with_context_integration(self):
        """Test detector with planning context."""
        from src.control.scenario_advanced import ScenarioType

        detector = PredictionAwareDetector(self.model)
        ctx_gen = PlanningContextGenerator(seed=42)

        # Test with flood context
        flood_ctx = ctx_gen.generate_flood_scenario(0.0)
        detector.set_context(flood_ctx)

        # Verify risk assessment reflects flood warning
        risks = detector.get_risk_assessment()
        self.assertGreater(risks['FLOOD_CONDITION'], 0.5)

        # Test with drought context
        drought_ctx = ctx_gen.generate_drought_scenario(0.0)
        detector.set_context(drought_ctx)

        risks = detector.get_risk_assessment()
        self.assertGreater(risks['DROUGHT_CONDITION'], 0.5)


if __name__ == '__main__':
    unittest.main()
