# -*- coding: utf-8 -*-
"""
Tests for Full Scenario Generation Framework.
"""

import unittest
import numpy as np

from src.control.scenario_generator import (
    FullScenarioGenerator,
    ParametricScenarioGenerator,
    FlowTransitionGenerator,
    FaultScenarioGenerator,
    CombinationScenarioGenerator,
    MonteCarloScenarioGenerator,
    TemporalScenarioGenerator,
    ScenarioSpec,
    ScenarioTestRunner,
    FlowRegime,
    HeadCondition,
    GateFaultType,
    TransitionPattern,
)
from src.simulation.physics import TangheSiphonModel
from src.control.integrated_controller import IntegratedController
from src.config import reset_config


class TestParametricGenerator(unittest.TestCase):
    """Test parametric scenario generator."""

    def setUp(self):
        self.gen = ParametricScenarioGenerator(
            parameters=['target_flow', 'head_upstream'],
            steps_per_param=3,
            seed=42
        )

    def test_count(self):
        """Test scenario count."""
        self.assertEqual(self.gen.count(), 9)  # 3 x 3

    def test_generate(self):
        """Test scenario generation."""
        scenarios = list(self.gen.generate())
        self.assertEqual(len(scenarios), 9)

    def test_scenario_structure(self):
        """Test generated scenario has correct structure."""
        scenario = next(self.gen.generate())
        self.assertIsInstance(scenario, ScenarioSpec)
        self.assertIsNotNone(scenario.scenario_id)
        self.assertIsNotNone(scenario.name)
        self.assertIsNotNone(scenario.category)

    def test_difficulty_range(self):
        """Test difficulty is in valid range."""
        for scenario in self.gen.generate():
            self.assertGreaterEqual(scenario.difficulty, 0.0)
            self.assertLessEqual(scenario.difficulty, 1.0)


class TestFlowTransitionGenerator(unittest.TestCase):
    """Test flow transition generator."""

    def setUp(self):
        self.gen = FlowTransitionGenerator(seed=42)

    def test_generate(self):
        """Test scenario generation."""
        scenarios = list(self.gen.generate())
        self.assertGreater(len(scenarios), 10)

    def test_transition_events(self):
        """Test transition scenarios have events."""
        for scenario in self.gen.generate():
            if scenario.flow_pattern in [
                TransitionPattern.STEP_UP, TransitionPattern.STEP_DOWN
            ]:
                self.assertGreater(len(scenario.events), 0)


class TestFaultScenarioGenerator(unittest.TestCase):
    """Test fault scenario generator."""

    def setUp(self):
        self.gen = FaultScenarioGenerator(seed=42)

    def test_generate(self):
        """Test scenario generation."""
        scenarios = list(self.gen.generate())
        self.assertGreater(len(scenarios), 100)

    def test_gate_faults(self):
        """Test gate fault scenarios."""
        gate_fault_scenarios = [
            s for s in self.gen.generate()
            if s.category == 'gate_fault'
        ]
        self.assertGreater(len(gate_fault_scenarios), 50)

    def test_multi_gate_faults(self):
        """Test multi-gate fault scenarios."""
        multi_fault_scenarios = [
            s for s in self.gen.generate()
            if s.category == 'multi_gate_fault'
        ]
        self.assertGreater(len(multi_fault_scenarios), 5)

    def test_sensor_faults(self):
        """Test sensor fault scenarios."""
        sensor_fault_scenarios = [
            s for s in self.gen.generate()
            if s.category == 'sensor_fault'
        ]
        self.assertGreater(len(sensor_fault_scenarios), 30)


class TestCombinationGenerator(unittest.TestCase):
    """Test combination scenario generator."""

    def setUp(self):
        self.gen = CombinationScenarioGenerator(seed=42)

    def test_generate(self):
        """Test scenario generation."""
        scenarios = list(self.gen.generate())
        self.assertGreater(len(scenarios), 1000)

    def test_unique_scenarios(self):
        """Test scenarios are unique."""
        scenarios = list(self.gen.generate())
        ids = [s.scenario_id for s in scenarios]
        self.assertEqual(len(ids), len(set(ids)))

    def test_difficulty_distribution(self):
        """Test difficulty has good distribution."""
        scenarios = list(self.gen.generate())
        difficulties = [s.difficulty for s in scenarios]

        # Should have scenarios across difficulty levels
        easy = sum(1 for d in difficulties if d < 0.3)
        medium = sum(1 for d in difficulties if 0.3 <= d < 0.6)
        hard = sum(1 for d in difficulties if d >= 0.6)

        self.assertGreater(easy, 0)
        self.assertGreater(medium, 0)
        self.assertGreater(hard, 0)


class TestMonteCarloGenerator(unittest.TestCase):
    """Test Monte Carlo scenario generator."""

    def setUp(self):
        self.gen = MonteCarloScenarioGenerator(n_scenarios=100, seed=42)

    def test_count(self):
        """Test correct number of scenarios."""
        self.assertEqual(self.gen.count(), 100)

    def test_generate(self):
        """Test scenario generation."""
        scenarios = list(self.gen.generate())
        self.assertEqual(len(scenarios), 100)

    def test_randomness(self):
        """Test scenarios are random (different values)."""
        scenarios = list(self.gen.generate())
        flows = [s.target_flow for s in scenarios]
        unique_flows = len(set(int(f) for f in flows))
        self.assertGreater(unique_flows, 20)

    def test_reproducibility(self):
        """Test same seed produces same scenarios."""
        gen1 = MonteCarloScenarioGenerator(n_scenarios=10, seed=123)
        gen2 = MonteCarloScenarioGenerator(n_scenarios=10, seed=123)

        scenarios1 = list(gen1.generate())
        scenarios2 = list(gen2.generate())

        for s1, s2 in zip(scenarios1, scenarios2):
            self.assertEqual(s1.target_flow, s2.target_flow)


class TestTemporalGenerator(unittest.TestCase):
    """Test temporal scenario generator."""

    def setUp(self):
        self.gen = TemporalScenarioGenerator(seed=42)

    def test_generate(self):
        """Test scenario generation."""
        scenarios = list(self.gen.generate())
        self.assertEqual(len(scenarios), 18)  # 6 patterns x 3 variations

    def test_events(self):
        """Test temporal scenarios have events."""
        scenarios = list(self.gen.generate())
        for scenario in scenarios:
            self.assertGreater(len(scenario.events), 0)


class TestFullScenarioGenerator(unittest.TestCase):
    """Test full scenario generator."""

    def setUp(self):
        self.gen = FullScenarioGenerator(seed=42)

    def test_count_total(self):
        """Test total scenario count."""
        counts = self.gen.count_total()
        self.assertIn('total', counts)
        self.assertGreater(counts['total'], 5000)

    def test_generate_all(self):
        """Test generating all scenarios (limited for speed)."""
        count = 0
        for scenario in self.gen.generate_all():
            count += 1
            if count >= 100:
                break
        self.assertEqual(count, 100)

    def test_generate_by_category(self):
        """Test filtering by category."""
        transition_scenarios = list(self.gen.generate_by_category('transition'))
        self.assertGreater(len(transition_scenarios), 10)
        for s in transition_scenarios:
            self.assertEqual(s.category, 'transition')

    def test_generate_by_difficulty(self):
        """Test filtering by difficulty."""
        count = 0
        for scenario in self.gen.generate_by_difficulty(0.5, 0.8):
            self.assertGreaterEqual(scenario.difficulty, 0.5)
            self.assertLessEqual(scenario.difficulty, 0.8)
            count += 1
            if count >= 50:
                break
        self.assertGreater(count, 0)

    def test_generate_random_subset(self):
        """Test random subset generation."""
        subset = self.gen.generate_random_subset(50)
        self.assertEqual(len(subset), 50)

    def test_statistics(self):
        """Test statistics collection."""
        stats = self.gen.get_statistics()
        self.assertIn('total_count', stats)
        self.assertIn('categories', stats)
        self.assertIn('difficulty', stats)
        self.assertIn('difficulty_distribution', stats)


class TestScenarioTestRunner(unittest.TestCase):
    """Test scenario test runner."""

    def setUp(self):
        reset_config()
        self.model = TangheSiphonModel()
        self.controller = IntegratedController(self.model)
        self.runner = ScenarioTestRunner(self.model, self.controller)

    def tearDown(self):
        reset_config()

    def test_run_simple_scenario(self):
        """Test running a simple scenario."""
        spec = ScenarioSpec(
            scenario_id="TEST_001",
            name="Simple Test",
            description="Basic test scenario",
            category="test",
            difficulty=0.2,
            target_flow=80.0,
            duration=5.0,  # Short duration for testing
        )

        result = self.runner.run_scenario(spec, dt=0.1)

        self.assertIsNotNone(result)
        self.assertEqual(result.scenario_id, "TEST_001")
        self.assertIn('max_flow_error', result.metrics)
        self.assertIn('max_vibration', result.metrics)

    def test_run_batch(self):
        """Test running batch of scenarios."""
        specs = [
            ScenarioSpec(
                scenario_id=f"BATCH_{i}",
                name=f"Batch Test {i}",
                description="Batch test",
                category="test",
                difficulty=0.3,
                target_flow=80.0 + i * 10,
                duration=2.0,
            )
            for i in range(3)
        ]

        results = self.runner.run_batch(specs)

        self.assertEqual(len(results), 3)

    def test_get_summary(self):
        """Test summary generation."""
        spec = ScenarioSpec(
            scenario_id="SUM_001",
            name="Summary Test",
            description="Test",
            category="test",
            difficulty=0.2,
            duration=2.0,
        )
        self.runner.run_scenario(spec, dt=0.1)

        summary = self.runner.get_summary()
        self.assertIn('total', summary)
        self.assertIn('passed', summary)
        self.assertIn('pass_rate', summary)


class TestScenarioSpecSerialization(unittest.TestCase):
    """Test scenario spec serialization."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        spec = ScenarioSpec(
            scenario_id="DICT_001",
            name="Dict Test",
            description="Test serialization",
            category="test",
            difficulty=0.5,
            target_flow=100.0,
            gate_faults=(GateFaultType.STUCK_PARTIAL, GateFaultType.NONE, GateFaultType.NONE),
        )

        d = spec.to_dict()

        self.assertEqual(d['id'], "DICT_001")
        self.assertEqual(d['name'], "Dict Test")
        self.assertEqual(d['initial']['target_flow'], 100.0)
        self.assertEqual(d['gate_faults'][0], 'STUCK_PARTIAL')


if __name__ == '__main__':
    unittest.main()
