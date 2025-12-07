# -*- coding: utf-8 -*-
"""
Tests for IntegratedController with scenario-aware adaptation.
"""

import unittest
import numpy as np

from src.simulation.physics import TangheSiphonModel
from src.control.integrated_controller import (
    IntegratedController,
    ScenarioType,
    MPCObjective,
    MPCConstraints,
    ScenarioControlConfig,
)
from src.config import reset_config


class TestIntegratedController(unittest.TestCase):
    """Test cases for IntegratedController."""

    def setUp(self):
        reset_config()
        self.model = TangheSiphonModel()
        self.controller = IntegratedController(self.model)

    def tearDown(self):
        reset_config()

    def test_initialization(self):
        """Test controller initializes correctly."""
        self.assertIsNotNone(self.controller._mpc)
        self.assertIsNotNone(self.controller._pid)
        self.assertIsNotNone(self.controller._scenario_mgr)

    def test_set_target_flow(self):
        """Test setting target flow."""
        self.controller.set_target_flow(120.0)
        self.assertEqual(self.controller._target_flow, 120.0)

    def test_update_returns_diagnostics(self):
        """Test update returns comprehensive diagnostics."""
        result = self.controller.update(dt=0.1)

        # Check all expected keys
        self.assertIn('timestamp', result)
        self.assertIn('scenario', result)
        self.assertIn('control', result)
        self.assertIn('mpc_config', result)
        self.assertIn('constraints', result)
        self.assertIn('state', result)
        self.assertIn('performance', result)
        self.assertIn('adaptation', result)

    def test_scenario_detection(self):
        """Test automatic scenario detection."""
        # Run enough steps to build history
        for _ in range(30):
            self.controller.update(dt=0.1)

        scenario = self.controller.get_current_scenario()
        self.assertIsInstance(scenario, ScenarioType)

    def test_scenario_config_application(self):
        """Test scenario-specific config is applied."""
        # Force high vibration scenario
        self.model.vibration_accel = np.array([0.8, 0.8, 0.8])

        for _ in range(30):
            result = self.controller.update(dt=0.1)

        # Should be in resonance crossing
        self.assertEqual(
            self.controller.get_current_scenario(),
            ScenarioType.RESONANCE_CROSSING
        )

        # Check MPC weights changed
        mpc_config = result['mpc_config']
        self.assertGreater(mpc_config['gamma'], 10.0)  # Higher spectral weight

    def test_force_scenario(self):
        """Test forcing a specific scenario."""
        self.controller.force_scenario(ScenarioType.FLOOD_CONDITION)
        self.assertEqual(
            self.controller.get_current_scenario(),
            ScenarioType.FLOOD_CONDITION
        )

    def test_target_flow_modifier(self):
        """Test target flow modifier is correctly defined in configs."""
        # Verify flood scenario config has modifier > 1
        flood_config = IntegratedController.SCENARIO_CONFIGS.get(ScenarioType.FLOOD_CONDITION)
        self.assertIsNotNone(flood_config)
        self.assertGreater(flood_config.target_flow_modifier, 1.0)

        # Verify drought scenario config has modifier < 1
        drought_config = IntegratedController.SCENARIO_CONFIGS.get(ScenarioType.DROUGHT_CONDITION)
        self.assertIsNotNone(drought_config)
        self.assertLess(drought_config.target_flow_modifier, 1.0)

    def test_constraint_enforcement(self):
        """Test constraints are enforced."""
        self.controller.set_target_flow(200.0)

        for _ in range(50):
            result = self.controller.update(dt=0.1)

        outputs = result['control']['outputs']
        max_opening = result['constraints']['max_opening']

        for output in outputs:
            self.assertLessEqual(output, max_opening)

    def test_history_storage(self):
        """Test history is stored."""
        for _ in range(20):
            self.controller.update(dt=0.1)

        history = self.controller.get_history(10)
        self.assertEqual(len(history), 10)

    def test_emergency_shutdown(self):
        """Test emergency shutdown scenario config is correct."""
        # Verify emergency shutdown config
        shutdown_config = IntegratedController.SCENARIO_CONFIGS.get(ScenarioType.EMERGENCY_SHUTDOWN)
        self.assertIsNotNone(shutdown_config)
        self.assertEqual(shutdown_config.target_flow_modifier, 0.0)
        self.assertEqual(shutdown_config.emergency_action, "shutdown")

    def test_gate_stuck_fallback_to_pid(self):
        """Test fallback to PID when gate stuck."""
        self.controller.force_scenario(ScenarioType.GATE_STUCK)

        config = self.controller.get_current_config()
        self.assertTrue(config.use_pid_backup)

    def test_callback_registration(self):
        """Test callback registration."""
        callback_called = [False]

        def on_scenario_change(old, new):
            callback_called[0] = True

        self.controller.register_callback('scenario_change', on_scenario_change)
        self.controller.force_scenario(ScenarioType.FLOOD_CONDITION)

        self.assertTrue(callback_called[0])

    def test_reset(self):
        """Test controller reset."""
        self.controller.set_target_flow(150.0)
        self.controller.force_scenario(ScenarioType.FLOOD_CONDITION)

        for _ in range(20):
            self.controller.update(dt=0.1)

        self.controller.reset()

        self.assertEqual(self.controller._time, 0.0)
        self.assertEqual(len(self.controller._history), 0)

    def test_get_scenario_configs(self):
        """Test getting all scenario configs."""
        configs = self.controller.get_scenario_configs()

        # Should have multiple scenarios
        self.assertGreater(len(configs), 10)

        # Check structure
        for name, config in configs.items():
            self.assertIn('description', config)
            self.assertIn('objective', config)


class TestMPCObjective(unittest.TestCase):
    """Test cases for MPCObjective dataclass."""

    def test_default_values(self):
        """Test default objective values."""
        obj = MPCObjective()
        self.assertEqual(obj.alpha, 1.0)
        self.assertEqual(obj.beta, 0.1)
        self.assertEqual(obj.gamma, 10.0)

    def test_custom_values(self):
        """Test custom objective values."""
        obj = MPCObjective(alpha=2.0, beta=0.5, gamma=20.0, delta=5.0)
        self.assertEqual(obj.alpha, 2.0)
        self.assertEqual(obj.delta, 5.0)


class TestMPCConstraints(unittest.TestCase):
    """Test cases for MPCConstraints dataclass."""

    def test_default_values(self):
        """Test default constraint values."""
        cons = MPCConstraints()
        self.assertEqual(cons.min_opening, 0.0)
        self.assertEqual(cons.max_opening, 5.0)
        self.assertEqual(cons.max_rate, 0.5)

    def test_resonance_band(self):
        """Test resonance band constraints."""
        cons = MPCConstraints()
        self.assertEqual(cons.resonance_low, 2.3)
        self.assertEqual(cons.resonance_high, 3.0)


class TestScenarioControlConfig(unittest.TestCase):
    """Test cases for ScenarioControlConfig dataclass."""

    def test_creation(self):
        """Test creating scenario config."""
        config = ScenarioControlConfig(
            scenario=ScenarioType.NORMAL_MEDIUM_FLOW,
            objective=MPCObjective(),
            constraints=MPCConstraints(),
            description="Test scenario"
        )
        self.assertEqual(config.scenario, ScenarioType.NORMAL_MEDIUM_FLOW)
        self.assertEqual(config.target_flow_modifier, 1.0)

    def test_emergency_config(self):
        """Test emergency config."""
        config = ScenarioControlConfig(
            scenario=ScenarioType.EMERGENCY_SHUTDOWN,
            objective=MPCObjective(alpha=0.0),
            constraints=MPCConstraints(),
            emergency_action="shutdown",
            target_flow_modifier=0.0
        )
        self.assertEqual(config.emergency_action, "shutdown")
        self.assertEqual(config.target_flow_modifier, 0.0)


class TestScenarioTransitions(unittest.TestCase):
    """Test scenario transition handling."""

    def setUp(self):
        reset_config()
        self.model = TangheSiphonModel()
        self.controller = IntegratedController(self.model)

    def tearDown(self):
        reset_config()

    def test_normal_to_resonance(self):
        """Test transition from normal to resonance."""
        # Start in normal
        for _ in range(20):
            self.controller.update(dt=0.1)

        # Induce vibration
        self.model.vibration_accel = np.array([0.5, 0.5, 0.5])

        for _ in range(30):
            self.controller.update(dt=0.1)

        scenario = self.controller.get_current_scenario()
        self.assertEqual(scenario, ScenarioType.RESONANCE_CROSSING)

    def test_transition_smooth_control(self):
        """Test control outputs remain smooth during transitions."""
        outputs_before = None

        for step in range(50):
            result = self.controller.update(dt=0.1)

            if step == 25:
                # Force scenario change mid-run
                self.controller.force_scenario(ScenarioType.HEAD_SURGE)

            if outputs_before is not None:
                # Check rate limit (no sudden jumps)
                outputs = np.array(result['control']['outputs'])
                max_change = np.max(np.abs(outputs - outputs_before))
                self.assertLess(max_change, 0.5)  # Max rate per step

            outputs_before = np.array(result['control']['outputs'])


if __name__ == '__main__':
    unittest.main()
