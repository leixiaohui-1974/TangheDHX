# -*- coding: utf-8 -*-
"""
Unit Tests for Control Modules.

Tests the MPC controller, local controller, and scenario manager.
"""

import unittest
import numpy as np

from src.simulation.physics import TangheSiphonModel
from src.control.mpc import SpectralMPC
from src.control.local import LocalController
from src.control.manager import ScenarioManager
from src.config import reset_config


class TestSpectralMPC(unittest.TestCase):
    """Test cases for SpectralMPC class."""

    def setUp(self):
        """Set up test fixtures."""
        reset_config()
        self.model = TangheSiphonModel()
        self.mpc = SpectralMPC(self.model)

    def tearDown(self):
        """Clean up after tests."""
        reset_config()

    def test_initialization(self):
        """Test MPC initializes with correct default weights."""
        self.assertEqual(self.mpc.alpha, 1.0)
        self.assertEqual(self.mpc.beta, 0.1)
        self.assertEqual(self.mpc.gamma, 10.0)

    def test_get_target_openings_returns_array(self):
        """Test get_target_openings returns numpy array."""
        target_flow = 100.0
        current_openings = np.array([0.5, 0.5, 0.5])
        head_diff = 2.0

        result = self.mpc.get_target_openings(target_flow, current_openings, head_diff)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 3)

    def test_get_target_openings_bounds(self):
        """Test MPC respects gate opening bounds."""
        target_flow = 500.0  # High target
        current_openings = np.array([0.5, 0.5, 0.5])
        head_diff = 2.0

        result = self.mpc.get_target_openings(target_flow, current_openings, head_diff)

        for opening in result:
            self.assertGreaterEqual(opening, 0.0)
            self.assertLessEqual(opening, 5.0)

    def test_mpc_resonance_avoidance(self):
        """Test MPC avoids resonance zone."""
        # Target flow that would require v=2.6 m/s if symmetric
        # v=2.6 -> Q = v * A_eff = 2.6 * 16.6 = 43.16 per gate
        # Total Q = 130
        target_flow = 130.0
        head_diff = 2.0
        current_openings = np.array([0.5, 0.5, 0.5])

        targets = self.mpc.get_target_openings(target_flow, current_openings, head_diff)

        # Calculate velocities for the targets
        Cd = 0.7
        g = 9.81
        A_eff = 16.6
        width = 6.0

        velocities = []
        for e in targets:
            q = Cd * width * e * np.sqrt(2 * g * head_diff)
            v = q / A_eff
            velocities.append(v)

        # With spectral avoidance, MPC should choose asymmetric openings
        # to avoid having all gates at resonance velocity
        self.assertGreater(np.std(targets), 0.01)

    def test_mpc_flow_tracking(self):
        """Test MPC achieves approximate flow target."""
        target_flow = 80.0  # Safe target (no resonance)
        head_diff = 2.0
        current_openings = np.array([1.0, 1.0, 1.0])

        targets = self.mpc.get_target_openings(target_flow, current_openings, head_diff)

        # Calculate total flow
        Cd = 0.7
        g = 9.81
        width = 6.0

        total_flow = sum(
            Cd * width * e * np.sqrt(2 * g * head_diff)
            for e in targets
        )

        # Should be reasonably close to target
        self.assertLess(abs(total_flow - target_flow), 10.0)

    def test_mpc_wrong_input_shape(self):
        """Test MPC raises error for wrong input shape."""
        with self.assertRaises(ValueError):
            self.mpc.get_target_openings(100.0, np.array([1.0, 1.0]), 2.0)

    def test_mpc_negative_head_diff(self):
        """Test MPC handles negative head difference."""
        # Should not raise, but will log warning
        result = self.mpc.get_target_openings(
            100.0, np.array([1.0, 1.0, 1.0]), -1.0
        )
        self.assertEqual(len(result), 3)

    def test_mpc_reset(self):
        """Test MPC reset functionality."""
        self.mpc.last_target_openings = np.array([1.0, 2.0, 3.0])
        self.mpc.reset()
        np.testing.assert_array_equal(
            self.mpc.last_target_openings, [0.0, 0.0, 0.0]
        )


class TestLocalController(unittest.TestCase):
    """Test cases for LocalController class."""

    def setUp(self):
        """Set up test fixtures."""
        reset_config()
        self.model = TangheSiphonModel()
        self.local_ctrl = LocalController(self.model)

    def tearDown(self):
        """Clean up after tests."""
        reset_config()

    def test_initialization(self):
        """Test local controller initializes correctly."""
        self.assertEqual(len(self.local_ctrl.dithering_active), 3)
        self.assertEqual(len(self.local_ctrl.dithering_phase), 3)
        self.assertFalse(any(self.local_ctrl.dithering_active))

    def test_update_pass_through(self):
        """Test pass-through behavior when no vibration."""
        mpc_targets = np.array([1.0, 2.0, 3.0])
        result = self.local_ctrl.update(mpc_targets, dt=0.1)

        # Without vibration, should pass through unchanged
        np.testing.assert_array_almost_equal(result, mpc_targets)

    def test_dithering_activation(self):
        """Test dithering activates on high vibration."""
        # Set high vibration on gate 0
        self.model.vibration_accel[0] = 0.2  # Above threshold

        mpc_targets = np.array([1.0, 1.0, 1.0])
        self.local_ctrl.update(mpc_targets, dt=0.1)

        self.assertTrue(self.local_ctrl.dithering_active[0])
        self.assertFalse(self.local_ctrl.dithering_active[1])
        self.assertFalse(self.local_ctrl.dithering_active[2])

    def test_dithering_perturbation(self):
        """Test dithering adds perturbation to target."""
        self.model.vibration_accel[0] = 0.2  # High vibration

        mpc_targets = np.array([1.0, 1.0, 1.0])

        # Collect multiple results to verify perturbation occurs
        results = []
        for _ in range(20):
            result = self.local_ctrl.update(mpc_targets, dt=0.1)
            results.append(result[0])

        # Gate 0 should have varying values due to dithering
        # Check that not all values are exactly 1.0
        self.assertTrue(
            any(abs(r - 1.0) > 0.001 for r in results),
            "Dithering should cause perturbation"
        )

    def test_dithering_deactivation(self):
        """Test dithering deactivates when vibration drops."""
        # Activate dithering
        self.model.vibration_accel[0] = 0.2
        self.local_ctrl.update(np.array([1.0, 1.0, 1.0]), dt=0.1)
        self.assertTrue(self.local_ctrl.dithering_active[0])

        # Lower vibration below threshold
        self.model.vibration_accel[0] = 0.03
        self.local_ctrl.update(np.array([1.0, 1.0, 1.0]), dt=0.1)
        self.assertFalse(self.local_ctrl.dithering_active[0])

    def test_reset(self):
        """Test local controller reset."""
        self.local_ctrl.dithering_active = [True, True, False]
        self.local_ctrl.dithering_phase = [1.0, 2.0, 3.0]

        self.local_ctrl.reset()

        self.assertFalse(any(self.local_ctrl.dithering_active))
        np.testing.assert_array_equal(self.local_ctrl.dithering_phase, [0.0, 0.0, 0.0])


class TestScenarioManager(unittest.TestCase):
    """Test cases for ScenarioManager class."""

    def setUp(self):
        """Set up test fixtures."""
        reset_config()
        self.model = TangheSiphonModel()
        self.mgr = ScenarioManager(self.model)

    def tearDown(self):
        """Clean up after tests."""
        reset_config()

    def test_initialization(self):
        """Test scenario manager initializes correctly."""
        self.assertIsNone(self.mgr.active_scenario)
        self.assertEqual(self.mgr.scenario_start_time, 0.0)

    def test_set_scenario_valid(self):
        """Test setting a valid scenario."""
        self.mgr.set_scenario('S1.1')
        self.assertEqual(self.mgr.active_scenario, 'S1.1')

    def test_set_scenario_invalid(self):
        """Test setting an invalid scenario raises error."""
        with self.assertRaises(ValueError):
            self.mgr.set_scenario('INVALID')

    def test_scenario_s1_1_normal(self):
        """Test S1.1 Normal Operation scenario."""
        self.mgr.set_scenario('S1.1')
        self.assertEqual(self.model.head_upstream, 10.0)

    def test_scenario_s4_1_blockage(self):
        """Test S4.1 Trash Rack Blockage scenario."""
        self.mgr.set_scenario('S4.1')
        self.assertEqual(self.model.head_upstream, 8.5)

    def test_scenario_s4_2_stuck_gate(self):
        """Test S4.2 Gate Stuck scenario."""
        self.mgr.set_scenario('S4.2')
        self.model.time = 0

        # Run for 15 seconds
        for _ in range(150):
            self.model.step(np.array([1.0, 1.0, 1.0]), dt=0.1)
            self.mgr.update()

        # Gate 2 (index 1) should be stuck
        self.assertTrue(self.model.gate_stuck[1])

    def test_scenario_resets_faults(self):
        """Test setting a new scenario resets faults."""
        self.model.gate_stuck = [True, True, True]
        self.mgr.set_scenario('S1.1')
        self.assertFalse(any(self.model.gate_stuck))

    def test_get_elapsed_time(self):
        """Test elapsed time calculation."""
        self.mgr.set_scenario('S1.1')
        self.model.time = 10.0

        elapsed = self.mgr.get_elapsed_time()
        self.assertEqual(elapsed, 10.0)

    def test_reset(self):
        """Test scenario manager reset."""
        self.mgr.set_scenario('S1.1')
        self.mgr.reset()

        self.assertIsNone(self.mgr.active_scenario)
        self.assertEqual(self.mgr.scenario_start_time, 0.0)

    def test_available_scenarios(self):
        """Test all available scenarios are defined."""
        expected = {'S1.1', 'S2.1', 'S4.1', 'S4.2'}
        self.assertEqual(set(ScenarioManager.SCENARIOS.keys()), expected)


if __name__ == '__main__':
    unittest.main()
