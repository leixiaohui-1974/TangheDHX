# -*- coding: utf-8 -*-
"""
Unit Tests for Physics Module.

Tests the physical simulation model including hydraulics,
vortex-induced vibration, and fault injection.
"""

import unittest
import numpy as np

from src.simulation.physics import TangheSiphonModel
from src.config import PhysicsConfig, reset_config


class TestTangheSiphonModel(unittest.TestCase):
    """Test cases for TangheSiphonModel class."""

    def setUp(self):
        """Set up test fixtures."""
        reset_config()
        self.model = TangheSiphonModel()

    def tearDown(self):
        """Clean up after tests."""
        reset_config()

    def test_initialization(self):
        """Test model initializes with correct default values."""
        self.assertEqual(self.model.num_gates, 3)
        self.assertEqual(self.model.width, 6.0)
        self.assertEqual(self.model.head_upstream, 10.0)
        self.assertEqual(self.model.head_downstream, 8.0)
        self.assertEqual(self.model.time, 0.0)

        np.testing.assert_array_equal(self.model.gate_openings, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(self.model.flow_rates, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(self.model.velocities, [0.0, 0.0, 0.0])

    def test_step_time_advance(self):
        """Test that step advances simulation time."""
        initial_time = self.model.time
        self.model.step(np.array([1.0, 1.0, 1.0]), dt=0.1)
        self.assertAlmostEqual(self.model.time, initial_time + 0.1)

    def test_step_wrong_input_shape(self):
        """Test that step raises error for wrong input shape."""
        with self.assertRaises(ValueError):
            self.model.step(np.array([1.0, 1.0]))  # Only 2 values

    def test_gate_opening_limits(self):
        """Test gate openings are clamped to valid range."""
        # Try to set opening above max (5.0m)
        self.model.step(np.array([10.0, 10.0, 10.0]), dt=1.0)
        for i in range(3):
            self.assertLessEqual(self.model.gate_openings[i], 5.0)

    def test_gate_actuator_speed(self):
        """Test gate movement respects speed limit."""
        # Gate starts at 0, command to 5.0
        # Max speed is 0.05 m/s, dt=0.1, so max move = 0.005
        self.model.step(np.array([5.0, 5.0, 5.0]), dt=0.1)
        for opening in self.model.gate_openings:
            self.assertLessEqual(opening, 0.01)  # Small move only

    def test_hydraulics_flow_calculation(self):
        """Test hydraulic flow rate calculation."""
        # Set opening directly
        self.model.gate_openings = np.array([1.0, 1.0, 1.0])
        self.model.step(np.array([1.0, 1.0, 1.0]), dt=0.1)

        # With dH=2m, e=1.0m, flow should be positive
        for flow in self.model.flow_rates:
            self.assertGreater(flow, 0)

    def test_velocity_calculation(self):
        """Test velocity calculation from flow."""
        self.model.gate_openings = np.array([2.0, 2.0, 2.0])
        self.model.step(np.array([2.0, 2.0, 2.0]), dt=0.1)

        for velocity in self.model.velocities:
            self.assertGreater(velocity, 0)

    def test_resonance_detection(self):
        """Test resonance is detected at expected velocity."""
        # v = 2.6 m/s should cause f = 2.8 Hz (resonance)
        # Calculate opening needed: e = v * A_eff / (Cd * B * sqrt(2*g*dH))
        Cd = 0.7
        B = 6.0
        g = 9.81
        dH = 2.0
        A_eff = 16.6
        K = Cd * B * np.sqrt(2 * g * dH) / A_eff
        target_v = 2.6
        target_e = target_v / K

        # Force opening directly
        self.model.gate_openings = np.array([target_e, target_e, target_e])
        self.model.step(self.model.gate_openings, dt=0.1)

        # Check frequency is in resonance range
        for freq in self.model.vortex_freqs:
            self.assertGreater(freq, 2.5)
            self.assertLess(freq, 3.1)

        # Check vibration is amplified (lock-in)
        for vib in self.model.vibration_accel:
            self.assertGreater(vib, 0.3)  # Should be amplified

    def test_no_resonance_at_low_velocity(self):
        """Test no resonance at low velocity."""
        self.model.gate_openings = np.array([0.5, 0.5, 0.5])
        self.model.step(self.model.gate_openings, dt=0.1)

        # At low velocity, vibration should be minimal
        for vib in self.model.vibration_accel:
            self.assertLess(vib, 0.1)

    def test_fault_injection_stuck(self):
        """Test gate stuck fault injection."""
        self.model.inject_fault(0, 'stuck')
        self.assertTrue(self.model.gate_stuck[0])

        # Try to move gate 0 - should not move
        initial_opening = self.model.gate_openings[0]
        self.model.step(np.array([5.0, 0.0, 0.0]), dt=1.0)
        self.assertEqual(self.model.gate_openings[0], initial_opening)

    def test_fault_injection_clear(self):
        """Test fault clearing."""
        self.model.inject_fault(1, 'stuck')
        self.assertTrue(self.model.gate_stuck[1])

        self.model.inject_fault(1, 'clear')
        self.assertFalse(self.model.gate_stuck[1])

    def test_fault_injection_invalid_gate(self):
        """Test fault injection with invalid gate index."""
        with self.assertRaises(ValueError):
            self.model.inject_fault(5, 'stuck')

    def test_fault_injection_invalid_type(self):
        """Test fault injection with invalid fault type."""
        with self.assertRaises(ValueError):
            self.model.inject_fault(0, 'invalid')

    def test_get_state(self):
        """Test get_state returns correct structure."""
        state = self.model.get_state()

        self.assertIn('timestamp', state)
        self.assertIn('openings', state)
        self.assertIn('flows', state)
        self.assertIn('velocities', state)
        self.assertIn('frequencies', state)
        self.assertIn('vibrations', state)
        self.assertIn('total_flow', state)

        self.assertEqual(len(state['openings']), 3)
        self.assertEqual(len(state['flows']), 3)

    def test_reset(self):
        """Test model reset functionality."""
        # Modify state
        self.model.gate_openings = np.array([2.0, 2.0, 2.0])
        self.model.time = 100.0
        self.model.gate_stuck[0] = True

        # Reset
        self.model.reset()

        # Verify reset
        self.assertEqual(self.model.time, 0.0)
        np.testing.assert_array_equal(self.model.gate_openings, [0.0, 0.0, 0.0])
        self.assertFalse(any(self.model.gate_stuck))

    def test_zero_head_difference(self):
        """Test behavior with zero head difference."""
        self.model.head_upstream = 8.0  # Same as downstream
        self.model.gate_openings = np.array([1.0, 1.0, 1.0])
        self.model.step(self.model.gate_openings, dt=0.1)

        # No flow with zero head difference
        for flow in self.model.flow_rates:
            self.assertEqual(flow, 0.0)

    def test_negative_head_difference(self):
        """Test behavior with negative head difference."""
        self.model.head_upstream = 6.0  # Less than downstream
        self.model.gate_openings = np.array([1.0, 1.0, 1.0])
        self.model.step(self.model.gate_openings, dt=0.1)

        # Flow should be zero (no reverse flow in model)
        for flow in self.model.flow_rates:
            self.assertEqual(flow, 0.0)


class TestPhysicsConfig(unittest.TestCase):
    """Test cases for physics configuration."""

    def test_custom_config(self):
        """Test model with custom configuration."""
        custom_cfg = PhysicsConfig(
            num_gates=4,
            gate_width=8.0,
            default_head_upstream=15.0
        )
        model = TangheSiphonModel(config=custom_cfg)

        self.assertEqual(model.num_gates, 4)
        self.assertEqual(model.width, 8.0)
        self.assertEqual(model.head_upstream, 15.0)


if __name__ == '__main__':
    unittest.main()
