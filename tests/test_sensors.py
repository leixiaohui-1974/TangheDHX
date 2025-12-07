# -*- coding: utf-8 -*-
"""
Unit Tests for Sensor Modules.

Tests sensor simulation including noise characteristics.
"""

import unittest
import numpy as np

from src.simulation.physics import TangheSiphonModel
from src.simulation.sensors import ADCPSensor, VibrationSensor
from src.simulation.actuators import GateController
from src.config import reset_config


class TestADCPSensor(unittest.TestCase):
    """Test cases for ADCPSensor class."""

    def setUp(self):
        """Set up test fixtures."""
        reset_config()
        self.model = TangheSiphonModel()
        self.sensor = ADCPSensor(self.model, 0)

    def tearDown(self):
        """Clean up after tests."""
        reset_config()

    def test_initialization(self):
        """Test sensor initializes correctly."""
        self.assertEqual(self.sensor.gate_index, 0)
        self.assertEqual(self.sensor.noise_std, 0.05)

    def test_invalid_gate_index(self):
        """Test invalid gate index raises error."""
        with self.assertRaises(ValueError):
            ADCPSensor(self.model, 5)

    def test_read_returns_float(self):
        """Test read returns a float value."""
        value = self.sensor.read()
        self.assertIsInstance(value, float)

    def test_read_non_negative(self):
        """Test read always returns non-negative value."""
        for _ in range(100):
            value = self.sensor.read()
            self.assertGreaterEqual(value, 0.0)

    def test_read_with_flow(self):
        """Test read returns velocity when flow is present."""
        self.model.gate_openings = np.array([2.0, 2.0, 2.0])
        self.model.step(np.array([2.0, 2.0, 2.0]), dt=0.1)

        # Take multiple readings to check noise
        readings = [self.sensor.read() for _ in range(50)]

        # Mean should be close to true velocity
        true_v = self.model.velocities[0]
        mean_reading = np.mean(readings)
        self.assertLess(abs(mean_reading - true_v), 0.1)

    def test_noise_characteristics(self):
        """Test noise has expected statistical properties."""
        self.model.velocities[0] = 2.0

        readings = [self.sensor.read() for _ in range(1000)]

        # Standard deviation should be approximately noise_std
        std = np.std(readings)
        self.assertLess(abs(std - 0.05), 0.02)


class TestVibrationSensor(unittest.TestCase):
    """Test cases for VibrationSensor class."""

    def setUp(self):
        """Set up test fixtures."""
        reset_config()
        self.model = TangheSiphonModel()
        self.sensor = VibrationSensor(self.model, 0)

    def tearDown(self):
        """Clean up after tests."""
        reset_config()

    def test_initialization(self):
        """Test sensor initializes correctly."""
        self.assertEqual(self.sensor.gate_index, 0)
        self.assertEqual(self.sensor.noise_std, 0.005)

    def test_invalid_gate_index(self):
        """Test invalid gate index raises error."""
        with self.assertRaises(ValueError):
            VibrationSensor(self.model, 10)

    def test_read_accel_returns_float(self):
        """Test read_accel returns a float value."""
        value = self.sensor.read_accel()
        self.assertIsInstance(value, float)

    def test_read_accel_non_negative(self):
        """Test read_accel always returns non-negative value."""
        for _ in range(100):
            value = self.sensor.read_accel()
            self.assertGreaterEqual(value, 0.0)

    def test_get_spectrum_peak_zero_when_no_vibration(self):
        """Test spectrum peak returns 0 when no vibration."""
        value = self.sensor.get_spectrum_peak()
        self.assertEqual(value, 0.0)

    def test_get_spectrum_peak_with_vibration(self):
        """Test spectrum peak returns frequency when vibrating."""
        self.model.vortex_freqs[0] = 2.8

        readings = [self.sensor.get_spectrum_peak() for _ in range(50)]
        mean_freq = np.mean(readings)

        self.assertLess(abs(mean_freq - 2.8), 0.1)

    def test_read_accel_with_vibration(self):
        """Test acceleration reading with vibration."""
        self.model.vibration_accel[0] = 0.5

        readings = [self.sensor.read_accel() for _ in range(50)]
        mean_accel = np.mean(readings)

        self.assertLess(abs(mean_accel - 0.5), 0.05)


class TestGateController(unittest.TestCase):
    """Test cases for GateController class."""

    def setUp(self):
        """Set up test fixtures."""
        reset_config()
        self.model = TangheSiphonModel()
        self.controller = GateController(self.model)

    def tearDown(self):
        """Clean up after tests."""
        reset_config()

    def test_initialization(self):
        """Test controller initializes correctly."""
        np.testing.assert_array_equal(
            self.controller.target_openings, [0.0, 0.0, 0.0]
        )

    def test_set_opening(self):
        """Test setting gate opening."""
        self.controller.set_opening(0, 2.5)
        self.assertEqual(self.controller.target_openings[0], 2.5)

    def test_set_opening_clamping(self):
        """Test opening is clamped to valid range."""
        self.controller.set_opening(0, 10.0)  # Above max
        self.assertEqual(self.controller.target_openings[0], 5.0)

        self.controller.set_opening(1, -1.0)  # Below min
        self.assertEqual(self.controller.target_openings[1], 0.0)

    def test_set_opening_invalid_gate(self):
        """Test invalid gate index raises error."""
        with self.assertRaises(ValueError):
            self.controller.set_opening(5, 1.0)

    def test_get_target_openings(self):
        """Test getting target openings returns copy."""
        self.controller.set_opening(0, 2.0)
        targets = self.controller.get_target_openings()

        # Modify returned array
        targets[0] = 99.0

        # Original should be unchanged
        self.assertEqual(self.controller.target_openings[0], 2.0)

    def test_set_all_openings(self):
        """Test setting all openings at once."""
        self.controller.set_all_openings(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(
            self.controller.target_openings, [1.0, 2.0, 3.0]
        )

    def test_set_all_openings_wrong_length(self):
        """Test wrong array length raises error."""
        with self.assertRaises(ValueError):
            self.controller.set_all_openings(np.array([1.0, 2.0]))

    def test_reset(self):
        """Test controller reset."""
        self.controller.set_all_openings(np.array([1.0, 2.0, 3.0]))
        self.controller.reset()
        np.testing.assert_array_equal(
            self.controller.target_openings, [0.0, 0.0, 0.0]
        )


if __name__ == '__main__':
    unittest.main()
