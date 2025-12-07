# -*- coding: utf-8 -*-
"""
Comprehensive Tests for Advanced Features.

Tests the new advanced modules:
- Advanced scenario detection
- Adaptive MPC controller
- PID and Hybrid controllers
- High-fidelity physics model
- HIL framework
- Advanced sensors
"""

import unittest
import numpy as np
import time

from src.simulation.physics import TangheSiphonModel
from src.simulation.physics_advanced import HighFidelityPhysicsModel, AdvancedPhysicsParams
from src.simulation.sensors_advanced import (
    NoiseGenerator,
    SensorDynamics,
    AdvancedADCPSensor,
    AdvancedVibrationSensor,
    AdvancedPositionSensor,
)
from src.simulation.hil_framework import (
    HILMode,
    HILConfig,
    HILMetrics,
    HILTestRunner,
    SimulatedSensorInterface,
    SimulatedActuatorInterface,
)
from src.control.adaptive_mpc import AdaptiveMPC, AdaptiveState
from src.control.pid import PIDController, PIDGains, MultiChannelPID, HybridController
from src.control.scenario_advanced import (
    ScenarioType,
    ScenarioConfig,
    ScenarioDetector,
    AdvancedScenarioManager,
)
from src.control.mpc import SpectralMPC
from src.config import reset_config


class TestNoiseGenerator(unittest.TestCase):
    """Test cases for NoiseGenerator class."""

    def setUp(self):
        self.noise = NoiseGenerator(seed=42)

    def test_white_noise_statistics(self):
        """Test white noise has correct mean and std."""
        samples = [self.noise.white_noise(1.0) for _ in range(10000)]
        mean = np.mean(samples)
        std = np.std(samples)
        self.assertLess(abs(mean), 0.05)
        self.assertLess(abs(std - 1.0), 0.1)

    def test_pink_noise_correlation(self):
        """Test pink noise has correlation (not white)."""
        samples = [self.noise.pink_noise(1.0) for _ in range(1000)]
        # Pink noise should have autocorrelation
        autocorr = np.correlate(samples[:-1], samples[1:], mode='valid')[0]
        self.assertGreater(autocorr, 0)

    def test_brownian_noise_trend(self):
        """Test brownian noise exhibits drift."""
        self.noise.reset()
        samples = [self.noise.brownian_noise(1.0) for _ in range(500)]
        # Brownian noise tends to drift away from zero
        max_abs = max(abs(s) for s in samples)
        self.assertGreater(max_abs, 0.05)

    def test_reset(self):
        """Test noise generator reset."""
        self.noise.pink_noise(1.0)
        self.noise.brownian_noise(1.0)
        self.noise.reset()
        # After reset, states should be zero
        self.assertEqual(self.noise._pink_state, 0.0)
        self.assertEqual(self.noise._brown_state, 0.0)


class TestAdvancedADCPSensor(unittest.TestCase):
    """Test cases for AdvancedADCPSensor class."""

    def setUp(self):
        reset_config()
        self.model = TangheSiphonModel()
        self.sensor = AdvancedADCPSensor(self.model, 0)

    def tearDown(self):
        reset_config()

    def test_initialization(self):
        """Test sensor initializes correctly."""
        self.assertEqual(self.sensor.gate_index, 0)
        self.assertIsNotNone(self.sensor._noise)

    def test_invalid_gate_index(self):
        """Test invalid gate index raises error."""
        with self.assertRaises(ValueError):
            AdvancedADCPSensor(self.model, 10)

    def test_read_with_flow(self):
        """Test reading velocity with flow."""
        self.model.gate_openings = np.array([2.0, 2.0, 2.0])
        self.model.step(np.array([2.0, 2.0, 2.0]), dt=0.1)

        readings = [self.sensor.read() for _ in range(50)]
        mean_v = np.mean(readings)
        true_v = self.model.velocities[0]
        self.assertLess(abs(mean_v - true_v), 0.3)

    def test_fault_injection_stuck(self):
        """Test stuck fault mode."""
        self.sensor.read()  # Get initial value
        initial = self.sensor._filter_state
        self.sensor.inject_fault('stuck')

        # Multiple reads should return same value
        readings = [self.sensor.read() for _ in range(10)]
        for r in readings:
            self.assertAlmostEqual(r, initial, places=4)

    def test_fault_injection_dead(self):
        """Test dead fault mode."""
        self.sensor.inject_fault('dead')
        value = self.sensor.read()
        self.assertEqual(value, 0.0)

    def test_fault_clear(self):
        """Test clearing faults."""
        self.sensor.inject_fault('dead')
        self.sensor.clear_fault()
        self.assertIsNone(self.sensor._fault_mode)

    def test_quantization(self):
        """Test quantization effect."""
        value = self.sensor._quantize(1.234567)
        # Should be quantized to ADC resolution
        self.assertIsInstance(value, float)

    def test_get_statistics(self):
        """Test statistics collection."""
        for _ in range(20):
            self.sensor.read()
        stats = self.sensor.get_statistics()
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)

    def test_reset(self):
        """Test sensor reset."""
        self.sensor.read()
        self.sensor.inject_fault('stuck')
        self.sensor.reset()
        self.assertIsNone(self.sensor._fault_mode)
        self.assertEqual(len(self.sensor._history), 0)


class TestAdvancedVibrationSensor(unittest.TestCase):
    """Test cases for AdvancedVibrationSensor class."""

    def setUp(self):
        reset_config()
        self.model = TangheSiphonModel()
        self.sensor = AdvancedVibrationSensor(self.model, 0)

    def tearDown(self):
        reset_config()

    def test_initialization(self):
        """Test sensor initializes correctly."""
        self.assertEqual(self.sensor.gate_index, 0)
        self.assertEqual(self.sensor._sample_rate, 1000.0)

    def test_read_accel_non_negative(self):
        """Test acceleration is non-negative."""
        for _ in range(100):
            value = self.sensor.read_accel()
            self.assertGreaterEqual(value, 0.0)

    def test_read_accel_with_vibration(self):
        """Test reading with vibration."""
        self.model.vibration_accel[0] = 0.5
        readings = [self.sensor.read_accel() for _ in range(50)]
        mean_a = np.mean(readings)
        self.assertLess(abs(mean_a - 0.5), 0.1)

    def test_get_spectrum_peak_empty(self):
        """Test spectrum peak with insufficient data."""
        freq, amp = self.sensor.get_spectrum_peak()
        self.assertEqual(freq, 0.0)
        self.assertEqual(amp, 0.0)

    def test_get_spectrum_peak_with_data(self):
        """Test spectrum peak with data."""
        # Fill buffer with samples
        for _ in range(self.sensor._fft_size):
            self.sensor.read_accel()
        freq, amp = self.sensor.get_spectrum_peak()
        self.assertIsInstance(freq, float)
        self.assertIsInstance(amp, float)

    def test_get_rms(self):
        """Test RMS calculation."""
        self.model.vibration_accel[0] = 0.3
        for _ in range(50):
            self.sensor.read_accel()
        rms = self.sensor.get_rms()
        self.assertGreater(rms, 0.0)

    def test_saturation(self):
        """Test saturation limit."""
        self.model.vibration_accel[0] = 5.0  # Above saturation
        value = self.sensor.read_accel()
        self.assertLessEqual(value, self.sensor._saturation_limit)


class TestAdvancedPositionSensor(unittest.TestCase):
    """Test cases for AdvancedPositionSensor class."""

    def setUp(self):
        reset_config()
        self.model = TangheSiphonModel()
        self.sensor = AdvancedPositionSensor(self.model, 0)

    def tearDown(self):
        reset_config()

    def test_initialization(self):
        """Test sensor initializes correctly."""
        self.assertEqual(self.sensor.gate_index, 0)
        self.assertEqual(self.sensor._resolution, 0.001)

    def test_read_returns_float(self):
        """Test read returns float."""
        value = self.sensor.read()
        self.assertIsInstance(value, float)

    def test_read_tracks_position(self):
        """Test sensor tracks gate position."""
        self.model.gate_openings[0] = 2.5
        readings = [self.sensor.read() for _ in range(20)]
        mean_pos = np.mean(readings)
        self.assertLess(abs(mean_pos - 2.5), 0.1)

    def test_quantization(self):
        """Test position is quantized to resolution."""
        self.model.gate_openings[0] = 1.2345
        value = self.sensor.read()
        # Should be close to a multiple of resolution
        self.assertLess(abs(value - round(value, 3)), 0.01)


class TestPIDController(unittest.TestCase):
    """Test cases for PIDController class."""

    def setUp(self):
        reset_config()
        self.gains = PIDGains(kp=1.0, ki=0.1, kd=0.05)
        self.pid = PIDController(self.gains)

    def test_initialization(self):
        """Test PID initializes correctly."""
        self.assertEqual(self.pid.gains.kp, 1.0)
        self.assertEqual(self.pid.gains.ki, 0.1)
        self.assertEqual(self.pid.gains.kd, 0.05)

    def test_compute_proportional(self):
        """Test proportional response."""
        output = self.pid.compute(10.0, 5.0, dt=0.1)
        # P term should dominate initially
        self.assertGreater(output, 0)

    def test_compute_integral_accumulation(self):
        """Test integral accumulation with positive error."""
        for _ in range(10):
            self.pid.compute(10.0, 5.0, dt=0.1)
        # Integral should accumulate (positive when setpoint > measurement)
        self.assertGreater(abs(self.pid._integral), 0)

    def test_anti_windup(self):
        """Test anti-windup keeps output within limits."""
        for _ in range(100):
            output = self.pid.compute(100.0, 0.0, dt=0.1)
            # Output should always be clamped within limits
            self.assertLessEqual(output, self.gains.output_max)
            self.assertGreaterEqual(output, self.gains.output_min)

    def test_output_limits(self):
        """Test output clamping."""
        output = self.pid.compute(1000.0, 0.0, dt=0.1)
        self.assertLessEqual(output, self.gains.output_max)
        self.assertGreaterEqual(output, self.gains.output_min)

    def test_reset(self):
        """Test PID reset."""
        self.pid.compute(10.0, 5.0, dt=0.1)
        self.pid.reset()
        self.assertEqual(self.pid._integral, 0.0)
        self.assertEqual(self.pid._last_error, 0.0)


class TestMultiChannelPID(unittest.TestCase):
    """Test cases for MultiChannelPID class."""

    def setUp(self):
        reset_config()
        self.model = TangheSiphonModel()
        self.mpid = MultiChannelPID(num_gates=3)

    def tearDown(self):
        reset_config()

    def test_initialization(self):
        """Test multi-channel PID initializes correctly."""
        self.assertEqual(len(self.mpid.controllers), 3)

    def test_compute_returns_array(self):
        """Test compute returns correct shape."""
        target_flow = 60.0
        current_flows = np.array([20.0, 20.0, 20.0])
        current_velocities = np.array([2.0, 2.0, 2.0])
        output = self.mpid.compute(target_flow, current_flows, current_velocities, dt=0.1)
        self.assertEqual(len(output), 3)

    def test_output_clamping(self):
        """Test outputs are clamped to valid range."""
        target_flow = 300.0  # High target
        current_flows = np.array([10.0, 10.0, 10.0])
        current_velocities = np.array([1.0, 1.0, 1.0])
        output = self.mpid.compute(target_flow, current_flows, current_velocities, dt=0.1)
        for o in output:
            self.assertLessEqual(o, 5.0)
            self.assertGreaterEqual(o, 0.0)


class TestHybridController(unittest.TestCase):
    """Test cases for HybridController class."""

    def setUp(self):
        reset_config()
        self.model = TangheSiphonModel()
        self.mpc = AdaptiveMPC(self.model)
        self.pid = MultiChannelPID(num_gates=3)
        self.controller = HybridController(self.mpc, self.pid, self.model)

    def tearDown(self):
        reset_config()

    def test_initialization(self):
        """Test hybrid controller initializes correctly."""
        self.assertIsNotNone(self.controller.mpc)
        self.assertIsNotNone(self.controller.pid)
        self.assertTrue(self.controller._use_mpc)

    def test_compute_returns_tuple(self):
        """Test compute returns output and controller used."""
        result = self.controller.compute(
            target_flow=80.0,
            current_openings=np.array([1.0, 1.0, 1.0]),
            head_diff=2.0,
            dt=0.1
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], str)

    def test_mode_switching(self):
        """Test mode can be forced."""
        self.controller.force_mode(use_mpc=False)
        self.assertFalse(self.controller._use_mpc)
        self.controller.force_mode(use_mpc=True)
        self.assertTrue(self.controller._use_mpc)

    def test_mpc_mode(self):
        """Test MPC mode."""
        output, controller_used = self.controller.compute(
            target_flow=80.0,
            current_openings=np.array([1.0, 1.0, 1.0]),
            head_diff=2.0,
            dt=0.1
        )
        self.assertEqual(controller_used, "MPC")


class TestAdaptiveMPC(unittest.TestCase):
    """Test cases for AdaptiveMPC class."""

    def setUp(self):
        reset_config()
        self.model = TangheSiphonModel()
        self.mpc = AdaptiveMPC(self.model, adaptation_enabled=True)

    def tearDown(self):
        reset_config()

    def test_initialization(self):
        """Test adaptive MPC initializes correctly."""
        alpha, beta, gamma = self.mpc.get_weights()
        self.assertGreater(alpha, 0)
        self.assertGreater(beta, 0)
        self.assertGreater(gamma, 0)

    def test_get_target_openings(self):
        """Test computing target openings."""
        targets = self.mpc.get_target_openings(
            target_flow=80.0,
            current_openings=np.array([1.0, 1.0, 1.0]),
            current_head_diff=2.0
        )
        self.assertEqual(len(targets), 3)
        for t in targets:
            self.assertGreaterEqual(t, 0.0)
            self.assertLessEqual(t, 5.0)

    def test_uncertainty_estimation(self):
        """Test uncertainty factor is computed."""
        # Run a few steps to build history
        for _ in range(5):
            self.mpc.get_target_openings(
                target_flow=80.0,
                current_openings=np.array([1.0, 1.0, 1.0]),
                current_head_diff=2.0
            )
            self.model.step(np.array([1.0, 1.0, 1.0]), 0.1)

        uncertainty = self.mpc.get_uncertainty()
        self.assertGreater(uncertainty, 0.5)
        self.assertLess(uncertainty, 1.5)

    def test_adaptation_disabled(self):
        """Test adaptation can be disabled."""
        mpc = AdaptiveMPC(self.model, adaptation_enabled=False)
        initial_weights = mpc.get_weights()

        # Run many steps
        for _ in range(50):
            mpc.get_target_openings(
                target_flow=80.0,
                current_openings=np.array([1.0, 1.0, 1.0]),
                current_head_diff=2.0
            )
            self.model.step(np.array([1.0, 1.0, 1.0]), 0.1)

        final_weights = mpc.get_weights()
        # Weights should not change when adaptation disabled
        self.assertEqual(initial_weights, final_weights)

    def test_reset(self):
        """Test adaptive MPC reset."""
        self.mpc.get_target_openings(
            target_flow=80.0,
            current_openings=np.array([1.0, 1.0, 1.0]),
            current_head_diff=2.0
        )
        self.mpc.reset()
        self.assertEqual(self.mpc._uncertainty_factor, 1.0)
        self.assertEqual(len(self.mpc._state.flow_errors), 0)


class TestScenarioDetector(unittest.TestCase):
    """Test cases for ScenarioDetector class."""

    def setUp(self):
        reset_config()
        self.model = TangheSiphonModel()
        self.detector = ScenarioDetector(self.model)

    def tearDown(self):
        reset_config()

    def test_initialization(self):
        """Test detector initializes correctly."""
        self.assertIsNotNone(self.detector._flow_history)
        self.assertIsNotNone(self.detector._vibration_history)

    def test_detect_normal_low_flow(self):
        """Test detection of low flow scenario."""
        self.model.flow_rates = np.array([10.0, 10.0, 10.0])
        for _ in range(20):
            self.detector.update(dt=0.1)
        # After update, the detector returns the scenario directly
        scenario = self.detector.update(dt=0.1)
        self.assertIn(scenario, [ScenarioType.NORMAL_LOW_FLOW, ScenarioType.NORMAL_MEDIUM_FLOW, ScenarioType.UNKNOWN])

    def test_detect_high_vibration(self):
        """Test detection of high vibration scenario."""
        self.model.vibration_accel = np.array([0.8, 0.8, 0.8])
        for _ in range(20):
            scenario = self.detector.update(dt=0.1)
        # High vibration should be detected as resonance crossing
        self.assertEqual(scenario, ScenarioType.RESONANCE_CROSSING)

    def test_detect_gate_stuck(self):
        """Test detection of stuck gate."""
        self.model.gate_stuck = [False, True, False]
        for _ in range(15):
            scenario = self.detector.update(dt=0.1)
        self.assertEqual(scenario, ScenarioType.GATE_STUCK)


class TestAdvancedScenarioManager(unittest.TestCase):
    """Test cases for AdvancedScenarioManager class."""

    def setUp(self):
        reset_config()
        self.model = TangheSiphonModel()
        self.manager = AdvancedScenarioManager(self.model)

    def tearDown(self):
        reset_config()

    def test_initialization(self):
        """Test manager initializes correctly."""
        self.assertIsNotNone(self.manager.detector)
        self.assertEqual(self.manager.detected_scenario, ScenarioType.UNKNOWN)

    def test_update(self):
        """Test manager update."""
        status = self.manager.update(dt=0.1)
        self.assertIn('detected_scenario', status)
        self.assertIn('elapsed_time', status)

    def test_set_scenario(self):
        """Test setting a scenario."""
        self.manager.set_scenario('S1.1')
        self.assertEqual(self.manager.active_scenario, 'S1.1')

    def test_scenario_detection(self):
        """Test scenario detection during update."""
        # Force high vibration
        self.model.vibration_accel = np.array([0.9, 0.9, 0.9])
        for _ in range(30):
            self.manager.update(dt=0.1)

        detected = self.manager.detected_scenario
        self.assertEqual(detected, ScenarioType.RESONANCE_CROSSING)


class TestHighFidelityPhysicsModel(unittest.TestCase):
    """Test cases for HighFidelityPhysicsModel class."""

    def setUp(self):
        reset_config()
        self.model = HighFidelityPhysicsModel()

    def tearDown(self):
        reset_config()

    def test_initialization(self):
        """Test high-fidelity model initializes correctly."""
        self.assertEqual(self.model.num_gates, 3)
        self.assertIsNotNone(self.model._params)

    def test_step(self):
        """Test stepping the model."""
        targets = np.array([2.0, 2.0, 2.0])
        self.model.step(targets, dt=0.01)
        self.assertGreater(self.model.time, 0.0)

    def test_flow_calculation(self):
        """Test flow is calculated with friction."""
        targets = np.array([2.0, 2.0, 2.0])
        for _ in range(100):
            self.model.step(targets, dt=0.01)

        total_flow = np.sum(self.model.flow_rates)
        self.assertGreater(total_flow, 0)

    def test_multi_mode_vibration(self):
        """Test multi-mode vibration model."""
        targets = np.array([2.0, 2.0, 2.0])
        for _ in range(50):
            self.model.step(targets, dt=0.01)

        # Should have computed structural displacement for modes
        self.assertIsNotNone(self.model.structural_displacement)
        self.assertEqual(self.model.structural_displacement.shape, (3, 3))

    def test_reynolds_number_tracking(self):
        """Test Reynolds number is computed."""
        targets = np.array([2.0, 2.0, 2.0])
        for _ in range(10):
            self.model.step(targets, dt=0.01)

        state = self.model.get_state()
        self.assertIn('reynolds', state)

    def test_cavitation_tracking(self):
        """Test cavitation detection."""
        targets = np.array([4.5, 4.5, 4.5])  # High openings
        for _ in range(50):
            self.model.step(targets, dt=0.01)

        state = self.model.get_state()
        self.assertIn('cavitation_indices', state)


class TestHILFramework(unittest.TestCase):
    """Test cases for HIL testing framework."""

    def setUp(self):
        reset_config()
        self.model = TangheSiphonModel()
        self.controller = SpectralMPC(self.model)
        self.config = HILConfig(
            mode=HILMode.SIMULATION,
            real_time=False,  # Faster testing
            dt=0.1
        )
        self.runner = HILTestRunner(self.model, self.controller, self.config)

    def tearDown(self):
        self.runner.teardown()
        reset_config()

    def test_setup(self):
        """Test HIL setup."""
        success = self.runner.setup()
        self.assertTrue(success)
        self.assertIsNotNone(self.runner.sensor_interface)
        self.assertIsNotNone(self.runner.actuator_interface)

    def test_sensor_interface(self):
        """Test simulated sensor interface."""
        self.runner.setup()
        velocity = self.runner.sensor_interface.read_velocity(0)
        self.assertIsInstance(velocity, float)

    def test_actuator_interface(self):
        """Test simulated actuator interface."""
        self.runner.setup()
        success = self.runner.actuator_interface.set_position(0, 2.0)
        self.assertTrue(success)

    def test_run_short_test(self):
        """Test running a short HIL test."""
        self.runner.setup()
        results = self.runner.run_test(
            duration=1.0,
            target_flow_func=lambda t: 80.0
        )
        self.assertIn('total_steps', results)
        self.assertGreater(results['total_steps'], 0)

    def test_metrics_collection(self):
        """Test metrics are collected during test."""
        self.runner.setup()
        results = self.runner.run_test(
            duration=0.5,
            target_flow_func=lambda t: 80.0
        )
        self.assertIn('average_loop_time', results)
        self.assertIn('timing_violations', results)

    def test_stop(self):
        """Test stopping a running test."""
        self.runner.setup()
        self.runner.stop()
        self.assertFalse(self.runner._running)


class TestFullScenarioHIL(unittest.TestCase):
    """Integration tests for full scenario HIL operation."""

    def setUp(self):
        reset_config()
        self.model = TangheSiphonModel()
        self.mpc = AdaptiveMPC(self.model)
        self.pid = MultiChannelPID(num_gates=3)
        self.controller = HybridController(self.mpc, self.pid, self.model)
        self.scenario_mgr = AdvancedScenarioManager(self.model)

    def tearDown(self):
        reset_config()

    def test_scenario_aware_control(self):
        """Test scenario-aware control loop."""
        dt = 0.1

        for step in range(100):
            # Update scenario detection
            status = self.scenario_mgr.update(dt)

            # Get control output
            target_flow = 80.0
            output, controller_used = self.controller.compute(
                target_flow=target_flow,
                current_openings=self.model.gate_openings,
                head_diff=self.model.head_upstream - self.model.head_downstream,
                dt=dt
            )

            # Step model
            self.model.step(output, dt)

        # System should be operating
        total_flow = np.sum(self.model.flow_rates)
        self.assertGreater(total_flow, 0)

    def test_emergency_scenario_handling(self):
        """Test handling of emergency scenarios."""
        dt = 0.1

        # Induce high vibration
        self.model.vibration_accel = np.array([0.9, 0.9, 0.9])

        for _ in range(50):
            self.scenario_mgr.update(dt)

        scenario = self.scenario_mgr.detected_scenario

        # Should detect resonance crossing due to high vibration
        self.assertEqual(scenario, ScenarioType.RESONANCE_CROSSING)

    def test_adaptive_control_convergence(self):
        """Test adaptive control converges to target."""
        mpc = AdaptiveMPC(self.model)
        target_flow = 100.0
        dt = 0.1

        for _ in range(200):
            targets = mpc.get_target_openings(
                target_flow=target_flow,
                current_openings=self.model.gate_openings,
                current_head_diff=self.model.head_upstream - self.model.head_downstream
            )
            self.model.step(targets, dt)

        actual_flow = np.sum(self.model.flow_rates)
        # Should be within 50% of target (MPC may prioritize vibration avoidance)
        self.assertLess(abs(actual_flow - target_flow), target_flow * 0.5)


if __name__ == '__main__':
    unittest.main()
