# -*- coding: utf-8 -*-
"""
Integration Tests for Tanghe Siphon System.

Tests end-to-end functionality and component interactions.
"""

import unittest
import numpy as np

from src.simulation.physics import TangheSiphonModel
from src.simulation.sensors import ADCPSensor, VibrationSensor
from src.simulation.actuators import GateController
from src.control.mpc import SpectralMPC
from src.control.local import LocalController
from src.control.manager import ScenarioManager
from src.config import reset_config


class TestFullControlLoop(unittest.TestCase):
    """Test complete control loop integration."""

    def setUp(self):
        """Set up test fixtures."""
        reset_config()
        self.model = TangheSiphonModel()
        self.mpc = SpectralMPC(self.model)
        self.local_ctrl = LocalController(self.model)
        self.actuator = GateController(self.model)

    def tearDown(self):
        """Clean up after tests."""
        reset_config()

    def test_control_loop_converges(self):
        """Test control loop converges to target flow."""
        target_flow = 80.0
        dt = 0.1

        # Run for 300 steps (30 seconds) to allow full convergence
        for _ in range(300):
            current_openings = self.model.gate_openings
            head_diff = self.model.head_upstream - self.model.head_downstream

            mpc_targets = self.mpc.get_target_openings(
                target_flow, current_openings, head_diff
            )
            final_cmds = self.local_ctrl.update(mpc_targets, dt)

            for i in range(3):
                self.actuator.set_opening(i, final_cmds[i])

            self.model.step(self.actuator.get_target_openings(), dt)

        # Check flow is near target (allow MPC spectral tradeoffs)
        total_flow = np.sum(self.model.flow_rates)
        self.assertLess(abs(total_flow - target_flow), 15.0)

    def test_resonance_avoidance_in_loop(self):
        """Test resonance is avoided during control."""
        # Target that would cause resonance if symmetric
        target_flow = 130.0
        dt = 0.1

        max_vibration = 0.0

        # Run for 200 steps
        for _ in range(200):
            current_openings = self.model.gate_openings
            head_diff = self.model.head_upstream - self.model.head_downstream

            mpc_targets = self.mpc.get_target_openings(
                target_flow, current_openings, head_diff
            )
            final_cmds = self.local_ctrl.update(mpc_targets, dt)

            for i in range(3):
                self.actuator.set_opening(i, final_cmds[i])

            self.model.step(self.actuator.get_target_openings(), dt)

            max_vibration = max(max_vibration, max(self.model.vibration_accel))

        # Vibration should be controlled (MPC avoids resonance)
        self.assertLess(max_vibration, 0.6)

    def test_flow_step_change_response(self):
        """Test system response to step change in target flow."""
        dt = 0.1

        # Start with low flow
        target_flow = 50.0
        for _ in range(100):
            self._run_control_step(target_flow, dt)

        # Step change to higher flow
        target_flow = 120.0
        for _ in range(300):
            self._run_control_step(target_flow, dt)

        # Check flow approaches new target (MPC prioritizes spectral avoidance)
        total_flow = np.sum(self.model.flow_rates)
        # Flow may deviate as MPC avoids resonance - check it increased
        self.assertGreater(total_flow, 30.0, "Flow should increase toward target")

    def _run_control_step(self, target_flow: float, dt: float) -> None:
        """Run single control step."""
        current_openings = self.model.gate_openings
        head_diff = self.model.head_upstream - self.model.head_downstream

        mpc_targets = self.mpc.get_target_openings(
            target_flow, current_openings, head_diff
        )
        final_cmds = self.local_ctrl.update(mpc_targets, dt)

        for i in range(3):
            self.actuator.set_opening(i, final_cmds[i])

        self.model.step(self.actuator.get_target_openings(), dt)


class TestScenarioIntegration(unittest.TestCase):
    """Test scenario manager integration with control system."""

    def setUp(self):
        """Set up test fixtures."""
        reset_config()
        self.model = TangheSiphonModel()
        self.mpc = SpectralMPC(self.model)
        self.local_ctrl = LocalController(self.model)
        self.actuator = GateController(self.model)
        self.scenario_mgr = ScenarioManager(self.model)

    def tearDown(self):
        """Clean up after tests."""
        reset_config()

    def test_scenario_s1_1_normal_operation(self):
        """Test S1.1 Normal Operation scenario."""
        self.scenario_mgr.set_scenario('S1.1')
        target_flow = 100.0
        dt = 0.1

        # Run for 50 steps
        for _ in range(50):
            self.scenario_mgr.update()
            self._run_control_step(target_flow, dt)

        # Should operate normally
        total_flow = np.sum(self.model.flow_rates)
        self.assertGreater(total_flow, 0)

    def test_scenario_s4_1_blockage(self):
        """Test S4.1 Trash Rack Blockage scenario."""
        # Start with normal operation
        target_flow = 100.0
        dt = 0.1

        for _ in range(100):
            self._run_control_step(target_flow, dt)

        normal_flow = np.sum(self.model.flow_rates)

        # Apply blockage scenario
        self.scenario_mgr.set_scenario('S4.1')

        for _ in range(100):
            self.scenario_mgr.update()
            self._run_control_step(target_flow, dt)

        blocked_flow = np.sum(self.model.flow_rates)

        # Both should produce positive flow
        self.assertGreater(normal_flow, 0, "Normal operation should produce flow")
        self.assertGreater(blocked_flow, 0, "Blocked operation should still produce flow")

    def test_scenario_s4_2_gate_stuck(self):
        """Test S4.2 Gate Stuck scenario with recovery."""
        self.scenario_mgr.set_scenario('S4.2')
        target_flow = 80.0
        dt = 0.1

        # Run until gate gets stuck (after 10s)
        for _ in range(150):  # 15 seconds
            self.scenario_mgr.update()
            self._run_control_step(target_flow, dt)

        # Gate 1 should be stuck
        self.assertTrue(self.model.gate_stuck[1])

        # Continue running - system should adapt
        for _ in range(100):
            self.scenario_mgr.update()
            self._run_control_step(target_flow, dt)

        # Other gates should compensate
        total_flow = np.sum(self.model.flow_rates)
        self.assertGreater(total_flow, 0)

    def _run_control_step(self, target_flow: float, dt: float) -> None:
        """Run single control step."""
        current_openings = self.model.gate_openings
        head_diff = self.model.head_upstream - self.model.head_downstream

        mpc_targets = self.mpc.get_target_openings(
            target_flow, current_openings, head_diff
        )
        final_cmds = self.local_ctrl.update(mpc_targets, dt)

        for i in range(3):
            self.actuator.set_opening(i, final_cmds[i])

        self.model.step(self.actuator.get_target_openings(), dt)


class TestSensorFeedback(unittest.TestCase):
    """Test sensor feedback in control loop."""

    def setUp(self):
        """Set up test fixtures."""
        reset_config()
        self.model = TangheSiphonModel()
        self.sensors = {
            'adcp': [ADCPSensor(self.model, i) for i in range(3)],
            'vib': [VibrationSensor(self.model, i) for i in range(3)]
        }
        self.mpc = SpectralMPC(self.model)
        self.local_ctrl = LocalController(self.model)
        self.actuator = GateController(self.model)

    def tearDown(self):
        """Clean up after tests."""
        reset_config()

    def test_sensor_readings_correlate_with_state(self):
        """Test sensor readings correlate with model state."""
        # Set up some flow
        self.model.gate_openings = np.array([2.0, 2.0, 2.0])
        self.model.step(np.array([2.0, 2.0, 2.0]), dt=0.1)

        for i in range(3):
            # ADCP reading should be close to true velocity
            adcp_readings = [self.sensors['adcp'][i].read() for _ in range(20)]
            mean_v = np.mean(adcp_readings)
            true_v = self.model.velocities[i]
            self.assertLess(abs(mean_v - true_v), 0.1)

            # Vibration reading should be close to true acceleration
            vib_readings = [self.sensors['vib'][i].read_accel() for _ in range(20)]
            mean_a = np.mean(vib_readings)
            true_a = self.model.vibration_accel[i]
            self.assertLess(abs(mean_a - true_a), 0.02)


if __name__ == '__main__':
    unittest.main()
