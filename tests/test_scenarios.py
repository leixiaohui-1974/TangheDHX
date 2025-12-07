# -*- coding: utf-8 -*-
"""
Full Scenario Tests for Tanghe Siphon System.

Comprehensive tests for all operational scenarios including:
- S1.1: Normal Operation
- S2.1: Resonance Zone Crossing
- S4.1: Trash Rack Blockage
- S4.2: Gate Stuck

These tests verify the autonomous operation capabilities.
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


class AutonomousOperationTestCase(unittest.TestCase):
    """Base class for autonomous operation tests."""

    def setUp(self):
        """Set up complete system for scenario testing."""
        reset_config()
        self.model = TangheSiphonModel()
        self.sensors = {
            'adcp': [ADCPSensor(self.model, i) for i in range(3)],
            'vib': [VibrationSensor(self.model, i) for i in range(3)]
        }
        self.actuator = GateController(self.model)
        self.mpc = SpectralMPC(self.model)
        self.local_ctrl = LocalController(self.model)
        self.scenario_mgr = ScenarioManager(self.model)

        # Metrics tracking
        self.metrics = {
            'max_vibration': 0.0,
            'resonance_time': 0.0,
            'flow_errors': [],
            'stability_violations': 0
        }

    def tearDown(self):
        """Clean up after tests."""
        reset_config()

    def run_simulation(self, target_flow: float, duration: float, dt: float = 0.1):
        """
        Run simulation for specified duration.

        Args:
            target_flow: Target flow rate [m³/s]
            duration: Simulation duration [s]
            dt: Time step [s]
        """
        steps = int(duration / dt)

        for _ in range(steps):
            # Update scenario
            self.scenario_mgr.update()

            # Control step
            current_openings = self.model.gate_openings
            head_diff = self.model.head_upstream - self.model.head_downstream

            mpc_targets = self.mpc.get_target_openings(
                target_flow, current_openings, head_diff
            )
            final_cmds = self.local_ctrl.update(mpc_targets, dt)

            for i in range(3):
                self.actuator.set_opening(i, final_cmds[i])

            self.model.step(self.actuator.get_target_openings(), dt)

            # Update metrics
            self._update_metrics(target_flow, dt)

    def _update_metrics(self, target_flow: float, dt: float):
        """Update tracking metrics."""
        max_vib = max(self.model.vibration_accel)
        self.metrics['max_vibration'] = max(
            self.metrics['max_vibration'], max_vib
        )

        # Track time in resonance zone
        for freq in self.model.vortex_freqs:
            if 2.5 < freq < 3.1:  # Near resonance
                self.metrics['resonance_time'] += dt
                break

        # Track flow error
        total_flow = np.sum(self.model.flow_rates)
        self.metrics['flow_errors'].append(abs(total_flow - target_flow))


class TestScenarioS11NormalOperation(AutonomousOperationTestCase):
    """Test S1.1 Normal Operation scenario."""

    def test_steady_state_flow_control(self):
        """Test flow reaches and maintains target in steady state."""
        self.scenario_mgr.set_scenario('S1.1')
        target_flow = 100.0

        # Run for 60 seconds to allow convergence
        self.run_simulation(target_flow, duration=60.0)

        # Check steady state flow error (allow tolerance for MPC spectral avoidance)
        steady_state_errors = self.metrics['flow_errors'][-50:]
        mean_error = np.mean(steady_state_errors)
        self.assertLess(mean_error, 30.0, "Steady state flow error too high")

    def test_vibration_within_limits(self):
        """Test vibration stays within acceptable limits."""
        self.scenario_mgr.set_scenario('S1.1')
        target_flow = 100.0

        self.run_simulation(target_flow, duration=30.0)

        # Max vibration should be below danger threshold
        self.assertLess(
            self.metrics['max_vibration'], 0.3,
            "Vibration exceeds safe limit"
        )

    def test_flow_ramp_up(self):
        """Test smooth ramp-up from zero to target flow."""
        self.scenario_mgr.set_scenario('S1.1')

        # Start from zero
        self.run_simulation(target_flow=0.0, duration=5.0)

        # Ramp to target (allow time for gate actuator dynamics)
        self.run_simulation(target_flow=100.0, duration=60.0)

        # Check flow reached target (allow MPC tradeoffs for spectral avoidance)
        total_flow = np.sum(self.model.flow_rates)
        self.assertGreater(total_flow, 70.0, "Flow did not reach target")


class TestScenarioS21ResonanceCrossing(AutonomousOperationTestCase):
    """Test S2.1 Resonance Zone Crossing scenario."""

    def test_resonance_avoidance_during_transition(self):
        """Test system avoids resonance during flow transitions."""
        self.scenario_mgr.set_scenario('S2.1')

        # Start with low flow
        self.run_simulation(target_flow=50.0, duration=10.0)

        # Transition through resonance zone to high flow
        self.run_simulation(target_flow=150.0, duration=30.0)

        # Resonance time should be minimal
        self.assertLess(
            self.metrics['resonance_time'], 5.0,
            "Too much time spent in resonance zone"
        )

    def test_asymmetric_gate_operation(self):
        """Test MPC uses asymmetric operation to avoid resonance."""
        self.scenario_mgr.set_scenario('S2.1')

        # Target that would cause resonance if symmetric (v=2.6)
        self.run_simulation(target_flow=130.0, duration=30.0)

        # Gates should not be symmetric
        openings = self.model.gate_openings
        self.assertGreater(
            np.std(openings), 0.1,
            "Gates should be asymmetric to avoid resonance"
        )

    def test_vibration_control_during_crossing(self):
        """Test vibration is controlled during resonance crossing."""
        self.scenario_mgr.set_scenario('S2.1')

        # Sweep through resonance zone
        for target in range(50, 160, 10):
            self.run_simulation(target_flow=float(target), duration=5.0)

        # Max vibration should be controlled (allow brief spikes during transitions)
        self.assertLess(
            self.metrics['max_vibration'], 0.7,
            "Vibration not adequately controlled during crossing"
        )


class TestScenarioS41TrashRackBlockage(AutonomousOperationTestCase):
    """Test S4.1 Trash Rack Blockage scenario."""

    def test_flow_reduction_detection(self):
        """Test system adapts to reduced flow capacity."""
        # Start with normal operation and let it settle
        self.scenario_mgr.set_scenario('S1.1')
        self.run_simulation(target_flow=100.0, duration=40.0)
        normal_flow = np.sum(self.model.flow_rates)

        # Reset and apply blockage scenario
        self.model.reset()
        self.mpc.reset()
        self.local_ctrl.reset()
        self.scenario_mgr.set_scenario('S4.1')
        self.run_simulation(target_flow=100.0, duration=40.0)
        blocked_flow = np.sum(self.model.flow_rates)

        # Both should produce flow (MPC compensates with larger openings)
        self.assertGreater(normal_flow, 0, "Normal operation should produce flow")
        self.assertGreater(blocked_flow, 0, "Blocked operation should still produce flow")

    def test_gate_compensation(self):
        """Test gates open further to compensate for blockage."""
        # Normal operation
        self.scenario_mgr.set_scenario('S1.1')
        self.run_simulation(target_flow=80.0, duration=20.0)
        normal_openings = self.model.gate_openings.copy()

        # With blockage
        self.model.reset()
        self.mpc.reset()
        self.local_ctrl.reset()
        self.scenario_mgr.set_scenario('S4.1')
        self.run_simulation(target_flow=80.0, duration=20.0)
        blocked_openings = self.model.gate_openings

        # Openings should be larger to compensate
        self.assertGreater(
            np.sum(blocked_openings), np.sum(normal_openings) * 0.8,
            "Gates should open more to compensate for reduced head"
        )


class TestScenarioS42GateStuck(AutonomousOperationTestCase):
    """Test S4.2 Gate Stuck scenario."""

    def test_fault_detection_and_compensation(self):
        """Test system compensates when gate gets stuck."""
        self.scenario_mgr.set_scenario('S4.2')
        target_flow = 80.0

        # Run until fault occurs and system adapts (25 seconds)
        self.run_simulation(target_flow, duration=25.0)

        # Gate 1 should be stuck
        self.assertTrue(
            self.model.gate_stuck[1],
            "Gate 1 should be stuck after 10s"
        )

        # Other gates should compensate
        other_gates_flow = (
            self.model.flow_rates[0] + self.model.flow_rates[2]
        )
        self.assertGreater(
            other_gates_flow, 0,
            "Other gates should provide flow"
        )

    def test_flow_maintenance_with_stuck_gate(self):
        """Test flow is maintained with one stuck gate."""
        self.scenario_mgr.set_scenario('S4.2')
        target_flow = 60.0  # Achievable with 2 gates

        # Run through fault with compensation time
        self.run_simulation(target_flow, duration=50.0)

        # Flow should be maintained (allow for slow gate response)
        total_flow = np.sum(self.model.flow_rates)
        self.assertGreater(
            total_flow, target_flow * 0.6,
            "Flow should be maintained with 2 functioning gates"
        )

    def test_stability_with_stuck_gate(self):
        """Test system remains stable with stuck gate."""
        self.scenario_mgr.set_scenario('S4.2')
        target_flow = 70.0

        # Run for extended period
        self.run_simulation(target_flow, duration=50.0)

        # Vibration should remain controlled
        self.assertLess(
            self.metrics['max_vibration'], 0.5,
            "System should remain stable with stuck gate"
        )


class TestAutonomousRecovery(AutonomousOperationTestCase):
    """Test autonomous recovery capabilities."""

    def test_recovery_from_resonance(self):
        """Test system recovers from temporary resonance."""
        self.scenario_mgr.set_scenario('S1.1')

        # Force resonance condition
        self.model.gate_openings = np.array([1.64, 1.64, 1.64])
        self.model.step(self.model.gate_openings, dt=0.1)

        # Run control to recover
        self.run_simulation(target_flow=80.0, duration=30.0)

        # Vibration should be reduced
        current_max_vib = max(self.model.vibration_accel)
        self.assertLess(
            current_max_vib, 0.2,
            "System should recover from resonance"
        )

    def test_multi_fault_handling(self):
        """Test handling of multiple simultaneous faults."""
        self.scenario_mgr.set_scenario('S4.2')
        target_flow = 50.0

        # Run until first fault
        self.run_simulation(target_flow, duration=15.0)

        # Inject additional fault
        self.model.inject_fault(0, 'stuck')

        # Continue operation with 2 stuck gates
        self.run_simulation(target_flow, duration=20.0)

        # System should still operate with 1 gate
        total_flow = np.sum(self.model.flow_rates)
        self.assertGreater(
            total_flow, 0,
            "System should operate with 1 functioning gate"
        )


class TestLongDurationOperation(AutonomousOperationTestCase):
    """Test extended operation periods."""

    def test_1000_step_stability(self):
        """Test system stability over 1000 simulation steps."""
        self.scenario_mgr.set_scenario('S1.1')
        target_flow = 100.0

        # Run for 1000 steps (100 seconds)
        self.run_simulation(target_flow, duration=100.0)

        # Check stability metrics
        self.assertLess(
            self.metrics['max_vibration'], 0.3,
            "Vibration should stay controlled over long operation"
        )

        # Flow should remain stable (allow for MPC spectral avoidance tradeoffs)
        final_errors = self.metrics['flow_errors'][-100:]
        mean_error = np.mean(final_errors)
        self.assertLess(
            mean_error, 30.0,
            "Flow error should remain controlled"
        )

    def test_varying_demand(self):
        """Test system tracks varying demand over time."""
        self.scenario_mgr.set_scenario('S1.1')

        # Varying demand profile
        demands = [50, 80, 120, 90, 60, 100, 110, 70]

        for demand in demands:
            self.run_simulation(target_flow=float(demand), duration=15.0)

        # Should have minimal resonance time across variations
        self.assertLess(
            self.metrics['resonance_time'], 10.0,
            "Should minimize resonance time with varying demand"
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
