# -*- coding: utf-8 -*-
"""
Tests for Specific Agent Types.

Tests for CentralCoordinator, ZoneManager, DeviceController, and GateAgent.
"""

import pytest
import time
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime

from src.agents.coordinator import CentralCoordinator, GlobalObjective, ResourceAllocation
from src.agents.zone_manager import ZoneManager, ZoneObjective, DeviceState
from src.agents.device_controller import DeviceController, GateAgent, ControlState
from src.agents.base import AgentState, AgentRole, Priority, MessageType


# =============================================================================
# GlobalObjective Tests
# =============================================================================

class TestGlobalObjective:
    """Tests for GlobalObjective dataclass."""

    def test_creation(self):
        """Test GlobalObjective creation."""
        obj = GlobalObjective(
            objective_id='flow',
            name='Total Flow',
            target_value=100.0,
        )
        assert obj.objective_id == 'flow'
        assert obj.target_value == 100.0
        assert obj.current_value == 0.0

    def test_error_calculation(self):
        """Test error property."""
        obj = GlobalObjective(
            objective_id='flow',
            name='Flow',
            target_value=100.0,
            current_value=80.0,
        )
        assert obj.error == 20.0

    def test_is_satisfied_true(self):
        """Test is_satisfied when within tolerance."""
        obj = GlobalObjective(
            objective_id='flow',
            name='Flow',
            target_value=100.0,
            current_value=98.0,
            tolerance=0.05,
        )
        # Error is 2, tolerance is 5% of 100 = 5
        assert obj.is_satisfied is True

    def test_is_satisfied_false(self):
        """Test is_satisfied when outside tolerance."""
        obj = GlobalObjective(
            objective_id='flow',
            name='Flow',
            target_value=100.0,
            current_value=80.0,
            tolerance=0.05,
        )
        assert obj.is_satisfied is False


# =============================================================================
# CentralCoordinator Tests
# =============================================================================

class TestCentralCoordinator:
    """Tests for CentralCoordinator agent."""

    def test_initialization(self):
        """Test coordinator initialization."""
        coordinator = CentralCoordinator(agent_id='coord_1')

        assert coordinator.agent_id == 'coord_1'
        assert coordinator.role == AgentRole.COORDINATOR
        assert 'total_flow' in coordinator._global_objectives
        assert 'vibration_limit' in coordinator._global_objectives
        assert 'efficiency' in coordinator._global_objectives

    def test_default_objectives(self):
        """Test default objectives are set."""
        coordinator = CentralCoordinator()

        flow_obj = coordinator._global_objectives.get('total_flow')
        assert flow_obj is not None
        assert flow_obj.target_value == 100.0
        assert flow_obj.priority == Priority.HIGH

    def test_set_global_objective_existing(self):
        """Test updating existing objective."""
        coordinator = CentralCoordinator()

        coordinator.set_global_objective('total_flow', 150.0, weight=2.0)

        obj = coordinator._global_objectives['total_flow']
        assert obj.target_value == 150.0
        assert obj.weight == 2.0

    def test_set_global_objective_new(self):
        """Test creating new objective."""
        coordinator = CentralCoordinator()

        coordinator.set_global_objective('custom_objective', 50.0)

        assert 'custom_objective' in coordinator._global_objectives
        obj = coordinator._global_objectives['custom_objective']
        assert obj.target_value == 50.0

    def test_perceive_without_model(self):
        """Test perceive without model."""
        coordinator = CentralCoordinator()

        perceptions = coordinator.perceive()

        # Should return empty or minimal perceptions
        assert isinstance(perceptions, dict)

    def test_perceive_with_model(self):
        """Test perceive with mock model."""
        mock_model = MagicMock()
        mock_model.flow_rates = np.array([10.0, 20.0, 30.0])
        mock_model.vibration_accel = np.array([0.1, 0.2, 0.3])
        mock_model.gate_openings = np.array([1.0, 2.0, 3.0])
        mock_model.head_upstream = 10.0
        mock_model.head_downstream = 5.0

        coordinator = CentralCoordinator(model=mock_model)
        perceptions = coordinator.perceive()

        assert perceptions['total_flow'] == 60.0
        assert perceptions['max_vibration'] == 0.3
        assert perceptions['head_difference'] == 5.0

    def test_evaluate_objectives(self):
        """Test objective evaluation."""
        coordinator = CentralCoordinator()
        coordinator.set_belief('total_flow', 90.0)

        status = coordinator._evaluate_objectives()

        assert 'total_flow' in status
        assert status['total_flow']['current'] == 90.0
        assert status['total_flow']['target'] == 100.0

    def test_detect_conflicts_empty(self):
        """Test conflict detection with no conflicts."""
        coordinator = CentralCoordinator()

        conflicts = coordinator._detect_conflicts()

        # No allocations, no conflicts
        assert len(conflicts) == 0

    def test_detect_conflicts_over_allocation(self):
        """Test over-allocation conflict detection."""
        coordinator = CentralCoordinator()

        # Create allocations exceeding target
        coordinator._allocations['zone1'] = ResourceAllocation(
            zone_id='zone1', flow_quota=80.0, gate_assignment=[0, 1]
        )
        coordinator._allocations['zone2'] = ResourceAllocation(
            zone_id='zone2', flow_quota=50.0, gate_assignment=[2]
        )

        conflicts = coordinator._detect_conflicts()

        # 130 > 100 * 1.1 = 110, should be over-allocation
        over_alloc = [c for c in conflicts if c['type'] == 'over_allocation']
        assert len(over_alloc) == 1

    def test_detect_conflicts_gate_conflict(self):
        """Test gate assignment conflict detection."""
        coordinator = CentralCoordinator()

        coordinator._allocations['zone1'] = ResourceAllocation(
            zone_id='zone1', flow_quota=50.0, gate_assignment=[0, 1]
        )
        coordinator._allocations['zone2'] = ResourceAllocation(
            zone_id='zone2', flow_quota=50.0, gate_assignment=[1, 2]  # Gate 1 conflict
        )

        conflicts = coordinator._detect_conflicts()

        gate_conflicts = [c for c in conflicts if c['type'] == 'gate_conflict']
        assert len(gate_conflicts) == 1
        assert gate_conflicts[0]['gate'] == 1

    def test_detect_anomalies_high_vibration(self):
        """Test high vibration anomaly detection."""
        coordinator = CentralCoordinator()
        coordinator.set_belief('max_vibration', 0.8)

        anomalies = coordinator._detect_anomalies()

        high_vib = [a for a in anomalies if a['type'] == 'high_vibration']
        assert len(high_vib) == 1
        assert high_vib[0]['severity'] == 'warning'

    def test_detect_anomalies_critical_vibration(self):
        """Test critical vibration anomaly detection."""
        coordinator = CentralCoordinator()
        coordinator.set_belief('max_vibration', 0.95)

        anomalies = coordinator._detect_anomalies()

        high_vib = [a for a in anomalies if a['type'] == 'high_vibration']
        assert len(high_vib) == 1
        assert high_vib[0]['severity'] == 'critical'

    def test_get_performance_metrics(self):
        """Test performance metrics retrieval."""
        coordinator = CentralCoordinator()

        metrics = coordinator.get_performance_metrics()

        assert 'objectives' in metrics
        assert 'allocations' in metrics
        assert 'conflicts_count' in metrics
        assert 'zones_count' in metrics


# =============================================================================
# ZoneObjective Tests
# =============================================================================

class TestZoneObjective:
    """Tests for ZoneObjective dataclass."""

    def test_creation(self):
        """Test ZoneObjective creation."""
        obj = ZoneObjective(
            objective_id='zone_flow',
            name='Zone Flow',
            target_value=30.0,
        )
        assert obj.objective_id == 'zone_flow'
        assert obj.target_value == 30.0
        assert obj.source == 'coordinator'


# =============================================================================
# ZoneManager Tests
# =============================================================================

class TestZoneManager:
    """Tests for ZoneManager agent."""

    def test_initialization(self):
        """Test zone manager initialization."""
        manager = ZoneManager(
            agent_id='zone_1',
            zone_name='North Zone',
            gates=[0, 1, 2],
        )

        assert manager.agent_id == 'zone_1'
        assert manager.zone_name == 'North Zone'
        assert manager.gates == [0, 1, 2]
        assert manager.role == AgentRole.ZONE_MANAGER

    def test_set_zone_objective(self):
        """Test setting zone objective."""
        manager = ZoneManager(
            agent_id='zone_1',
            zone_name='Zone 1',
            gates=[0],
        )

        manager.set_zone_objective('zone_flow', 50.0, weight=1.5)

        assert 'zone_flow' in manager._zone_objectives
        obj = manager._zone_objectives['zone_flow']
        assert obj.target_value == 50.0
        assert obj.weight == 1.5

    def test_perceive_without_model(self):
        """Test perceive without model."""
        manager = ZoneManager(
            agent_id='zone_1',
            zone_name='Zone 1',
            gates=[0, 1],
        )

        perceptions = manager.perceive()

        assert isinstance(perceptions, dict)

    def test_perceive_with_model(self):
        """Test perceive with mock model."""
        mock_model = MagicMock()
        mock_model.flow_rates = np.array([10.0, 20.0, 30.0])
        mock_model.vibration_accel = np.array([0.1, 0.2, 0.3])
        mock_model.gate_openings = np.array([1.0, 2.0, 3.0])
        mock_model.head_upstream = 10.0
        mock_model.head_downstream = 5.0

        manager = ZoneManager(
            agent_id='zone_1',
            zone_name='Zone 1',
            gates=[0, 1],
            model=mock_model,
        )

        perceptions = manager.perceive()

        assert perceptions['zone_flow'] == 30.0  # 10 + 20
        assert perceptions['zone_vibration'] == 0.2  # max(0.1, 0.2)

    def test_evaluate_objectives_no_gap(self):
        """Test objective evaluation with no gap."""
        manager = ZoneManager(
            agent_id='zone_1',
            zone_name='Zone 1',
            gates=[0],
        )
        manager.set_zone_objective('zone_flow', 50.0)
        manager.set_belief('zone_flow', 49.0)  # Within 5% tolerance

        gaps = manager._evaluate_objectives()

        # 49 is within 5% of 50, so no gap
        assert len(gaps) == 0

    def test_evaluate_objectives_with_gap(self):
        """Test objective evaluation with gap."""
        manager = ZoneManager(
            agent_id='zone_1',
            zone_name='Zone 1',
            gates=[0],
        )
        manager.set_zone_objective('zone_flow', 50.0)
        manager.set_belief('zone_flow', 40.0)  # 20% off

        gaps = manager._evaluate_objectives()

        assert 'zone_flow' in gaps
        assert gaps['zone_flow'] == 10.0  # 50 - 40

    def test_collect_zone_status(self):
        """Test zone status collection."""
        manager = ZoneManager(
            agent_id='zone_1',
            zone_name='Zone 1',
            gates=[0, 1],
        )
        manager.set_belief('zone_flow', 30.0)
        manager.set_belief('zone_vibration', 0.2)

        status = manager._collect_zone_status()

        assert status['zone_id'] == 'zone_1'
        assert status['zone_name'] == 'Zone 1'
        assert status['gates'] == [0, 1]
        assert status['current_flow'] == 30.0

    def test_request_quota_increase(self):
        """Test quota increase request."""
        manager = ZoneManager(
            agent_id='zone_1',
            zone_name='Zone 1',
            gates=[0],
        )
        manager._parent_id = 'coordinator'

        # Should not raise
        manager.request_quota_increase(10.0)

        # Check message was sent
        assert len(manager._outbox) > 0


# =============================================================================
# ControlState Tests
# =============================================================================

class TestControlState:
    """Tests for ControlState dataclass."""

    def test_creation(self):
        """Test ControlState creation."""
        state = ControlState(setpoint=10.0, measured=8.0)

        assert state.setpoint == 10.0
        assert state.measured == 8.0
        assert state.error == 0.0
        assert state.last_update is not None


# =============================================================================
# DeviceController Tests
# =============================================================================

class TestDeviceController:
    """Tests for DeviceController agent."""

    def test_initialization(self):
        """Test device controller initialization."""
        controller = DeviceController(
            agent_id='device_1',
            device_type='gate',
        )

        assert controller.agent_id == 'device_1'
        assert controller.device_type == 'gate'
        assert controller.role == AgentRole.DEVICE_CONTROLLER

    def test_set_pid_gains(self):
        """Test PID gains configuration."""
        controller = DeviceController(
            agent_id='device_1',
            device_type='gate',
        )

        controller.set_pid_gains(2.0, 0.5, 0.1)

        assert controller._kp == 2.0
        assert controller._ki == 0.5
        assert controller._kd == 0.1

    def test_set_output_limits(self):
        """Test output limits configuration."""
        controller = DeviceController(
            agent_id='device_1',
            device_type='gate',
        )

        controller.set_output_limits(0.5, 4.5)

        assert controller._output_min == 0.5
        assert controller._output_max == 4.5

    def test_set_rate_limit(self):
        """Test rate limit configuration."""
        controller = DeviceController(
            agent_id='device_1',
            device_type='gate',
        )

        controller.set_rate_limit(0.3)

        assert controller._rate_limit == 0.3

    def test_perceive(self):
        """Test device perception."""
        controller = DeviceController(
            agent_id='device_1',
            device_type='gate',
        )
        controller._control_state.setpoint = 10.0
        controller._control_state.measured = 8.0

        perceptions = controller.perceive()

        assert perceptions['setpoint'] == 10.0
        assert perceptions['measured'] == 8.0

    def test_compute_control_proportional(self):
        """Test PID proportional control."""
        controller = DeviceController(
            agent_id='device_1',
            device_type='gate',
        )
        controller.set_pid_gains(1.0, 0.001, 0.0)  # P-dominant with small I
        controller._control_state.setpoint = 10.0
        controller._control_state.measured = 8.0
        controller._control_state.output = 0.0  # Reset output

        output = controller._compute_control()

        # P-dominant: error = 2, output should be close to 2 (but rate limited)
        assert 0.0 < output <= 2.5  # Allow for rate limiting

    def test_compute_control_output_limits(self):
        """Test output limiting."""
        controller = DeviceController(
            agent_id='device_1',
            device_type='gate',
        )
        controller.set_pid_gains(10.0, 0.01, 0.0)  # High gain with small I
        controller._control_state.setpoint = 10.0
        controller._control_state.measured = 0.0
        controller._control_state.output = 0.0

        output = controller._compute_control()

        # Should be clamped or rate-limited
        assert output <= controller._output_max

    def test_detect_fault_large_error(self):
        """Test fault detection for large error."""
        controller = DeviceController(
            agent_id='device_1',
            device_type='gate',
        )
        controller._control_state.error = 15.0  # > 10.0 threshold
        controller._fault_thresholds['error'] = 10.0

        fault = controller._detect_fault()

        assert fault is not None
        assert fault['type'] == 'control_error'

    def test_get_status(self):
        """Test status retrieval."""
        controller = DeviceController(
            agent_id='device_1',
            device_type='gate',
        )
        controller._control_state.setpoint = 10.0
        controller._control_state.output = 8.0

        status = controller.get_status()

        assert status['device_type'] == 'gate'
        assert 'control_state' in status
        assert status['control_state']['setpoint'] == 10.0


# =============================================================================
# GateAgent Tests
# =============================================================================

class TestGateAgent:
    """Tests for GateAgent."""

    def test_initialization(self):
        """Test gate agent initialization."""
        agent = GateAgent(
            agent_id='gate_0',
            gate_index=0,
        )

        assert agent.agent_id == 'gate_0'
        assert agent.gate_index == 0
        assert agent.device_type == 'gate'

    def test_perceive_without_model(self):
        """Test perceive without model."""
        agent = GateAgent(
            agent_id='gate_0',
            gate_index=0,
        )

        perceptions = agent.perceive()

        assert isinstance(perceptions, dict)

    def test_perceive_with_model(self):
        """Test perceive with mock model."""
        mock_model = MagicMock()
        mock_model.gate_openings = np.array([1.5, 2.0, 2.5])
        mock_model.flow_rates = np.array([15.0, 20.0, 25.0])
        mock_model.vibration_accel = np.array([0.1, 0.2, 0.15])
        mock_model.gate_stuck = [False, False, False]

        agent = GateAgent(
            agent_id='gate_1',
            gate_index=1,
            model=mock_model,
        )

        perceptions = agent.perceive()

        assert perceptions['opening'] == 2.0
        assert perceptions['flow'] == 20.0
        assert perceptions['vibration'] == 0.2
        assert perceptions['stuck'] is False

    def test_detect_fault_gate_stuck(self):
        """Test gate stuck fault detection."""
        mock_model = MagicMock()
        mock_model.gate_openings = np.array([1.0])
        mock_model.flow_rates = np.array([10.0])
        mock_model.vibration_accel = np.array([0.1])
        mock_model.gate_stuck = [True]

        agent = GateAgent(
            agent_id='gate_0',
            gate_index=0,
            model=mock_model,
        )
        agent._control_state.error = 0  # No control error
        agent._control_state.output = 2.5  # Not at limits

        fault = agent._detect_fault()

        assert fault is not None
        assert fault['type'] == 'gate_stuck'

    def test_detect_fault_high_vibration(self):
        """Test high vibration fault detection."""
        mock_model = MagicMock()
        mock_model.gate_openings = np.array([1.0])
        mock_model.flow_rates = np.array([10.0])
        mock_model.vibration_accel = np.array([0.8])  # > 0.7 threshold
        mock_model.gate_stuck = [False]

        agent = GateAgent(
            agent_id='gate_0',
            gate_index=0,
            model=mock_model,
        )
        agent._control_state.error = 0
        agent._control_state.output = 2.5  # Not at limits

        fault = agent._detect_fault()

        assert fault is not None
        assert fault['type'] == 'high_vibration'

    def test_get_status(self):
        """Test gate status retrieval."""
        agent = GateAgent(
            agent_id='gate_0',
            gate_index=0,
        )
        agent.set_belief('opening', 2.0)
        agent.set_belief('flow', 20.0)

        status = agent.get_status()

        assert status['gate_index'] == 0
        assert status['opening'] == 2.0
        assert status['flow'] == 20.0

    def test_check_resonance_in_range(self):
        """Test resonance detection in danger zone."""
        mock_model = MagicMock()
        mock_model.flow_rates = np.array([85.0])  # In resonance range 70-100
        mock_model.vibration_accel = np.array([0.25])  # > 0.2 threshold

        agent = GateAgent(
            agent_id='gate_0',
            gate_index=0,
            model=mock_model,
        )

        action = agent._check_resonance()

        assert action is not None
        assert action['action'] == 'avoid_resonance'

    def test_check_resonance_outside_range(self):
        """Test no resonance detection outside danger zone."""
        mock_model = MagicMock()
        mock_model.flow_rates = np.array([50.0])  # Outside resonance range
        mock_model.vibration_accel = np.array([0.1])

        agent = GateAgent(
            agent_id='gate_0',
            gate_index=0,
            model=mock_model,
        )

        action = agent._check_resonance()

        assert action is None


# =============================================================================
# Integration Tests
# =============================================================================

class TestAgentHierarchyIntegration:
    """Integration tests for agent hierarchy."""

    def test_coordinator_zone_hierarchy(self):
        """Test coordinator-zone manager hierarchy."""
        coordinator = CentralCoordinator(agent_id='coord')
        zone = ZoneManager(
            agent_id='zone_1',
            zone_name='Zone 1',
            gates=[0, 1],
        )

        coordinator.add_child(zone)

        assert 'zone_1' in coordinator._children
        assert zone.parent_id == 'coord'

    def test_zone_device_hierarchy(self):
        """Test zone manager-device controller hierarchy."""
        zone = ZoneManager(
            agent_id='zone_1',
            zone_name='Zone 1',
            gates=[0],
        )
        device = GateAgent(
            agent_id='gate_0',
            gate_index=0,
        )

        zone.add_child(device)

        assert 'gate_0' in zone._children
        assert device.parent_id == 'zone_1'

    def test_full_hierarchy(self):
        """Test full three-level hierarchy."""
        coordinator = CentralCoordinator(agent_id='coord')
        zone = ZoneManager(
            agent_id='zone_1',
            zone_name='Zone 1',
            gates=[0, 1],
        )
        gate0 = GateAgent(agent_id='gate_0', gate_index=0)
        gate1 = GateAgent(agent_id='gate_1', gate_index=1)

        coordinator.add_child(zone)
        zone.add_child(gate0)
        zone.add_child(gate1)

        assert coordinator._children['zone_1'] == zone
        assert zone._children['gate_0'] == gate0
        assert zone._children['gate_1'] == gate1
        assert gate0.parent_id == 'zone_1'
        assert zone.parent_id == 'coord'

    def test_hierarchical_update(self):
        """Test hierarchical update propagation."""
        mock_model = MagicMock()
        mock_model.flow_rates = np.array([10.0, 20.0, 30.0])
        mock_model.vibration_accel = np.array([0.1, 0.2, 0.15])
        mock_model.gate_openings = np.array([1.0, 2.0, 3.0])
        mock_model.gate_stuck = [False, False, False]
        mock_model.head_upstream = 10.0
        mock_model.head_downstream = 5.0

        coordinator = CentralCoordinator(agent_id='coord', model=mock_model)
        zone = ZoneManager(
            agent_id='zone_1',
            zone_name='Zone 1',
            gates=[0, 1],
            model=mock_model,
        )
        gate0 = GateAgent(agent_id='gate_0', gate_index=0, model=mock_model)

        coordinator.add_child(zone)
        zone.add_child(gate0)

        coordinator.start()
        zone.start()
        gate0.start()

        # Update hierarchy with dt parameter
        dt = 0.1
        coordinator.update(dt)
        zone.update(dt)
        gate0.update(dt)

        # Check all agents updated (decisions_made incremented on update)
        assert coordinator._stats['decisions_made'] >= 1
        assert zone._stats['decisions_made'] >= 1
        assert gate0._stats['decisions_made'] >= 1
