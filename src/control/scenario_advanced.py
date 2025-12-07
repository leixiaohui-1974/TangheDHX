# -*- coding: utf-8 -*-
"""
Advanced Scenario Manager with Automatic Detection.

This module provides comprehensive scenario management including:
- Extended scenario library covering all operational conditions
- Automatic scenario detection based on sensor data
- Scenario transition management
- Event-driven fault injection
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Enumeration of all supported scenario types."""
    # Normal operations
    NORMAL_LOW_FLOW = auto()      # S1.1 - 低流量正常运行
    NORMAL_MEDIUM_FLOW = auto()   # S1.2 - 中流量正常运行
    NORMAL_HIGH_FLOW = auto()     # S1.3 - 高流量正常运行

    # Flow transitions
    RAMP_UP = auto()              # S2.1 - 流量递增
    RAMP_DOWN = auto()            # S2.2 - 流量递减
    RESONANCE_CROSSING = auto()   # S2.3 - 共鸣区穿越
    STEP_CHANGE = auto()          # S2.4 - 阶跃变化

    # Environmental variations
    HEAD_SURGE = auto()           # S3.1 - 水头波动
    HEAD_DROP = auto()            # S3.2 - 水头骤降
    SEASONAL_VARIATION = auto()   # S3.3 - 季节性变化

    # Fault conditions
    TRASH_BLOCKAGE = auto()       # S4.1 - 格栅堵塞
    GATE_STUCK = auto()           # S4.2 - 闸门卡住
    GATE_DRIFT = auto()           # S4.3 - 闸门漂移
    SENSOR_FAULT = auto()         # S4.4 - 传感器故障
    MULTI_GATE_FAULT = auto()     # S4.5 - 多闸门故障

    # Emergency conditions
    EMERGENCY_SHUTDOWN = auto()   # S5.1 - 紧急停机
    FLOOD_CONDITION = auto()      # S5.2 - 洪水工况
    DROUGHT_CONDITION = auto()    # S5.3 - 枯水工况

    # Unknown/Transition
    UNKNOWN = auto()              # 未知工况


@dataclass
class ScenarioConfig:
    """Configuration for a specific scenario."""
    scenario_type: ScenarioType
    name: str
    description: str
    head_upstream: float = 10.0
    head_downstream: float = 8.0
    target_flow_range: Tuple[float, float] = (0.0, 200.0)
    fault_gates: List[int] = field(default_factory=list)
    fault_delay: float = 0.0  # seconds
    sensor_noise_multiplier: float = 1.0
    duration: Optional[float] = None  # None = indefinite
    transition_events: List[Dict[str, Any]] = field(default_factory=list)


class ScenarioDetector:
    """
    Automatic scenario detection based on sensor data and system state.

    Uses pattern matching and threshold detection to identify
    the current operational scenario.
    """

    def __init__(self, model: 'TangheSiphonModel'):
        """
        Initialize the scenario detector.

        Args:
            model: Reference to the physics model
        """
        self.model = model
        self._history_length = 100
        self._flow_history: List[float] = []
        self._vibration_history: List[float] = []
        self._head_history: List[float] = []

        # Detection thresholds
        self._resonance_vib_threshold = 0.3  # g
        self._high_flow_threshold = 120.0  # m³/s
        self._low_flow_threshold = 50.0  # m³/s
        self._head_surge_threshold = 0.5  # m change rate
        self._stuck_gate_threshold = 0.001  # minimal movement

        logger.debug("ScenarioDetector initialized")

    def update(self, dt: float) -> ScenarioType:
        """
        Update history and detect current scenario.

        Args:
            dt: Time step [s]

        Returns:
            Detected scenario type
        """
        # Update histories
        total_flow = float(np.sum(self.model.flow_rates))
        max_vib = float(np.max(self.model.vibration_accel))
        head_diff = self.model.head_upstream - self.model.head_downstream

        self._flow_history.append(total_flow)
        self._vibration_history.append(max_vib)
        self._head_history.append(head_diff)

        # Trim to history length
        if len(self._flow_history) > self._history_length:
            self._flow_history.pop(0)
            self._vibration_history.pop(0)
            self._head_history.pop(0)

        # Detect scenario
        return self._detect_scenario()

    def _detect_scenario(self) -> ScenarioType:
        """Analyze history and detect current scenario."""
        if len(self._flow_history) < 10:
            return ScenarioType.UNKNOWN

        # Current values
        current_flow = self._flow_history[-1]
        current_vib = self._vibration_history[-1]
        current_head = self._head_history[-1]

        # Historical statistics
        flow_mean = np.mean(self._flow_history[-20:])
        flow_std = np.std(self._flow_history[-20:])
        flow_trend = self._compute_trend(self._flow_history[-20:])
        head_trend = self._compute_trend(self._head_history[-20:])

        # Check for fault conditions first (highest priority)
        if self._detect_stuck_gate():
            return ScenarioType.GATE_STUCK

        if any(self.model.gate_stuck):
            return ScenarioType.GATE_STUCK

        # Check for emergency conditions
        if current_head < 0.5:
            return ScenarioType.DROUGHT_CONDITION

        if current_head > 5.0:
            return ScenarioType.FLOOD_CONDITION

        # Check for resonance crossing
        if current_vib > self._resonance_vib_threshold:
            return ScenarioType.RESONANCE_CROSSING

        # Check for head variations
        if abs(head_trend) > self._head_surge_threshold:
            return ScenarioType.HEAD_SURGE if head_trend > 0 else ScenarioType.HEAD_DROP

        # Check for flow transitions
        if flow_trend > 2.0:  # Rapid increase
            return ScenarioType.RAMP_UP
        elif flow_trend < -2.0:  # Rapid decrease
            return ScenarioType.RAMP_DOWN
        elif flow_std > 10.0:  # Step change
            return ScenarioType.STEP_CHANGE

        # Normal operation classification
        if current_flow < self._low_flow_threshold:
            return ScenarioType.NORMAL_LOW_FLOW
        elif current_flow > self._high_flow_threshold:
            return ScenarioType.NORMAL_HIGH_FLOW
        else:
            return ScenarioType.NORMAL_MEDIUM_FLOW

    def _compute_trend(self, data: List[float]) -> float:
        """Compute linear trend of data (slope)."""
        if len(data) < 2:
            return 0.0
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        return float(coeffs[0])

    def _detect_stuck_gate(self) -> bool:
        """Detect if any gate appears to be stuck."""
        if len(self._flow_history) < 20:
            return False

        # Check for unusual flow distribution
        flows = self.model.flow_rates
        if np.std(flows) > 0:
            # One gate producing much less than others
            min_flow = np.min(flows)
            mean_flow = np.mean(flows)
            if min_flow < mean_flow * 0.3 and mean_flow > 10:
                return True

        return False

    def get_detection_confidence(self) -> float:
        """
        Get confidence level of current detection.

        Returns:
            Confidence value between 0 and 1
        """
        if len(self._flow_history) < self._history_length:
            return len(self._flow_history) / self._history_length
        return 1.0

    def reset(self) -> None:
        """Reset detector state."""
        self._flow_history.clear()
        self._vibration_history.clear()
        self._head_history.clear()


class AdvancedScenarioManager:
    """
    Advanced scenario manager with automatic detection and transition handling.

    Features:
    - Comprehensive scenario library
    - Automatic scenario detection
    - Event-driven scenario transitions
    - Fault injection scheduling
    """

    # Complete scenario library
    SCENARIO_LIBRARY: Dict[str, ScenarioConfig] = {
        # Normal operations
        'S1.1': ScenarioConfig(
            ScenarioType.NORMAL_LOW_FLOW,
            "Normal Low Flow",
            "低流量正常运行 (Q < 50 m³/s)",
            head_upstream=10.0,
            target_flow_range=(20.0, 50.0)
        ),
        'S1.2': ScenarioConfig(
            ScenarioType.NORMAL_MEDIUM_FLOW,
            "Normal Medium Flow",
            "中流量正常运行 (50 < Q < 120 m³/s)",
            head_upstream=10.0,
            target_flow_range=(50.0, 120.0)
        ),
        'S1.3': ScenarioConfig(
            ScenarioType.NORMAL_HIGH_FLOW,
            "Normal High Flow",
            "高流量正常运行 (Q > 120 m³/s)",
            head_upstream=10.0,
            target_flow_range=(120.0, 180.0)
        ),

        # Flow transitions
        'S2.1': ScenarioConfig(
            ScenarioType.RAMP_UP,
            "Flow Ramp Up",
            "流量递增过程",
            head_upstream=10.0,
            transition_events=[
                {'time': 0.0, 'action': 'set_target_flow', 'value': 30.0},
                {'time': 30.0, 'action': 'set_target_flow', 'value': 150.0},
            ]
        ),
        'S2.2': ScenarioConfig(
            ScenarioType.RAMP_DOWN,
            "Flow Ramp Down",
            "流量递减过程",
            head_upstream=10.0,
            transition_events=[
                {'time': 0.0, 'action': 'set_target_flow', 'value': 150.0},
                {'time': 30.0, 'action': 'set_target_flow', 'value': 30.0},
            ]
        ),
        'S2.3': ScenarioConfig(
            ScenarioType.RESONANCE_CROSSING,
            "Resonance Zone Crossing",
            "共鸣区穿越测试",
            head_upstream=10.0,
            transition_events=[
                {'time': 0.0, 'action': 'set_target_flow', 'value': 80.0},
                {'time': 20.0, 'action': 'set_target_flow', 'value': 130.0},
                {'time': 40.0, 'action': 'set_target_flow', 'value': 80.0},
            ]
        ),
        'S2.4': ScenarioConfig(
            ScenarioType.STEP_CHANGE,
            "Flow Step Change",
            "流量阶跃变化",
            head_upstream=10.0,
            transition_events=[
                {'time': 0.0, 'action': 'set_target_flow', 'value': 50.0},
                {'time': 15.0, 'action': 'set_target_flow', 'value': 120.0},
                {'time': 30.0, 'action': 'set_target_flow', 'value': 70.0},
            ]
        ),

        # Environmental variations
        'S3.1': ScenarioConfig(
            ScenarioType.HEAD_SURGE,
            "Head Surge",
            "上游水头波动",
            head_upstream=10.0,
            transition_events=[
                {'time': 10.0, 'action': 'set_head_upstream', 'value': 12.0},
                {'time': 20.0, 'action': 'set_head_upstream', 'value': 9.0},
                {'time': 30.0, 'action': 'set_head_upstream', 'value': 10.0},
            ]
        ),
        'S3.2': ScenarioConfig(
            ScenarioType.HEAD_DROP,
            "Head Drop",
            "水头骤降",
            head_upstream=10.0,
            transition_events=[
                {'time': 10.0, 'action': 'set_head_upstream', 'value': 7.0},
            ]
        ),

        # Fault conditions
        'S4.1': ScenarioConfig(
            ScenarioType.TRASH_BLOCKAGE,
            "Trash Rack Blockage",
            "格栅堵塞导致水头损失",
            head_upstream=8.5,
            sensor_noise_multiplier=1.5
        ),
        'S4.2': ScenarioConfig(
            ScenarioType.GATE_STUCK,
            "Single Gate Stuck",
            "单闸门卡住故障",
            fault_gates=[1],
            fault_delay=10.0
        ),
        'S4.3': ScenarioConfig(
            ScenarioType.GATE_DRIFT,
            "Gate Position Drift",
            "闸门位置漂移",
            transition_events=[
                {'time': 10.0, 'action': 'inject_drift', 'gate': 0, 'value': 0.1},
            ]
        ),
        'S4.4': ScenarioConfig(
            ScenarioType.SENSOR_FAULT,
            "Sensor Fault",
            "传感器故障",
            sensor_noise_multiplier=5.0
        ),
        'S4.5': ScenarioConfig(
            ScenarioType.MULTI_GATE_FAULT,
            "Multiple Gate Fault",
            "多闸门同时故障",
            fault_gates=[0, 2],
            fault_delay=10.0
        ),

        # Emergency conditions
        'S5.1': ScenarioConfig(
            ScenarioType.EMERGENCY_SHUTDOWN,
            "Emergency Shutdown",
            "紧急停机",
            transition_events=[
                {'time': 10.0, 'action': 'emergency_close'},
            ]
        ),
        'S5.2': ScenarioConfig(
            ScenarioType.FLOOD_CONDITION,
            "Flood Condition",
            "洪水工况 - 高水头",
            head_upstream=14.0,
            head_downstream=10.0
        ),
        'S5.3': ScenarioConfig(
            ScenarioType.DROUGHT_CONDITION,
            "Drought Condition",
            "枯水工况 - 低水头",
            head_upstream=6.0,
            head_downstream=5.5
        ),
    }

    def __init__(self, model: 'TangheSiphonModel'):
        """
        Initialize the advanced scenario manager.

        Args:
            model: Reference to the physics model
        """
        self.model = model
        self.detector = ScenarioDetector(model)

        self.active_scenario: Optional[str] = None
        self.active_config: Optional[ScenarioConfig] = None
        self.scenario_start_time: float = 0.0
        self.detected_scenario: ScenarioType = ScenarioType.UNKNOWN

        # Callbacks for scenario events
        self._event_callbacks: Dict[str, Callable] = {}
        self._target_flow: float = 100.0

        # Fault tracking
        self._faults_injected: Dict[int, bool] = {}
        self._drift_values: Dict[int, float] = {}

        logger.info("AdvancedScenarioManager initialized with %d scenarios",
                    len(self.SCENARIO_LIBRARY))

    def set_scenario(self, scenario_id: str) -> None:
        """
        Set and initialize a scenario.

        Args:
            scenario_id: Scenario identifier (e.g., 'S1.1', 'S4.2')

        Raises:
            ValueError: If scenario_id is unknown
        """
        if scenario_id not in self.SCENARIO_LIBRARY:
            valid = ', '.join(sorted(self.SCENARIO_LIBRARY.keys()))
            raise ValueError(f"Unknown scenario '{scenario_id}'. Valid: {valid}")

        config = self.SCENARIO_LIBRARY[scenario_id]
        self.active_scenario = scenario_id
        self.active_config = config
        self.scenario_start_time = self.model.time

        # Reset state
        self._faults_injected.clear()
        self._drift_values.clear()
        self.model.gate_stuck = [False] * self.model.num_gates
        self.model.gate_noise = [0.0] * self.model.num_gates

        # Apply initial conditions
        self.model.head_upstream = config.head_upstream
        self.model.head_downstream = config.head_downstream

        logger.info(
            "Scenario %s (%s) started: %s",
            scenario_id, config.name, config.description
        )

    def update(self, dt: float = 0.1) -> Dict[str, Any]:
        """
        Update scenario state and detect conditions.

        Args:
            dt: Time step [s]

        Returns:
            Dictionary with scenario status information
        """
        # Update detector
        self.detected_scenario = self.detector.update(dt)

        status = {
            'active_scenario': self.active_scenario,
            'detected_scenario': self.detected_scenario.name,
            'detection_confidence': self.detector.get_detection_confidence(),
            'elapsed_time': self.get_elapsed_time(),
        }

        if self.active_config is None:
            return status

        elapsed = self.get_elapsed_time()

        # Process scheduled events
        for event in self.active_config.transition_events:
            if event['time'] <= elapsed < event['time'] + dt:
                self._process_event(event)

        # Process delayed faults
        if self.active_config.fault_delay > 0:
            if elapsed >= self.active_config.fault_delay:
                for gate_idx in self.active_config.fault_gates:
                    if gate_idx not in self._faults_injected:
                        self.model.inject_fault(gate_idx, 'stuck')
                        self._faults_injected[gate_idx] = True
                        logger.warning(
                            "Scheduled fault: Gate %d stuck at t=%.2fs",
                            gate_idx, self.model.time
                        )

        # Apply drift
        for gate_idx, drift in self._drift_values.items():
            self.model.gate_noise[gate_idx] = drift

        return status

    def _process_event(self, event: Dict[str, Any]) -> None:
        """Process a scheduled event."""
        action = event['action']

        if action == 'set_target_flow':
            self._target_flow = event['value']
            logger.info("Event: Target flow set to %.1f m³/s", event['value'])
            if 'set_target_flow' in self._event_callbacks:
                self._event_callbacks['set_target_flow'](event['value'])

        elif action == 'set_head_upstream':
            self.model.head_upstream = event['value']
            logger.info("Event: Head upstream set to %.1f m", event['value'])

        elif action == 'inject_drift':
            gate = event['gate']
            drift = event['value']
            self._drift_values[gate] = drift
            logger.info("Event: Gate %d drift set to %.3f m", gate, drift)

        elif action == 'emergency_close':
            # Set target to close all gates
            self._target_flow = 0.0
            logger.warning("Event: Emergency shutdown initiated")
            if 'emergency' in self._event_callbacks:
                self._event_callbacks['emergency']()

    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register a callback for scenario events."""
        self._event_callbacks[event_type] = callback

    def get_target_flow(self) -> float:
        """Get current target flow for active scenario."""
        return self._target_flow

    def get_elapsed_time(self) -> float:
        """Get elapsed time since scenario started."""
        return self.model.time - self.scenario_start_time

    def get_scenario_info(self, scenario_id: str) -> Optional[ScenarioConfig]:
        """Get information about a specific scenario."""
        return self.SCENARIO_LIBRARY.get(scenario_id)

    def list_scenarios(self) -> Dict[str, str]:
        """List all available scenarios with descriptions."""
        return {
            sid: f"{cfg.name}: {cfg.description}"
            for sid, cfg in self.SCENARIO_LIBRARY.items()
        }

    def reset(self) -> None:
        """Reset scenario manager state."""
        self.active_scenario = None
        self.active_config = None
        self.scenario_start_time = 0.0
        self._faults_injected.clear()
        self._drift_values.clear()
        self._target_flow = 100.0
        self.detector.reset()
        logger.debug("AdvancedScenarioManager reset")
