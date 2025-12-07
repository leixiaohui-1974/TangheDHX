# -*- coding: utf-8 -*-
"""
Integrated Controller with Scenario-Aware Adaptation.

This module provides a unified controller that automatically adjusts
MPC objective functions, constraints, and control strategies based on
detected scenarios.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List, Tuple
from enum import Enum

import numpy as np

from src.control.scenario_advanced import (
    ScenarioType, ScenarioDetector, AdvancedScenarioManager
)
from src.control.adaptive_mpc import AdaptiveMPC
from src.control.pid import MultiChannelPID, HybridController
from src.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class MPCObjective:
    """MPC objective function configuration."""
    alpha: float = 1.0      # Flow tracking weight
    beta: float = 0.1       # Control action penalty
    gamma: float = 10.0     # Spectral avoidance weight
    delta: float = 0.0      # Vibration suppression weight
    epsilon: float = 0.0    # Energy efficiency weight


@dataclass
class MPCConstraints:
    """MPC constraint configuration."""
    min_opening: float = 0.0
    max_opening: float = 5.0
    max_rate: float = 0.5           # m/s max gate speed
    min_velocity: float = 0.5       # m/s avoid sedimentation
    max_velocity: float = 4.5       # m/s avoid cavitation
    resonance_low: float = 2.3      # m/s resonance band lower
    resonance_high: float = 3.0     # m/s resonance band upper


@dataclass
class ScenarioControlConfig:
    """Control configuration for a specific scenario."""
    scenario: ScenarioType
    objective: MPCObjective
    constraints: MPCConstraints
    use_pid_backup: bool = False
    target_flow_modifier: float = 1.0
    emergency_action: Optional[str] = None
    description: str = ""


class IntegratedController:
    """
    Scenario-aware integrated controller.

    Automatically adjusts MPC objective function and constraints
    based on detected operational scenarios.

    Features:
    - Automatic scenario detection
    - Dynamic objective function adaptation
    - Constraint modification based on conditions
    - Smooth transitions between control modes
    - Real-time diagnostics and state tracking
    """

    # Scenario-specific control configurations
    SCENARIO_CONFIGS: Dict[ScenarioType, ScenarioControlConfig] = {
        # Normal operations - balanced control
        ScenarioType.NORMAL_LOW_FLOW: ScenarioControlConfig(
            scenario=ScenarioType.NORMAL_LOW_FLOW,
            objective=MPCObjective(alpha=1.0, beta=0.05, gamma=5.0),
            constraints=MPCConstraints(min_velocity=0.3),
            description="低流量正常运行 - 注重节能"
        ),
        ScenarioType.NORMAL_MEDIUM_FLOW: ScenarioControlConfig(
            scenario=ScenarioType.NORMAL_MEDIUM_FLOW,
            objective=MPCObjective(alpha=1.0, beta=0.1, gamma=10.0),
            constraints=MPCConstraints(),
            description="中流量正常运行 - 平衡控制"
        ),
        ScenarioType.NORMAL_HIGH_FLOW: ScenarioControlConfig(
            scenario=ScenarioType.NORMAL_HIGH_FLOW,
            objective=MPCObjective(alpha=1.0, beta=0.2, gamma=15.0),
            constraints=MPCConstraints(max_velocity=4.0),
            description="高流量正常运行 - 注重安全"
        ),

        # Resonance crossing - prioritize vibration avoidance
        ScenarioType.RESONANCE_CROSSING: ScenarioControlConfig(
            scenario=ScenarioType.RESONANCE_CROSSING,
            objective=MPCObjective(alpha=0.5, beta=0.1, gamma=50.0, delta=20.0),
            constraints=MPCConstraints(max_rate=0.3),  # Slower transitions
            description="共鸣区穿越 - 优先避振"
        ),

        # Flow transitions
        ScenarioType.RAMP_UP: ScenarioControlConfig(
            scenario=ScenarioType.RAMP_UP,
            objective=MPCObjective(alpha=1.5, beta=0.05, gamma=15.0),
            constraints=MPCConstraints(max_rate=0.4),
            description="流量递增 - 快速响应"
        ),
        ScenarioType.RAMP_DOWN: ScenarioControlConfig(
            scenario=ScenarioType.RAMP_DOWN,
            objective=MPCObjective(alpha=1.5, beta=0.05, gamma=15.0),
            constraints=MPCConstraints(max_rate=0.4),
            description="流量递减 - 平稳下降"
        ),

        # Environmental variations
        ScenarioType.HEAD_SURGE: ScenarioControlConfig(
            scenario=ScenarioType.HEAD_SURGE,
            objective=MPCObjective(alpha=2.0, beta=0.2, gamma=10.0),
            constraints=MPCConstraints(max_rate=0.3),
            description="水头波动 - 快速调节"
        ),
        ScenarioType.HEAD_DROP: ScenarioControlConfig(
            scenario=ScenarioType.HEAD_DROP,
            objective=MPCObjective(alpha=2.0, beta=0.15, gamma=8.0),
            constraints=MPCConstraints(min_velocity=0.2),
            target_flow_modifier=0.8,
            description="水头骤降 - 降低目标流量"
        ),

        # Fault conditions - use PID backup
        ScenarioType.GATE_STUCK: ScenarioControlConfig(
            scenario=ScenarioType.GATE_STUCK,
            objective=MPCObjective(alpha=1.0, beta=0.3, gamma=5.0),
            constraints=MPCConstraints(),
            use_pid_backup=True,
            target_flow_modifier=0.7,
            description="闸门卡住 - 切换PID + 降载"
        ),
        ScenarioType.MULTI_GATE_FAULT: ScenarioControlConfig(
            scenario=ScenarioType.MULTI_GATE_FAULT,
            objective=MPCObjective(alpha=0.5, beta=0.5, gamma=3.0),
            constraints=MPCConstraints(),
            use_pid_backup=True,
            target_flow_modifier=0.5,
            emergency_action="reduce_load",
            description="多闸门故障 - 紧急降载"
        ),
        ScenarioType.SENSOR_FAULT: ScenarioControlConfig(
            scenario=ScenarioType.SENSOR_FAULT,
            objective=MPCObjective(alpha=0.8, beta=0.3, gamma=5.0),
            constraints=MPCConstraints(max_rate=0.2),
            use_pid_backup=True,
            description="传感器故障 - 保守控制"
        ),

        # Emergency conditions
        ScenarioType.FLOOD_CONDITION: ScenarioControlConfig(
            scenario=ScenarioType.FLOOD_CONDITION,
            objective=MPCObjective(alpha=2.0, beta=0.05, gamma=5.0),
            constraints=MPCConstraints(max_opening=4.5, max_velocity=5.0),
            target_flow_modifier=1.5,
            description="洪水工况 - 最大泄洪"
        ),
        ScenarioType.DROUGHT_CONDITION: ScenarioControlConfig(
            scenario=ScenarioType.DROUGHT_CONDITION,
            objective=MPCObjective(alpha=1.0, beta=0.1, gamma=3.0, epsilon=5.0),
            constraints=MPCConstraints(min_velocity=0.2, max_opening=2.0),
            target_flow_modifier=0.5,
            description="枯水工况 - 节水运行"
        ),
        ScenarioType.EMERGENCY_SHUTDOWN: ScenarioControlConfig(
            scenario=ScenarioType.EMERGENCY_SHUTDOWN,
            objective=MPCObjective(alpha=0.0, beta=1.0, gamma=0.0),
            constraints=MPCConstraints(max_rate=1.0),  # Fast close
            target_flow_modifier=0.0,
            emergency_action="shutdown",
            description="紧急停机 - 快速关闭"
        ),
    }

    def __init__(self, model: 'TangheSiphonModel'):
        """
        Initialize integrated controller.

        Args:
            model: Reference to physics model
        """
        self.model = model
        self.config = get_config()

        # Create sub-controllers
        self._mpc = AdaptiveMPC(model, adaptation_enabled=True)
        self._pid = MultiChannelPID(num_gates=model.num_gates)
        self._hybrid = HybridController(self._mpc, self._pid, model)

        # Create scenario manager
        self._scenario_mgr = AdvancedScenarioManager(model)

        # Current state
        self._current_scenario: ScenarioType = ScenarioType.NORMAL_LOW_FLOW
        self._current_config: ScenarioControlConfig = self.SCENARIO_CONFIGS[
            ScenarioType.NORMAL_MEDIUM_FLOW
        ]
        self._target_flow: float = 100.0
        self._time: float = 0.0

        # Diagnostics history
        self._history: List[Dict[str, Any]] = []
        self._max_history: int = 1000

        # Callbacks for external systems
        self._callbacks: Dict[str, Callable] = {}

        logger.info("IntegratedController initialized with %d scenario configs",
                    len(self.SCENARIO_CONFIGS))

    def set_target_flow(self, target: float) -> None:
        """Set base target flow rate."""
        self._target_flow = target
        logger.debug("Target flow set to %.1f m³/s", target)

    def update(self, dt: float = 0.1) -> Dict[str, Any]:
        """
        Main control loop update.

        Performs:
        1. Scenario detection
        2. Configuration update if scenario changed
        3. Control computation
        4. Diagnostics collection

        Args:
            dt: Time step [s]

        Returns:
            Control output and diagnostics
        """
        self._time += dt

        # 1. Update scenario detection
        scenario_status = self._scenario_mgr.update(dt)
        detected = self._scenario_mgr.detected_scenario

        # 2. Handle scenario transitions
        if detected != self._current_scenario:
            self._handle_scenario_transition(detected)

        # 3. Get effective target flow
        effective_target = self._target_flow * self._current_config.target_flow_modifier

        # 4. Apply updated MPC weights
        self._apply_mpc_configuration()

        # 5. Compute control output
        if self._current_config.use_pid_backup:
            self._hybrid.force_mode(use_mpc=False)
        else:
            self._hybrid.force_mode(use_mpc=True)

        outputs, controller_used = self._hybrid.compute(
            target_flow=effective_target,
            current_openings=self.model.gate_openings,
            head_diff=self.model.head_upstream - self.model.head_downstream,
            dt=dt
        )

        # 6. Apply constraint enforcement
        outputs = self._enforce_constraints(outputs, dt)

        # 7. Handle emergency actions
        if self._current_config.emergency_action:
            outputs = self._handle_emergency(outputs)

        # 8. Collect diagnostics
        diagnostics = self._collect_diagnostics(
            outputs, controller_used, effective_target, scenario_status
        )

        # Store history
        self._store_history(diagnostics)

        return diagnostics

    def _handle_scenario_transition(self, new_scenario: ScenarioType) -> None:
        """Handle transition to a new scenario."""
        old_scenario = self._current_scenario

        # Get new configuration (use default if not defined)
        if new_scenario in self.SCENARIO_CONFIGS:
            self._current_config = self.SCENARIO_CONFIGS[new_scenario]
        else:
            # Unknown scenario - use conservative defaults
            self._current_config = ScenarioControlConfig(
                scenario=new_scenario,
                objective=MPCObjective(alpha=0.8, beta=0.2, gamma=10.0),
                constraints=MPCConstraints(max_rate=0.3),
                description="未知场景 - 保守控制"
            )

        self._current_scenario = new_scenario

        logger.info(
            "Scenario transition: %s -> %s (%s)",
            old_scenario.name, new_scenario.name,
            self._current_config.description
        )

        # Trigger callback
        if 'scenario_change' in self._callbacks:
            self._callbacks['scenario_change'](old_scenario, new_scenario)

    def _apply_mpc_configuration(self) -> None:
        """Apply current scenario's MPC configuration."""
        obj = self._current_config.objective

        # Update MPC weights
        self._mpc._alpha = obj.alpha
        self._mpc._beta = obj.beta
        self._mpc._gamma = obj.gamma

    def _enforce_constraints(
        self,
        outputs: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """Enforce scenario-specific constraints."""
        constraints = self._current_config.constraints

        # Opening limits
        outputs = np.clip(outputs, constraints.min_opening, constraints.max_opening)

        # Rate limiting
        if hasattr(self, '_last_outputs'):
            max_change = constraints.max_rate * dt
            change = outputs - self._last_outputs
            change = np.clip(change, -max_change, max_change)
            outputs = self._last_outputs + change

        self._last_outputs = outputs.copy()

        return outputs

    def _handle_emergency(self, outputs: np.ndarray) -> np.ndarray:
        """Handle emergency actions."""
        action = self._current_config.emergency_action

        if action == "shutdown":
            # Rapidly close all gates
            return np.zeros_like(outputs)
        elif action == "reduce_load":
            # Limit openings to 50%
            return np.clip(outputs, 0, 2.5)

        return outputs

    def _collect_diagnostics(
        self,
        outputs: np.ndarray,
        controller_used: str,
        effective_target: float,
        scenario_status: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collect comprehensive diagnostics."""
        obj = self._current_config.objective
        constraints = self._current_config.constraints

        actual_flow = float(np.sum(self.model.flow_rates))
        flow_error = effective_target - actual_flow

        return {
            # Time
            'timestamp': self._time,

            # Scenario
            'scenario': {
                'detected': self._current_scenario.name,
                'confidence': scenario_status.get('detection_confidence', 0.0),
                'description': self._current_config.description,
            },

            # Control outputs
            'control': {
                'outputs': outputs.tolist(),
                'controller': controller_used,
                'target_flow': effective_target,
                'target_modifier': self._current_config.target_flow_modifier,
            },

            # MPC configuration
            'mpc_config': {
                'alpha': obj.alpha,
                'beta': obj.beta,
                'gamma': obj.gamma,
                'delta': obj.delta,
                'epsilon': obj.epsilon,
            },

            # Constraints
            'constraints': {
                'max_opening': constraints.max_opening,
                'max_rate': constraints.max_rate,
                'velocity_band': [constraints.resonance_low, constraints.resonance_high],
            },

            # System state
            'state': {
                'openings': self.model.gate_openings.tolist(),
                'flows': self.model.flow_rates.tolist(),
                'velocities': self.model.velocities.tolist(),
                'vibrations': self.model.vibration_accel.tolist(),
                'total_flow': actual_flow,
                'flow_error': flow_error,
                'head_diff': self.model.head_upstream - self.model.head_downstream,
            },

            # Performance
            'performance': {
                'flow_tracking_error': abs(flow_error),
                'max_vibration': float(np.max(self.model.vibration_accel)),
                'in_resonance_band': self._check_resonance_band(),
            },

            # Adaptive MPC state
            'adaptation': {
                'uncertainty': self._mpc.get_uncertainty(),
                'weights': self._mpc.get_weights(),
            },
        }

    def _check_resonance_band(self) -> bool:
        """Check if any velocity is in resonance band."""
        constraints = self._current_config.constraints
        for v in self.model.velocities:
            if constraints.resonance_low <= v <= constraints.resonance_high:
                return True
        return False

    def _store_history(self, diagnostics: Dict[str, Any]) -> None:
        """Store diagnostics in history buffer."""
        self._history.append(diagnostics)
        if len(self._history) > self._max_history:
            self._history.pop(0)

    def get_history(self, n: int = 100) -> List[Dict[str, Any]]:
        """Get recent history."""
        return self._history[-n:]

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for events."""
        self._callbacks[event] = callback

    def get_current_scenario(self) -> ScenarioType:
        """Get current detected scenario."""
        return self._current_scenario

    def get_current_config(self) -> ScenarioControlConfig:
        """Get current control configuration."""
        return self._current_config

    def get_scenario_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all scenario configurations for display."""
        result = {}
        for scenario_type, config in self.SCENARIO_CONFIGS.items():
            result[scenario_type.name] = {
                'description': config.description,
                'objective': {
                    'alpha': config.objective.alpha,
                    'beta': config.objective.beta,
                    'gamma': config.objective.gamma,
                },
                'target_modifier': config.target_flow_modifier,
                'use_pid': config.use_pid_backup,
                'emergency': config.emergency_action,
            }
        return result

    def force_scenario(self, scenario: ScenarioType) -> None:
        """Force a specific scenario (for testing)."""
        self._handle_scenario_transition(scenario)
        logger.warning("Scenario forced to %s", scenario.name)

    def reset(self) -> None:
        """Reset controller state."""
        self._current_scenario = ScenarioType.NORMAL_LOW_FLOW
        self._current_config = self.SCENARIO_CONFIGS[ScenarioType.NORMAL_MEDIUM_FLOW]
        self._target_flow = 100.0
        self._time = 0.0
        self._history.clear()
        self._mpc.reset()
        self._pid.reset()
        self._scenario_mgr.reset()
        if hasattr(self, '_last_outputs'):
            del self._last_outputs
        logger.info("IntegratedController reset")
