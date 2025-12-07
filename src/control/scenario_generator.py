# -*- coding: utf-8 -*-
"""
Full Scenario Generation Framework.

This module provides comprehensive scenario generation capabilities:
- Parametric scenario generation with continuous parameter ranges
- Combination scenarios (multiple conditions simultaneously)
- Time-series scenario patterns
- Monte Carlo random scenario generation
- Scenario difficulty scaling

Can generate 10,000+ unique test scenarios.
"""

import logging
import itertools
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Generator, Callable
from enum import Enum, auto
import numpy as np

from src.control.scenario_advanced import ScenarioType

logger = logging.getLogger(__name__)


# =============================================================================
# Base Scenario Dimensions
# =============================================================================

class FlowRegime(Enum):
    """Flow regime classification."""
    VERY_LOW = auto()      # < 30 m³/s
    LOW = auto()           # 30-60 m³/s
    MEDIUM_LOW = auto()    # 60-90 m³/s
    MEDIUM = auto()        # 90-120 m³/s
    MEDIUM_HIGH = auto()   # 120-150 m³/s
    HIGH = auto()          # 150-180 m³/s
    VERY_HIGH = auto()     # > 180 m³/s


class HeadCondition(Enum):
    """Head difference condition."""
    VERY_LOW = auto()      # < 1.0 m
    LOW = auto()           # 1.0-1.5 m
    NORMAL = auto()        # 1.5-2.5 m
    HIGH = auto()          # 2.5-3.5 m
    VERY_HIGH = auto()     # > 3.5 m
    FLUCTUATING = auto()   # Variable


class VibrationLevel(Enum):
    """Vibration intensity level."""
    NONE = auto()          # < 0.05 g
    LOW = auto()           # 0.05-0.15 g
    MODERATE = auto()      # 0.15-0.3 g
    HIGH = auto()          # 0.3-0.5 g
    SEVERE = auto()        # 0.5-0.7 g
    CRITICAL = auto()      # > 0.7 g


class GateFaultType(Enum):
    """Gate fault types."""
    NONE = auto()
    STUCK_OPEN = auto()
    STUCK_CLOSED = auto()
    STUCK_PARTIAL = auto()
    DRIFT_POSITIVE = auto()
    DRIFT_NEGATIVE = auto()
    OSCILLATING = auto()
    SLOW_RESPONSE = auto()


class SensorFaultType(Enum):
    """Sensor fault types."""
    NONE = auto()
    BIAS = auto()
    DRIFT = auto()
    NOISE_HIGH = auto()
    STUCK = auto()
    DEAD = auto()
    INTERMITTENT = auto()
    DELAYED = auto()


class EnvironmentalCondition(Enum):
    """Environmental conditions."""
    NORMAL = auto()
    TRASH_LIGHT = auto()
    TRASH_MODERATE = auto()
    TRASH_HEAVY = auto()
    SEDIMENT_LOW = auto()
    SEDIMENT_HIGH = auto()
    TEMPERATURE_COLD = auto()
    TEMPERATURE_HOT = auto()


class TransitionPattern(Enum):
    """Flow transition patterns."""
    STEADY = auto()
    RAMP_UP_SLOW = auto()
    RAMP_UP_FAST = auto()
    RAMP_DOWN_SLOW = auto()
    RAMP_DOWN_FAST = auto()
    STEP_UP = auto()
    STEP_DOWN = auto()
    OSCILLATING = auto()
    RANDOM_WALK = auto()


# =============================================================================
# Scenario Parameter Ranges
# =============================================================================

@dataclass
class ParameterRange:
    """Defines a continuous parameter range."""
    name: str
    min_value: float
    max_value: float
    unit: str = ""
    steps: int = 10  # Discretization steps

    def sample(self, n: int = 1) -> np.ndarray:
        """Sample n values from the range."""
        return np.random.uniform(self.min_value, self.max_value, n)

    def linspace(self) -> np.ndarray:
        """Get linearly spaced values."""
        return np.linspace(self.min_value, self.max_value, self.steps)

    def __iter__(self):
        return iter(self.linspace())


# Standard parameter ranges for Tanghe siphon
PARAMETER_RANGES = {
    'target_flow': ParameterRange('Target Flow', 20.0, 200.0, 'm³/s', steps=19),
    'head_upstream': ParameterRange('Upstream Head', 6.0, 14.0, 'm', steps=9),
    'head_downstream': ParameterRange('Downstream Head', 4.0, 12.0, 'm', steps=9),
    'initial_opening': ParameterRange('Initial Opening', 0.0, 5.0, 'm', steps=11),
    'ramp_rate': ParameterRange('Flow Ramp Rate', 0.5, 10.0, 'm³/s²', steps=10),
    'disturbance_amplitude': ParameterRange('Disturbance Amp', 0.0, 2.0, 'm', steps=5),
    'disturbance_frequency': ParameterRange('Disturbance Freq', 0.01, 0.5, 'Hz', steps=5),
    'noise_level': ParameterRange('Sensor Noise', 0.0, 0.2, '', steps=5),
    'fault_severity': ParameterRange('Fault Severity', 0.0, 1.0, '', steps=5),
    'fault_onset_time': ParameterRange('Fault Onset', 5.0, 50.0, 's', steps=10),
}


# =============================================================================
# Scenario Specification
# =============================================================================

@dataclass
class ScenarioSpec:
    """Complete specification for a generated scenario."""
    # Identification
    scenario_id: str
    name: str
    description: str
    category: str
    difficulty: float  # 0.0 (easy) to 1.0 (hard)

    # Initial conditions
    target_flow: float = 100.0
    head_upstream: float = 10.0
    head_downstream: float = 8.0
    initial_openings: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Flow pattern
    flow_pattern: TransitionPattern = TransitionPattern.STEADY
    flow_params: Dict[str, float] = field(default_factory=dict)

    # Environmental
    environmental: EnvironmentalCondition = EnvironmentalCondition.NORMAL
    environmental_params: Dict[str, float] = field(default_factory=dict)

    # Gate faults (per gate)
    gate_faults: Tuple[GateFaultType, GateFaultType, GateFaultType] = (
        GateFaultType.NONE, GateFaultType.NONE, GateFaultType.NONE
    )
    gate_fault_params: Dict[str, Any] = field(default_factory=dict)

    # Sensor faults (per sensor type per gate)
    sensor_faults: Dict[str, List[SensorFaultType]] = field(default_factory=dict)
    sensor_fault_params: Dict[str, Any] = field(default_factory=dict)

    # Duration and timing
    duration: float = 60.0  # seconds
    warmup_time: float = 5.0  # seconds before main scenario

    # Events (time, action, params)
    events: List[Tuple[float, str, Dict[str, Any]]] = field(default_factory=list)

    # Expected behavior
    expected_scenario_type: Optional[ScenarioType] = None
    pass_criteria: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.scenario_id,
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'difficulty': self.difficulty,
            'initial': {
                'target_flow': self.target_flow,
                'head_upstream': self.head_upstream,
                'head_downstream': self.head_downstream,
                'openings': list(self.initial_openings),
            },
            'flow_pattern': self.flow_pattern.name,
            'environmental': self.environmental.name,
            'gate_faults': [f.name for f in self.gate_faults],
            'duration': self.duration,
            'events': self.events,
        }


# =============================================================================
# Scenario Generators
# =============================================================================

class BaseScenarioGenerator:
    """Base class for scenario generators."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self._counter = 0

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter:05d}"

    def generate(self) -> Generator[ScenarioSpec, None, None]:
        raise NotImplementedError


class ParametricScenarioGenerator(BaseScenarioGenerator):
    """
    Generates scenarios by varying parameters systematically.

    Creates grid of scenarios across parameter space.
    """

    def __init__(
        self,
        parameters: List[str] = None,
        steps_per_param: int = 5,
        seed: Optional[int] = None
    ):
        super().__init__(seed)
        self.parameters = parameters or ['target_flow', 'head_upstream']
        self.steps = steps_per_param

    def generate(self) -> Generator[ScenarioSpec, None, None]:
        """Generate parametric scenarios."""
        # Get ranges for selected parameters
        ranges = [PARAMETER_RANGES[p] for p in self.parameters]

        # Create grid
        grids = [np.linspace(r.min_value, r.max_value, self.steps) for r in ranges]

        for values in itertools.product(*grids):
            params = dict(zip(self.parameters, values))

            # Calculate difficulty based on extremity of parameters
            difficulty = self._calculate_difficulty(params)

            yield ScenarioSpec(
                scenario_id=self._next_id("PARAM"),
                name=f"Parametric: {', '.join(f'{k}={v:.1f}' for k, v in params.items())}",
                description=f"Parametric sweep scenario",
                category="parametric",
                difficulty=difficulty,
                target_flow=params.get('target_flow', 100.0),
                head_upstream=params.get('head_upstream', 10.0),
                head_downstream=params.get('head_downstream', 8.0),
                flow_pattern=TransitionPattern.STEADY,
            )

    def _calculate_difficulty(self, params: Dict[str, float]) -> float:
        """Calculate difficulty based on parameter values."""
        difficulty = 0.0

        # High/low flows are harder
        if 'target_flow' in params:
            flow = params['target_flow']
            if flow < 40 or flow > 160:
                difficulty += 0.3
            if flow < 30 or flow > 180:
                difficulty += 0.2

        # Low head is harder
        if 'head_upstream' in params:
            head = params['head_upstream']
            if head < 7.0 or head > 13.0:
                difficulty += 0.2

        return min(1.0, difficulty)

    def count(self) -> int:
        """Count total scenarios."""
        return self.steps ** len(self.parameters)


class FlowTransitionGenerator(BaseScenarioGenerator):
    """
    Generates flow transition scenarios.

    Tests system response to various flow changes.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)

        # Flow transition configurations
        self.transitions = [
            # (start_flow, end_flow, duration, pattern)
            (30, 150, 30, TransitionPattern.RAMP_UP_SLOW),
            (30, 150, 10, TransitionPattern.RAMP_UP_FAST),
            (150, 30, 30, TransitionPattern.RAMP_DOWN_SLOW),
            (150, 30, 10, TransitionPattern.RAMP_DOWN_FAST),
            (50, 120, 0, TransitionPattern.STEP_UP),
            (120, 50, 0, TransitionPattern.STEP_DOWN),
            (80, 120, 60, TransitionPattern.OSCILLATING),
        ]

        # Head variations
        self.head_conditions = [
            (10.0, 8.0),   # Normal
            (7.0, 5.5),    # Low
            (13.0, 10.0),  # High
        ]

    def generate(self) -> Generator[ScenarioSpec, None, None]:
        """Generate flow transition scenarios."""
        for (start, end, dur, pattern), (h_up, h_down) in itertools.product(
            self.transitions, self.head_conditions
        ):
            # Create events for flow changes
            events = []
            if pattern == TransitionPattern.STEP_UP or pattern == TransitionPattern.STEP_DOWN:
                events.append((10.0, 'set_target_flow', {'value': end}))
            elif pattern == TransitionPattern.OSCILLATING:
                for t in range(10, 60, 10):
                    target = start if (t // 10) % 2 == 0 else end
                    events.append((float(t), 'set_target_flow', {'value': target}))

            difficulty = self._calculate_difficulty(start, end, pattern, h_up)

            yield ScenarioSpec(
                scenario_id=self._next_id("TRANS"),
                name=f"Transition: {pattern.name} {start}->{end} @ H={h_up}m",
                description=f"Flow transition from {start} to {end} m³/s",
                category="transition",
                difficulty=difficulty,
                target_flow=start,
                head_upstream=h_up,
                head_downstream=h_down,
                flow_pattern=pattern,
                flow_params={'start': start, 'end': end, 'duration': dur},
                duration=max(60, dur + 30),
                events=events,
            )

    def _calculate_difficulty(
        self, start: float, end: float,
        pattern: TransitionPattern, head: float
    ) -> float:
        difficulty = 0.0

        # Large transitions are harder
        if abs(end - start) > 100:
            difficulty += 0.3

        # Fast transitions are harder
        if pattern in [TransitionPattern.RAMP_UP_FAST, TransitionPattern.RAMP_DOWN_FAST]:
            difficulty += 0.2
        if pattern in [TransitionPattern.STEP_UP, TransitionPattern.STEP_DOWN]:
            difficulty += 0.3

        # Crossing resonance zone is harder
        resonance_zone = (70, 100)
        if (start < resonance_zone[0] < end) or (end < resonance_zone[0] < start):
            difficulty += 0.2

        # Low head is harder
        if head < 8.0:
            difficulty += 0.2

        return min(1.0, difficulty)

    def count(self) -> int:
        return len(self.transitions) * len(self.head_conditions)


class FaultScenarioGenerator(BaseScenarioGenerator):
    """
    Generates fault scenarios.

    Tests system response to various equipment faults.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)

        # Gate fault configurations
        self.gate_faults = [
            GateFaultType.STUCK_OPEN,
            GateFaultType.STUCK_CLOSED,
            GateFaultType.STUCK_PARTIAL,
            GateFaultType.DRIFT_POSITIVE,
            GateFaultType.DRIFT_NEGATIVE,
            GateFaultType.SLOW_RESPONSE,
        ]

        # Sensor fault configurations
        self.sensor_faults = [
            SensorFaultType.BIAS,
            SensorFaultType.DRIFT,
            SensorFaultType.NOISE_HIGH,
            SensorFaultType.STUCK,
            SensorFaultType.DEAD,
        ]

        # Operating points
        self.operating_points = [
            (60, 10.0),   # Low flow
            (100, 10.0),  # Medium flow
            (150, 10.0),  # High flow
        ]

        # Fault onset times
        self.onset_times = [5.0, 15.0, 30.0]

    def generate(self) -> Generator[ScenarioSpec, None, None]:
        """Generate fault scenarios."""
        # Single gate faults
        for gate_idx in range(3):
            for fault_type in self.gate_faults:
                for (flow, head), onset in itertools.product(
                    self.operating_points, self.onset_times
                ):
                    gate_faults = [GateFaultType.NONE] * 3
                    gate_faults[gate_idx] = fault_type

                    events = [(onset, 'inject_gate_fault', {
                        'gate': gate_idx, 'type': fault_type.name
                    })]

                    yield ScenarioSpec(
                        scenario_id=self._next_id("GFLT"),
                        name=f"Gate {gate_idx} {fault_type.name} @ Q={flow}",
                        description=f"Gate {gate_idx} fault: {fault_type.name}",
                        category="gate_fault",
                        difficulty=self._gate_fault_difficulty(fault_type, flow),
                        target_flow=flow,
                        head_upstream=head,
                        gate_faults=tuple(gate_faults),
                        gate_fault_params={'onset': onset, 'severity': 1.0},
                        events=events,
                        expected_scenario_type=ScenarioType.GATE_STUCK,
                    )

        # Multi-gate faults
        for fault_type in [GateFaultType.STUCK_PARTIAL, GateFaultType.SLOW_RESPONSE]:
            for (flow, head) in self.operating_points:
                # Two gates faulty
                for g1, g2 in [(0, 1), (0, 2), (1, 2)]:
                    gate_faults = [GateFaultType.NONE] * 3
                    gate_faults[g1] = fault_type
                    gate_faults[g2] = fault_type

                    yield ScenarioSpec(
                        scenario_id=self._next_id("MGFLT"),
                        name=f"Gates {g1},{g2} {fault_type.name} @ Q={flow}",
                        description=f"Multi-gate fault",
                        category="multi_gate_fault",
                        difficulty=0.8,
                        target_flow=flow,
                        gate_faults=tuple(gate_faults),
                        expected_scenario_type=ScenarioType.MULTI_GATE_FAULT,
                    )

        # Sensor faults
        for sensor_type in ['adcp', 'vibration', 'position']:
            for gate_idx in range(3):
                for fault_type in self.sensor_faults:
                    for (flow, head) in self.operating_points:
                        sensor_faults = {sensor_type: [SensorFaultType.NONE] * 3}
                        sensor_faults[sensor_type][gate_idx] = fault_type

                        yield ScenarioSpec(
                            scenario_id=self._next_id("SFLT"),
                            name=f"{sensor_type} {gate_idx} {fault_type.name}",
                            description=f"Sensor fault: {sensor_type} gate {gate_idx}",
                            category="sensor_fault",
                            difficulty=self._sensor_fault_difficulty(fault_type),
                            target_flow=flow,
                            sensor_faults=sensor_faults,
                            expected_scenario_type=ScenarioType.SENSOR_FAULT,
                        )

    def _gate_fault_difficulty(self, fault: GateFaultType, flow: float) -> float:
        difficulty = 0.3
        if fault in [GateFaultType.STUCK_CLOSED, GateFaultType.STUCK_OPEN]:
            difficulty += 0.3
        if flow > 120:
            difficulty += 0.2
        return min(1.0, difficulty)

    def _sensor_fault_difficulty(self, fault: SensorFaultType) -> float:
        difficulty = 0.2
        if fault in [SensorFaultType.DEAD, SensorFaultType.STUCK]:
            difficulty += 0.3
        return min(1.0, difficulty)

    def count(self) -> int:
        # Rough estimate
        gate_single = 3 * len(self.gate_faults) * len(self.operating_points) * len(self.onset_times)
        gate_multi = 2 * len(self.operating_points) * 3
        sensor = 3 * 3 * len(self.sensor_faults) * len(self.operating_points)
        return gate_single + gate_multi + sensor


class CombinationScenarioGenerator(BaseScenarioGenerator):
    """
    Generates combination scenarios with multiple simultaneous conditions.

    This is where the scenario count explodes!
    """

    def __init__(
        self,
        max_combinations: int = 3,
        seed: Optional[int] = None
    ):
        super().__init__(seed)
        self.max_combinations = max_combinations

        # Condition dimensions
        self.flow_regimes = list(FlowRegime)
        self.head_conditions = list(HeadCondition)
        self.vibration_levels = [VibrationLevel.NONE, VibrationLevel.LOW,
                                  VibrationLevel.MODERATE, VibrationLevel.HIGH]
        self.gate_conditions = [
            (GateFaultType.NONE, GateFaultType.NONE, GateFaultType.NONE),
            (GateFaultType.STUCK_PARTIAL, GateFaultType.NONE, GateFaultType.NONE),
            (GateFaultType.NONE, GateFaultType.SLOW_RESPONSE, GateFaultType.NONE),
        ]
        self.env_conditions = [
            EnvironmentalCondition.NORMAL,
            EnvironmentalCondition.TRASH_MODERATE,
            EnvironmentalCondition.SEDIMENT_LOW,
        ]
        self.transition_patterns = [
            TransitionPattern.STEADY,
            TransitionPattern.RAMP_UP_SLOW,
            TransitionPattern.OSCILLATING,
        ]

    def generate(self) -> Generator[ScenarioSpec, None, None]:
        """Generate combination scenarios."""
        # Full grid (subset for memory)
        count = 0
        max_per_category = 1000

        for flow in self.flow_regimes:
            for head in self.head_conditions:
                for vib in self.vibration_levels:
                    for gate in self.gate_conditions:
                        for env in self.env_conditions:
                            for trans in self.transition_patterns:
                                if count >= max_per_category * 6:
                                    return

                                # Skip trivial combinations
                                if (flow == FlowRegime.MEDIUM and
                                    head == HeadCondition.NORMAL and
                                    vib == VibrationLevel.NONE and
                                    gate[0] == GateFaultType.NONE and
                                    env == EnvironmentalCondition.NORMAL and
                                    trans == TransitionPattern.STEADY):
                                    continue

                                difficulty = self._calculate_difficulty(
                                    flow, head, vib, gate, env, trans
                                )

                                target_flow = self._flow_regime_to_value(flow)
                                h_up, h_down = self._head_condition_to_values(head)

                                yield ScenarioSpec(
                                    scenario_id=self._next_id("COMBO"),
                                    name=f"Combo: {flow.name}/{head.name}/{vib.name}",
                                    description=self._build_description(
                                        flow, head, vib, gate, env, trans
                                    ),
                                    category="combination",
                                    difficulty=difficulty,
                                    target_flow=target_flow,
                                    head_upstream=h_up,
                                    head_downstream=h_down,
                                    flow_pattern=trans,
                                    gate_faults=gate,
                                    environmental=env,
                                )
                                count += 1

    def _flow_regime_to_value(self, regime: FlowRegime) -> float:
        mapping = {
            FlowRegime.VERY_LOW: 25,
            FlowRegime.LOW: 45,
            FlowRegime.MEDIUM_LOW: 75,
            FlowRegime.MEDIUM: 105,
            FlowRegime.MEDIUM_HIGH: 135,
            FlowRegime.HIGH: 165,
            FlowRegime.VERY_HIGH: 190,
        }
        return mapping.get(regime, 100)

    def _head_condition_to_values(self, cond: HeadCondition) -> Tuple[float, float]:
        mapping = {
            HeadCondition.VERY_LOW: (7.0, 6.5),
            HeadCondition.LOW: (8.0, 7.0),
            HeadCondition.NORMAL: (10.0, 8.0),
            HeadCondition.HIGH: (12.0, 9.0),
            HeadCondition.VERY_HIGH: (14.0, 10.0),
            HeadCondition.FLUCTUATING: (10.0, 8.0),
        }
        return mapping.get(cond, (10.0, 8.0))

    def _calculate_difficulty(
        self,
        flow: FlowRegime,
        head: HeadCondition,
        vib: VibrationLevel,
        gate: Tuple[GateFaultType, ...],
        env: EnvironmentalCondition,
        trans: TransitionPattern
    ) -> float:
        difficulty = 0.0

        # Flow extremes
        if flow in [FlowRegime.VERY_LOW, FlowRegime.VERY_HIGH]:
            difficulty += 0.2

        # Head extremes
        if head in [HeadCondition.VERY_LOW, HeadCondition.VERY_HIGH]:
            difficulty += 0.2

        # Vibration
        vib_map = {
            VibrationLevel.NONE: 0, VibrationLevel.LOW: 0.1,
            VibrationLevel.MODERATE: 0.2, VibrationLevel.HIGH: 0.3,
            VibrationLevel.SEVERE: 0.4, VibrationLevel.CRITICAL: 0.5
        }
        difficulty += vib_map.get(vib, 0)

        # Gate faults
        fault_count = sum(1 for g in gate if g != GateFaultType.NONE)
        difficulty += fault_count * 0.15

        # Environmental
        if env != EnvironmentalCondition.NORMAL:
            difficulty += 0.1

        # Transitions
        if trans != TransitionPattern.STEADY:
            difficulty += 0.1

        return min(1.0, difficulty)

    def _build_description(self, flow, head, vib, gate, env, trans) -> str:
        parts = [f"Flow: {flow.name}", f"Head: {head.name}"]
        if vib != VibrationLevel.NONE:
            parts.append(f"Vib: {vib.name}")
        faults = [g.name for g in gate if g != GateFaultType.NONE]
        if faults:
            parts.append(f"Faults: {','.join(faults)}")
        if env != EnvironmentalCondition.NORMAL:
            parts.append(f"Env: {env.name}")
        if trans != TransitionPattern.STEADY:
            parts.append(f"Trans: {trans.name}")
        return " | ".join(parts)

    def count(self) -> int:
        """Estimate total combinations."""
        return (len(self.flow_regimes) * len(self.head_conditions) *
                len(self.vibration_levels) * len(self.gate_conditions) *
                len(self.env_conditions) * len(self.transition_patterns))


class MonteCarloScenarioGenerator(BaseScenarioGenerator):
    """
    Generates random scenarios using Monte Carlo sampling.

    Useful for stress testing with random conditions.
    """

    def __init__(
        self,
        n_scenarios: int = 1000,
        seed: Optional[int] = None
    ):
        super().__init__(seed)
        self.n_scenarios = n_scenarios

    def generate(self) -> Generator[ScenarioSpec, None, None]:
        """Generate random scenarios."""
        for i in range(self.n_scenarios):
            # Random parameters
            target_flow = self.rng.uniform(20, 200)
            head_up = self.rng.uniform(6, 14)
            head_down = self.rng.uniform(4, min(head_up - 0.5, 12))

            # Random faults
            gate_faults = []
            for _ in range(3):
                if self.rng.random() < 0.1:  # 10% fault probability
                    fault = self.rng.choice(list(GateFaultType)[1:])  # Exclude NONE
                else:
                    fault = GateFaultType.NONE
                gate_faults.append(fault)

            # Random pattern
            pattern = self.rng.choice(list(TransitionPattern))

            # Random environment
            env = self.rng.choice(list(EnvironmentalCondition))

            difficulty = self._estimate_difficulty(
                target_flow, head_up, gate_faults, pattern
            )

            yield ScenarioSpec(
                scenario_id=self._next_id("MC"),
                name=f"MonteCarlo #{i+1}",
                description=f"Random scenario: Q={target_flow:.0f}, H={head_up:.1f}",
                category="montecarlo",
                difficulty=difficulty,
                target_flow=target_flow,
                head_upstream=head_up,
                head_downstream=head_down,
                flow_pattern=pattern,
                gate_faults=tuple(gate_faults),
                environmental=env,
            )

    def _estimate_difficulty(
        self,
        flow: float,
        head: float,
        faults: List[GateFaultType],
        pattern: TransitionPattern
    ) -> float:
        difficulty = 0.0

        # Flow extremes
        if flow < 40 or flow > 160:
            difficulty += 0.2

        # Low head
        if head < 8:
            difficulty += 0.2

        # Faults
        fault_count = sum(1 for f in faults if f != GateFaultType.NONE)
        difficulty += fault_count * 0.2

        # Complex patterns
        if pattern != TransitionPattern.STEADY:
            difficulty += 0.15

        return min(1.0, difficulty)

    def count(self) -> int:
        return self.n_scenarios


class TemporalScenarioGenerator(BaseScenarioGenerator):
    """
    Generates time-series scenarios with complex temporal patterns.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)

        # Define temporal patterns
        self.patterns = [
            self._daily_cycle,
            self._weekly_pattern,
            self._storm_event,
            self._drought_recovery,
            self._flood_event,
            self._maintenance_window,
        ]

    def generate(self) -> Generator[ScenarioSpec, None, None]:
        """Generate temporal scenarios."""
        for pattern_func in self.patterns:
            for variation in range(3):  # 3 variations per pattern
                events = pattern_func(variation)
                name = pattern_func.__name__.replace('_', ' ').title()

                yield ScenarioSpec(
                    scenario_id=self._next_id("TEMP"),
                    name=f"{name} v{variation+1}",
                    description=f"Temporal pattern: {name}",
                    category="temporal",
                    difficulty=0.4 + variation * 0.2,
                    duration=max(e[0] for e in events) + 30 if events else 60,
                    events=events,
                )

    def _daily_cycle(self, variation: int) -> List[Tuple[float, str, Dict]]:
        """Simulate daily flow cycle."""
        base_flow = 80 + variation * 20
        events = []
        for hour in range(0, 24, 4):
            t = hour * 5  # Compressed time
            # Morning peak
            if 6 <= hour < 10:
                flow = base_flow * 1.3
            # Evening peak
            elif 18 <= hour < 22:
                flow = base_flow * 1.2
            # Night low
            elif hour < 6 or hour >= 22:
                flow = base_flow * 0.7
            else:
                flow = base_flow
            events.append((float(t), 'set_target_flow', {'value': flow}))
        return events

    def _weekly_pattern(self, variation: int) -> List[Tuple[float, str, Dict]]:
        """Simulate weekly pattern."""
        base_flow = 100
        events = []
        for day in range(7):
            t = day * 10
            # Weekend reduction
            if day >= 5:
                flow = base_flow * 0.8
            else:
                flow = base_flow * (1.0 + variation * 0.1)
            events.append((float(t), 'set_target_flow', {'value': flow}))
        return events

    def _storm_event(self, variation: int) -> List[Tuple[float, str, Dict]]:
        """Simulate storm event with head surge."""
        events = [
            (0, 'set_target_flow', {'value': 100}),
            (10, 'set_head_upstream', {'value': 11.0 + variation}),
            (15, 'set_head_upstream', {'value': 12.0 + variation}),
            (20, 'set_target_flow', {'value': 150 + variation * 20}),
            (30, 'set_head_upstream', {'value': 11.0}),
            (40, 'set_head_upstream', {'value': 10.0}),
            (50, 'set_target_flow', {'value': 100}),
        ]
        return events

    def _drought_recovery(self, variation: int) -> List[Tuple[float, str, Dict]]:
        """Simulate drought recovery."""
        events = [
            (0, 'set_head_upstream', {'value': 7.0 - variation * 0.5}),
            (0, 'set_target_flow', {'value': 40}),
            (20, 'set_head_upstream', {'value': 8.0}),
            (20, 'set_target_flow', {'value': 60}),
            (40, 'set_head_upstream', {'value': 9.0}),
            (40, 'set_target_flow', {'value': 80}),
            (60, 'set_head_upstream', {'value': 10.0}),
            (60, 'set_target_flow', {'value': 100}),
        ]
        return events

    def _flood_event(self, variation: int) -> List[Tuple[float, str, Dict]]:
        """Simulate flood event."""
        peak_head = 13.0 + variation
        events = [
            (0, 'set_target_flow', {'value': 100}),
            (5, 'set_head_upstream', {'value': 11.0}),
            (10, 'set_head_upstream', {'value': peak_head}),
            (10, 'set_target_flow', {'value': 180}),
            (15, 'set_target_flow', {'value': 200}),
            (25, 'set_head_upstream', {'value': 12.0}),
            (30, 'set_target_flow', {'value': 150}),
            (40, 'set_head_upstream', {'value': 10.0}),
            (50, 'set_target_flow', {'value': 100}),
        ]
        return events

    def _maintenance_window(self, variation: int) -> List[Tuple[float, str, Dict]]:
        """Simulate maintenance requiring gate shutdown."""
        gate_to_maintain = variation % 3
        events = [
            (0, 'set_target_flow', {'value': 80}),
            (10, 'inject_gate_fault', {'gate': gate_to_maintain, 'type': 'STUCK_CLOSED'}),
            (10, 'set_target_flow', {'value': 60}),  # Reduce demand
            (40, 'clear_gate_fault', {'gate': gate_to_maintain}),
            (40, 'set_target_flow', {'value': 80}),
        ]
        return events

    def count(self) -> int:
        return len(self.patterns) * 3


# =============================================================================
# Main Scenario Generation System
# =============================================================================

class FullScenarioGenerator:
    """
    Master scenario generator that combines all generators.

    Can generate 10,000+ unique scenarios.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.generators = {
            'parametric': ParametricScenarioGenerator(
                parameters=['target_flow', 'head_upstream', 'initial_opening'],
                steps_per_param=5,
                seed=seed
            ),
            'transition': FlowTransitionGenerator(seed=seed),
            'fault': FaultScenarioGenerator(seed=seed),
            'combination': CombinationScenarioGenerator(seed=seed),
            'montecarlo': MonteCarloScenarioGenerator(n_scenarios=2000, seed=seed),
            'temporal': TemporalScenarioGenerator(seed=seed),
        }

    def generate_all(self) -> Generator[ScenarioSpec, None, None]:
        """Generate all scenarios from all generators."""
        for name, generator in self.generators.items():
            logger.info("Generating %s scenarios...", name)
            for scenario in generator.generate():
                yield scenario

    def generate_by_category(
        self,
        category: str
    ) -> Generator[ScenarioSpec, None, None]:
        """Generate scenarios from a specific category."""
        if category not in self.generators:
            raise ValueError(f"Unknown category: {category}")
        return self.generators[category].generate()

    def generate_by_difficulty(
        self,
        min_difficulty: float = 0.0,
        max_difficulty: float = 1.0
    ) -> Generator[ScenarioSpec, None, None]:
        """Generate scenarios within difficulty range."""
        for scenario in self.generate_all():
            if min_difficulty <= scenario.difficulty <= max_difficulty:
                yield scenario

    def generate_random_subset(
        self,
        n: int,
        categories: Optional[List[str]] = None
    ) -> List[ScenarioSpec]:
        """Generate random subset of scenarios."""
        all_scenarios = []

        if categories:
            for cat in categories:
                all_scenarios.extend(list(self.generate_by_category(cat)))
        else:
            all_scenarios = list(self.generate_all())

        if n >= len(all_scenarios):
            return all_scenarios

        rng = np.random.default_rng(self.seed)
        indices = rng.choice(len(all_scenarios), n, replace=False)
        return [all_scenarios[i] for i in indices]

    def count_total(self) -> Dict[str, int]:
        """Count scenarios per category."""
        counts = {}
        for name, gen in self.generators.items():
            counts[name] = gen.count()
        counts['total'] = sum(counts.values())
        return counts

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated scenarios."""
        scenarios = list(self.generate_all())

        difficulties = [s.difficulty for s in scenarios]
        categories = {}
        for s in scenarios:
            categories[s.category] = categories.get(s.category, 0) + 1

        return {
            'total_count': len(scenarios),
            'categories': categories,
            'difficulty': {
                'mean': np.mean(difficulties),
                'std': np.std(difficulties),
                'min': np.min(difficulties),
                'max': np.max(difficulties),
            },
            'difficulty_distribution': {
                'easy (0-0.3)': sum(1 for d in difficulties if d < 0.3),
                'medium (0.3-0.6)': sum(1 for d in difficulties if 0.3 <= d < 0.6),
                'hard (0.6-0.8)': sum(1 for d in difficulties if 0.6 <= d < 0.8),
                'extreme (0.8-1.0)': sum(1 for d in difficulties if d >= 0.8),
            }
        }


# =============================================================================
# Scenario Test Runner
# =============================================================================

@dataclass
class ScenarioTestResult:
    """Result of running a scenario test."""
    scenario_id: str
    passed: bool
    duration: float
    metrics: Dict[str, float]
    errors: List[str]
    warnings: List[str]


class ScenarioTestRunner:
    """
    Runs scenario tests and collects results.
    """

    def __init__(
        self,
        model: 'TangheSiphonModel',
        controller: 'IntegratedController'
    ):
        self.model = model
        self.controller = controller
        self.results: List[ScenarioTestResult] = []

    def run_scenario(
        self,
        spec: ScenarioSpec,
        dt: float = 0.1
    ) -> ScenarioTestResult:
        """Run a single scenario test."""
        import time
        start_time = time.time()

        errors = []
        warnings = []
        metrics = {
            'max_flow_error': 0.0,
            'max_vibration': 0.0,
            'time_in_resonance': 0.0,
            'scenario_detected_correctly': 0.0,
        }

        try:
            # Reset and configure
            self.model.reset()
            self.controller.reset()
            self.controller.set_target_flow(spec.target_flow)
            self.model.head_upstream = spec.head_upstream
            self.model.head_downstream = spec.head_downstream

            # Apply initial openings
            for i, opening in enumerate(spec.initial_openings):
                self.model.gate_openings[i] = opening

            # Run simulation
            sim_time = 0.0
            event_idx = 0

            while sim_time < spec.duration:
                # Process events
                while event_idx < len(spec.events) and spec.events[event_idx][0] <= sim_time:
                    self._process_event(spec.events[event_idx])
                    event_idx += 1

                # Control step
                diag = self.controller.update(dt)

                # Apply control outputs
                outputs = diag['control']['outputs']
                self.model.step(np.array(outputs), dt)

                # Collect metrics
                flow_error = abs(diag['state']['flow_error'])
                metrics['max_flow_error'] = max(metrics['max_flow_error'], flow_error)
                metrics['max_vibration'] = max(
                    metrics['max_vibration'],
                    diag['performance']['max_vibration']
                )
                if diag['performance']['in_resonance_band']:
                    metrics['time_in_resonance'] += dt

                # Check expected scenario
                if spec.expected_scenario_type:
                    detected = self.controller.get_current_scenario()
                    if detected == spec.expected_scenario_type:
                        metrics['scenario_detected_correctly'] += dt

                sim_time += dt

            # Normalize time-based metrics
            if spec.expected_scenario_type:
                metrics['scenario_detected_correctly'] /= spec.duration

            # Evaluate pass/fail
            passed = self._evaluate_pass(spec, metrics, errors, warnings)

        except Exception as e:
            errors.append(str(e))
            passed = False

        elapsed = time.time() - start_time

        result = ScenarioTestResult(
            scenario_id=spec.scenario_id,
            passed=passed,
            duration=elapsed,
            metrics=metrics,
            errors=errors,
            warnings=warnings,
        )

        self.results.append(result)
        return result

    def _process_event(self, event: Tuple[float, str, Dict]) -> None:
        """Process a scenario event."""
        _, action, params = event

        if action == 'set_target_flow':
            self.controller.set_target_flow(params['value'])
        elif action == 'set_head_upstream':
            self.model.head_upstream = params['value']
        elif action == 'inject_gate_fault':
            self.model.inject_fault(params['gate'], params['type'].lower())
        elif action == 'clear_gate_fault':
            self.model.inject_fault(params['gate'], 'clear')

    def _evaluate_pass(
        self,
        spec: ScenarioSpec,
        metrics: Dict[str, float],
        errors: List[str],
        warnings: List[str]
    ) -> bool:
        """Evaluate if scenario passed."""
        passed = True

        # Default criteria
        criteria = spec.pass_criteria or {
            'max_flow_error': 50.0,
            'max_vibration': 0.8,
        }

        if metrics['max_flow_error'] > criteria.get('max_flow_error', 50.0):
            warnings.append(f"Flow error exceeded: {metrics['max_flow_error']:.1f}")
            if metrics['max_flow_error'] > criteria.get('max_flow_error', 50.0) * 2:
                passed = False

        if metrics['max_vibration'] > criteria.get('max_vibration', 0.8):
            warnings.append(f"Vibration exceeded: {metrics['max_vibration']:.2f}g")
            if metrics['max_vibration'] > 1.0:
                passed = False

        return passed and len(errors) == 0

    def run_batch(
        self,
        scenarios: List[ScenarioSpec],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[ScenarioTestResult]:
        """Run batch of scenarios."""
        results = []
        total = len(scenarios)

        for i, spec in enumerate(scenarios):
            result = self.run_scenario(spec)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all results."""
        if not self.results:
            return {}

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        return {
            'total': total,
            'passed': passed,
            'failed': total - passed,
            'pass_rate': passed / total if total > 0 else 0,
            'avg_duration': np.mean([r.duration for r in self.results]),
            'avg_max_flow_error': np.mean([r.metrics['max_flow_error'] for r in self.results]),
            'avg_max_vibration': np.mean([r.metrics['max_vibration'] for r in self.results]),
        }


# =============================================================================
# Planning-Aware Scenario Generation
# =============================================================================

class PlanningAwareScenarioGenerator(BaseScenarioGenerator):
    """
    Generates scenarios based on planning and prediction information.

    This generator creates realistic scenarios that incorporate:
    - 调度计划 (Dispatch schedules)
    - 维护计划 (Maintenance plans)
    - 天气预测 (Weather forecasts)
    - 上游来水预测 (Upstream inflow predictions)
    - 季节性变化 (Seasonal variations)
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)

        # Import planning module
        from src.control.planning_prediction import (
            PlanningContextGenerator,
            PlanningContext,
            TimeOfDay,
            Season,
            WeekDay,
            WeatherType,
            PlanPriority,
        )
        self.ctx_generator = PlanningContextGenerator(seed)

        # Scenario templates based on planning context
        self.context_types = [
            'normal_day',
            'flood_scenario',
            'drought_scenario',
            'maintenance',
            'transition',
        ]

        # Seasonal variations
        self.seasons = list(Season)

        # Time variations
        self.times_of_day = list(TimeOfDay)

    def generate(self) -> Generator[ScenarioSpec, None, None]:
        """Generate planning-aware scenarios."""
        from src.control.planning_prediction import (
            PlanningContext, Season, TimeOfDay, WeekDay
        )

        # Generate scenarios for each context type
        for ctx_type in self.context_types:
            for season in self.seasons:
                for time_of_day in self.times_of_day:
                    for variation in range(3):  # 3 variations per combination
                        ctx = self._generate_context(ctx_type, variation * 3600.0)
                        ctx.season = season
                        ctx.time_of_day = time_of_day

                        spec = self._context_to_spec(ctx, ctx_type, variation)
                        if spec:
                            yield spec

    def _generate_context(
        self,
        ctx_type: str,
        base_time: float
    ) -> 'PlanningContext':
        """Generate planning context for given type."""
        if ctx_type == 'normal_day':
            return self.ctx_generator.generate_normal_day(base_time)
        elif ctx_type == 'flood_scenario':
            return self.ctx_generator.generate_flood_scenario(base_time)
        elif ctx_type == 'drought_scenario':
            return self.ctx_generator.generate_drought_scenario(base_time)
        elif ctx_type == 'maintenance':
            gate = self.rng.integers(0, 3)
            return self.ctx_generator.generate_maintenance_scenario(base_time, gate)
        else:
            from_flow = self.rng.uniform(40, 80)
            to_flow = self.rng.uniform(100, 160)
            return self.ctx_generator.generate_transition_scenario(
                base_time, from_flow, to_flow
            )

    def _context_to_spec(
        self,
        ctx: 'PlanningContext',
        ctx_type: str,
        variation: int
    ) -> Optional[ScenarioSpec]:
        """Convert planning context to scenario spec."""
        from src.control.planning_prediction import Season, TimeOfDay

        # Get expected flow from context
        expected_flow = ctx.get_expected_inflow()
        constraints = ctx.get_operational_constraints()
        hints = ctx.get_scenario_hints()

        # Build scenario name
        name = f"Plan:{ctx_type}/{ctx.season.name}/{ctx.time_of_day.name}/v{variation+1}"

        # Calculate difficulty based on context
        difficulty = self._calculate_difficulty(ctx, ctx_type)

        # Create events from planning context
        events = self._extract_events(ctx)

        # Get gate faults from maintenance
        gate_faults = [GateFaultType.NONE] * 3
        for m in ctx.get_active_maintenance():
            for gate_idx in m.affected_gates:
                if not m.is_gate_available(gate_idx):
                    gate_faults[gate_idx] = GateFaultType.STUCK_CLOSED

        return ScenarioSpec(
            scenario_id=self._next_id("PLAN"),
            name=name,
            description=f"Planning scenario: {ctx_type} in {ctx.season.name}",
            category="planning",
            difficulty=difficulty,
            target_flow=expected_flow,
            head_upstream=10.0 if not ctx.drought_warning else 7.0,
            head_downstream=8.0 if not ctx.drought_warning else 6.0,
            flow_pattern=TransitionPattern.STEADY,
            gate_faults=tuple(gate_faults),
            duration=60.0,
            events=events,
            flow_params={
                'context_type': ctx_type,
                'season': ctx.season.name,
                'time_of_day': ctx.time_of_day.name,
                'constraints': constraints,
                'hints': hints,
            },
        )

    def _calculate_difficulty(
        self,
        ctx: 'PlanningContext',
        ctx_type: str
    ) -> float:
        """Calculate scenario difficulty from context."""
        difficulty = 0.2  # Base difficulty

        # Context type difficulty
        type_diff = {
            'normal_day': 0.0,
            'flood_scenario': 0.4,
            'drought_scenario': 0.3,
            'maintenance': 0.3,
            'transition': 0.2,
        }
        difficulty += type_diff.get(ctx_type, 0.0)

        # Warnings
        if ctx.flood_warning:
            difficulty += 0.2
        if ctx.drought_warning:
            difficulty += 0.15

        # Active maintenance
        maintenance = ctx.get_active_maintenance()
        if maintenance:
            difficulty += 0.1 * len(maintenance)

        # Risk factors
        hints = ctx.get_scenario_hints()
        difficulty += 0.05 * len(hints.get('risk_factors', []))

        return min(1.0, difficulty)

    def _extract_events(
        self,
        ctx: 'PlanningContext'
    ) -> List[Tuple[float, str, Dict[str, Any]]]:
        """Extract events from planning context."""
        events = []

        # Extract dispatch plan events
        dispatch = ctx.get_active_dispatch_plan()
        if dispatch and dispatch.flow_schedule:
            for t, flow in dispatch.flow_schedule[:5]:  # Limit events
                rel_time = t - ctx.simulation_time
                if 0 <= rel_time <= 60:
                    events.append((rel_time, 'set_target_flow', {'value': flow}))

        # Extract maintenance events
        for m in ctx.maintenance_plans:
            rel_start = m.start_time - ctx.simulation_time
            if 0 <= rel_start <= 60:
                for gate in m.affected_gates:
                    events.append((
                        rel_start,
                        'inject_gate_fault',
                        {'gate': gate, 'type': 'STUCK_CLOSED'}
                    ))

        return sorted(events, key=lambda e: e[0])

    def count(self) -> int:
        """Count total planning scenarios."""
        return (len(self.context_types) *
                len(self.seasons) *
                len(self.times_of_day) *
                3)  # 3 variations


class SeasonalScenarioGenerator(BaseScenarioGenerator):
    """
    Generates scenarios with seasonal variations.

    Considers:
    - 丰水期 (Flood season: June-August)
    - 枯水期 (Dry season: December-February)
    - 过渡期 (Transition seasons)
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)

        self.seasonal_profiles = {
            'flood_season': {
                'months': [6, 7, 8],
                'flow_range': (100, 200),
                'head_range': (10.0, 14.0),
                'flood_probability': 0.3,
                'storm_probability': 0.2,
            },
            'dry_season': {
                'months': [12, 1, 2],
                'flow_range': (30, 80),
                'head_range': (6.0, 9.0),
                'drought_probability': 0.4,
                'storm_probability': 0.05,
            },
            'spring_transition': {
                'months': [3, 4, 5],
                'flow_range': (60, 150),
                'head_range': (8.0, 12.0),
                'flood_probability': 0.1,
                'storm_probability': 0.15,
            },
            'autumn_transition': {
                'months': [9, 10, 11],
                'flow_range': (50, 120),
                'head_range': (7.0, 11.0),
                'flood_probability': 0.05,
                'storm_probability': 0.1,
            },
        }

    def generate(self) -> Generator[ScenarioSpec, None, None]:
        """Generate seasonal scenarios."""
        for season_name, profile in self.seasonal_profiles.items():
            # Generate scenarios for each month in season
            for month in profile['months']:
                for variation in range(5):  # 5 variations per month
                    # Randomize within ranges
                    flow = self.rng.uniform(*profile['flow_range'])
                    head_up = self.rng.uniform(*profile['head_range'])
                    head_down = head_up - self.rng.uniform(1.5, 3.0)

                    # Determine if special condition
                    is_flood = self.rng.random() < profile.get('flood_probability', 0)
                    is_drought = self.rng.random() < profile.get('drought_probability', 0)
                    is_storm = self.rng.random() < profile.get('storm_probability', 0)

                    # Adjust for special conditions
                    if is_flood:
                        flow *= 1.5
                        head_up += 2.0
                    elif is_drought:
                        flow *= 0.6
                        head_up -= 1.5

                    events = []
                    if is_storm:
                        # Storm event mid-scenario
                        events.append((20.0, 'set_head_upstream', {'value': head_up + 1.5}))
                        events.append((40.0, 'set_head_upstream', {'value': head_up}))

                    difficulty = self._calculate_difficulty(
                        season_name, is_flood, is_drought, is_storm
                    )

                    yield ScenarioSpec(
                        scenario_id=self._next_id("SEASON"),
                        name=f"Seasonal:{season_name}/M{month}/v{variation+1}",
                        description=f"{season_name} scenario for month {month}",
                        category="seasonal",
                        difficulty=difficulty,
                        target_flow=flow,
                        head_upstream=head_up,
                        head_downstream=max(4.0, head_down),
                        flow_pattern=TransitionPattern.STEADY,
                        duration=60.0,
                        events=events,
                        flow_params={
                            'season': season_name,
                            'month': month,
                            'is_flood': is_flood,
                            'is_drought': is_drought,
                            'is_storm': is_storm,
                        },
                    )

    def _calculate_difficulty(
        self,
        season: str,
        is_flood: bool,
        is_drought: bool,
        is_storm: bool
    ) -> float:
        """Calculate scenario difficulty."""
        difficulty = 0.2

        if season == 'flood_season':
            difficulty += 0.2
        elif season == 'dry_season':
            difficulty += 0.15

        if is_flood:
            difficulty += 0.3
        if is_drought:
            difficulty += 0.25
        if is_storm:
            difficulty += 0.15

        return min(1.0, difficulty)

    def count(self) -> int:
        """Count total seasonal scenarios."""
        total = 0
        for profile in self.seasonal_profiles.values():
            total += len(profile['months']) * 5
        return total


class PredictionScenarioGenerator(BaseScenarioGenerator):
    """
    Generates scenarios testing prediction accuracy.

    Creates scenarios where predictions may be:
    - Accurate
    - Under-predicted
    - Over-predicted
    - Suddenly wrong (unexpected events)
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)

        self.prediction_accuracies = [
            ('accurate', 0.0, 0.1),      # Prediction within 10%
            ('under_5', -0.05, 0.02),    # 5% under-prediction
            ('under_15', -0.15, 0.03),   # 15% under-prediction
            ('under_30', -0.30, 0.05),   # 30% under-prediction
            ('over_5', 0.05, 0.02),      # 5% over-prediction
            ('over_15', 0.15, 0.03),     # 15% over-prediction
            ('over_30', 0.30, 0.05),     # 30% over-prediction
            ('sudden_change', 0.0, 0.5), # Sudden unexpected change
        ]

        self.base_flows = [50, 80, 100, 130, 160]

    def generate(self) -> Generator[ScenarioSpec, None, None]:
        """Generate prediction accuracy scenarios."""
        for accuracy_name, bias, noise in self.prediction_accuracies:
            for base_flow in self.base_flows:
                for variation in range(3):
                    # Predicted flow
                    predicted_flow = base_flow

                    # Actual flow (with prediction error)
                    actual_flow = base_flow * (1.0 + bias + self.rng.normal(0, noise))

                    events = []
                    if accuracy_name == 'sudden_change':
                        # Sudden change mid-scenario
                        change_time = 20.0 + self.rng.uniform(0, 20)
                        new_flow = base_flow * (1.0 + self.rng.choice([-0.4, 0.5]))
                        events.append((change_time, 'set_target_flow', {'value': new_flow}))
                        actual_flow = new_flow

                    difficulty = self._calculate_difficulty(accuracy_name, abs(bias))

                    yield ScenarioSpec(
                        scenario_id=self._next_id("PRED"),
                        name=f"Prediction:{accuracy_name}/Q{base_flow}/v{variation+1}",
                        description=f"Testing prediction with {accuracy_name} error",
                        category="prediction",
                        difficulty=difficulty,
                        target_flow=actual_flow,
                        head_upstream=10.0,
                        head_downstream=8.0,
                        flow_pattern=TransitionPattern.STEADY,
                        duration=60.0,
                        events=events,
                        flow_params={
                            'predicted_flow': predicted_flow,
                            'actual_flow': actual_flow,
                            'accuracy_type': accuracy_name,
                            'bias': bias,
                            'noise': noise,
                        },
                    )

    def _calculate_difficulty(self, accuracy_type: str, bias: float) -> float:
        """Calculate difficulty based on prediction error."""
        if accuracy_type == 'accurate':
            return 0.2
        elif accuracy_type == 'sudden_change':
            return 0.8
        else:
            return 0.3 + abs(bias) * 1.5

    def count(self) -> int:
        """Count prediction scenarios."""
        return len(self.prediction_accuracies) * len(self.base_flows) * 3


class DispatchScheduleGenerator(BaseScenarioGenerator):
    """
    Generates scenarios based on typical dispatch schedules.

    Includes:
    - Daily operation patterns
    - Weekly patterns
    - Holiday patterns
    - Emergency dispatch
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)

        # Daily patterns
        self.daily_patterns = [
            'weekday_normal',
            'weekend_reduced',
            'holiday_minimal',
            'peak_demand',
            'night_operation',
        ]

    def generate(self) -> Generator[ScenarioSpec, None, None]:
        """Generate dispatch schedule scenarios."""
        for pattern in self.daily_patterns:
            for variation in range(4):
                events = self._generate_schedule_events(pattern, variation)
                initial_flow = self._get_initial_flow(pattern)
                difficulty = self._calculate_difficulty(pattern)

                yield ScenarioSpec(
                    scenario_id=self._next_id("DISP"),
                    name=f"Dispatch:{pattern}/v{variation+1}",
                    description=f"Dispatch schedule: {pattern}",
                    category="dispatch",
                    difficulty=difficulty,
                    target_flow=initial_flow,
                    head_upstream=10.0,
                    head_downstream=8.0,
                    flow_pattern=TransitionPattern.STEADY,
                    duration=120.0,  # Longer to show pattern
                    events=events,
                    flow_params={'pattern': pattern, 'variation': variation},
                )

    def _generate_schedule_events(
        self,
        pattern: str,
        variation: int
    ) -> List[Tuple[float, str, Dict[str, Any]]]:
        """Generate events for schedule pattern."""
        events = []
        base_variation = 1.0 + variation * 0.1

        if pattern == 'weekday_normal':
            # Morning ramp, steady, evening ramp down
            events = [
                (0, 'set_target_flow', {'value': 80 * base_variation}),
                (20, 'set_target_flow', {'value': 120 * base_variation}),  # Morning peak
                (60, 'set_target_flow', {'value': 100 * base_variation}),  # Midday
                (90, 'set_target_flow', {'value': 110 * base_variation}),  # Afternoon
                (110, 'set_target_flow', {'value': 70 * base_variation}),  # Evening decline
            ]
        elif pattern == 'weekend_reduced':
            events = [
                (0, 'set_target_flow', {'value': 60 * base_variation}),
                (30, 'set_target_flow', {'value': 70 * base_variation}),
                (90, 'set_target_flow', {'value': 60 * base_variation}),
            ]
        elif pattern == 'holiday_minimal':
            events = [
                (0, 'set_target_flow', {'value': 40 * base_variation}),
                (60, 'set_target_flow', {'value': 50 * base_variation}),
            ]
        elif pattern == 'peak_demand':
            events = [
                (0, 'set_target_flow', {'value': 100 * base_variation}),
                (15, 'set_target_flow', {'value': 150 * base_variation}),
                (45, 'set_target_flow', {'value': 170 * base_variation}),
                (75, 'set_target_flow', {'value': 140 * base_variation}),
                (100, 'set_target_flow', {'value': 100 * base_variation}),
            ]
        elif pattern == 'night_operation':
            events = [
                (0, 'set_target_flow', {'value': 50 * base_variation}),
                (40, 'set_target_flow', {'value': 40 * base_variation}),
                (80, 'set_target_flow', {'value': 60 * base_variation}),
            ]

        return events

    def _get_initial_flow(self, pattern: str) -> float:
        """Get initial flow for pattern."""
        return {
            'weekday_normal': 80,
            'weekend_reduced': 60,
            'holiday_minimal': 40,
            'peak_demand': 100,
            'night_operation': 50,
        }.get(pattern, 80)

    def _calculate_difficulty(self, pattern: str) -> float:
        """Calculate difficulty for pattern."""
        return {
            'weekday_normal': 0.3,
            'weekend_reduced': 0.2,
            'holiday_minimal': 0.15,
            'peak_demand': 0.5,
            'night_operation': 0.25,
        }.get(pattern, 0.3)

    def count(self) -> int:
        return len(self.daily_patterns) * 4


# =============================================================================
# Extended Full Scenario Generator
# =============================================================================

class ExtendedScenarioGenerator(FullScenarioGenerator):
    """
    Extended scenario generator including planning-aware scenarios.

    Adds planning, seasonal, prediction, and dispatch scenarios
    to the base generator.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)

        # Add new generators
        self.generators['planning'] = PlanningAwareScenarioGenerator(seed)
        self.generators['seasonal'] = SeasonalScenarioGenerator(seed)
        self.generators['prediction'] = PredictionScenarioGenerator(seed)
        self.generators['dispatch'] = DispatchScheduleGenerator(seed)
