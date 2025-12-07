# -*- coding: utf-8 -*-
"""
System State Real-time Evaluation Module.

This module provides real-time evaluation of system state including:
- Control target deviation assessment
- Performance index calculation
- Multi-objective evaluation
- Alarm and warning generation
- Trend analysis
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
from collections import deque
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class EvaluationLevel(Enum):
    """System evaluation levels."""
    EXCELLENT = "excellent"    # All objectives met, optimal performance
    GOOD = "good"             # Objectives met, acceptable performance
    ACCEPTABLE = "acceptable"  # Minor deviations, within tolerance
    WARNING = "warning"        # Significant deviation, attention needed
    CRITICAL = "critical"      # Critical deviation, immediate action required
    EMERGENCY = "emergency"    # Emergency condition


class ObjectiveType(Enum):
    """Control objective types."""
    FLOW_TRACKING = "flow_tracking"
    VIBRATION_SUPPRESSION = "vibration_suppression"
    ENERGY_EFFICIENCY = "energy_efficiency"
    GATE_BALANCE = "gate_balance"
    RESONANCE_AVOIDANCE = "resonance_avoidance"
    WATER_LEVEL = "water_level"
    SAFETY = "safety"


@dataclass
class ControlObjective:
    """Definition of a control objective."""
    name: str
    type: ObjectiveType
    target_value: float
    tolerance: float  # Acceptable deviation
    weight: float = 1.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    unit: str = ""
    description: str = ""


@dataclass
class ObjectiveDeviation:
    """Deviation from control objective."""
    objective: ControlObjective
    current_value: float
    deviation: float
    normalized_deviation: float  # 0-1 scale
    within_tolerance: bool
    timestamp: float = 0.0


@dataclass
class PerformanceIndex:
    """System performance index."""
    overall: float  # 0-100 scale
    by_objective: Dict[str, float] = field(default_factory=dict)
    weighted_deviation: float = 0.0
    level: EvaluationLevel = EvaluationLevel.GOOD
    timestamp: float = 0.0


@dataclass
class StateEvaluation:
    """Complete state evaluation result."""
    performance: PerformanceIndex
    deviations: List[ObjectiveDeviation]
    alarms: List[Dict[str, Any]]
    recommendations: List[str]
    timestamp: float = 0.0


class ControlObjectiveManager:
    """
    Manages control objectives and their evaluation.
    """

    def __init__(self):
        self._objectives: Dict[str, ControlObjective] = {}
        self._objective_history: Dict[str, deque] = {}

        # Initialize default objectives
        self._init_default_objectives()

        logger.debug("ControlObjectiveManager initialized")

    def _init_default_objectives(self) -> None:
        """Initialize default control objectives."""
        # Flow tracking objective
        self.add_objective(ControlObjective(
            name="target_flow",
            type=ObjectiveType.FLOW_TRACKING,
            target_value=50.0,
            tolerance=5.0,  # ±5 m³/s
            weight=1.0,
            min_value=0.0,
            max_value=200.0,
            unit="m³/s",
            description="Total flow rate tracking"
        ))

        # Vibration suppression objective
        self.add_objective(ControlObjective(
            name="max_vibration",
            type=ObjectiveType.VIBRATION_SUPPRESSION,
            target_value=0.0,
            tolerance=0.1,  # Max 0.1g
            weight=0.8,
            min_value=0.0,
            max_value=5.0,
            unit="g",
            description="Maximum vibration level"
        ))

        # Resonance avoidance objective
        self.add_objective(ControlObjective(
            name="resonance_margin",
            type=ObjectiveType.RESONANCE_AVOIDANCE,
            target_value=0.5,  # 50% margin from resonance
            tolerance=0.2,
            weight=0.9,
            min_value=0.0,
            max_value=1.0,
            unit="-",
            description="Distance from resonance frequency"
        ))

        # Gate balance objective
        self.add_objective(ControlObjective(
            name="gate_balance",
            type=ObjectiveType.GATE_BALANCE,
            target_value=0.0,  # Zero imbalance
            tolerance=0.2,  # Max 20% imbalance
            weight=0.5,
            min_value=0.0,
            max_value=1.0,
            unit="-",
            description="Gate opening balance"
        ))

        # Upstream water level
        self.add_objective(ControlObjective(
            name="upstream_level",
            type=ObjectiveType.WATER_LEVEL,
            target_value=10.0,
            tolerance=0.5,
            weight=0.7,
            min_value=5.0,
            max_value=15.0,
            unit="m",
            description="Upstream water level"
        ))

    def add_objective(self, objective: ControlObjective) -> None:
        """Add a control objective."""
        self._objectives[objective.name] = objective
        self._objective_history[objective.name] = deque(maxlen=1000)
        logger.debug("Added objective: %s", objective.name)

    def remove_objective(self, name: str) -> None:
        """Remove a control objective."""
        if name in self._objectives:
            del self._objectives[name]
            del self._objective_history[name]

    def update_target(self, name: str, target_value: float) -> None:
        """Update objective target value."""
        if name in self._objectives:
            self._objectives[name].target_value = target_value
            logger.info("Updated %s target to %.2f", name, target_value)

    def get_objective(self, name: str) -> Optional[ControlObjective]:
        """Get objective by name."""
        return self._objectives.get(name)

    def get_all_objectives(self) -> Dict[str, ControlObjective]:
        """Get all objectives."""
        return self._objectives.copy()


class StateEvaluator:
    """
    Real-time system state evaluator.

    Evaluates current system state against control objectives
    and generates performance indices and alarms.
    """

    def __init__(
        self,
        objective_manager: Optional[ControlObjectiveManager] = None
    ):
        self._obj_manager = objective_manager or ControlObjectiveManager()

        # Evaluation history
        self._evaluation_history: deque = deque(maxlen=1000)
        self._alarm_history: deque = deque(maxlen=100)

        # Thresholds for evaluation levels
        self._level_thresholds = {
            EvaluationLevel.EXCELLENT: 95.0,
            EvaluationLevel.GOOD: 85.0,
            EvaluationLevel.ACCEPTABLE: 70.0,
            EvaluationLevel.WARNING: 50.0,
            EvaluationLevel.CRITICAL: 30.0,
            EvaluationLevel.EMERGENCY: 0.0
        }

        # Alarm configuration
        self._alarm_thresholds = {
            'vibration': 0.3,    # g
            'flow_error': 0.2,   # 20%
            'resonance': 0.9     # 90% of f_struct
        }

        logger.info("StateEvaluator initialized")

    def evaluate(
        self,
        state: Dict[str, Any],
        targets: Optional[Dict[str, float]] = None,
        timestamp: float = 0.0
    ) -> StateEvaluation:
        """
        Evaluate current system state.

        Args:
            state: Current system state dictionary
            targets: Optional target overrides
            timestamp: Current timestamp

        Returns:
            Complete state evaluation
        """
        # Update targets if provided
        if targets:
            for name, value in targets.items():
                self._obj_manager.update_target(name, value)

        # Calculate deviations for each objective
        deviations = self._calculate_deviations(state, timestamp)

        # Calculate performance index
        performance = self._calculate_performance(deviations, timestamp)

        # Generate alarms
        alarms = self._check_alarms(state, deviations, timestamp)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            performance, deviations, state
        )

        evaluation = StateEvaluation(
            performance=performance,
            deviations=deviations,
            alarms=alarms,
            recommendations=recommendations,
            timestamp=timestamp
        )

        # Record history
        self._evaluation_history.append({
            'timestamp': timestamp,
            'performance': performance.overall,
            'level': performance.level.value,
            'num_alarms': len(alarms)
        })

        return evaluation

    def _calculate_deviations(
        self,
        state: Dict[str, Any],
        timestamp: float
    ) -> List[ObjectiveDeviation]:
        """Calculate deviations from all objectives."""
        deviations = []

        for name, objective in self._obj_manager.get_all_objectives().items():
            current_value = self._extract_value(state, objective)

            if current_value is None:
                continue

            # Calculate raw deviation
            deviation = abs(current_value - objective.target_value)

            # Normalized deviation (0-1 scale)
            if objective.tolerance > 0:
                normalized = min(1.0, deviation / objective.tolerance)
            else:
                normalized = 0.0 if deviation == 0 else 1.0

            # Check if within tolerance
            within_tolerance = deviation <= objective.tolerance

            dev = ObjectiveDeviation(
                objective=objective,
                current_value=current_value,
                deviation=deviation,
                normalized_deviation=normalized,
                within_tolerance=within_tolerance,
                timestamp=timestamp
            )
            deviations.append(dev)

        return deviations

    def _extract_value(
        self,
        state: Dict[str, Any],
        objective: ControlObjective
    ) -> Optional[float]:
        """Extract relevant value from state for objective."""
        if objective.type == ObjectiveType.FLOW_TRACKING:
            return state.get('total_flow')

        elif objective.type == ObjectiveType.VIBRATION_SUPPRESSION:
            if 'vibrations' in state:
                return max(state['vibrations'])
            return None

        elif objective.type == ObjectiveType.RESONANCE_AVOIDANCE:
            if 'frequencies' in state:
                freqs = np.array(state['frequencies'])
                f_struct = 2.8  # Reference structural frequency
                if np.max(freqs) > 0:
                    min_margin = min(abs(f - f_struct) / f_struct for f in freqs if f > 0)
                    return min_margin
            return 1.0  # Safe if no frequencies

        elif objective.type == ObjectiveType.GATE_BALANCE:
            if 'openings' in state:
                openings = np.array(state['openings'])
                if np.mean(openings) > 0:
                    return np.std(openings) / np.mean(openings)
                return 0.0
            return None

        elif objective.type == ObjectiveType.WATER_LEVEL:
            if objective.name == 'upstream_level':
                return state.get('head_upstream')
            elif objective.name == 'downstream_level':
                return state.get('head_downstream')
            return None

        return None

    def _calculate_performance(
        self,
        deviations: List[ObjectiveDeviation],
        timestamp: float
    ) -> PerformanceIndex:
        """Calculate overall performance index."""
        if not deviations:
            return PerformanceIndex(
                overall=100.0,
                level=EvaluationLevel.EXCELLENT,
                timestamp=timestamp
            )

        # Weighted sum of normalized deviations
        total_weight = sum(d.objective.weight for d in deviations)
        if total_weight == 0:
            total_weight = 1.0

        weighted_deviation = sum(
            d.normalized_deviation * d.objective.weight
            for d in deviations
        ) / total_weight

        # Convert to 0-100 performance score
        overall = 100.0 * (1.0 - weighted_deviation)
        overall = max(0.0, min(100.0, overall))

        # Individual objective scores
        by_objective = {
            d.objective.name: 100.0 * (1.0 - d.normalized_deviation)
            for d in deviations
        }

        # Determine evaluation level
        level = EvaluationLevel.EMERGENCY
        for eval_level, threshold in sorted(
            self._level_thresholds.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if overall >= threshold:
                level = eval_level
                break

        return PerformanceIndex(
            overall=overall,
            by_objective=by_objective,
            weighted_deviation=weighted_deviation,
            level=level,
            timestamp=timestamp
        )

    def _check_alarms(
        self,
        state: Dict[str, Any],
        deviations: List[ObjectiveDeviation],
        timestamp: float
    ) -> List[Dict[str, Any]]:
        """Check for alarm conditions."""
        alarms = []

        # Vibration alarm
        if 'vibrations' in state:
            max_vib = max(state['vibrations'])
            if max_vib > self._alarm_thresholds['vibration']:
                alarms.append({
                    'type': 'vibration',
                    'severity': 'warning' if max_vib < 0.5 else 'critical',
                    'message': f"High vibration detected: {max_vib:.3f} g",
                    'value': max_vib,
                    'threshold': self._alarm_thresholds['vibration'],
                    'timestamp': timestamp
                })

        # Flow error alarm
        target_flow_obj = self._obj_manager.get_objective('target_flow')
        if target_flow_obj and 'total_flow' in state:
            flow_error = abs(state['total_flow'] - target_flow_obj.target_value)
            rel_error = flow_error / max(target_flow_obj.target_value, 1.0)
            if rel_error > self._alarm_thresholds['flow_error']:
                alarms.append({
                    'type': 'flow_tracking',
                    'severity': 'warning',
                    'message': f"Flow tracking error: {rel_error*100:.1f}%",
                    'value': flow_error,
                    'threshold': self._alarm_thresholds['flow_error'],
                    'timestamp': timestamp
                })

        # Resonance alarm
        if 'frequencies' in state:
            freqs = state['frequencies']
            f_struct = 2.8
            for i, f in enumerate(freqs):
                if f > 0:
                    ratio = f / f_struct
                    if 0.9 < ratio < 1.1:  # Near resonance
                        alarms.append({
                            'type': 'resonance',
                            'severity': 'critical',
                            'message': f"Gate {i} near resonance: f={f:.2f} Hz",
                            'value': ratio,
                            'threshold': self._alarm_thresholds['resonance'],
                            'timestamp': timestamp
                        })

        # Critical deviation alarms
        for dev in deviations:
            if not dev.within_tolerance and dev.normalized_deviation > 0.5:
                alarms.append({
                    'type': 'objective_violation',
                    'severity': 'warning',
                    'message': f"{dev.objective.name} deviation: {dev.deviation:.2f} {dev.objective.unit}",
                    'value': dev.deviation,
                    'threshold': dev.objective.tolerance,
                    'timestamp': timestamp
                })

        # Record alarms
        for alarm in alarms:
            self._alarm_history.append(alarm)

        return alarms

    def _generate_recommendations(
        self,
        performance: PerformanceIndex,
        deviations: List[ObjectiveDeviation],
        state: Dict[str, Any]
    ) -> List[str]:
        """Generate control recommendations based on evaluation."""
        recommendations = []

        # Performance-based recommendations
        if performance.level == EvaluationLevel.CRITICAL:
            recommendations.append("Consider emergency control action")

        if performance.level == EvaluationLevel.WARNING:
            recommendations.append("Adjust control parameters to improve performance")

        # Specific recommendations based on deviations
        for dev in deviations:
            if not dev.within_tolerance:
                if dev.objective.type == ObjectiveType.VIBRATION_SUPPRESSION:
                    recommendations.append(
                        "Reduce gate opening velocities to suppress vibration"
                    )
                elif dev.objective.type == ObjectiveType.FLOW_TRACKING:
                    if dev.current_value < dev.objective.target_value:
                        recommendations.append("Increase gate openings to meet flow target")
                    else:
                        recommendations.append("Decrease gate openings to meet flow target")
                elif dev.objective.type == ObjectiveType.RESONANCE_AVOIDANCE:
                    recommendations.append(
                        "Adjust velocities to move away from structural frequency"
                    )
                elif dev.objective.type == ObjectiveType.GATE_BALANCE:
                    recommendations.append("Rebalance gate openings for uniform operation")

        # State-specific recommendations
        if 'vibrations' in state and max(state['vibrations']) > 0.2:
            recommendations.append("Consider enabling dithering control for vibration mitigation")

        return recommendations

    def get_performance_trend(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent performance trend."""
        return list(self._evaluation_history)[-count:]

    def get_alarm_history(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get recent alarm history."""
        return list(self._alarm_history)[-count:]

    def get_objective_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all objectives."""
        return {
            name: {
                'target': obj.target_value,
                'tolerance': obj.tolerance,
                'weight': obj.weight,
                'unit': obj.unit,
                'type': obj.type.value
            }
            for name, obj in self._obj_manager.get_all_objectives().items()
        }


class DeviationAnalyzer:
    """
    Analyzes control deviation patterns and trends.
    """

    def __init__(self, window_size: int = 100):
        self._window_size = window_size
        self._deviation_history: Dict[str, deque] = {}

    def add_deviation(
        self,
        objective_name: str,
        deviation: float,
        timestamp: float
    ) -> None:
        """Add deviation measurement."""
        if objective_name not in self._deviation_history:
            self._deviation_history[objective_name] = deque(maxlen=self._window_size)

        self._deviation_history[objective_name].append({
            'value': deviation,
            'timestamp': timestamp
        })

    def analyze_trend(self, objective_name: str) -> Dict[str, Any]:
        """Analyze deviation trend for an objective."""
        if objective_name not in self._deviation_history:
            return {'sufficient_data': False}

        history = list(self._deviation_history[objective_name])
        if len(history) < 10:
            return {'sufficient_data': False}

        values = np.array([h['value'] for h in history])
        timestamps = np.array([h['timestamp'] for h in history])

        # Statistical analysis
        mean = np.mean(values)
        std = np.std(values)
        current = values[-1]

        # Trend analysis (linear regression)
        if len(values) > 20:
            t_norm = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0] + 1e-10)
            slope = np.polyfit(t_norm, values, 1)[0]
        else:
            slope = 0.0

        # Determine trend direction
        if slope > 0.1 * std:
            trend = 'increasing'
        elif slope < -0.1 * std:
            trend = 'decreasing'
        else:
            trend = 'stable'

        return {
            'sufficient_data': True,
            'mean': float(mean),
            'std': float(std),
            'current': float(current),
            'slope': float(slope),
            'trend': trend,
            'samples': len(values)
        }

    def detect_oscillation(self, objective_name: str) -> Dict[str, Any]:
        """Detect oscillatory behavior in deviation."""
        if objective_name not in self._deviation_history:
            return {'oscillating': False}

        history = list(self._deviation_history[objective_name])
        if len(history) < 20:
            return {'oscillating': False, 'reason': 'insufficient_data'}

        values = np.array([h['value'] for h in history])

        # Check for sign changes (zero crossings)
        centered = values - np.mean(values)
        sign_changes = np.sum(np.abs(np.diff(np.sign(centered))) > 0)

        # High sign change rate indicates oscillation
        oscillation_rate = sign_changes / len(values)

        oscillating = oscillation_rate > 0.3

        return {
            'oscillating': oscillating,
            'oscillation_rate': float(oscillation_rate),
            'frequency_estimate': oscillation_rate * 2,  # Approximate
            'amplitude': float(np.std(values) * 2)
        }


class RealTimeStateEvaluator:
    """
    High-level real-time state evaluator combining all evaluation components.
    """

    def __init__(self):
        self.objective_manager = ControlObjectiveManager()
        self.state_evaluator = StateEvaluator(self.objective_manager)
        self.deviation_analyzer = DeviationAnalyzer()

        self._last_evaluation: Optional[StateEvaluation] = None

        logger.info("RealTimeStateEvaluator initialized")

    def evaluate(
        self,
        model: 'TangheSiphonModel',
        target_flow: Optional[float] = None
    ) -> StateEvaluation:
        """
        Perform real-time evaluation of model state.

        Args:
            model: Physics model
            target_flow: Optional flow target override

        Returns:
            Complete state evaluation
        """
        # Extract state from model
        state = model.get_state()
        state['head_upstream'] = model.head_upstream
        state['head_downstream'] = model.head_downstream

        # Update target if provided
        targets = {}
        if target_flow is not None:
            targets['target_flow'] = target_flow

        # Perform evaluation
        evaluation = self.state_evaluator.evaluate(
            state, targets, model.time
        )

        # Update deviation analyzer
        for dev in evaluation.deviations:
            self.deviation_analyzer.add_deviation(
                dev.objective.name,
                dev.deviation,
                model.time
            )

        self._last_evaluation = evaluation

        return evaluation

    def get_deviation_analysis(self) -> Dict[str, Dict[str, Any]]:
        """Get deviation analysis for all objectives."""
        analysis = {}
        for name in self.objective_manager.get_all_objectives():
            analysis[name] = {
                'trend': self.deviation_analyzer.analyze_trend(name),
                'oscillation': self.deviation_analyzer.detect_oscillation(name)
            }
        return analysis

    def get_summary(self) -> Dict[str, Any]:
        """Get evaluation summary."""
        if self._last_evaluation is None:
            return {'status': 'no_evaluation'}

        return {
            'performance': self._last_evaluation.performance.overall,
            'level': self._last_evaluation.performance.level.value,
            'active_alarms': len(self._last_evaluation.alarms),
            'recommendations': len(self._last_evaluation.recommendations),
            'by_objective': self._last_evaluation.performance.by_objective,
            'timestamp': self._last_evaluation.timestamp
        }

    def set_flow_target(self, target: float) -> None:
        """Set flow tracking target."""
        self.objective_manager.update_target('target_flow', target)

    def set_vibration_limit(self, limit: float) -> None:
        """Set vibration limit."""
        self.objective_manager.update_target('max_vibration', limit)
