# -*- coding: utf-8 -*-
"""
System State Real-time Prediction Module.

This module provides real-time state prediction including:
- Short-term state prediction
- Trend analysis and extrapolation
- Predictive alerts
- Scenario-based prediction
- Uncertainty quantification
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
from collections import deque
from enum import Enum
import numpy as np
from scipy import interpolate, signal

logger = logging.getLogger(__name__)


class PredictionMethod(Enum):
    """State prediction methods."""
    PHYSICS_BASED = "physics_based"      # Use physics model
    DATA_DRIVEN = "data_driven"          # Statistical/ML methods
    HYBRID = "hybrid"                    # Combined approach
    ENSEMBLE = "ensemble"                # Multiple model ensemble


class PredictionHorizon(Enum):
    """Prediction time horizons."""
    IMMEDIATE = "immediate"    # 0-1 second
    SHORT = "short"           # 1-60 seconds
    MEDIUM = "medium"         # 1-30 minutes
    LONG = "long"            # 30+ minutes


@dataclass
class PredictionResult:
    """Result of state prediction."""
    predicted_state: Dict[str, np.ndarray]
    prediction_times: np.ndarray
    confidence_intervals: Dict[str, Tuple[np.ndarray, np.ndarray]]
    method: PredictionMethod
    horizon: PredictionHorizon
    uncertainty: Dict[str, float]
    timestamp: float = 0.0


@dataclass
class TrendAnalysis:
    """Trend analysis result."""
    variable: str
    current_value: float
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_rate: float     # Rate of change per second
    predicted_value: float
    time_to_threshold: Optional[float] = None  # Time until threshold breach
    confidence: float = 0.0


@dataclass
class PredictiveAlert:
    """Predictive alert for anticipated issues."""
    alert_type: str
    severity: str  # 'info', 'warning', 'critical'
    variable: str
    current_value: float
    predicted_value: float
    threshold: float
    time_to_event: float
    message: str
    timestamp: float = 0.0


class StatePredictor:
    """
    Physics-based state predictor.

    Uses the physics model to predict future states based on
    current conditions and planned control actions.
    """

    def __init__(
        self,
        model: 'TangheSiphonModel',
        dt: float = 0.1  # Prediction time step
    ):
        self.model = model
        self.dt = dt

        # State history for data-driven methods
        self._state_history: deque = deque(maxlen=1000)
        self._prediction_history: deque = deque(maxlen=100)

        logger.debug("StatePredictor initialized with dt=%.2f s", dt)

    def predict_physics(
        self,
        horizon_seconds: float,
        target_openings: Optional[np.ndarray] = None,
        num_steps: Optional[int] = None
    ) -> PredictionResult:
        """
        Predict future states using physics model.

        Args:
            horizon_seconds: Prediction horizon in seconds
            target_openings: Target gate openings (if None, use current)
            num_steps: Number of prediction steps (if None, computed from horizon)

        Returns:
            PredictionResult with predicted states
        """
        if num_steps is None:
            num_steps = int(horizon_seconds / self.dt)

        if target_openings is None:
            target_openings = self.model.gate_openings.copy()

        # Save current model state
        saved_state = self._save_model_state()

        # Prediction arrays
        times = np.zeros(num_steps + 1)
        openings = np.zeros((num_steps + 1, self.model.num_gates))
        flows = np.zeros((num_steps + 1, self.model.num_gates))
        velocities = np.zeros((num_steps + 1, self.model.num_gates))
        vibrations = np.zeros((num_steps + 1, self.model.num_gates))
        frequencies = np.zeros((num_steps + 1, self.model.num_gates))

        # Initial state
        times[0] = self.model.time
        openings[0] = self.model.gate_openings.copy()
        flows[0] = self.model.flow_rates.copy()
        velocities[0] = self.model.velocities.copy()
        vibrations[0] = self.model.vibration_accel.copy()
        frequencies[0] = self.model.vortex_freqs.copy()

        # Propagate model forward
        for i in range(num_steps):
            self.model.step(target_openings, self.dt)

            times[i + 1] = self.model.time
            openings[i + 1] = self.model.gate_openings.copy()
            flows[i + 1] = self.model.flow_rates.copy()
            velocities[i + 1] = self.model.velocities.copy()
            vibrations[i + 1] = self.model.vibration_accel.copy()
            frequencies[i + 1] = self.model.vortex_freqs.copy()

        # Restore model state
        self._restore_model_state(saved_state)

        # Calculate confidence intervals (simplified)
        confidence = self._calculate_confidence_intervals(
            flows, velocities, vibrations, horizon_seconds
        )

        predicted_state = {
            'times': times,
            'openings': openings,
            'flows': flows,
            'velocities': velocities,
            'vibrations': vibrations,
            'frequencies': frequencies,
            'total_flow': np.sum(flows, axis=1)
        }

        # Determine horizon category
        if horizon_seconds <= 1.0:
            horizon = PredictionHorizon.IMMEDIATE
        elif horizon_seconds <= 60.0:
            horizon = PredictionHorizon.SHORT
        elif horizon_seconds <= 1800.0:
            horizon = PredictionHorizon.MEDIUM
        else:
            horizon = PredictionHorizon.LONG

        return PredictionResult(
            predicted_state=predicted_state,
            prediction_times=times,
            confidence_intervals=confidence,
            method=PredictionMethod.PHYSICS_BASED,
            horizon=horizon,
            uncertainty=self._estimate_uncertainty(horizon_seconds),
            timestamp=saved_state['time']
        )

    def _save_model_state(self) -> Dict[str, Any]:
        """Save current model state."""
        return {
            'time': self.model.time,
            'gate_openings': self.model.gate_openings.copy(),
            'flow_rates': self.model.flow_rates.copy(),
            'velocities': self.model.velocities.copy(),
            'vortex_freqs': self.model.vortex_freqs.copy(),
            'vibration_accel': self.model.vibration_accel.copy(),
            'head_upstream': self.model.head_upstream,
            'head_downstream': self.model.head_downstream
        }

    def _restore_model_state(self, state: Dict[str, Any]) -> None:
        """Restore model state."""
        self.model.time = state['time']
        self.model.gate_openings = state['gate_openings']
        self.model.flow_rates = state['flow_rates']
        self.model.velocities = state['velocities']
        self.model.vortex_freqs = state['vortex_freqs']
        self.model.vibration_accel = state['vibration_accel']
        self.model.head_upstream = state['head_upstream']
        self.model.head_downstream = state['head_downstream']

    def _calculate_confidence_intervals(
        self,
        flows: np.ndarray,
        velocities: np.ndarray,
        vibrations: np.ndarray,
        horizon: float
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Calculate confidence intervals for predictions."""
        # Uncertainty grows with prediction horizon
        base_uncertainty = 0.05  # 5% base uncertainty
        growth_rate = 0.01  # 1% per second

        num_steps = len(flows)
        uncertainty = base_uncertainty + growth_rate * horizon

        # For total flow
        total_flow = np.sum(flows, axis=1)
        flow_std = total_flow * uncertainty

        # Calculate intervals
        return {
            'total_flow': (total_flow - 2 * flow_std, total_flow + 2 * flow_std),
            'max_vibration': (
                np.max(vibrations, axis=1) * (1 - uncertainty),
                np.max(vibrations, axis=1) * (1 + uncertainty)
            )
        }

    def _estimate_uncertainty(self, horizon: float) -> Dict[str, float]:
        """Estimate prediction uncertainty."""
        base = 0.05
        growth = 0.01 * horizon

        return {
            'flow': base + growth,
            'velocity': base + growth * 0.8,
            'vibration': base + growth * 1.5,  # Vibration harder to predict
            'frequency': base + growth * 0.5
        }


class TrendPredictor:
    """
    Data-driven trend analysis and extrapolation.
    """

    def __init__(self, window_size: int = 100):
        self._window_size = window_size
        self._history: Dict[str, deque] = {}
        self._thresholds: Dict[str, Tuple[float, float]] = {
            'total_flow': (0.0, 200.0),
            'max_vibration': (0.0, 0.5),
            'velocity': (0.0, 10.0)
        }

    def add_observation(
        self,
        variable: str,
        value: float,
        timestamp: float
    ) -> None:
        """Add observation for trend analysis."""
        if variable not in self._history:
            self._history[variable] = deque(maxlen=self._window_size)

        self._history[variable].append({
            'value': value,
            'timestamp': timestamp
        })

    def analyze_trend(
        self,
        variable: str,
        prediction_horizon: float = 60.0
    ) -> TrendAnalysis:
        """
        Analyze trend and predict future value.

        Args:
            variable: Variable name to analyze
            prediction_horizon: Seconds to predict ahead

        Returns:
            TrendAnalysis result
        """
        if variable not in self._history or len(self._history[variable]) < 5:
            return TrendAnalysis(
                variable=variable,
                current_value=0.0,
                trend_direction='unknown',
                trend_rate=0.0,
                predicted_value=0.0,
                confidence=0.0
            )

        history = list(self._history[variable])
        values = np.array([h['value'] for h in history])
        times = np.array([h['timestamp'] for h in history])

        current_value = values[-1]
        current_time = times[-1]

        # Linear regression for trend
        if len(values) >= 10:
            # Use recent data for trend
            n_trend = min(50, len(values))
            t = times[-n_trend:]
            v = values[-n_trend:]

            # Normalize time
            t_norm = t - t[0]

            # Fit linear model
            coeffs = np.polyfit(t_norm, v, 1)
            trend_rate = coeffs[0]  # Units per second

            # Confidence from R-squared
            predicted = np.polyval(coeffs, t_norm)
            ss_res = np.sum((v - predicted) ** 2)
            ss_tot = np.sum((v - np.mean(v)) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-10))
            confidence = max(0.0, min(1.0, r_squared))
        else:
            # Simple difference for short history
            trend_rate = (values[-1] - values[0]) / (times[-1] - times[0] + 1e-10)
            confidence = 0.3

        # Determine trend direction
        if abs(trend_rate) < 0.01 * np.std(values):
            trend_direction = 'stable'
        elif trend_rate > 0:
            trend_direction = 'increasing'
        else:
            trend_direction = 'decreasing'

        # Predict future value
        predicted_value = current_value + trend_rate * prediction_horizon

        # Calculate time to threshold breach
        time_to_threshold = None
        if variable in self._thresholds:
            low, high = self._thresholds[variable]

            if trend_direction == 'increasing' and predicted_value > high:
                if trend_rate > 0:
                    time_to_threshold = (high - current_value) / trend_rate

            elif trend_direction == 'decreasing' and predicted_value < low:
                if trend_rate < 0:
                    time_to_threshold = (low - current_value) / trend_rate

        return TrendAnalysis(
            variable=variable,
            current_value=current_value,
            trend_direction=trend_direction,
            trend_rate=trend_rate,
            predicted_value=predicted_value,
            time_to_threshold=time_to_threshold,
            confidence=confidence
        )

    def set_threshold(self, variable: str, low: float, high: float) -> None:
        """Set threshold bounds for a variable."""
        self._thresholds[variable] = (low, high)


class PredictiveAlertEngine:
    """
    Generates predictive alerts based on trend analysis.
    """

    def __init__(self):
        self._alert_rules: List[Dict[str, Any]] = []
        self._active_alerts: Dict[str, PredictiveAlert] = {}
        self._alert_history: deque = deque(maxlen=100)

        # Initialize default rules
        self._init_default_rules()

        logger.debug("PredictiveAlertEngine initialized")

    def _init_default_rules(self) -> None:
        """Initialize default alert rules."""
        self._alert_rules = [
            {
                'variable': 'max_vibration',
                'threshold': 0.3,
                'condition': 'above',
                'severity': 'warning',
                'message_template': "Vibration predicted to exceed {threshold}g in {time:.0f}s"
            },
            {
                'variable': 'max_vibration',
                'threshold': 0.5,
                'condition': 'above',
                'severity': 'critical',
                'message_template': "Critical vibration level predicted in {time:.0f}s"
            },
            {
                'variable': 'total_flow',
                'threshold': 180.0,
                'condition': 'above',
                'severity': 'warning',
                'message_template': "Flow rate approaching maximum capacity in {time:.0f}s"
            },
            {
                'variable': 'total_flow',
                'threshold': 5.0,
                'condition': 'below',
                'severity': 'warning',
                'message_template': "Flow rate dropping critically low in {time:.0f}s"
            }
        ]

    def check_alerts(
        self,
        trend_analyses: Dict[str, TrendAnalysis],
        timestamp: float
    ) -> List[PredictiveAlert]:
        """
        Check for predictive alerts based on trends.

        Args:
            trend_analyses: Dict of variable -> TrendAnalysis
            timestamp: Current timestamp

        Returns:
            List of new alerts
        """
        new_alerts = []

        for rule in self._alert_rules:
            variable = rule['variable']

            if variable not in trend_analyses:
                continue

            trend = trend_analyses[variable]

            if trend.time_to_threshold is None:
                continue

            # Check if threshold will be crossed
            threshold = rule['threshold']
            condition = rule['condition']

            trigger = False
            if condition == 'above' and trend.trend_direction == 'increasing':
                if trend.predicted_value > threshold:
                    trigger = True
            elif condition == 'below' and trend.trend_direction == 'decreasing':
                if trend.predicted_value < threshold:
                    trigger = True

            if trigger and trend.time_to_threshold < 300:  # Within 5 minutes
                alert = PredictiveAlert(
                    alert_type='predicted_threshold_breach',
                    severity=rule['severity'],
                    variable=variable,
                    current_value=trend.current_value,
                    predicted_value=trend.predicted_value,
                    threshold=threshold,
                    time_to_event=trend.time_to_threshold,
                    message=rule['message_template'].format(
                        threshold=threshold,
                        time=trend.time_to_threshold
                    ),
                    timestamp=timestamp
                )

                # Check if this is a new or updated alert
                alert_key = f"{variable}_{condition}_{threshold}"
                if alert_key not in self._active_alerts:
                    new_alerts.append(alert)
                    self._active_alerts[alert_key] = alert
                    self._alert_history.append(alert)

        return new_alerts

    def clear_resolved_alerts(
        self,
        trend_analyses: Dict[str, TrendAnalysis]
    ) -> List[str]:
        """Clear alerts that are no longer relevant."""
        resolved = []

        for key, alert in list(self._active_alerts.items()):
            variable = alert.variable

            if variable in trend_analyses:
                trend = trend_analyses[variable]

                # Check if threat has passed
                if alert.threshold > alert.current_value:
                    # Was above threshold
                    if trend.trend_direction != 'increasing':
                        resolved.append(key)
                else:
                    # Was below threshold
                    if trend.trend_direction != 'decreasing':
                        resolved.append(key)

        for key in resolved:
            del self._active_alerts[key]

        return resolved

    def get_active_alerts(self) -> List[PredictiveAlert]:
        """Get all active alerts."""
        return list(self._active_alerts.values())


class ScenarioPredictor:
    """
    Scenario-based prediction for what-if analysis.
    """

    def __init__(self, state_predictor: StatePredictor):
        self._predictor = state_predictor
        self._scenario_results: Dict[str, PredictionResult] = {}

    def predict_scenario(
        self,
        scenario_name: str,
        target_openings: np.ndarray,
        horizon_seconds: float = 60.0,
        head_upstream: Optional[float] = None,
        head_downstream: Optional[float] = None
    ) -> PredictionResult:
        """
        Predict system behavior under a specific scenario.

        Args:
            scenario_name: Name for this scenario
            target_openings: Target gate openings
            horizon_seconds: Prediction horizon
            head_upstream: Optional upstream head override
            head_downstream: Optional downstream head override

        Returns:
            PredictionResult for the scenario
        """
        # Temporarily modify environmental conditions if specified
        saved_upstream = self._predictor.model.head_upstream
        saved_downstream = self._predictor.model.head_downstream

        if head_upstream is not None:
            self._predictor.model.head_upstream = head_upstream
        if head_downstream is not None:
            self._predictor.model.head_downstream = head_downstream

        try:
            result = self._predictor.predict_physics(
                horizon_seconds, target_openings
            )
            self._scenario_results[scenario_name] = result
            return result
        finally:
            # Restore original conditions
            self._predictor.model.head_upstream = saved_upstream
            self._predictor.model.head_downstream = saved_downstream

    def compare_scenarios(
        self,
        scenario_names: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple predicted scenarios.

        Args:
            scenario_names: Names of scenarios to compare

        Returns:
            Comparison metrics for each scenario
        """
        comparison = {}

        for name in scenario_names:
            if name not in self._scenario_results:
                continue

            result = self._scenario_results[name]

            # Calculate key metrics
            total_flow = result.predicted_state['total_flow']
            vibrations = result.predicted_state['vibrations']

            comparison[name] = {
                'final_flow': float(total_flow[-1]),
                'mean_flow': float(np.mean(total_flow)),
                'flow_variability': float(np.std(total_flow)),
                'max_vibration': float(np.max(vibrations)),
                'mean_vibration': float(np.mean(vibrations)),
                'settling_time': self._estimate_settling_time(total_flow),
                'prediction_horizon': result.horizon.value
            }

        return comparison

    def _estimate_settling_time(
        self,
        values: np.ndarray,
        tolerance: float = 0.02
    ) -> Optional[float]:
        """Estimate time to reach steady state."""
        final_value = values[-1]
        threshold = final_value * tolerance

        for i in range(len(values) - 1, -1, -1):
            if abs(values[i] - final_value) > threshold:
                return float(i * 0.1)  # Assuming 0.1s timestep

        return 0.0


class RealTimeStatePredictor:
    """
    High-level real-time state prediction system.

    Combines physics-based and data-driven prediction methods
    with predictive alerting.
    """

    def __init__(self, model: 'TangheSiphonModel'):
        self.model = model
        self.physics_predictor = StatePredictor(model)
        self.trend_predictor = TrendPredictor()
        self.alert_engine = PredictiveAlertEngine()
        self.scenario_predictor = ScenarioPredictor(self.physics_predictor)

        self._last_prediction: Optional[PredictionResult] = None

        logger.info("RealTimeStatePredictor initialized")

    def update_observations(self) -> None:
        """Update trend predictor with current observations."""
        state = self.model.get_state()

        self.trend_predictor.add_observation(
            'total_flow', state['total_flow'], self.model.time
        )
        self.trend_predictor.add_observation(
            'max_vibration', max(state['vibrations']), self.model.time
        )

        for i, v in enumerate(state['velocities']):
            self.trend_predictor.add_observation(
                f'velocity_{i}', v, self.model.time
            )

    def predict(
        self,
        horizon_seconds: float = 60.0,
        target_openings: Optional[np.ndarray] = None,
        method: PredictionMethod = PredictionMethod.HYBRID
    ) -> Dict[str, Any]:
        """
        Perform state prediction.

        Args:
            horizon_seconds: Prediction horizon
            target_openings: Optional target gate openings
            method: Prediction method to use

        Returns:
            Complete prediction results
        """
        # Update observations
        self.update_observations()

        # Physics-based prediction
        physics_result = self.physics_predictor.predict_physics(
            horizon_seconds, target_openings
        )
        self._last_prediction = physics_result

        # Trend analysis
        trend_analyses = {
            'total_flow': self.trend_predictor.analyze_trend(
                'total_flow', horizon_seconds
            ),
            'max_vibration': self.trend_predictor.analyze_trend(
                'max_vibration', horizon_seconds
            )
        }

        # Check for predictive alerts
        new_alerts = self.alert_engine.check_alerts(
            trend_analyses, self.model.time
        )
        self.alert_engine.clear_resolved_alerts(trend_analyses)

        return {
            'physics_prediction': physics_result,
            'trend_analyses': trend_analyses,
            'new_alerts': new_alerts,
            'active_alerts': self.alert_engine.get_active_alerts(),
            'timestamp': self.model.time
        }

    def get_predicted_state_at(
        self,
        time_ahead: float
    ) -> Optional[Dict[str, float]]:
        """
        Get predicted state at specific future time.

        Args:
            time_ahead: Seconds ahead to predict

        Returns:
            Predicted state values or None if not available
        """
        if self._last_prediction is None:
            return None

        times = self._last_prediction.prediction_times
        state = self._last_prediction.predicted_state

        if time_ahead > times[-1] - times[0]:
            return None

        # Find closest time index
        target_time = times[0] + time_ahead
        idx = np.argmin(np.abs(times - target_time))

        return {
            'time': float(times[idx]),
            'total_flow': float(state['total_flow'][idx]),
            'gate_openings': state['openings'][idx].tolist(),
            'velocities': state['velocities'][idx].tolist(),
            'vibrations': state['vibrations'][idx].tolist(),
            'frequencies': state['frequencies'][idx].tolist()
        }

    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get summary of current predictions."""
        if self._last_prediction is None:
            return {'status': 'no_prediction'}

        pred = self._last_prediction

        return {
            'method': pred.method.value,
            'horizon': pred.horizon.value,
            'start_time': float(pred.prediction_times[0]),
            'end_time': float(pred.prediction_times[-1]),
            'final_flow': float(pred.predicted_state['total_flow'][-1]),
            'max_predicted_vibration': float(
                np.max(pred.predicted_state['vibrations'])
            ),
            'uncertainty': pred.uncertainty,
            'active_alerts': len(self.alert_engine.get_active_alerts())
        }

    def what_if_analysis(
        self,
        target_openings: np.ndarray,
        horizon_seconds: float = 60.0
    ) -> Dict[str, Any]:
        """
        Perform what-if analysis for proposed control action.

        Args:
            target_openings: Proposed gate openings
            horizon_seconds: Analysis horizon

        Returns:
            Analysis results including predicted outcomes
        """
        # Predict with proposed openings
        prediction = self.physics_predictor.predict_physics(
            horizon_seconds, target_openings
        )

        # Compare with current trajectory (no change)
        current_prediction = self.physics_predictor.predict_physics(
            horizon_seconds, None  # Use current openings
        )

        return {
            'proposed': {
                'final_flow': float(prediction.predicted_state['total_flow'][-1]),
                'max_vibration': float(np.max(prediction.predicted_state['vibrations'])),
                'mean_vibration': float(np.mean(prediction.predicted_state['vibrations']))
            },
            'current': {
                'final_flow': float(current_prediction.predicted_state['total_flow'][-1]),
                'max_vibration': float(np.max(current_prediction.predicted_state['vibrations'])),
                'mean_vibration': float(np.mean(current_prediction.predicted_state['vibrations']))
            },
            'improvement': {
                'flow_delta': float(
                    prediction.predicted_state['total_flow'][-1] -
                    current_prediction.predicted_state['total_flow'][-1]
                ),
                'vibration_reduction': float(
                    np.max(current_prediction.predicted_state['vibrations']) -
                    np.max(prediction.predicted_state['vibrations'])
                )
            }
        }
