# -*- coding: utf-8 -*-
"""
Planning and Prediction Information Module.

This module provides comprehensive data structures and utilities for:
- 调度计划 (Dispatching schedule)
- 运行计划 (Operation plan)
- 维护计划 (Maintenance plan)
- 流量预测 (Flow forecast)
- 水位预测 (Water level forecast)
- 天气预测 (Weather forecast)
- 上游来水预测 (Upstream inflow prediction)

These information sources are used by the scenario generator and detector
to create more realistic scenarios and improve recognition accuracy.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Callable
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Time-related Enums
# =============================================================================

class TimeOfDay(Enum):
    """Time of day classification."""
    NIGHT = auto()        # 00:00 - 06:00
    MORNING = auto()      # 06:00 - 12:00
    AFTERNOON = auto()    # 12:00 - 18:00
    EVENING = auto()      # 18:00 - 24:00


class Season(Enum):
    """Season classification."""
    SPRING = auto()       # 3-5月
    SUMMER = auto()       # 6-8月 (丰水期)
    AUTUMN = auto()       # 9-11月
    WINTER = auto()       # 12-2月 (枯水期)


class WeekDay(Enum):
    """Day of week classification."""
    WEEKDAY = auto()
    WEEKEND = auto()
    HOLIDAY = auto()


# =============================================================================
# Weather Information
# =============================================================================

class WeatherType(Enum):
    """Weather type classification."""
    CLEAR = auto()           # 晴天
    CLOUDY = auto()          # 多云
    OVERCAST = auto()        # 阴天
    LIGHT_RAIN = auto()      # 小雨
    MODERATE_RAIN = auto()   # 中雨
    HEAVY_RAIN = auto()      # 大雨
    STORM = auto()           # 暴风雨
    SNOW = auto()            # 雪
    FOG = auto()             # 雾


@dataclass
class WeatherForecast:
    """Weather forecast for a time period."""
    start_time: float              # 开始时间 (simulation time)
    end_time: float                # 结束时间
    weather_type: WeatherType      # 天气类型
    temperature: float = 20.0      # 温度 (°C)
    humidity: float = 60.0         # 湿度 (%)
    wind_speed: float = 3.0        # 风速 (m/s)
    precipitation: float = 0.0     # 降水量 (mm/h)
    probability: float = 0.8       # 预测置信度

    def affects_inflow(self) -> float:
        """Calculate expected inflow effect multiplier."""
        if self.weather_type in [WeatherType.HEAVY_RAIN, WeatherType.STORM]:
            return 1.5 + self.precipitation / 50.0
        elif self.weather_type == WeatherType.MODERATE_RAIN:
            return 1.2 + self.precipitation / 100.0
        elif self.weather_type == WeatherType.LIGHT_RAIN:
            return 1.05 + self.precipitation / 200.0
        elif self.weather_type == WeatherType.SNOW:
            return 0.9  # Delayed effect
        else:
            return 1.0


# =============================================================================
# Inflow and Water Level Prediction
# =============================================================================

@dataclass
class InflowPrediction:
    """Upstream inflow prediction."""
    time: float                    # 预测时间点
    expected_flow: float           # 预期流量 (m³/s)
    lower_bound: float             # 下界 (90%置信区间)
    upper_bound: float             # 上界 (90%置信区间)
    source: str = "model"          # 预测来源 (model/historical/manual)
    confidence: float = 0.85       # 置信度

    def get_range(self) -> Tuple[float, float]:
        """Get prediction range."""
        return (self.lower_bound, self.upper_bound)

    def contains(self, value: float) -> bool:
        """Check if value is within prediction range."""
        return self.lower_bound <= value <= self.upper_bound


@dataclass
class WaterLevelPrediction:
    """Water level prediction for upstream/downstream."""
    time: float                    # 预测时间点
    upstream_level: float          # 上游水位 (m)
    downstream_level: float        # 下游水位 (m)
    upstream_uncertainty: float = 0.2   # 上游不确定性 (m)
    downstream_uncertainty: float = 0.15  # 下游不确定性 (m)
    source: str = "model"

    def get_head_difference(self) -> float:
        """Calculate expected head difference."""
        return self.upstream_level - self.downstream_level

    def get_head_range(self) -> Tuple[float, float]:
        """Get head difference range."""
        min_head = (self.upstream_level - self.upstream_uncertainty -
                    self.downstream_level - self.downstream_uncertainty)
        max_head = (self.upstream_level + self.upstream_uncertainty -
                    self.downstream_level + self.downstream_uncertainty)
        return (max(0.0, min_head), max_head)


# =============================================================================
# Operation Plans
# =============================================================================

class PlanType(Enum):
    """Type of operation plan."""
    DISPATCH = auto()         # 调度计划 (流量调度)
    MAINTENANCE = auto()      # 维护计划
    INSPECTION = auto()       # 检修计划
    EMERGENCY_DRILL = auto()  # 应急演练
    TEST = auto()             # 测试计划


class PlanPriority(Enum):
    """Plan priority level."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    EMERGENCY = 5


@dataclass
class OperationEvent:
    """Single operation event in a plan."""
    time: float                    # 事件时间
    event_type: str               # 事件类型
    target_value: Optional[float] = None   # 目标值
    gate_index: Optional[int] = None       # 闸门索引
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class DispatchPlan:
    """
    Flow dispatch plan (调度计划).

    Defines scheduled flow targets and timing.
    """
    plan_id: str
    name: str
    start_time: float
    end_time: float
    priority: PlanPriority = PlanPriority.NORMAL

    # Flow schedule: list of (time, target_flow) tuples
    flow_schedule: List[Tuple[float, float]] = field(default_factory=list)

    # Constraints
    min_flow: float = 20.0         # 最小流量
    max_flow: float = 200.0        # 最大流量
    max_ramp_rate: float = 10.0    # 最大变化率 (m³/s per minute)

    # Metadata
    source: str = "scheduler"      # 来源
    approved: bool = True          # 是否批准

    def get_target_flow_at(self, time: float) -> Optional[float]:
        """Get scheduled target flow at given time."""
        if not self.flow_schedule:
            return None

        # Find the applicable schedule point
        for i, (t, flow) in enumerate(self.flow_schedule):
            if t > time:
                if i == 0:
                    return self.flow_schedule[0][1]
                # Interpolate between previous and current
                t_prev, flow_prev = self.flow_schedule[i - 1]
                ratio = (time - t_prev) / (t - t_prev) if t != t_prev else 0
                return flow_prev + ratio * (flow - flow_prev)

        # Past the last schedule point
        return self.flow_schedule[-1][1] if self.flow_schedule else None

    def is_active_at(self, time: float) -> bool:
        """Check if plan is active at given time."""
        return self.start_time <= time <= self.end_time


@dataclass
class MaintenancePlan:
    """
    Maintenance plan (维护计划).

    Defines scheduled maintenance activities that affect operation.
    """
    plan_id: str
    name: str
    start_time: float
    end_time: float
    priority: PlanPriority = PlanPriority.NORMAL

    # Affected equipment
    affected_gates: List[int] = field(default_factory=list)
    affected_sensors: List[str] = field(default_factory=list)

    # Operational constraints during maintenance
    gate_availability: Dict[int, bool] = field(default_factory=dict)
    reduced_capacity: float = 1.0  # Capacity multiplier (0.0-1.0)

    # Maintenance actions
    events: List[OperationEvent] = field(default_factory=list)

    # Metadata
    maintenance_type: str = "preventive"  # preventive/corrective
    requires_shutdown: bool = False

    def is_gate_available(self, gate_idx: int) -> bool:
        """Check if gate is available during maintenance."""
        return self.gate_availability.get(gate_idx, True)

    def get_available_gates(self, total_gates: int = 3) -> List[int]:
        """Get list of available gates during maintenance."""
        return [i for i in range(total_gates) if self.is_gate_available(i)]


@dataclass
class InspectionPlan:
    """
    Inspection plan (检修计划).

    Defines scheduled inspection activities.
    """
    plan_id: str
    name: str
    scheduled_time: float
    duration: float

    # Inspection scope
    components: List[str] = field(default_factory=list)
    gates_to_inspect: List[int] = field(default_factory=list)

    # Operational impact
    flow_reduction: float = 0.0    # Flow reduction during inspection (%)
    gate_isolation: List[int] = field(default_factory=list)  # Gates to isolate

    # Results (filled after inspection)
    completed: bool = False
    findings: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Comprehensive Planning Context
# =============================================================================

@dataclass
class PlanningContext:
    """
    Complete planning context combining all information sources.

    This is the main interface for scenario generators and detectors
    to access planning and prediction information.
    """
    # Time context
    simulation_time: float = 0.0
    time_of_day: TimeOfDay = TimeOfDay.MORNING
    season: Season = Season.SUMMER
    day_type: WeekDay = WeekDay.WEEKDAY

    # Active plans
    dispatch_plans: List[DispatchPlan] = field(default_factory=list)
    maintenance_plans: List[MaintenancePlan] = field(default_factory=list)
    inspection_plans: List[InspectionPlan] = field(default_factory=list)

    # Predictions
    weather_forecasts: List[WeatherForecast] = field(default_factory=list)
    inflow_predictions: List[InflowPrediction] = field(default_factory=list)
    water_level_predictions: List[WaterLevelPrediction] = field(default_factory=list)

    # Historical context
    recent_flow_mean: float = 100.0    # 近期平均流量
    recent_flow_std: float = 10.0      # 近期流量标准差
    recent_head_mean: float = 2.0      # 近期平均水头差

    # Alerts and warnings
    active_alerts: List[str] = field(default_factory=list)
    flood_warning: bool = False
    drought_warning: bool = False

    def get_active_dispatch_plan(self) -> Optional[DispatchPlan]:
        """Get currently active dispatch plan with highest priority."""
        active = [p for p in self.dispatch_plans
                  if p.is_active_at(self.simulation_time)]
        if not active:
            return None
        return max(active, key=lambda p: p.priority.value)

    def get_active_maintenance(self) -> List[MaintenancePlan]:
        """Get all active maintenance plans."""
        return [p for p in self.maintenance_plans
                if p.start_time <= self.simulation_time <= p.end_time]

    def get_current_weather(self) -> Optional[WeatherForecast]:
        """Get current weather forecast."""
        for w in self.weather_forecasts:
            if w.start_time <= self.simulation_time <= w.end_time:
                return w
        return None

    def get_inflow_prediction(self, horizon: float = 60.0) -> List[InflowPrediction]:
        """Get inflow predictions within horizon (seconds)."""
        return [p for p in self.inflow_predictions
                if self.simulation_time <= p.time <= self.simulation_time + horizon]

    def get_expected_inflow(self) -> float:
        """Get expected inflow based on all available information."""
        # Base from predictions
        predictions = self.get_inflow_prediction(30)
        if predictions:
            base_flow = np.mean([p.expected_flow for p in predictions])
        else:
            base_flow = self.recent_flow_mean

        # Adjust for weather
        weather = self.get_current_weather()
        if weather:
            base_flow *= weather.affects_inflow()

        # Adjust for season
        season_factor = {
            Season.SPRING: 1.0,
            Season.SUMMER: 1.3,    # 丰水期
            Season.AUTUMN: 0.9,
            Season.WINTER: 0.7,    # 枯水期
        }.get(self.season, 1.0)

        return base_flow * season_factor

    def is_maintenance_affecting_gate(self, gate_idx: int) -> bool:
        """Check if any maintenance affects the given gate."""
        for m in self.get_active_maintenance():
            if gate_idx in m.affected_gates:
                return True
        return False

    def get_operational_constraints(self) -> Dict[str, Any]:
        """Get current operational constraints from all plans."""
        constraints = {
            'available_gates': [0, 1, 2],
            'max_flow': 200.0,
            'min_flow': 0.0,
            'max_ramp_rate': 10.0,
            'reduced_capacity': 1.0,
        }

        # Apply dispatch plan constraints
        dispatch = self.get_active_dispatch_plan()
        if dispatch:
            constraints['max_flow'] = min(constraints['max_flow'], dispatch.max_flow)
            constraints['min_flow'] = max(constraints['min_flow'], dispatch.min_flow)
            constraints['max_ramp_rate'] = min(constraints['max_ramp_rate'],
                                               dispatch.max_ramp_rate)

        # Apply maintenance constraints
        for m in self.get_active_maintenance():
            constraints['available_gates'] = [
                g for g in constraints['available_gates']
                if m.is_gate_available(g)
            ]
            constraints['reduced_capacity'] = min(
                constraints['reduced_capacity'],
                m.reduced_capacity
            )

        return constraints

    def get_scenario_hints(self) -> Dict[str, Any]:
        """Get hints about expected scenario from planning info."""
        hints = {
            'expected_flow_range': (self.recent_flow_mean * 0.8,
                                     self.recent_flow_mean * 1.2),
            'possible_scenarios': [],
            'risk_factors': [],
        }

        # Check for flood/drought warnings
        if self.flood_warning:
            hints['possible_scenarios'].append('FLOOD_CONDITION')
            hints['risk_factors'].append('flood_warning')

        if self.drought_warning:
            hints['possible_scenarios'].append('DROUGHT_CONDITION')
            hints['risk_factors'].append('drought_warning')

        # Check weather
        weather = self.get_current_weather()
        if weather:
            if weather.weather_type in [WeatherType.HEAVY_RAIN, WeatherType.STORM]:
                hints['possible_scenarios'].append('HEAD_SURGE')
                hints['risk_factors'].append('heavy_rain')
                hints['expected_flow_range'] = (
                    hints['expected_flow_range'][0] * 1.3,
                    hints['expected_flow_range'][1] * 1.5
                )

        # Check maintenance
        if self.get_active_maintenance():
            hints['possible_scenarios'].append('GATE_STUCK')
            hints['risk_factors'].append('maintenance_in_progress')

        # Check season
        if self.season == Season.SUMMER:
            hints['risk_factors'].append('flood_season')
        elif self.season == Season.WINTER:
            hints['risk_factors'].append('dry_season')

        return hints

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'time': self.simulation_time,
            'time_of_day': self.time_of_day.name,
            'season': self.season.name,
            'day_type': self.day_type.name,
            'active_dispatch': (self.get_active_dispatch_plan().plan_id
                               if self.get_active_dispatch_plan() else None),
            'active_maintenance_count': len(self.get_active_maintenance()),
            'current_weather': (self.get_current_weather().weather_type.name
                               if self.get_current_weather() else None),
            'expected_inflow': self.get_expected_inflow(),
            'flood_warning': self.flood_warning,
            'drought_warning': self.drought_warning,
            'constraints': self.get_operational_constraints(),
        }


# =============================================================================
# Planning Context Generators
# =============================================================================

class PlanningContextGenerator:
    """
    Generates realistic planning contexts for scenario testing.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def generate_normal_day(self, base_time: float = 0.0) -> PlanningContext:
        """Generate planning context for a normal operation day."""
        ctx = PlanningContext(simulation_time=base_time)

        # Set time context
        hour = (base_time / 3600) % 24
        ctx.time_of_day = self._hour_to_time_of_day(hour)

        # Generate typical dispatch plan
        flow_schedule = self._generate_daily_flow_schedule(base_time)
        ctx.dispatch_plans.append(DispatchPlan(
            plan_id="DP_NORMAL_001",
            name="日常调度",
            start_time=base_time,
            end_time=base_time + 86400,  # 24 hours
            flow_schedule=flow_schedule,
        ))

        # Clear weather
        ctx.weather_forecasts.append(WeatherForecast(
            start_time=base_time,
            end_time=base_time + 86400,
            weather_type=WeatherType.CLEAR,
        ))

        # Stable inflow predictions
        for t in np.arange(base_time, base_time + 3600, 300):  # 5-min intervals
            flow = 100 + self.rng.normal(0, 5)
            ctx.inflow_predictions.append(InflowPrediction(
                time=t,
                expected_flow=flow,
                lower_bound=flow * 0.9,
                upper_bound=flow * 1.1,
            ))

        return ctx

    def generate_flood_scenario(self, base_time: float = 0.0) -> PlanningContext:
        """Generate planning context for flood conditions."""
        ctx = PlanningContext(
            simulation_time=base_time,
            season=Season.SUMMER,
            flood_warning=True,
        )

        # Emergency dispatch plan
        flow_schedule = [
            (base_time, 120.0),
            (base_time + 600, 150.0),
            (base_time + 1200, 180.0),
            (base_time + 1800, 200.0),
        ]
        ctx.dispatch_plans.append(DispatchPlan(
            plan_id="DP_FLOOD_001",
            name="防汛调度",
            start_time=base_time,
            end_time=base_time + 86400,
            priority=PlanPriority.URGENT,
            flow_schedule=flow_schedule,
            max_flow=220.0,
        ))

        # Storm weather
        ctx.weather_forecasts.append(WeatherForecast(
            start_time=base_time,
            end_time=base_time + 14400,  # 4 hours
            weather_type=WeatherType.STORM,
            precipitation=50.0,
        ))

        # Rising inflow predictions
        for i, t in enumerate(np.arange(base_time, base_time + 3600, 300)):
            base_flow = 150 + i * 10
            ctx.inflow_predictions.append(InflowPrediction(
                time=t,
                expected_flow=base_flow,
                lower_bound=base_flow * 0.85,
                upper_bound=base_flow * 1.2,
                confidence=0.7,
            ))

        ctx.active_alerts.append("FLOOD_LEVEL_1")

        return ctx

    def generate_drought_scenario(self, base_time: float = 0.0) -> PlanningContext:
        """Generate planning context for drought conditions."""
        ctx = PlanningContext(
            simulation_time=base_time,
            season=Season.WINTER,
            drought_warning=True,
        )

        # Reduced flow dispatch
        flow_schedule = [
            (base_time, 50.0),
            (base_time + 3600, 40.0),
            (base_time + 7200, 35.0),
        ]
        ctx.dispatch_plans.append(DispatchPlan(
            plan_id="DP_DROUGHT_001",
            name="枯水期调度",
            start_time=base_time,
            end_time=base_time + 86400,
            flow_schedule=flow_schedule,
            min_flow=25.0,
            max_flow=80.0,
        ))

        # Low inflow predictions
        for t in np.arange(base_time, base_time + 3600, 300):
            flow = 40 + self.rng.normal(0, 3)
            ctx.inflow_predictions.append(InflowPrediction(
                time=t,
                expected_flow=max(20, flow),
                lower_bound=max(15, flow * 0.8),
                upper_bound=flow * 1.1,
            ))

        ctx.active_alerts.append("LOW_WATER_LEVEL")

        return ctx

    def generate_maintenance_scenario(
        self,
        base_time: float = 0.0,
        gate_to_maintain: int = 1
    ) -> PlanningContext:
        """Generate planning context with scheduled maintenance."""
        ctx = PlanningContext(simulation_time=base_time)

        # Maintenance plan
        ctx.maintenance_plans.append(MaintenancePlan(
            plan_id="MP_GATE_001",
            name=f"闸门{gate_to_maintain}号定期维护",
            start_time=base_time + 600,  # Start in 10 minutes
            end_time=base_time + 4200,   # 1 hour maintenance
            affected_gates=[gate_to_maintain],
            gate_availability={gate_to_maintain: False},
            reduced_capacity=0.67,  # 2/3 capacity
            requires_shutdown=False,
        ))

        # Adjusted dispatch for maintenance
        flow_schedule = [
            (base_time, 100.0),
            (base_time + 600, 70.0),   # Reduce before maintenance
            (base_time + 4200, 100.0), # Restore after
        ]
        ctx.dispatch_plans.append(DispatchPlan(
            plan_id="DP_MAINT_001",
            name="维护期间调度",
            start_time=base_time,
            end_time=base_time + 7200,
            flow_schedule=flow_schedule,
        ))

        return ctx

    def generate_transition_scenario(
        self,
        base_time: float = 0.0,
        from_flow: float = 50.0,
        to_flow: float = 150.0
    ) -> PlanningContext:
        """Generate planning context for planned flow transition."""
        ctx = PlanningContext(simulation_time=base_time)

        # Planned transition
        transition_time = base_time + 600  # 10 minutes
        flow_schedule = [
            (base_time, from_flow),
            (transition_time, from_flow),
            (transition_time + 1, to_flow),  # Step change
        ]

        ctx.dispatch_plans.append(DispatchPlan(
            plan_id="DP_TRANS_001",
            name="计划流量切换",
            start_time=base_time,
            end_time=base_time + 3600,
            flow_schedule=flow_schedule,
        ))

        return ctx

    def generate_random_context(self, base_time: float = 0.0) -> PlanningContext:
        """Generate random planning context."""
        scenario_type = self.rng.choice([
            'normal', 'flood', 'drought', 'maintenance', 'transition'
        ], p=[0.5, 0.15, 0.1, 0.15, 0.1])

        if scenario_type == 'normal':
            return self.generate_normal_day(base_time)
        elif scenario_type == 'flood':
            return self.generate_flood_scenario(base_time)
        elif scenario_type == 'drought':
            return self.generate_drought_scenario(base_time)
        elif scenario_type == 'maintenance':
            gate = self.rng.integers(0, 3)
            return self.generate_maintenance_scenario(base_time, gate)
        else:
            from_flow = self.rng.uniform(40, 80)
            to_flow = self.rng.uniform(100, 180)
            return self.generate_transition_scenario(base_time, from_flow, to_flow)

    def _hour_to_time_of_day(self, hour: float) -> TimeOfDay:
        """Convert hour to time of day."""
        if hour < 6:
            return TimeOfDay.NIGHT
        elif hour < 12:
            return TimeOfDay.MORNING
        elif hour < 18:
            return TimeOfDay.AFTERNOON
        else:
            return TimeOfDay.EVENING

    def _generate_daily_flow_schedule(
        self,
        base_time: float
    ) -> List[Tuple[float, float]]:
        """Generate typical daily flow schedule."""
        schedule = []
        base_flow = 100.0

        # Every hour
        for hour in range(24):
            t = base_time + hour * 3600

            # Morning peak
            if 7 <= hour < 10:
                flow = base_flow * 1.3
            # Evening peak
            elif 18 <= hour < 21:
                flow = base_flow * 1.2
            # Night low
            elif hour < 6 or hour >= 22:
                flow = base_flow * 0.7
            else:
                flow = base_flow

            schedule.append((t, flow + self.rng.normal(0, 5)))

        return schedule


# =============================================================================
# Prediction-Aware Scenario Detector
# =============================================================================

class PredictionAwareDetector:
    """
    Enhanced scenario detector that uses prediction information.

    Combines real-time sensor data with planning and prediction
    information for more accurate scenario detection.
    """

    def __init__(self, model: 'TangheSiphonModel'):
        self.model = model
        self._context: Optional[PlanningContext] = None

        # Detection weights
        self.sensor_weight = 0.6
        self.prediction_weight = 0.4

        # History for trend analysis
        self._flow_history: List[float] = []
        self._prediction_errors: List[float] = []

    def set_context(self, context: PlanningContext) -> None:
        """Set the current planning context."""
        self._context = context

    def detect_with_context(
        self,
        sensor_scenario: 'ScenarioType',
        dt: float
    ) -> Tuple['ScenarioType', float]:
        """
        Detect scenario using both sensor data and predictions.

        Args:
            sensor_scenario: Scenario detected from sensor data
            dt: Time step

        Returns:
            Tuple of (detected scenario, confidence)
        """
        if self._context is None:
            return sensor_scenario, 0.5

        hints = self._context.get_scenario_hints()

        # Track prediction accuracy
        current_flow = float(np.sum(self.model.flow_rates))
        self._flow_history.append(current_flow)

        expected_flow = self._context.get_expected_inflow()
        error = abs(current_flow - expected_flow) / max(expected_flow, 1.0)
        self._prediction_errors.append(error)

        # Limit history
        if len(self._flow_history) > 100:
            self._flow_history.pop(0)
            self._prediction_errors.pop(0)

        # Calculate confidence based on prediction accuracy
        if self._prediction_errors:
            avg_error = np.mean(self._prediction_errors[-20:])
            prediction_confidence = max(0.0, 1.0 - avg_error)
        else:
            prediction_confidence = 0.5

        # Check if sensor detection matches planning hints
        possible_scenarios = hints.get('possible_scenarios', [])

        if sensor_scenario.name in possible_scenarios:
            # Sensor and planning agree
            confidence = 0.9
        elif possible_scenarios:
            # Planning suggests different scenario - use weighted decision
            if self._context.flood_warning and sensor_scenario.name != 'FLOOD_CONDITION':
                # Override with flood scenario if warning is active
                from src.control.scenario_advanced import ScenarioType
                return ScenarioType.FLOOD_CONDITION, 0.8
            if self._context.drought_warning and sensor_scenario.name != 'DROUGHT_CONDITION':
                from src.control.scenario_advanced import ScenarioType
                return ScenarioType.DROUGHT_CONDITION, 0.75
            confidence = 0.6
        else:
            confidence = prediction_confidence

        return sensor_scenario, confidence

    def get_predicted_scenario(self, horizon: float = 300.0) -> Optional[str]:
        """
        Predict scenario for future time horizon.

        Args:
            horizon: Prediction horizon in seconds

        Returns:
            Predicted scenario name or None
        """
        if self._context is None:
            return None

        # Get future context
        future_time = self._context.simulation_time + horizon

        # Check dispatch plan
        dispatch = self._context.get_active_dispatch_plan()
        if dispatch:
            future_flow = dispatch.get_target_flow_at(future_time)
            if future_flow:
                if future_flow > 150:
                    return 'NORMAL_HIGH_FLOW'
                elif future_flow < 50:
                    return 'NORMAL_LOW_FLOW'

        # Check maintenance plans
        for m in self._context.maintenance_plans:
            if m.start_time <= future_time <= m.end_time:
                if m.requires_shutdown:
                    return 'EMERGENCY_SHUTDOWN'
                elif m.affected_gates:
                    return 'GATE_STUCK'  # Expected reduced operation

        # Check weather forecasts
        for w in self._context.weather_forecasts:
            if w.start_time <= future_time <= w.end_time:
                if w.weather_type == WeatherType.STORM:
                    return 'HEAD_SURGE'

        return None

    def get_risk_assessment(self) -> Dict[str, float]:
        """
        Get risk assessment based on predictions.

        Returns:
            Dictionary of scenario to risk probability
        """
        risks = {
            'FLOOD_CONDITION': 0.0,
            'DROUGHT_CONDITION': 0.0,
            'RESONANCE_CROSSING': 0.0,
            'GATE_STUCK': 0.0,
            'HEAD_SURGE': 0.0,
        }

        if self._context is None:
            return risks

        # Flood risk
        if self._context.flood_warning:
            risks['FLOOD_CONDITION'] = 0.7
        weather = self._context.get_current_weather()
        if weather and weather.weather_type in [WeatherType.HEAVY_RAIN, WeatherType.STORM]:
            risks['FLOOD_CONDITION'] += 0.2
            risks['HEAD_SURGE'] += 0.4

        # Drought risk
        if self._context.drought_warning:
            risks['DROUGHT_CONDITION'] = 0.7
        if self._context.season == Season.WINTER:
            risks['DROUGHT_CONDITION'] += 0.15

        # Gate stuck risk (maintenance)
        for m in self._context.get_active_maintenance():
            risks['GATE_STUCK'] += 0.3 * len(m.affected_gates)

        # Resonance risk
        expected_flow = self._context.get_expected_inflow()
        if 70 <= expected_flow <= 110:  # Resonance zone
            risks['RESONANCE_CROSSING'] = 0.3

        # Clamp to [0, 1]
        return {k: min(1.0, v) for k, v in risks.items()}
