# -*- coding: utf-8 -*-
"""
Control Module

Contains MPC controller, local controller, and scenario manager.
"""

from .mpc import SpectralMPC
from .local import LocalController
from .manager import ScenarioManager
from .adaptive_mpc import AdaptiveMPC, AdaptiveState
from .pid import PIDController, PIDGains, MultiChannelPID, HybridController
from .scenario_advanced import (
    ScenarioType,
    ScenarioConfig,
    ScenarioDetector,
    AdvancedScenarioManager,
)
from .integrated_controller import (
    IntegratedController,
    MPCObjective,
    MPCConstraints,
    ScenarioControlConfig,
)
from .scenario_generator import (
    FullScenarioGenerator,
    ScenarioSpec,
    ScenarioTestRunner,
    FlowRegime,
    HeadCondition,
    VibrationLevel,
    GateFaultType,
    SensorFaultType,
    TransitionPattern,
    ExtendedScenarioGenerator,
    PlanningAwareScenarioGenerator,
    SeasonalScenarioGenerator,
    PredictionScenarioGenerator,
    DispatchScheduleGenerator,
)
from .planning_prediction import (
    PlanningContext,
    PlanningContextGenerator,
    DispatchPlan,
    MaintenancePlan,
    InspectionPlan,
    WeatherForecast,
    InflowPrediction,
    WaterLevelPrediction,
    PredictionAwareDetector,
    TimeOfDay,
    Season,
    WeekDay,
    WeatherType,
    PlanType,
    PlanPriority,
)

__all__ = [
    'SpectralMPC',
    'LocalController',
    'ScenarioManager',
    'AdaptiveMPC',
    'AdaptiveState',
    'PIDController',
    'PIDGains',
    'MultiChannelPID',
    'HybridController',
    'ScenarioType',
    'ScenarioConfig',
    'ScenarioDetector',
    'AdvancedScenarioManager',
    'IntegratedController',
    'MPCObjective',
    'MPCConstraints',
    'ScenarioControlConfig',
    'FullScenarioGenerator',
    'ScenarioSpec',
    'ScenarioTestRunner',
    'FlowRegime',
    'HeadCondition',
    'VibrationLevel',
    'GateFaultType',
    'SensorFaultType',
    'TransitionPattern',
    'ExtendedScenarioGenerator',
    'PlanningAwareScenarioGenerator',
    'SeasonalScenarioGenerator',
    'PredictionScenarioGenerator',
    'DispatchScheduleGenerator',
    # Planning and Prediction
    'PlanningContext',
    'PlanningContextGenerator',
    'DispatchPlan',
    'MaintenancePlan',
    'InspectionPlan',
    'WeatherForecast',
    'InflowPrediction',
    'WaterLevelPrediction',
    'PredictionAwareDetector',
    'TimeOfDay',
    'Season',
    'WeekDay',
    'WeatherType',
    'PlanType',
    'PlanPriority',
]
