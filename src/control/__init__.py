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
]
