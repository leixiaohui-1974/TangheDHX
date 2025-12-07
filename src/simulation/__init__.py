# -*- coding: utf-8 -*-
"""
Simulation Module

Contains physical models, sensors, and actuators for the digital twin.
"""

from .physics import TangheSiphonModel
from .sensors import ADCPSensor, VibrationSensor
from .actuators import GateController
from .physics_advanced import HighFidelityPhysicsModel, AdvancedPhysicsParams
from .sensors_advanced import (
    NoiseGenerator,
    SensorDynamics,
    AdvancedADCPSensor,
    AdvancedVibrationSensor,
    AdvancedPositionSensor,
)
from .hil_framework import (
    HILMode,
    HILConfig,
    HILMetrics,
    HILTestRunner,
    SimulatedSensorInterface,
    SimulatedActuatorInterface,
)

__all__ = [
    'TangheSiphonModel',
    'ADCPSensor',
    'VibrationSensor',
    'GateController',
    'HighFidelityPhysicsModel',
    'AdvancedPhysicsParams',
    'NoiseGenerator',
    'SensorDynamics',
    'AdvancedADCPSensor',
    'AdvancedVibrationSensor',
    'AdvancedPositionSensor',
    'HILMode',
    'HILConfig',
    'HILMetrics',
    'HILTestRunner',
    'SimulatedSensorInterface',
    'SimulatedActuatorInterface',
]
