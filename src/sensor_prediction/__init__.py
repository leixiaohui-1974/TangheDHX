# -*- coding: utf-8 -*-
"""
传感器仿真与状态预测模块

提供传感器仿真、状态预测、虚拟传感器合成和传感器健康监测功能。
"""

from .predictor import (
    # 枚举类型
    SensorType,
    NoiseModel,
    FailureMode,
    PredictionMethod,
    SensorHealth,

    # 数据类
    SensorConfig,
    SensorState,
    SensorReading,
    PredictionResult,
    VirtualSensorConfig,
    HealthReport,

    # 核心类
    SensorSimulator,
    StatePredictor,
    VirtualSensorSynthesizer,
    SensorHealthMonitor,
    SensorPredictionManager,
)

__all__ = [
    # 枚举
    'SensorType',
    'NoiseModel',
    'FailureMode',
    'PredictionMethod',
    'SensorHealth',

    # 数据类
    'SensorConfig',
    'SensorState',
    'SensorReading',
    'PredictionResult',
    'VirtualSensorConfig',
    'HealthReport',

    # 核心类
    'SensorSimulator',
    'StatePredictor',
    'VirtualSensorSynthesizer',
    'SensorHealthMonitor',
    'SensorPredictionManager',
]
