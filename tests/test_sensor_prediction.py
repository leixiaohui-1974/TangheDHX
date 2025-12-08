# -*- coding: utf-8 -*-
"""
传感器仿真与状态预测模块单元测试
"""

import pytest
import time
import math
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np

from src.sensor_prediction import (
    SensorType,
    NoiseModel,
    FailureMode,
    PredictionMethod,
    SensorHealth,
    SensorConfig,
    SensorState,
    SensorReading,
    PredictionResult,
    VirtualSensorConfig,
    HealthReport,
    SensorSimulator,
    StatePredictor,
    VirtualSensorSynthesizer,
    SensorHealthMonitor,
    SensorPredictionManager,
)


class TestSensorType:
    """传感器类型测试"""

    def test_sensor_types_exist(self):
        """测试传感器类型存在"""
        assert SensorType.FLOW_METER is not None
        assert SensorType.PRESSURE_SENSOR is not None
        assert SensorType.LEVEL_SENSOR is not None
        assert SensorType.VELOCITY_SENSOR is not None
        assert SensorType.TEMPERATURE_SENSOR is not None
        assert SensorType.TURBIDITY_SENSOR is not None
        assert SensorType.PH_SENSOR is not None
        assert SensorType.CONDUCTIVITY_SENSOR is not None
        assert SensorType.GENERIC is not None


class TestNoiseModel:
    """噪声模型测试"""

    def test_noise_models_exist(self):
        """测试噪声模型存在"""
        assert NoiseModel.GAUSSIAN is not None
        assert NoiseModel.UNIFORM is not None
        assert NoiseModel.PINK is not None
        assert NoiseModel.BROWN is not None
        assert NoiseModel.IMPULSE is not None
        assert NoiseModel.QUANTIZATION is not None


class TestFailureMode:
    """故障模式测试"""

    def test_failure_modes_exist(self):
        """测试故障模式存在"""
        assert FailureMode.NONE is not None
        assert FailureMode.STUCK is not None
        assert FailureMode.DRIFT is not None
        assert FailureMode.BIAS is not None
        assert FailureMode.NOISE_INCREASE is not None
        assert FailureMode.INTERMITTENT is not None
        assert FailureMode.COMPLETE_FAILURE is not None
        assert FailureMode.SPIKE is not None


class TestPredictionMethod:
    """预测方法测试"""

    def test_prediction_methods_exist(self):
        """测试预测方法存在"""
        assert PredictionMethod.LINEAR is not None
        assert PredictionMethod.POLYNOMIAL is not None
        assert PredictionMethod.EXPONENTIAL_SMOOTHING is not None
        assert PredictionMethod.KALMAN_FILTER is not None
        assert PredictionMethod.ARIMA is not None
        assert PredictionMethod.NEURAL_NETWORK is not None
        assert PredictionMethod.ENSEMBLE is not None


class TestSensorHealth:
    """传感器健康状态测试"""

    def test_health_states_exist(self):
        """测试健康状态存在"""
        assert SensorHealth.EXCELLENT is not None
        assert SensorHealth.GOOD is not None
        assert SensorHealth.FAIR is not None
        assert SensorHealth.POOR is not None
        assert SensorHealth.CRITICAL is not None
        assert SensorHealth.FAILED is not None


class TestSensorConfig:
    """传感器配置测试"""

    def test_create_config(self):
        """测试创建配置"""
        config = SensorConfig(
            sensor_id='flow_1',
            sensor_type=SensorType.FLOW_METER,
            name='Flow Meter 1',
            unit='m³/s',
            min_value=0.0,
            max_value=100.0
        )

        assert config.sensor_id == 'flow_1'
        assert config.sensor_type == SensorType.FLOW_METER
        assert config.name == 'Flow Meter 1'
        assert config.unit == 'm³/s'
        assert config.min_value == 0.0
        assert config.max_value == 100.0

    def test_config_defaults(self):
        """测试配置默认值"""
        config = SensorConfig(
            sensor_id='test',
            sensor_type=SensorType.GENERIC,
            name='Test',
            unit='unit',
            min_value=0.0,
            max_value=100.0
        )

        assert config.resolution == 0.01
        assert config.accuracy == 0.01
        assert config.response_time_ms == 100.0
        assert config.noise_model == NoiseModel.GAUSSIAN
        assert config.noise_level == 0.01
        assert config.drift_rate == 0.0


class TestSensorSimulator:
    """传感器仿真器测试"""

    @pytest.fixture
    def config(self):
        """创建测试配置"""
        return SensorConfig(
            sensor_id='test_sensor',
            sensor_type=SensorType.FLOW_METER,
            name='Test Flow Meter',
            unit='m³/s',
            min_value=0.0,
            max_value=100.0,
            noise_level=0.01,
            resolution=0.1
        )

    @pytest.fixture
    def simulator(self, config):
        """创建仿真器"""
        return SensorSimulator(config)

    def test_create_simulator(self, simulator):
        """测试创建仿真器"""
        assert simulator is not None
        assert simulator.config.sensor_id == 'test_sensor'

    def test_set_true_value(self, simulator):
        """测试设置真实值"""
        simulator.set_true_value(50.0)
        assert simulator.get_true_value() == 50.0

    def test_true_value_clamping(self, simulator):
        """测试真实值限幅"""
        simulator.set_true_value(150.0)
        assert simulator.get_true_value() == 100.0

        simulator.set_true_value(-10.0)
        assert simulator.get_true_value() == 0.0

    def test_simulate_reading(self, simulator):
        """测试仿真读数"""
        reading = simulator.simulate_reading(50.0)

        assert reading.sensor_id == 'test_sensor'
        assert isinstance(reading.timestamp, datetime)
        assert reading.raw_value == 50.0
        assert 0.0 <= reading.value <= 100.0
        assert 0.0 <= reading.quality <= 1.0
        assert reading.is_valid

    def test_reading_with_noise(self, simulator):
        """测试带噪声的读数"""
        readings = [simulator.simulate_reading(50.0) for _ in range(100)]
        values = [r.value for r in readings]

        # 值应该在50附近波动
        assert all(40 <= v <= 60 for v in values)
        # 应该有变异
        assert len(set(values)) > 1

    def test_inject_failure_stuck(self, simulator):
        """测试注入卡死故障"""
        simulator.simulate_reading(50.0)
        simulator.inject_failure(FailureMode.STUCK)

        readings = [simulator.simulate_reading(60.0) for _ in range(5)]
        values = [r.value for r in readings]

        # 卡死后值应该相同
        assert all(v == values[0] for v in values)

    def test_inject_failure_complete(self, simulator):
        """测试注入完全失效"""
        simulator.inject_failure(FailureMode.COMPLETE_FAILURE)
        reading = simulator.simulate_reading(50.0)

        assert math.isnan(reading.value)
        assert not reading.is_valid

    def test_clear_failure(self, simulator):
        """测试清除故障"""
        simulator.inject_failure(FailureMode.STUCK)
        simulator.clear_failure()

        state = simulator.get_state()
        assert state.failure_mode == FailureMode.NONE

    def test_calibrate(self, simulator):
        """测试校准"""
        # 先运行一段时间让漂移累积
        for _ in range(10):
            simulator.simulate_reading(50.0)

        simulator.calibrate()
        state = simulator.get_state()

        assert state.drift == 0.0
        assert state.bias == 0.0

    def test_get_state(self, simulator):
        """测试获取状态"""
        simulator.simulate_reading(50.0)
        state = simulator.get_state()

        assert isinstance(state, SensorState)
        assert state.sensor_id == 'test_sensor'
        assert isinstance(state.health, SensorHealth)

    def test_reading_history(self, simulator):
        """测试读数历史"""
        for i in range(10):
            simulator.simulate_reading(float(i * 10))

        history = simulator.get_reading_history(5)
        assert len(history) == 5

    def test_different_noise_models(self):
        """测试不同噪声模型"""
        noise_models = [
            NoiseModel.GAUSSIAN,
            NoiseModel.UNIFORM,
            NoiseModel.PINK,
            NoiseModel.BROWN,
            NoiseModel.IMPULSE,
            NoiseModel.QUANTIZATION
        ]

        for model in noise_models:
            config = SensorConfig(
                sensor_id=f'test_{model.name}',
                sensor_type=SensorType.GENERIC,
                name='Test',
                unit='unit',
                min_value=0.0,
                max_value=100.0,
                noise_model=model,
                noise_level=0.05
            )
            simulator = SensorSimulator(config)
            reading = simulator.simulate_reading(50.0)
            assert 0.0 <= reading.value <= 100.0


class TestStatePredictor:
    """状态预测器测试"""

    @pytest.fixture
    def predictor(self):
        """创建预测器"""
        return StatePredictor(
            method=PredictionMethod.LINEAR,
            history_size=100
        )

    def test_create_predictor(self, predictor):
        """测试创建预测器"""
        assert predictor is not None
        assert predictor.method == PredictionMethod.LINEAR

    def test_add_observation(self, predictor):
        """测试添加观测值"""
        current_time = time.time()
        predictor.add_observation('sensor1', current_time, 50.0)
        # 不应抛出异常

    def test_predict_insufficient_data(self, predictor):
        """测试数据不足时的预测"""
        current_time = time.time()
        predictor.add_observation('sensor1', current_time, 50.0)

        result = predictor.predict('sensor1', 10.0)
        assert result is None

    def test_predict_linear(self, predictor):
        """测试线性预测"""
        base_time = time.time()

        # 添加线性增长的数据
        for i in range(10):
            predictor.add_observation('sensor1', base_time + i, float(i * 10))

        result = predictor.predict('sensor1', 5.0, num_points=5)

        assert result is not None
        assert result.sensor_id == 'sensor1'
        assert len(result.predicted_values) == 5
        assert len(result.confidence_intervals) == 5
        assert result.method == PredictionMethod.LINEAR

        # 预测值应该继续增长
        assert result.predicted_values[-1] > result.predicted_values[0]

    def test_predict_polynomial(self):
        """测试多项式预测"""
        predictor = StatePredictor(method=PredictionMethod.POLYNOMIAL)
        base_time = time.time()

        for i in range(20):
            value = i ** 2  # 二次函数
            predictor.add_observation('sensor1', base_time + i, float(value))

        result = predictor.predict('sensor1', 5.0)
        assert result is not None
        assert result.method == PredictionMethod.POLYNOMIAL

    def test_predict_exponential_smoothing(self):
        """测试指数平滑预测"""
        predictor = StatePredictor(method=PredictionMethod.EXPONENTIAL_SMOOTHING)
        base_time = time.time()

        for i in range(10):
            predictor.add_observation('sensor1', base_time + i, 50.0 + np.sin(i) * 5)

        result = predictor.predict('sensor1', 5.0)
        assert result is not None
        assert result.method == PredictionMethod.EXPONENTIAL_SMOOTHING

    def test_predict_kalman(self):
        """测试卡尔曼滤波预测"""
        predictor = StatePredictor(method=PredictionMethod.KALMAN_FILTER)
        base_time = time.time()

        for i in range(10):
            predictor.add_observation('sensor1', base_time + i, float(i * 5))

        result = predictor.predict('sensor1', 5.0)
        assert result is not None
        assert result.method == PredictionMethod.KALMAN_FILTER

    def test_predict_ensemble(self):
        """测试集成预测"""
        predictor = StatePredictor(method=PredictionMethod.ENSEMBLE)
        base_time = time.time()

        for i in range(20):
            predictor.add_observation('sensor1', base_time + i, float(i * 3))

        result = predictor.predict('sensor1', 10.0)
        assert result is not None
        assert result.method == PredictionMethod.ENSEMBLE

    def test_set_method(self, predictor):
        """测试设置预测方法"""
        predictor.set_method(PredictionMethod.KALMAN_FILTER)
        assert predictor.method == PredictionMethod.KALMAN_FILTER

    def test_clear_history(self, predictor):
        """测试清除历史"""
        base_time = time.time()
        for i in range(10):
            predictor.add_observation('sensor1', base_time + i, float(i))

        predictor.clear_history('sensor1')
        result = predictor.predict('sensor1', 5.0)
        assert result is None

    def test_clear_all_history(self, predictor):
        """测试清除所有历史"""
        base_time = time.time()
        for i in range(10):
            predictor.add_observation('sensor1', base_time + i, float(i))
            predictor.add_observation('sensor2', base_time + i, float(i * 2))

        predictor.clear_history()

        assert predictor.predict('sensor1', 5.0) is None
        assert predictor.predict('sensor2', 5.0) is None

    def test_prediction_accuracy_estimate(self, predictor):
        """测试预测精度估计"""
        base_time = time.time()
        for i in range(20):
            predictor.add_observation('sensor1', base_time + i, 50.0)  # 稳定值

        result = predictor.predict('sensor1', 5.0)
        assert result is not None
        assert 0.0 <= result.accuracy_estimate <= 1.0


class TestVirtualSensorSynthesizer:
    """虚拟传感器合成器测试"""

    @pytest.fixture
    def synthesizer(self):
        """创建合成器"""
        return VirtualSensorSynthesizer()

    def test_create_synthesizer(self, synthesizer):
        """测试创建合成器"""
        assert synthesizer is not None

    def test_register_virtual_sensor(self, synthesizer):
        """测试注册虚拟传感器"""
        config = VirtualSensorConfig(
            sensor_id='virtual_1',
            name='Virtual Sensor 1',
            unit='m³/s',
            source_sensors=['sensor1', 'sensor2'],
            fusion_method='weighted_average'
        )

        synthesizer.register_virtual_sensor(config)
        assert 'virtual_1' in synthesizer.list_virtual_sensors()

    def test_unregister_virtual_sensor(self, synthesizer):
        """测试注销虚拟传感器"""
        config = VirtualSensorConfig(
            sensor_id='virtual_1',
            name='Virtual Sensor 1',
            unit='m³/s',
            source_sensors=['sensor1'],
            fusion_method='weighted_average'
        )

        synthesizer.register_virtual_sensor(config)
        synthesizer.unregister_virtual_sensor('virtual_1')
        assert 'virtual_1' not in synthesizer.list_virtual_sensors()

    def test_update_weighted_average(self, synthesizer):
        """测试加权平均更新"""
        config = VirtualSensorConfig(
            sensor_id='virtual_1',
            name='Virtual Sensor 1',
            unit='unit',
            source_sensors=['s1', 's2'],
            fusion_method='weighted_average',
            weights=[0.6, 0.4]
        )
        synthesizer.register_virtual_sensor(config)

        source_values = {'s1': 100.0, 's2': 50.0}
        synthesizer.update(source_values)

        value = synthesizer.get_value('virtual_1')
        expected = 0.6 * 100.0 + 0.4 * 50.0
        assert abs(value - expected) < 0.01

    def test_update_kalman_fusion(self, synthesizer):
        """测试卡尔曼融合"""
        config = VirtualSensorConfig(
            sensor_id='virtual_1',
            name='Virtual Sensor 1',
            unit='unit',
            source_sensors=['s1', 's2'],
            fusion_method='kalman'
        )
        synthesizer.register_virtual_sensor(config)

        for _ in range(10):
            source_values = {'s1': 50.0, 's2': 52.0}
            synthesizer.update(source_values)

        value = synthesizer.get_value('virtual_1')
        assert 49.0 <= value <= 53.0

    def test_update_formula(self, synthesizer):
        """测试公式计算"""
        config = VirtualSensorConfig(
            sensor_id='virtual_1',
            name='Virtual Sensor 1',
            unit='unit',
            source_sensors=['a', 'b'],
            fusion_method='formula',
            formula='a + b * 2'
        )
        synthesizer.register_virtual_sensor(config)

        source_values = {'a': 10.0, 'b': 5.0}
        synthesizer.update(source_values)

        value = synthesizer.get_value('virtual_1')
        assert value == 20.0

    def test_update_max(self, synthesizer):
        """测试最大值融合"""
        config = VirtualSensorConfig(
            sensor_id='virtual_1',
            name='Virtual Sensor 1',
            unit='unit',
            source_sensors=['s1', 's2', 's3'],
            fusion_method='max'
        )
        synthesizer.register_virtual_sensor(config)

        source_values = {'s1': 10.0, 's2': 30.0, 's3': 20.0}
        synthesizer.update(source_values)

        value = synthesizer.get_value('virtual_1')
        assert value == 30.0

    def test_update_min(self, synthesizer):
        """测试最小值融合"""
        config = VirtualSensorConfig(
            sensor_id='virtual_1',
            name='Virtual Sensor 1',
            unit='unit',
            source_sensors=['s1', 's2', 's3'],
            fusion_method='min'
        )
        synthesizer.register_virtual_sensor(config)

        source_values = {'s1': 10.0, 's2': 30.0, 's3': 20.0}
        synthesizer.update(source_values)

        value = synthesizer.get_value('virtual_1')
        assert value == 10.0

    def test_update_median(self, synthesizer):
        """测试中位数融合"""
        config = VirtualSensorConfig(
            sensor_id='virtual_1',
            name='Virtual Sensor 1',
            unit='unit',
            source_sensors=['s1', 's2', 's3'],
            fusion_method='median'
        )
        synthesizer.register_virtual_sensor(config)

        source_values = {'s1': 10.0, 's2': 30.0, 's3': 20.0}
        synthesizer.update(source_values)

        value = synthesizer.get_value('virtual_1')
        assert value == 20.0

    def test_get_all_values(self, synthesizer):
        """测试获取所有值"""
        for i in range(3):
            config = VirtualSensorConfig(
                sensor_id=f'virtual_{i}',
                name=f'Virtual {i}',
                unit='unit',
                source_sensors=['s1'],
                fusion_method='weighted_average'
            )
            synthesizer.register_virtual_sensor(config)

        synthesizer.update({'s1': 50.0})
        values = synthesizer.get_all_values()

        assert len(values) == 3

    def test_get_config(self, synthesizer):
        """测试获取配置"""
        config = VirtualSensorConfig(
            sensor_id='virtual_1',
            name='Virtual Sensor 1',
            unit='unit',
            source_sensors=['s1'],
            fusion_method='weighted_average'
        )
        synthesizer.register_virtual_sensor(config)

        retrieved = synthesizer.get_config('virtual_1')
        assert retrieved is not None
        assert retrieved.sensor_id == 'virtual_1'

    def test_quality_weighted_average(self, synthesizer):
        """测试质量加权平均"""
        config = VirtualSensorConfig(
            sensor_id='virtual_1',
            name='Virtual Sensor 1',
            unit='unit',
            source_sensors=['s1', 's2'],
            fusion_method='weighted_average'
        )
        synthesizer.register_virtual_sensor(config)

        source_values = {'s1': 100.0, 's2': 0.0}
        source_qualities = {'s1': 1.0, 's2': 0.0}  # s2 质量为0

        synthesizer.update(source_values, source_qualities)
        value = synthesizer.get_value('virtual_1')

        # 应该更接近 s1 的值
        assert value > 50.0


class TestSensorHealthMonitor:
    """传感器健康监测器测试"""

    @pytest.fixture
    def monitor(self):
        """创建监测器"""
        return SensorHealthMonitor(window_size=50)

    def test_create_monitor(self, monitor):
        """测试创建监测器"""
        assert monitor is not None

    def test_add_reading(self, monitor):
        """测试添加读数"""
        reading = SensorReading(
            sensor_id='sensor1',
            timestamp=datetime.now(),
            value=50.0,
            raw_value=50.0,
            quality=1.0,
            is_valid=True
        )
        monitor.add_reading(reading)
        # 不应抛出异常

    def test_evaluate_health_insufficient_data(self, monitor):
        """测试数据不足时的健康评估"""
        reading = SensorReading(
            sensor_id='sensor1',
            timestamp=datetime.now(),
            value=50.0,
            raw_value=50.0,
            quality=1.0,
            is_valid=True
        )
        monitor.add_reading(reading)

        report = monitor.evaluate_health('sensor1')
        assert report is None

    def test_evaluate_health_excellent(self, monitor):
        """测试优秀健康状态评估"""
        for i in range(20):
            reading = SensorReading(
                sensor_id='sensor1',
                timestamp=datetime.now(),
                value=50.0 + np.random.normal(0, 0.1),
                raw_value=50.0,
                quality=1.0,
                is_valid=True
            )
            monitor.add_reading(reading)

        report = monitor.evaluate_health('sensor1')
        assert report is not None
        assert report.health in [SensorHealth.EXCELLENT, SensorHealth.GOOD]
        assert report.health_score >= 75

    def test_evaluate_health_with_failures(self, monitor):
        """测试有故障时的健康评估"""
        for i in range(20):
            reading = SensorReading(
                sensor_id='sensor1',
                timestamp=datetime.now(),
                value=50.0,
                raw_value=50.0,
                quality=0.5,
                is_valid=True,
                failure_mode=FailureMode.DRIFT if i % 3 == 0 else FailureMode.NONE
            )
            monitor.add_reading(reading)

        report = monitor.evaluate_health('sensor1')
        assert report is not None
        assert report.health_score < 90
        assert len(report.issues) > 0

    def test_evaluate_health_with_invalid_readings(self, monitor):
        """测试有无效读数时的健康评估"""
        for i in range(20):
            reading = SensorReading(
                sensor_id='sensor1',
                timestamp=datetime.now(),
                value=50.0 if i % 4 != 0 else float('nan'),
                raw_value=50.0,
                quality=0.8,
                is_valid=i % 4 != 0
            )
            monitor.add_reading(reading)

        report = monitor.evaluate_health('sensor1')
        assert report is not None
        assert report.health_score < 100

    def test_health_report_contains_metrics(self, monitor):
        """测试健康报告包含指标"""
        for i in range(20):
            reading = SensorReading(
                sensor_id='sensor1',
                timestamp=datetime.now(),
                value=50.0 + i * 0.1,
                raw_value=50.0,
                quality=0.9,
                is_valid=True
            )
            monitor.add_reading(reading)

        report = monitor.evaluate_health('sensor1')
        assert report is not None
        assert 'data_completeness' in report.metrics
        assert 'avg_quality' in report.metrics
        assert 'failure_rate' in report.metrics

    def test_health_trend(self, monitor):
        """测试健康趋势"""
        for i in range(30):
            reading = SensorReading(
                sensor_id='sensor1',
                timestamp=datetime.now(),
                value=50.0,
                raw_value=50.0,
                quality=1.0,
                is_valid=True
            )
            monitor.add_reading(reading)
            monitor.evaluate_health('sensor1')

        report = monitor.evaluate_health('sensor1')
        assert report is not None
        assert report.trend in ['improving', 'stable', 'degrading']

    def test_get_health_history(self, monitor):
        """测试获取健康历史"""
        for i in range(30):
            reading = SensorReading(
                sensor_id='sensor1',
                timestamp=datetime.now(),
                value=50.0,
                raw_value=50.0,
                quality=1.0,
                is_valid=True
            )
            monitor.add_reading(reading)
            monitor.evaluate_health('sensor1')

        history = monitor.get_health_history('sensor1', 10)
        assert len(history) <= 10


class TestSensorPredictionManager:
    """传感器仿真与状态预测管理器测试"""

    @pytest.fixture
    def manager(self):
        """创建管理器"""
        return SensorPredictionManager({
            'prediction_method': PredictionMethod.ENSEMBLE.value,
            'history_size': 50,
            'update_interval_ms': 100
        })

    @pytest.fixture
    def sensor_config(self):
        """创建传感器配置"""
        return SensorConfig(
            sensor_id='flow_1',
            sensor_type=SensorType.FLOW_METER,
            name='Flow Meter 1',
            unit='m³/s',
            min_value=0.0,
            max_value=100.0,
            noise_level=0.01
        )

    def test_create_manager(self, manager):
        """测试创建管理器"""
        assert manager is not None

    def test_register_sensor(self, manager, sensor_config):
        """测试注册传感器"""
        manager.register_sensor(sensor_config)
        assert 'flow_1' in manager.list_sensors()

    def test_unregister_sensor(self, manager, sensor_config):
        """测试注销传感器"""
        manager.register_sensor(sensor_config)
        manager.unregister_sensor('flow_1')
        assert 'flow_1' not in manager.list_sensors()

    def test_simulate_reading(self, manager, sensor_config):
        """测试仿真读数"""
        manager.register_sensor(sensor_config)
        reading = manager.simulate_reading('flow_1', 50.0)

        assert reading is not None
        assert reading.sensor_id == 'flow_1'
        assert 0.0 <= reading.value <= 100.0

    def test_simulate_all(self, manager):
        """测试仿真所有传感器"""
        for i in range(3):
            config = SensorConfig(
                sensor_id=f'sensor_{i}',
                sensor_type=SensorType.GENERIC,
                name=f'Sensor {i}',
                unit='unit',
                min_value=0.0,
                max_value=100.0
            )
            manager.register_sensor(config)

        readings = manager.simulate_all({'sensor_0': 30.0, 'sensor_1': 50.0, 'sensor_2': 70.0})
        assert len(readings) == 3

    def test_predict(self, manager, sensor_config):
        """测试预测"""
        manager.register_sensor(sensor_config)

        # 生成足够的历史数据
        for i in range(20):
            manager.simulate_reading('flow_1', 50.0 + i * 0.5)

        result = manager.predict('flow_1', 5.0, 5)
        assert result is not None
        assert len(result.predicted_values) == 5

    def test_predict_all(self, manager):
        """测试预测所有传感器"""
        for i in range(2):
            config = SensorConfig(
                sensor_id=f'sensor_{i}',
                sensor_type=SensorType.GENERIC,
                name=f'Sensor {i}',
                unit='unit',
                min_value=0.0,
                max_value=100.0
            )
            manager.register_sensor(config)

        for _ in range(20):
            manager.simulate_all()

        results = manager.predict_all(5.0)
        assert len(results) == 2

    def test_register_virtual_sensor(self, manager):
        """测试注册虚拟传感器"""
        # 先注册物理传感器
        for i in range(2):
            config = SensorConfig(
                sensor_id=f'sensor_{i}',
                sensor_type=SensorType.GENERIC,
                name=f'Sensor {i}',
                unit='unit',
                min_value=0.0,
                max_value=100.0
            )
            manager.register_sensor(config)

        # 注册虚拟传感器
        virtual_config = VirtualSensorConfig(
            sensor_id='virtual_1',
            name='Virtual 1',
            unit='unit',
            source_sensors=['sensor_0', 'sensor_1'],
            fusion_method='weighted_average'
        )
        manager.register_virtual_sensor(virtual_config)

        assert 'virtual_1' in manager.list_virtual_sensors()

    def test_get_virtual_sensor_value(self, manager):
        """测试获取虚拟传感器值"""
        for i in range(2):
            config = SensorConfig(
                sensor_id=f'sensor_{i}',
                sensor_type=SensorType.GENERIC,
                name=f'Sensor {i}',
                unit='unit',
                min_value=0.0,
                max_value=100.0
            )
            manager.register_sensor(config)

        virtual_config = VirtualSensorConfig(
            sensor_id='virtual_1',
            name='Virtual 1',
            unit='unit',
            source_sensors=['sensor_0', 'sensor_1'],
            fusion_method='weighted_average'
        )
        manager.register_virtual_sensor(virtual_config)

        manager.simulate_all({'sensor_0': 40.0, 'sensor_1': 60.0})
        value = manager.get_virtual_sensor_value('virtual_1')

        assert value is not None
        assert 40.0 <= value <= 60.0

    def test_get_health_report(self, manager, sensor_config):
        """测试获取健康报告"""
        manager.register_sensor(sensor_config)

        for _ in range(20):
            manager.simulate_reading('flow_1', 50.0)

        report = manager.get_health_report('flow_1')
        assert report is not None
        assert isinstance(report.health, SensorHealth)

    def test_inject_and_clear_failure(self, manager, sensor_config):
        """测试注入和清除故障"""
        manager.register_sensor(sensor_config)

        success = manager.inject_failure('flow_1', FailureMode.STUCK)
        assert success

        state = manager.get_sensor_state('flow_1')
        assert state.failure_mode == FailureMode.STUCK

        manager.clear_failure('flow_1')
        state = manager.get_sensor_state('flow_1')
        assert state.failure_mode == FailureMode.NONE

    def test_calibrate_sensor(self, manager, sensor_config):
        """测试校准传感器"""
        manager.register_sensor(sensor_config)

        for _ in range(10):
            manager.simulate_reading('flow_1', 50.0)

        success = manager.calibrate_sensor('flow_1')
        assert success

    def test_get_sensor_state(self, manager, sensor_config):
        """测试获取传感器状态"""
        manager.register_sensor(sensor_config)
        manager.simulate_reading('flow_1', 50.0)

        state = manager.get_sensor_state('flow_1')
        assert state is not None
        assert state.sensor_id == 'flow_1'

    def test_get_all_sensor_states(self, manager):
        """测试获取所有传感器状态"""
        for i in range(3):
            config = SensorConfig(
                sensor_id=f'sensor_{i}',
                sensor_type=SensorType.GENERIC,
                name=f'Sensor {i}',
                unit='unit',
                min_value=0.0,
                max_value=100.0
            )
            manager.register_sensor(config)

        manager.simulate_all()
        states = manager.get_all_sensor_states()

        assert len(states) == 3

    def test_set_prediction_method(self, manager):
        """测试设置预测方法"""
        manager.set_prediction_method(PredictionMethod.KALMAN_FILTER)
        # 应该不抛出异常

    def test_get_status(self, manager, sensor_config):
        """测试获取状态"""
        manager.register_sensor(sensor_config)

        status = manager.get_status()
        assert 'running' in status
        assert 'sensor_count' in status
        assert status['sensor_count'] == 1

    def test_start_stop(self, manager, sensor_config):
        """测试启动停止"""
        manager.register_sensor(sensor_config)

        manager.start()
        assert manager._running

        time.sleep(0.2)
        manager.stop()
        assert not manager._running

    def test_background_update(self, manager, sensor_config):
        """测试后台更新"""
        manager.register_sensor(sensor_config)
        manager._simulators['flow_1'].set_true_value(50.0)

        manager.start()
        time.sleep(0.3)
        manager.stop()

        # 检查历史是否有更新
        history = manager._simulators['flow_1'].get_reading_history(10)
        assert len(history) > 0


class TestIntegration:
    """集成测试"""

    def test_full_workflow(self):
        """测试完整工作流程"""
        # 创建管理器
        manager = SensorPredictionManager({
            'prediction_method': PredictionMethod.ENSEMBLE.value
        })

        # 注册物理传感器
        sensors = [
            SensorConfig(
                sensor_id='upstream_flow',
                sensor_type=SensorType.FLOW_METER,
                name='Upstream Flow',
                unit='m³/s',
                min_value=0.0,
                max_value=50.0,
                noise_level=0.02
            ),
            SensorConfig(
                sensor_id='downstream_flow',
                sensor_type=SensorType.FLOW_METER,
                name='Downstream Flow',
                unit='m³/s',
                min_value=0.0,
                max_value=50.0,
                noise_level=0.02
            ),
            SensorConfig(
                sensor_id='pressure_1',
                sensor_type=SensorType.PRESSURE_SENSOR,
                name='Pressure 1',
                unit='kPa',
                min_value=0.0,
                max_value=500.0,
                noise_level=0.01
            )
        ]

        for config in sensors:
            manager.register_sensor(config)

        # 注册虚拟传感器
        virtual_config = VirtualSensorConfig(
            sensor_id='avg_flow',
            name='Average Flow',
            unit='m³/s',
            source_sensors=['upstream_flow', 'downstream_flow'],
            fusion_method='weighted_average',
            weights=[0.5, 0.5]
        )
        manager.register_virtual_sensor(virtual_config)

        # 仿真一段时间
        for i in range(50):
            true_values = {
                'upstream_flow': 20.0 + np.sin(i * 0.1) * 5,
                'downstream_flow': 19.0 + np.sin(i * 0.1) * 5,
                'pressure_1': 200.0 + i * 0.5
            }
            manager.simulate_all(true_values)

        # 预测
        predictions = manager.predict_all(10.0)
        assert len(predictions) == 3

        # 健康报告
        reports = manager.get_all_health_reports()
        assert len(reports) == 3
        for report in reports.values():
            assert report.health in [SensorHealth.EXCELLENT, SensorHealth.GOOD]

        # 虚拟传感器值
        avg_flow = manager.get_virtual_sensor_value('avg_flow')
        assert avg_flow is not None

        # 系统状态
        status = manager.get_status()
        assert status['sensor_count'] == 3
        assert status['virtual_sensor_count'] == 1

    def test_failure_injection_workflow(self):
        """测试故障注入工作流程"""
        manager = SensorPredictionManager()

        config = SensorConfig(
            sensor_id='sensor1',
            sensor_type=SensorType.GENERIC,
            name='Sensor 1',
            unit='unit',
            min_value=0.0,
            max_value=100.0
        )
        manager.register_sensor(config)

        # 正常运行
        for _ in range(20):
            manager.simulate_reading('sensor1', 50.0)

        report_before = manager.get_health_report('sensor1')
        assert report_before is not None

        # 注入故障
        manager.inject_failure('sensor1', FailureMode.DRIFT)

        for _ in range(20):
            manager.simulate_reading('sensor1', 50.0)

        report_after = manager.get_health_report('sensor1')
        assert report_after is not None

        # 健康应该下降
        assert report_after.health_score <= report_before.health_score

        # 清除故障并校准
        manager.clear_failure('sensor1')
        manager.calibrate_sensor('sensor1')

    def test_prediction_accuracy(self):
        """测试预测精度"""
        manager = SensorPredictionManager({
            'prediction_method': PredictionMethod.LINEAR.value
        })

        config = SensorConfig(
            sensor_id='sensor1',
            sensor_type=SensorType.GENERIC,
            name='Sensor 1',
            unit='unit',
            min_value=0.0,
            max_value=100.0,
            noise_level=0.001  # 低噪声
        )
        manager.register_sensor(config)

        # 生成线性增长数据
        for i in range(30):
            manager.simulate_reading('sensor1', float(i))

        # 预测
        result = manager.predict('sensor1', 5.0, num_points=5)
        assert result is not None

        # 预测值应该继续增长
        assert result.predicted_values[-1] > result.predicted_values[0]

    def test_multi_sensor_virtual_fusion(self):
        """测试多传感器虚拟融合"""
        manager = SensorPredictionManager()

        # 注册多个冗余传感器
        for i in range(4):
            config = SensorConfig(
                sensor_id=f'redundant_{i}',
                sensor_type=SensorType.FLOW_METER,
                name=f'Redundant {i}',
                unit='m³/s',
                min_value=0.0,
                max_value=100.0,
                noise_level=0.05
            )
            manager.register_sensor(config)

        # 使用中位数融合（抗干扰）
        virtual_config = VirtualSensorConfig(
            sensor_id='fused_flow',
            name='Fused Flow',
            unit='m³/s',
            source_sensors=['redundant_0', 'redundant_1', 'redundant_2', 'redundant_3'],
            fusion_method='median'
        )
        manager.register_virtual_sensor(virtual_config)

        # 仿真，其中一个传感器有异常
        manager.inject_failure('redundant_2', FailureMode.BIAS)

        for _ in range(10):
            true_values = {f'redundant_{i}': 50.0 for i in range(4)}
            manager.simulate_all(true_values)

        # 融合值应该接近真实值
        fused = manager.get_virtual_sensor_value('fused_flow')
        assert fused is not None
        assert 40.0 <= fused <= 60.0
