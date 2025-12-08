# -*- coding: utf-8 -*-
"""
传感器仿真与状态预测核心模块

实现传感器仿真、状态预测、虚拟传感器合成和健康监测。
"""

import threading
import time
import math
import random
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple, Deque
from collections import deque
from datetime import datetime
import numpy as np


class SensorType(Enum):
    """传感器类型"""
    FLOW_METER = auto()          # 流量计
    PRESSURE_SENSOR = auto()     # 压力传感器
    LEVEL_SENSOR = auto()        # 液位传感器
    VELOCITY_SENSOR = auto()     # 流速传感器
    TEMPERATURE_SENSOR = auto()  # 温度传感器
    TURBIDITY_SENSOR = auto()    # 浊度传感器
    PH_SENSOR = auto()           # pH传感器
    CONDUCTIVITY_SENSOR = auto() # 电导率传感器
    GENERIC = auto()             # 通用传感器


class NoiseModel(Enum):
    """噪声模型"""
    GAUSSIAN = auto()      # 高斯白噪声
    UNIFORM = auto()       # 均匀分布噪声
    PINK = auto()          # 粉红噪声 (1/f)
    BROWN = auto()         # 布朗噪声
    IMPULSE = auto()       # 脉冲噪声
    QUANTIZATION = auto()  # 量化噪声


class FailureMode(Enum):
    """故障模式"""
    NONE = auto()           # 无故障
    STUCK = auto()          # 卡死
    DRIFT = auto()          # 漂移
    BIAS = auto()           # 偏置
    NOISE_INCREASE = auto() # 噪声增大
    INTERMITTENT = auto()   # 间歇性故障
    COMPLETE_FAILURE = auto() # 完全失效
    SPIKE = auto()          # 尖峰干扰


class PredictionMethod(Enum):
    """预测方法"""
    LINEAR = auto()           # 线性外推
    POLYNOMIAL = auto()       # 多项式拟合
    EXPONENTIAL_SMOOTHING = auto()  # 指数平滑
    KALMAN_FILTER = auto()    # 卡尔曼滤波
    ARIMA = auto()            # 自回归移动平均
    NEURAL_NETWORK = auto()   # 神经网络
    ENSEMBLE = auto()         # 集成方法


class SensorHealth(Enum):
    """传感器健康状态"""
    EXCELLENT = auto()   # 优秀
    GOOD = auto()        # 良好
    FAIR = auto()        # 一般
    POOR = auto()        # 较差
    CRITICAL = auto()    # 危险
    FAILED = auto()      # 失效


@dataclass
class SensorConfig:
    """传感器配置"""
    sensor_id: str
    sensor_type: SensorType
    name: str
    unit: str
    min_value: float
    max_value: float
    resolution: float = 0.01
    accuracy: float = 0.01  # 精度百分比
    response_time_ms: float = 100.0
    noise_model: NoiseModel = NoiseModel.GAUSSIAN
    noise_level: float = 0.01  # 噪声水平
    drift_rate: float = 0.0  # 漂移率 (每小时)
    calibration_interval_hours: float = 720.0  # 校准间隔
    failure_probability: float = 0.0001  # 故障概率


@dataclass
class SensorState:
    """传感器状态"""
    sensor_id: str
    true_value: float
    measured_value: float
    noise: float
    bias: float
    drift: float
    failure_mode: FailureMode = FailureMode.NONE
    last_calibration: datetime = field(default_factory=datetime.now)
    operating_hours: float = 0.0
    health: SensorHealth = SensorHealth.EXCELLENT


@dataclass
class SensorReading:
    """传感器读数"""
    sensor_id: str
    timestamp: datetime
    value: float
    raw_value: float
    quality: float  # 0-1 质量指标
    is_valid: bool
    failure_mode: FailureMode = FailureMode.NONE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """预测结果"""
    sensor_id: str
    timestamp: datetime
    prediction_horizon_seconds: float
    predicted_values: List[float]
    timestamps: List[datetime]
    confidence_intervals: List[Tuple[float, float]]
    method: PredictionMethod
    accuracy_estimate: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VirtualSensorConfig:
    """虚拟传感器配置"""
    sensor_id: str
    name: str
    unit: str
    source_sensors: List[str]
    fusion_method: str  # 'weighted_average', 'kalman', 'neural', 'formula'
    formula: Optional[str] = None  # 计算公式
    weights: Optional[List[float]] = None
    update_interval_ms: float = 100.0


@dataclass
class HealthReport:
    """健康报告"""
    sensor_id: str
    timestamp: datetime
    health: SensorHealth
    health_score: float  # 0-100
    issues: List[str]
    recommendations: List[str]
    metrics: Dict[str, float]
    trend: str  # 'improving', 'stable', 'degrading'


class SensorSimulator:
    """传感器仿真器"""

    def __init__(self, config: SensorConfig):
        self.config = config
        self._state = SensorState(
            sensor_id=config.sensor_id,
            true_value=0.0,
            measured_value=0.0,
            noise=0.0,
            bias=0.0,
            drift=0.0
        )
        self._lock = threading.RLock()
        self._start_time = time.time()
        self._last_update = time.time()
        self._pink_noise_state = 0.0
        self._brown_noise_state = 0.0
        self._reading_history: Deque[SensorReading] = deque(maxlen=1000)
        self._failure_start_time: Optional[float] = None
        self._stuck_value: Optional[float] = None

    def set_true_value(self, value: float) -> None:
        """设置真实值"""
        with self._lock:
            self._state.true_value = np.clip(
                value, self.config.min_value, self.config.max_value
            )

    def get_true_value(self) -> float:
        """获取真实值"""
        with self._lock:
            return self._state.true_value

    def simulate_reading(self, true_value: Optional[float] = None) -> SensorReading:
        """仿真传感器读数"""
        with self._lock:
            current_time = time.time()
            elapsed = current_time - self._last_update
            self._last_update = current_time

            # 更新运行时间
            self._state.operating_hours += elapsed / 3600.0

            # 设置真实值
            if true_value is not None:
                self._state.true_value = np.clip(
                    true_value, self.config.min_value, self.config.max_value
                )

            # 检查并模拟故障
            self._check_failure()

            # 计算测量值
            measured = self._calculate_measurement()

            # 创建读数
            reading = SensorReading(
                sensor_id=self.config.sensor_id,
                timestamp=datetime.now(),
                value=measured,
                raw_value=self._state.true_value,
                quality=self._calculate_quality(),
                is_valid=self._state.failure_mode not in [
                    FailureMode.COMPLETE_FAILURE,
                    FailureMode.STUCK
                ],
                failure_mode=self._state.failure_mode,
                metadata={
                    'noise': self._state.noise,
                    'bias': self._state.bias,
                    'drift': self._state.drift
                }
            )

            self._reading_history.append(reading)
            return reading

    def _calculate_measurement(self) -> float:
        """计算测量值"""
        true_val = self._state.true_value

        # 处理故障模式
        if self._state.failure_mode == FailureMode.COMPLETE_FAILURE:
            return float('nan')

        if self._state.failure_mode == FailureMode.STUCK:
            if self._stuck_value is None:
                self._stuck_value = self._state.measured_value
            return self._stuck_value

        if self._state.failure_mode == FailureMode.INTERMITTENT:
            if random.random() < 0.3:
                return float('nan')

        # 计算漂移
        hours_since_cal = (datetime.now() - self._state.last_calibration).total_seconds() / 3600
        self._state.drift = self.config.drift_rate * hours_since_cal

        # 计算偏置
        if self._state.failure_mode == FailureMode.BIAS:
            self._state.bias = self.config.accuracy * (self.config.max_value - self.config.min_value) * 5
        elif self._state.failure_mode == FailureMode.DRIFT:
            self._state.drift *= 3  # 加速漂移

        # 计算噪声
        noise_level = self.config.noise_level
        if self._state.failure_mode == FailureMode.NOISE_INCREASE:
            noise_level *= 5

        self._state.noise = self._generate_noise(noise_level)

        # 添加尖峰
        if self._state.failure_mode == FailureMode.SPIKE:
            if random.random() < 0.1:
                spike = random.choice([-1, 1]) * (self.config.max_value - self.config.min_value) * 0.3
                self._state.noise += spike

        # 组合测量值
        measured = true_val + self._state.noise + self._state.bias + self._state.drift

        # 应用量化
        measured = round(measured / self.config.resolution) * self.config.resolution

        # 限幅
        measured = np.clip(measured, self.config.min_value, self.config.max_value)

        self._state.measured_value = measured
        return measured

    def _generate_noise(self, level: float) -> float:
        """生成噪声"""
        range_val = self.config.max_value - self.config.min_value

        if self.config.noise_model == NoiseModel.GAUSSIAN:
            return random.gauss(0, level * range_val)

        elif self.config.noise_model == NoiseModel.UNIFORM:
            return random.uniform(-level * range_val, level * range_val)

        elif self.config.noise_model == NoiseModel.PINK:
            # 1/f 噪声近似
            white = random.gauss(0, level * range_val)
            self._pink_noise_state = 0.99 * self._pink_noise_state + 0.01 * white
            return self._pink_noise_state

        elif self.config.noise_model == NoiseModel.BROWN:
            # 积分白噪声
            white = random.gauss(0, level * range_val * 0.1)
            self._brown_noise_state += white
            self._brown_noise_state *= 0.99  # 衰减防止发散
            return self._brown_noise_state

        elif self.config.noise_model == NoiseModel.IMPULSE:
            if random.random() < 0.05:
                return random.choice([-1, 1]) * level * range_val * 3
            return 0.0

        elif self.config.noise_model == NoiseModel.QUANTIZATION:
            return random.uniform(-0.5, 0.5) * self.config.resolution

        return 0.0

    def _check_failure(self) -> None:
        """检查并模拟故障"""
        if self._state.failure_mode != FailureMode.NONE:
            return

        if random.random() < self.config.failure_probability:
            # 随机选择故障模式
            failure_modes = [
                FailureMode.STUCK,
                FailureMode.DRIFT,
                FailureMode.BIAS,
                FailureMode.NOISE_INCREASE,
                FailureMode.INTERMITTENT,
                FailureMode.SPIKE
            ]
            self._state.failure_mode = random.choice(failure_modes)
            self._failure_start_time = time.time()

    def _calculate_quality(self) -> float:
        """计算读数质量"""
        quality = 1.0

        # 故障降低质量
        failure_penalties = {
            FailureMode.NONE: 0.0,
            FailureMode.STUCK: 0.8,
            FailureMode.DRIFT: 0.3,
            FailureMode.BIAS: 0.4,
            FailureMode.NOISE_INCREASE: 0.3,
            FailureMode.INTERMITTENT: 0.5,
            FailureMode.COMPLETE_FAILURE: 1.0,
            FailureMode.SPIKE: 0.2
        }
        quality -= failure_penalties.get(self._state.failure_mode, 0.0)

        # 噪声降低质量
        range_val = self.config.max_value - self.config.min_value
        if range_val > 0:
            noise_ratio = abs(self._state.noise) / range_val
            quality -= min(0.3, noise_ratio * 3)

        # 漂移降低质量
        if range_val > 0:
            drift_ratio = abs(self._state.drift) / range_val
            quality -= min(0.2, drift_ratio * 2)

        return max(0.0, min(1.0, quality))

    def inject_failure(self, failure_mode: FailureMode) -> None:
        """注入故障"""
        with self._lock:
            self._state.failure_mode = failure_mode
            self._failure_start_time = time.time()
            if failure_mode == FailureMode.STUCK:
                self._stuck_value = self._state.measured_value

    def clear_failure(self) -> None:
        """清除故障"""
        with self._lock:
            self._state.failure_mode = FailureMode.NONE
            self._failure_start_time = None
            self._stuck_value = None

    def calibrate(self) -> None:
        """校准传感器"""
        with self._lock:
            self._state.last_calibration = datetime.now()
            self._state.bias = 0.0
            self._state.drift = 0.0

    def get_state(self) -> SensorState:
        """获取传感器状态"""
        with self._lock:
            return SensorState(
                sensor_id=self._state.sensor_id,
                true_value=self._state.true_value,
                measured_value=self._state.measured_value,
                noise=self._state.noise,
                bias=self._state.bias,
                drift=self._state.drift,
                failure_mode=self._state.failure_mode,
                last_calibration=self._state.last_calibration,
                operating_hours=self._state.operating_hours,
                health=self._evaluate_health()
            )

    def _evaluate_health(self) -> SensorHealth:
        """评估传感器健康状态"""
        if self._state.failure_mode == FailureMode.COMPLETE_FAILURE:
            return SensorHealth.FAILED
        if self._state.failure_mode in [FailureMode.STUCK, FailureMode.INTERMITTENT]:
            return SensorHealth.CRITICAL

        # 基于漂移和偏置评估
        range_val = self.config.max_value - self.config.min_value
        if range_val > 0:
            error_ratio = (abs(self._state.drift) + abs(self._state.bias)) / range_val
            if error_ratio < 0.01:
                return SensorHealth.EXCELLENT
            elif error_ratio < 0.03:
                return SensorHealth.GOOD
            elif error_ratio < 0.05:
                return SensorHealth.FAIR
            elif error_ratio < 0.1:
                return SensorHealth.POOR
            else:
                return SensorHealth.CRITICAL

        return SensorHealth.GOOD

    def get_reading_history(self, count: int = 100) -> List[SensorReading]:
        """获取历史读数"""
        with self._lock:
            return list(self._reading_history)[-count:]


class StatePredictor:
    """状态预测器"""

    def __init__(
        self,
        method: PredictionMethod = PredictionMethod.EXPONENTIAL_SMOOTHING,
        history_size: int = 100
    ):
        self.method = method
        self.history_size = history_size
        self._histories: Dict[str, Deque[Tuple[float, float]]] = {}  # sensor_id -> [(timestamp, value)]
        self._lock = threading.RLock()

        # Kalman滤波器状态
        self._kalman_states: Dict[str, Dict[str, float]] = {}

        # 指数平滑状态
        self._ema_states: Dict[str, float] = {}
        self._ema_alpha = 0.3

    def add_observation(self, sensor_id: str, timestamp: float, value: float) -> None:
        """添加观测值"""
        with self._lock:
            if sensor_id not in self._histories:
                self._histories[sensor_id] = deque(maxlen=self.history_size)
            self._histories[sensor_id].append((timestamp, value))

            # 更新EMA
            if sensor_id not in self._ema_states:
                self._ema_states[sensor_id] = value
            else:
                self._ema_states[sensor_id] = (
                    self._ema_alpha * value +
                    (1 - self._ema_alpha) * self._ema_states[sensor_id]
                )

    def predict(
        self,
        sensor_id: str,
        horizon_seconds: float,
        num_points: int = 10
    ) -> Optional[PredictionResult]:
        """预测未来状态"""
        with self._lock:
            if sensor_id not in self._histories:
                return None

            history = list(self._histories[sensor_id])
            if len(history) < 3:
                return None

            timestamps = [h[0] for h in history]
            values = [h[1] for h in history]

            # 生成预测时间点
            current_time = timestamps[-1]
            dt = horizon_seconds / num_points
            future_times = [current_time + dt * (i + 1) for i in range(num_points)]

            # 根据方法进行预测
            if self.method == PredictionMethod.LINEAR:
                predicted, confidence = self._predict_linear(timestamps, values, future_times)
            elif self.method == PredictionMethod.POLYNOMIAL:
                predicted, confidence = self._predict_polynomial(timestamps, values, future_times)
            elif self.method == PredictionMethod.EXPONENTIAL_SMOOTHING:
                predicted, confidence = self._predict_ema(sensor_id, values, future_times, current_time)
            elif self.method == PredictionMethod.KALMAN_FILTER:
                predicted, confidence = self._predict_kalman(sensor_id, timestamps, values, future_times)
            elif self.method == PredictionMethod.ENSEMBLE:
                predicted, confidence = self._predict_ensemble(sensor_id, timestamps, values, future_times)
            else:
                predicted, confidence = self._predict_linear(timestamps, values, future_times)

            # 计算精度估计
            accuracy = self._estimate_accuracy(values, predicted)

            return PredictionResult(
                sensor_id=sensor_id,
                timestamp=datetime.now(),
                prediction_horizon_seconds=horizon_seconds,
                predicted_values=predicted,
                timestamps=[datetime.fromtimestamp(t) for t in future_times],
                confidence_intervals=confidence,
                method=self.method,
                accuracy_estimate=accuracy,
                metadata={
                    'history_length': len(history),
                    'current_value': values[-1]
                }
            )

    def _predict_linear(
        self,
        timestamps: List[float],
        values: List[float],
        future_times: List[float]
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
        """线性预测"""
        # 简单线性回归
        n = len(timestamps)
        t_array = np.array(timestamps)
        v_array = np.array(values)

        t_mean = np.mean(t_array)
        v_mean = np.mean(v_array)

        slope = np.sum((t_array - t_mean) * (v_array - v_mean)) / max(np.sum((t_array - t_mean) ** 2), 1e-10)
        intercept = v_mean - slope * t_mean

        # 预测
        predicted = [slope * t + intercept for t in future_times]

        # 计算置信区间
        residuals = v_array - (slope * t_array + intercept)
        std_error = np.std(residuals)

        confidence = []
        for i, t in enumerate(future_times):
            # 距离越远，不确定性越大
            distance = t - timestamps[-1]
            uncertainty = std_error * (1 + distance / (timestamps[-1] - timestamps[0] + 1e-10))
            confidence.append((predicted[i] - 2 * uncertainty, predicted[i] + 2 * uncertainty))

        return predicted, confidence

    def _predict_polynomial(
        self,
        timestamps: List[float],
        values: List[float],
        future_times: List[float],
        degree: int = 2
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
        """多项式预测"""
        # 归一化时间
        t_min = timestamps[0]
        t_range = max(timestamps[-1] - t_min, 1e-10)
        t_norm = [(t - t_min) / t_range for t in timestamps]
        future_norm = [(t - t_min) / t_range for t in future_times]

        # 拟合多项式
        try:
            coeffs = np.polyfit(t_norm, values, min(degree, len(timestamps) - 1))
            poly = np.poly1d(coeffs)

            predicted = [float(poly(t)) for t in future_norm]

            # 计算置信区间
            residuals = np.array(values) - poly(t_norm)
            std_error = np.std(residuals)

            confidence = []
            for i, t in enumerate(future_norm):
                uncertainty = std_error * (1 + abs(t - t_norm[-1]))
                confidence.append((predicted[i] - 2 * uncertainty, predicted[i] + 2 * uncertainty))

            return predicted, confidence
        except Exception:
            return self._predict_linear(timestamps, values, future_times)

    def _predict_ema(
        self,
        sensor_id: str,
        values: List[float],
        future_times: List[float],
        current_time: float
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
        """指数平滑预测"""
        if sensor_id not in self._ema_states:
            self._ema_states[sensor_id] = values[-1]

        ema = self._ema_states[sensor_id]

        # 计算趋势
        if len(values) >= 2:
            trend = values[-1] - values[-2]
        else:
            trend = 0

        # 预测（假设趋势逐渐衰减）
        predicted = []
        current = ema
        for i, t in enumerate(future_times):
            current = current + trend * (0.9 ** i)
            predicted.append(current)

        # 置信区间
        std_val = np.std(values) if len(values) > 1 else abs(values[-1]) * 0.1
        confidence = []
        for i, p in enumerate(predicted):
            uncertainty = std_val * (1 + 0.5 * i)
            confidence.append((p - 2 * uncertainty, p + 2 * uncertainty))

        return predicted, confidence

    def _predict_kalman(
        self,
        sensor_id: str,
        timestamps: List[float],
        values: List[float],
        future_times: List[float]
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
        """卡尔曼滤波预测"""
        # 初始化状态
        if sensor_id not in self._kalman_states:
            self._kalman_states[sensor_id] = {
                'x': values[-1],  # 状态估计
                'v': 0.0,         # 速度估计
                'P_x': 1.0,       # 位置方差
                'P_v': 1.0,       # 速度方差
            }

        state = self._kalman_states[sensor_id]

        # 使用最新观测更新状态
        dt = timestamps[-1] - timestamps[-2] if len(timestamps) > 1 else 1.0

        # 预测步
        x_pred = state['x'] + state['v'] * dt
        P_x_pred = state['P_x'] + state['P_v'] * dt ** 2

        # 更新步
        K = P_x_pred / (P_x_pred + 0.1)  # 观测噪声
        state['x'] = x_pred + K * (values[-1] - x_pred)
        state['P_x'] = (1 - K) * P_x_pred

        # 更新速度估计
        if len(values) >= 2:
            measured_v = (values[-1] - values[-2]) / dt
            state['v'] = 0.8 * state['v'] + 0.2 * measured_v

        # 预测未来
        predicted = []
        uncertainties = []
        x = state['x']
        v = state['v']
        P = state['P_x']

        for t in future_times:
            dt_future = t - timestamps[-1]
            x_future = x + v * dt_future
            P_future = P + state['P_v'] * dt_future ** 2
            predicted.append(x_future)
            uncertainties.append(math.sqrt(P_future))

        confidence = [(p - 2 * u, p + 2 * u) for p, u in zip(predicted, uncertainties)]

        return predicted, confidence

    def _predict_ensemble(
        self,
        sensor_id: str,
        timestamps: List[float],
        values: List[float],
        future_times: List[float]
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
        """集成预测"""
        # 使用多种方法进行预测
        linear_pred, linear_conf = self._predict_linear(timestamps, values, future_times)
        poly_pred, poly_conf = self._predict_polynomial(timestamps, values, future_times)
        ema_pred, ema_conf = self._predict_ema(sensor_id, values, future_times, timestamps[-1])

        # 加权平均
        weights = [0.3, 0.3, 0.4]  # linear, poly, ema
        predicted = []
        confidence = []

        for i in range(len(future_times)):
            avg = (
                weights[0] * linear_pred[i] +
                weights[1] * poly_pred[i] +
                weights[2] * ema_pred[i]
            )
            predicted.append(avg)

            # 置信区间取并集
            lower = min(linear_conf[i][0], poly_conf[i][0], ema_conf[i][0])
            upper = max(linear_conf[i][1], poly_conf[i][1], ema_conf[i][1])
            confidence.append((lower, upper))

        return predicted, confidence

    def _estimate_accuracy(self, historical_values: List[float], predicted: List[float]) -> float:
        """估计预测精度"""
        if len(historical_values) < 2:
            return 0.5

        # 基于历史数据的变异性
        std = np.std(historical_values)
        mean = np.mean(historical_values)

        if mean == 0:
            return 0.5

        cv = std / abs(mean)  # 变异系数

        # 变异系数越小，精度越高
        accuracy = max(0.0, min(1.0, 1.0 - cv))
        return accuracy

    def set_method(self, method: PredictionMethod) -> None:
        """设置预测方法"""
        with self._lock:
            self.method = method

    def clear_history(self, sensor_id: Optional[str] = None) -> None:
        """清除历史数据"""
        with self._lock:
            if sensor_id:
                if sensor_id in self._histories:
                    self._histories[sensor_id].clear()
                if sensor_id in self._kalman_states:
                    del self._kalman_states[sensor_id]
                if sensor_id in self._ema_states:
                    del self._ema_states[sensor_id]
            else:
                self._histories.clear()
                self._kalman_states.clear()
                self._ema_states.clear()


class VirtualSensorSynthesizer:
    """虚拟传感器合成器"""

    def __init__(self):
        self._configs: Dict[str, VirtualSensorConfig] = {}
        self._values: Dict[str, float] = {}
        self._kalman_states: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def register_virtual_sensor(self, config: VirtualSensorConfig) -> None:
        """注册虚拟传感器"""
        with self._lock:
            self._configs[config.sensor_id] = config
            self._values[config.sensor_id] = 0.0

    def unregister_virtual_sensor(self, sensor_id: str) -> None:
        """注销虚拟传感器"""
        with self._lock:
            self._configs.pop(sensor_id, None)
            self._values.pop(sensor_id, None)
            self._kalman_states.pop(sensor_id, None)

    def update(
        self,
        source_values: Dict[str, float],
        source_qualities: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """更新所有虚拟传感器"""
        with self._lock:
            results = {}
            for sensor_id, config in self._configs.items():
                value = self._compute_virtual_value(config, source_values, source_qualities)
                if value is not None:
                    self._values[sensor_id] = value
                    results[sensor_id] = value
            return results

    def _compute_virtual_value(
        self,
        config: VirtualSensorConfig,
        source_values: Dict[str, float],
        source_qualities: Optional[Dict[str, float]] = None
    ) -> Optional[float]:
        """计算虚拟传感器值"""
        # 获取源传感器值
        values = []
        qualities = []
        for src_id in config.source_sensors:
            if src_id in source_values:
                values.append(source_values[src_id])
                q = source_qualities.get(src_id, 1.0) if source_qualities else 1.0
                qualities.append(q)

        if not values:
            return None

        if config.fusion_method == 'weighted_average':
            return self._weighted_average(values, qualities, config.weights)
        elif config.fusion_method == 'kalman':
            return self._kalman_fusion(config.sensor_id, values, qualities)
        elif config.fusion_method == 'formula':
            return self._formula_compute(config.formula, source_values)
        elif config.fusion_method == 'max':
            return max(values)
        elif config.fusion_method == 'min':
            return min(values)
        elif config.fusion_method == 'median':
            return float(np.median(values))
        else:
            return np.mean(values)

    def _weighted_average(
        self,
        values: List[float],
        qualities: List[float],
        weights: Optional[List[float]] = None
    ) -> float:
        """加权平均"""
        if weights and len(weights) == len(values):
            w = np.array(weights) * np.array(qualities)
        else:
            w = np.array(qualities)

        w_sum = np.sum(w)
        if w_sum == 0:
            return np.mean(values)

        return float(np.sum(np.array(values) * w) / w_sum)

    def _kalman_fusion(
        self,
        sensor_id: str,
        values: List[float],
        qualities: List[float]
    ) -> float:
        """卡尔曼融合"""
        if sensor_id not in self._kalman_states:
            self._kalman_states[sensor_id] = {
                'x': np.mean(values),
                'P': 1.0
            }

        state = self._kalman_states[sensor_id]

        # 依次融合每个测量
        for val, q in zip(values, qualities):
            R = 1.0 / max(q, 0.01)  # 观测噪声与质量成反比
            K = state['P'] / (state['P'] + R)
            state['x'] = state['x'] + K * (val - state['x'])
            state['P'] = (1 - K) * state['P']

        # 增加过程噪声
        state['P'] += 0.01

        return state['x']

    def _formula_compute(
        self,
        formula: Optional[str],
        source_values: Dict[str, float]
    ) -> Optional[float]:
        """公式计算"""
        if not formula:
            return None

        try:
            # 安全的数学函数
            safe_dict = {
                'sqrt': math.sqrt,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'log': math.log,
                'exp': math.exp,
                'abs': abs,
                'min': min,
                'max': max,
                'pow': pow,
            }
            safe_dict.update(source_values)
            return float(eval(formula, {"__builtins__": {}}, safe_dict))
        except Exception:
            return None

    def get_value(self, sensor_id: str) -> Optional[float]:
        """获取虚拟传感器值"""
        with self._lock:
            return self._values.get(sensor_id)

    def get_all_values(self) -> Dict[str, float]:
        """获取所有虚拟传感器值"""
        with self._lock:
            return dict(self._values)

    def get_config(self, sensor_id: str) -> Optional[VirtualSensorConfig]:
        """获取虚拟传感器配置"""
        with self._lock:
            return self._configs.get(sensor_id)

    def list_virtual_sensors(self) -> List[str]:
        """列出所有虚拟传感器"""
        with self._lock:
            return list(self._configs.keys())


class SensorHealthMonitor:
    """传感器健康监测器"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._reading_histories: Dict[str, Deque[SensorReading]] = {}
        self._health_histories: Dict[str, Deque[Tuple[datetime, float]]] = {}
        self._lock = threading.RLock()

    def add_reading(self, reading: SensorReading) -> None:
        """添加传感器读数"""
        with self._lock:
            sensor_id = reading.sensor_id
            if sensor_id not in self._reading_histories:
                self._reading_histories[sensor_id] = deque(maxlen=self.window_size)
            self._reading_histories[sensor_id].append(reading)

    def evaluate_health(self, sensor_id: str) -> Optional[HealthReport]:
        """评估传感器健康状态"""
        with self._lock:
            if sensor_id not in self._reading_histories:
                return None

            readings = list(self._reading_histories[sensor_id])
            if len(readings) < 5:
                return None

            # 计算各项指标
            metrics = self._compute_metrics(readings)

            # 评估健康得分
            health_score = self._compute_health_score(metrics)

            # 确定健康状态
            health = self._score_to_health(health_score)

            # 生成问题和建议
            issues, recommendations = self._generate_insights(metrics, readings)

            # 确定趋势
            trend = self._determine_trend(sensor_id, health_score)

            # 记录健康历史
            if sensor_id not in self._health_histories:
                self._health_histories[sensor_id] = deque(maxlen=100)
            self._health_histories[sensor_id].append((datetime.now(), health_score))

            return HealthReport(
                sensor_id=sensor_id,
                timestamp=datetime.now(),
                health=health,
                health_score=health_score,
                issues=issues,
                recommendations=recommendations,
                metrics=metrics,
                trend=trend
            )

    def _compute_metrics(self, readings: List[SensorReading]) -> Dict[str, float]:
        """计算健康指标"""
        values = [r.value for r in readings if not math.isnan(r.value)]
        qualities = [r.quality for r in readings]

        metrics = {}

        # 数据完整性
        valid_count = sum(1 for r in readings if r.is_valid and not math.isnan(r.value))
        metrics['data_completeness'] = valid_count / len(readings)

        # 平均质量
        metrics['avg_quality'] = np.mean(qualities)

        # 值的统计
        if values:
            metrics['mean'] = np.mean(values)
            metrics['std'] = np.std(values)
            metrics['min'] = min(values)
            metrics['max'] = max(values)

            # 变异系数
            if metrics['mean'] != 0:
                metrics['cv'] = metrics['std'] / abs(metrics['mean'])
            else:
                metrics['cv'] = 0

            # 检测异常值
            if len(values) >= 10:
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                outliers = sum(1 for v in values if v < q1 - 1.5 * iqr or v > q3 + 1.5 * iqr)
                metrics['outlier_ratio'] = outliers / len(values)
            else:
                metrics['outlier_ratio'] = 0

        # 故障频率
        failure_count = sum(1 for r in readings if r.failure_mode != FailureMode.NONE)
        metrics['failure_rate'] = failure_count / len(readings)

        # 噪声估计（相邻值差异）
        if len(values) >= 2:
            diffs = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
            metrics['noise_estimate'] = np.mean(diffs)
        else:
            metrics['noise_estimate'] = 0

        return metrics

    def _compute_health_score(self, metrics: Dict[str, float]) -> float:
        """计算健康得分 (0-100)"""
        score = 100.0

        # 数据完整性 (权重 30%)
        score -= (1 - metrics.get('data_completeness', 1)) * 30

        # 平均质量 (权重 25%)
        score -= (1 - metrics.get('avg_quality', 1)) * 25

        # 故障率 (权重 25%)
        score -= metrics.get('failure_rate', 0) * 25

        # 异常值比例 (权重 10%)
        score -= metrics.get('outlier_ratio', 0) * 10

        # 变异系数 (权重 10%)
        cv = metrics.get('cv', 0)
        if cv > 0.5:
            score -= min(10, (cv - 0.5) * 20)

        return max(0, min(100, score))

    def _score_to_health(self, score: float) -> SensorHealth:
        """得分转健康状态"""
        if score >= 90:
            return SensorHealth.EXCELLENT
        elif score >= 75:
            return SensorHealth.GOOD
        elif score >= 60:
            return SensorHealth.FAIR
        elif score >= 40:
            return SensorHealth.POOR
        elif score > 0:
            return SensorHealth.CRITICAL
        else:
            return SensorHealth.FAILED

    def _generate_insights(
        self,
        metrics: Dict[str, float],
        readings: List[SensorReading]
    ) -> Tuple[List[str], List[str]]:
        """生成问题和建议"""
        issues = []
        recommendations = []

        # 检查数据完整性
        if metrics.get('data_completeness', 1) < 0.95:
            issues.append(f"数据完整性低: {metrics['data_completeness']:.1%}")
            recommendations.append("检查传感器连接和数据采集系统")

        # 检查故障率
        if metrics.get('failure_rate', 0) > 0.05:
            issues.append(f"故障率过高: {metrics['failure_rate']:.1%}")
            recommendations.append("进行传感器诊断和维护")

        # 检查异常值
        if metrics.get('outlier_ratio', 0) > 0.1:
            issues.append(f"异常值过多: {metrics['outlier_ratio']:.1%}")
            recommendations.append("检查环境干扰或考虑数据滤波")

        # 检查噪声水平
        if 'noise_estimate' in metrics and 'mean' in metrics:
            if metrics['mean'] != 0:
                noise_ratio = metrics['noise_estimate'] / abs(metrics['mean'])
                if noise_ratio > 0.1:
                    issues.append(f"噪声水平过高: {noise_ratio:.1%}")
                    recommendations.append("考虑增加滤波或检查接地")

        # 检查质量下降
        if metrics.get('avg_quality', 1) < 0.8:
            issues.append(f"平均读数质量低: {metrics['avg_quality']:.1%}")
            recommendations.append("考虑重新校准传感器")

        # 检查特定故障模式
        failure_modes = [r.failure_mode for r in readings if r.failure_mode != FailureMode.NONE]
        if failure_modes:
            mode_counts = {}
            for mode in failure_modes:
                mode_counts[mode] = mode_counts.get(mode, 0) + 1

            for mode, count in mode_counts.items():
                if count > len(readings) * 0.05:
                    issues.append(f"检测到{mode.name}故障模式")
                    if mode == FailureMode.DRIFT:
                        recommendations.append("需要重新校准")
                    elif mode == FailureMode.STUCK:
                        recommendations.append("传感器可能卡死，需要检修")
                    elif mode == FailureMode.NOISE_INCREASE:
                        recommendations.append("检查电磁干扰源")

        if not issues:
            issues.append("未发现明显问题")

        if not recommendations:
            recommendations.append("继续正常监测")

        return issues, recommendations

    def _determine_trend(self, sensor_id: str, current_score: float) -> str:
        """确定健康趋势"""
        if sensor_id not in self._health_histories:
            return 'stable'

        history = list(self._health_histories[sensor_id])
        if len(history) < 5:
            return 'stable'

        # 计算最近趋势
        recent_scores = [h[1] for h in history[-10:]]
        if len(recent_scores) < 2:
            return 'stable'

        # 简单线性回归
        x = np.arange(len(recent_scores))
        slope = np.polyfit(x, recent_scores, 1)[0]

        if slope > 1:
            return 'improving'
        elif slope < -1:
            return 'degrading'
        else:
            return 'stable'

    def get_health_history(
        self,
        sensor_id: str,
        count: int = 50
    ) -> List[Tuple[datetime, float]]:
        """获取健康历史"""
        with self._lock:
            if sensor_id not in self._health_histories:
                return []
            return list(self._health_histories[sensor_id])[-count:]


class SensorPredictionManager:
    """传感器仿真与状态预测管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._simulators: Dict[str, SensorSimulator] = {}
        self._predictor = StatePredictor(
            method=PredictionMethod(self.config.get('prediction_method', PredictionMethod.ENSEMBLE.value)),
            history_size=self.config.get('history_size', 100)
        )
        self._synthesizer = VirtualSensorSynthesizer()
        self._health_monitor = SensorHealthMonitor(
            window_size=self.config.get('health_window', 100)
        )
        self._lock = threading.RLock()
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        self._update_interval_ms = self.config.get('update_interval_ms', 100.0)

    def register_sensor(self, config: SensorConfig) -> None:
        """注册传感器"""
        with self._lock:
            simulator = SensorSimulator(config)
            self._simulators[config.sensor_id] = simulator

    def unregister_sensor(self, sensor_id: str) -> None:
        """注销传感器"""
        with self._lock:
            self._simulators.pop(sensor_id, None)

    def register_virtual_sensor(self, config: VirtualSensorConfig) -> None:
        """注册虚拟传感器"""
        self._synthesizer.register_virtual_sensor(config)

    def unregister_virtual_sensor(self, sensor_id: str) -> None:
        """注销虚拟传感器"""
        self._synthesizer.unregister_virtual_sensor(sensor_id)

    def simulate_reading(
        self,
        sensor_id: str,
        true_value: Optional[float] = None
    ) -> Optional[SensorReading]:
        """仿真单个传感器读数"""
        with self._lock:
            if sensor_id not in self._simulators:
                return None

            reading = self._simulators[sensor_id].simulate_reading(true_value)

            # 添加到预测器和健康监测
            self._predictor.add_observation(
                sensor_id,
                reading.timestamp.timestamp(),
                reading.value
            )
            self._health_monitor.add_reading(reading)

            return reading

    def simulate_all(
        self,
        true_values: Optional[Dict[str, float]] = None
    ) -> Dict[str, SensorReading]:
        """仿真所有传感器"""
        with self._lock:
            readings = {}
            source_values = {}
            source_qualities = {}

            for sensor_id, simulator in self._simulators.items():
                true_val = true_values.get(sensor_id) if true_values else None
                reading = simulator.simulate_reading(true_val)
                readings[sensor_id] = reading

                if not math.isnan(reading.value):
                    source_values[sensor_id] = reading.value
                    source_qualities[sensor_id] = reading.quality

                # 添加到预测器和健康监测
                self._predictor.add_observation(
                    sensor_id,
                    reading.timestamp.timestamp(),
                    reading.value
                )
                self._health_monitor.add_reading(reading)

            # 更新虚拟传感器
            self._synthesizer.update(source_values, source_qualities)

            return readings

    def predict(
        self,
        sensor_id: str,
        horizon_seconds: float,
        num_points: int = 10
    ) -> Optional[PredictionResult]:
        """预测传感器状态"""
        return self._predictor.predict(sensor_id, horizon_seconds, num_points)

    def predict_all(
        self,
        horizon_seconds: float,
        num_points: int = 10
    ) -> Dict[str, PredictionResult]:
        """预测所有传感器状态"""
        with self._lock:
            results = {}
            for sensor_id in self._simulators:
                result = self._predictor.predict(sensor_id, horizon_seconds, num_points)
                if result:
                    results[sensor_id] = result
            return results

    def get_health_report(self, sensor_id: str) -> Optional[HealthReport]:
        """获取传感器健康报告"""
        return self._health_monitor.evaluate_health(sensor_id)

    def get_all_health_reports(self) -> Dict[str, HealthReport]:
        """获取所有传感器健康报告"""
        with self._lock:
            reports = {}
            for sensor_id in self._simulators:
                report = self._health_monitor.evaluate_health(sensor_id)
                if report:
                    reports[sensor_id] = report
            return reports

    def inject_failure(self, sensor_id: str, failure_mode: FailureMode) -> bool:
        """注入传感器故障"""
        with self._lock:
            if sensor_id in self._simulators:
                self._simulators[sensor_id].inject_failure(failure_mode)
                return True
            return False

    def clear_failure(self, sensor_id: str) -> bool:
        """清除传感器故障"""
        with self._lock:
            if sensor_id in self._simulators:
                self._simulators[sensor_id].clear_failure()
                return True
            return False

    def calibrate_sensor(self, sensor_id: str) -> bool:
        """校准传感器"""
        with self._lock:
            if sensor_id in self._simulators:
                self._simulators[sensor_id].calibrate()
                return True
            return False

    def get_sensor_state(self, sensor_id: str) -> Optional[SensorState]:
        """获取传感器状态"""
        with self._lock:
            if sensor_id in self._simulators:
                return self._simulators[sensor_id].get_state()
            return None

    def get_all_sensor_states(self) -> Dict[str, SensorState]:
        """获取所有传感器状态"""
        with self._lock:
            return {
                sid: sim.get_state()
                for sid, sim in self._simulators.items()
            }

    def get_virtual_sensor_value(self, sensor_id: str) -> Optional[float]:
        """获取虚拟传感器值"""
        return self._synthesizer.get_value(sensor_id)

    def get_all_virtual_sensor_values(self) -> Dict[str, float]:
        """获取所有虚拟传感器值"""
        return self._synthesizer.get_all_values()

    def set_prediction_method(self, method: PredictionMethod) -> None:
        """设置预测方法"""
        self._predictor.set_method(method)

    def start(self) -> None:
        """启动后台更新"""
        if self._running:
            return

        self._running = True
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()

    def stop(self) -> None:
        """停止后台更新"""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=1.0)
            self._update_thread = None

    def _update_loop(self) -> None:
        """后台更新循环"""
        while self._running:
            self.simulate_all()
            time.sleep(self._update_interval_ms / 1000.0)

    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        with self._lock:
            sensor_count = len(self._simulators)
            virtual_count = len(self._synthesizer.list_virtual_sensors())

            # 统计健康状态
            health_stats = {h.name: 0 for h in SensorHealth}
            for sensor_id in self._simulators:
                state = self._simulators[sensor_id].get_state()
                health_stats[state.health.name] += 1

            return {
                'running': self._running,
                'sensor_count': sensor_count,
                'virtual_sensor_count': virtual_count,
                'prediction_method': self._predictor.method.name,
                'update_interval_ms': self._update_interval_ms,
                'health_distribution': health_stats
            }

    def list_sensors(self) -> List[str]:
        """列出所有传感器"""
        with self._lock:
            return list(self._simulators.keys())

    def list_virtual_sensors(self) -> List[str]:
        """列出所有虚拟传感器"""
        return self._synthesizer.list_virtual_sensors()
