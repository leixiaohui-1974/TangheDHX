# -*- coding: utf-8 -*-
"""
Anomaly Detection Module.

异常检测模块，提供：
- 统计异常检测
- 阈值检测
- 趋势异常检测
- 模式异常检测
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable
import numpy as np

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """异常类型."""
    THRESHOLD_HIGH = auto()      # 超过上限
    THRESHOLD_LOW = auto()       # 低于下限
    SPIKE = auto()               # 尖峰
    DROP = auto()                # 骤降
    DRIFT = auto()               # 漂移
    FLATLINE = auto()            # 无变化
    OSCILLATION = auto()         # 振荡
    PATTERN_BREAK = auto()       # 模式断裂
    RATE_OF_CHANGE = auto()      # 变化率异常
    STATISTICAL = auto()         # 统计异常


class Severity(Enum):
    """异常严重程度."""
    INFO = 1
    WARNING = 2
    CRITICAL = 3
    EMERGENCY = 4


@dataclass
class Anomaly:
    """异常记录."""
    anomaly_id: str
    anomaly_type: AnomalyType
    severity: Severity
    series_name: str
    timestamp: float
    value: float
    expected_value: Optional[float] = None
    threshold: Optional[float] = None
    message: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'anomaly_id': self.anomaly_id,
            'type': self.anomaly_type.name,
            'severity': self.severity.name,
            'series': self.series_name,
            'timestamp': self.timestamp,
            'value': self.value,
            'expected': self.expected_value,
            'threshold': self.threshold,
            'message': self.message,
            'context': self.context,
            'acknowledged': self.acknowledged,
        }


@dataclass
class ThresholdRule:
    """阈值规则."""
    name: str
    series_name: str
    high: Optional[float] = None
    low: Optional[float] = None
    high_high: Optional[float] = None  # 紧急上限
    low_low: Optional[float] = None    # 紧急下限
    deadband: float = 0.0              # 死区
    enabled: bool = True


@dataclass
class RateRule:
    """变化率规则."""
    name: str
    series_name: str
    max_rate: float           # 最大变化率 (单位/秒)
    window: float = 1.0       # 检测窗口 (秒)
    enabled: bool = True


class AnomalyDetector:
    """
    异常检测器.

    提供多种异常检测方法。
    """

    def __init__(self):
        # 阈值规则
        self._threshold_rules: Dict[str, ThresholdRule] = {}

        # 变化率规则
        self._rate_rules: Dict[str, RateRule] = {}

        # 历史数据缓存
        self._history: Dict[str, List[tuple]] = {}  # {series: [(timestamp, value), ...]}
        self._history_size = 1000

        # 检测状态
        self._last_values: Dict[str, float] = {}
        self._alarm_states: Dict[str, bool] = {}

        # 异常记录
        self._anomalies: List[Anomaly] = []
        self._max_anomalies = 10000

        # 回调
        self._callbacks: List[Callable[[Anomaly], None]] = []

        # 统计模型参数
        self._stats: Dict[str, Dict[str, float]] = {}

        # 异常ID计数器
        self._anomaly_counter = 0

    # -------------------------------------------------------------------------
    # Rule Configuration
    # -------------------------------------------------------------------------

    def add_threshold_rule(self, rule: ThresholdRule) -> None:
        """添加阈值规则."""
        self._threshold_rules[rule.name] = rule
        logger.info(f"Added threshold rule: {rule.name}")

    def add_rate_rule(self, rule: RateRule) -> None:
        """添加变化率规则."""
        self._rate_rules[rule.name] = rule
        logger.info(f"Added rate rule: {rule.name}")

    def remove_rule(self, rule_name: str) -> None:
        """移除规则."""
        self._threshold_rules.pop(rule_name, None)
        self._rate_rules.pop(rule_name, None)

    def configure_statistical(
        self,
        series_name: str,
        mean: float,
        std: float,
        n_sigma: float = 3.0,
    ) -> None:
        """配置统计检测参数."""
        self._stats[series_name] = {
            'mean': mean,
            'std': std,
            'n_sigma': n_sigma,
            'upper': mean + n_sigma * std,
            'lower': mean - n_sigma * std,
        }

    # -------------------------------------------------------------------------
    # Detection
    # -------------------------------------------------------------------------

    def detect(
        self,
        series_name: str,
        value: float,
        timestamp: Optional[float] = None,
    ) -> List[Anomaly]:
        """
        检测单个数据点的异常.

        Args:
            series_name: 序列名称
            value: 当前值
            timestamp: 时间戳

        Returns:
            检测到的异常列表
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()

        anomalies = []

        # 更新历史
        if series_name not in self._history:
            self._history[series_name] = []
        self._history[series_name].append((timestamp, value))
        if len(self._history[series_name]) > self._history_size:
            self._history[series_name].pop(0)

        # 阈值检测
        threshold_anomalies = self._detect_threshold(series_name, value, timestamp)
        anomalies.extend(threshold_anomalies)

        # 变化率检测
        rate_anomalies = self._detect_rate(series_name, value, timestamp)
        anomalies.extend(rate_anomalies)

        # 统计异常检测
        stat_anomalies = self._detect_statistical(series_name, value, timestamp)
        anomalies.extend(stat_anomalies)

        # 模式异常检测
        pattern_anomalies = self._detect_pattern(series_name, value, timestamp)
        anomalies.extend(pattern_anomalies)

        # 更新状态
        self._last_values[series_name] = value

        # 记录并触发回调
        for anomaly in anomalies:
            self._record_anomaly(anomaly)

        return anomalies

    def detect_batch(
        self,
        series_name: str,
        values: List[tuple],
    ) -> List[Anomaly]:
        """
        批量检测.

        Args:
            series_name: 序列名称
            values: [(timestamp, value), ...] 列表

        Returns:
            异常列表
        """
        all_anomalies = []
        for timestamp, value in values:
            anomalies = self.detect(series_name, value, timestamp)
            all_anomalies.extend(anomalies)
        return all_anomalies

    # -------------------------------------------------------------------------
    # Specific Detection Methods
    # -------------------------------------------------------------------------

    def _detect_threshold(
        self,
        series_name: str,
        value: float,
        timestamp: float,
    ) -> List[Anomaly]:
        """阈值检测."""
        anomalies = []

        for rule in self._threshold_rules.values():
            if not rule.enabled or rule.series_name != series_name:
                continue

            alarm_key = f"{rule.name}_{series_name}"

            # 紧急上限
            if rule.high_high is not None and value >= rule.high_high:
                anomalies.append(self._create_anomaly(
                    AnomalyType.THRESHOLD_HIGH,
                    Severity.EMERGENCY,
                    series_name,
                    timestamp,
                    value,
                    threshold=rule.high_high,
                    message=f"值 {value:.2f} 超过紧急上限 {rule.high_high:.2f}",
                ))

            # 高限
            elif rule.high is not None and value >= rule.high:
                if not self._alarm_states.get(alarm_key + '_high', False):
                    anomalies.append(self._create_anomaly(
                        AnomalyType.THRESHOLD_HIGH,
                        Severity.WARNING,
                        series_name,
                        timestamp,
                        value,
                        threshold=rule.high,
                        message=f"值 {value:.2f} 超过上限 {rule.high:.2f}",
                    ))
                    self._alarm_states[alarm_key + '_high'] = True

            else:
                # 检查是否恢复 (考虑死区)
                if rule.high is not None and value < rule.high - rule.deadband:
                    self._alarm_states[alarm_key + '_high'] = False

            # 紧急下限
            if rule.low_low is not None and value <= rule.low_low:
                anomalies.append(self._create_anomaly(
                    AnomalyType.THRESHOLD_LOW,
                    Severity.EMERGENCY,
                    series_name,
                    timestamp,
                    value,
                    threshold=rule.low_low,
                    message=f"值 {value:.2f} 低于紧急下限 {rule.low_low:.2f}",
                ))

            # 低限
            elif rule.low is not None and value <= rule.low:
                if not self._alarm_states.get(alarm_key + '_low', False):
                    anomalies.append(self._create_anomaly(
                        AnomalyType.THRESHOLD_LOW,
                        Severity.WARNING,
                        series_name,
                        timestamp,
                        value,
                        threshold=rule.low,
                        message=f"值 {value:.2f} 低于下限 {rule.low:.2f}",
                    ))
                    self._alarm_states[alarm_key + '_low'] = True

            else:
                if rule.low is not None and value > rule.low + rule.deadband:
                    self._alarm_states[alarm_key + '_low'] = False

        return anomalies

    def _detect_rate(
        self,
        series_name: str,
        value: float,
        timestamp: float,
    ) -> List[Anomaly]:
        """变化率检测."""
        anomalies = []

        history = self._history.get(series_name, [])
        if len(history) < 2:
            return anomalies

        for rule in self._rate_rules.values():
            if not rule.enabled or rule.series_name != series_name:
                continue

            # 找到窗口内的历史点
            window_start = timestamp - rule.window
            window_points = [
                (t, v) for t, v in history
                if t >= window_start and t < timestamp
            ]

            if window_points:
                # 计算最大变化率
                prev_time, prev_value = window_points[-1]
                dt = timestamp - prev_time
                if dt > 0:
                    rate = abs(value - prev_value) / dt

                    if rate > rule.max_rate:
                        if value > prev_value:
                            anomaly_type = AnomalyType.SPIKE
                        else:
                            anomaly_type = AnomalyType.DROP

                        anomalies.append(self._create_anomaly(
                            anomaly_type,
                            Severity.WARNING,
                            series_name,
                            timestamp,
                            value,
                            expected_value=prev_value,
                            threshold=rule.max_rate,
                            message=f"变化率 {rate:.2f} 超过限制 {rule.max_rate:.2f}",
                            context={'rate': rate, 'window': rule.window},
                        ))

        return anomalies

    def _detect_statistical(
        self,
        series_name: str,
        value: float,
        timestamp: float,
    ) -> List[Anomaly]:
        """统计异常检测."""
        anomalies = []

        if series_name not in self._stats:
            return anomalies

        stats = self._stats[series_name]
        upper = stats['upper']
        lower = stats['lower']

        if value > upper or value < lower:
            z_score = (value - stats['mean']) / max(stats['std'], 0.001)

            anomalies.append(self._create_anomaly(
                AnomalyType.STATISTICAL,
                Severity.WARNING if abs(z_score) < 4 else Severity.CRITICAL,
                series_name,
                timestamp,
                value,
                expected_value=stats['mean'],
                message=f"统计异常: z-score={z_score:.2f}",
                context={
                    'z_score': z_score,
                    'mean': stats['mean'],
                    'std': stats['std'],
                },
            ))

        return anomalies

    def _detect_pattern(
        self,
        series_name: str,
        value: float,
        timestamp: float,
    ) -> List[Anomaly]:
        """模式异常检测."""
        anomalies = []

        history = self._history.get(series_name, [])
        if len(history) < 10:
            return anomalies

        # 获取最近的值
        recent_values = [v for _, v in history[-10:]]

        # Flatline检测 (无变化)
        if len(set(recent_values)) == 1:
            anomalies.append(self._create_anomaly(
                AnomalyType.FLATLINE,
                Severity.WARNING,
                series_name,
                timestamp,
                value,
                message="信号无变化，可能传感器故障",
            ))

        # 振荡检测
        if len(recent_values) >= 6:
            diffs = np.diff(recent_values)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)

            if sign_changes >= len(diffs) - 2:  # 频繁方向变化
                anomalies.append(self._create_anomaly(
                    AnomalyType.OSCILLATION,
                    Severity.WARNING,
                    series_name,
                    timestamp,
                    value,
                    message="检测到振荡模式",
                    context={'sign_changes': sign_changes},
                ))

        return anomalies

    # -------------------------------------------------------------------------
    # Anomaly Management
    # -------------------------------------------------------------------------

    def _create_anomaly(
        self,
        anomaly_type: AnomalyType,
        severity: Severity,
        series_name: str,
        timestamp: float,
        value: float,
        expected_value: Optional[float] = None,
        threshold: Optional[float] = None,
        message: str = "",
        context: Optional[Dict] = None,
    ) -> Anomaly:
        """创建异常记录."""
        self._anomaly_counter += 1
        return Anomaly(
            anomaly_id=f"A{self._anomaly_counter:08d}",
            anomaly_type=anomaly_type,
            severity=severity,
            series_name=series_name,
            timestamp=timestamp,
            value=value,
            expected_value=expected_value,
            threshold=threshold,
            message=message,
            context=context or {},
        )

    def _record_anomaly(self, anomaly: Anomaly) -> None:
        """记录异常."""
        self._anomalies.append(anomaly)

        # 限制记录数量
        if len(self._anomalies) > self._max_anomalies:
            self._anomalies = self._anomalies[-self._max_anomalies:]

        # 触发回调
        for callback in self._callbacks:
            try:
                callback(anomaly)
            except Exception as e:
                logger.error(f"Anomaly callback error: {e}")

        # 日志
        log_func = {
            Severity.INFO: logger.info,
            Severity.WARNING: logger.warning,
            Severity.CRITICAL: logger.error,
            Severity.EMERGENCY: logger.critical,
        }.get(anomaly.severity, logger.warning)

        log_func(f"Anomaly [{anomaly.anomaly_type.name}] {anomaly.series_name}: {anomaly.message}")

    def acknowledge_anomaly(self, anomaly_id: str) -> bool:
        """确认异常."""
        for anomaly in self._anomalies:
            if anomaly.anomaly_id == anomaly_id:
                anomaly.acknowledged = True
                return True
        return False

    def get_anomalies(
        self,
        series_name: Optional[str] = None,
        severity: Optional[Severity] = None,
        since: Optional[float] = None,
        unacknowledged_only: bool = False,
    ) -> List[Anomaly]:
        """获取异常列表."""
        result = self._anomalies

        if series_name:
            result = [a for a in result if a.series_name == series_name]

        if severity:
            result = [a for a in result if a.severity.value >= severity.value]

        if since:
            result = [a for a in result if a.timestamp >= since]

        if unacknowledged_only:
            result = [a for a in result if not a.acknowledged]

        return result

    def get_active_alarms(self) -> Dict[str, bool]:
        """获取当前活动告警状态."""
        return self._alarm_states.copy()

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def add_callback(self, callback: Callable[[Anomaly], None]) -> None:
        """添加异常回调."""
        self._callbacks.append(callback)

    def clear_callbacks(self) -> None:
        """清除回调."""
        self._callbacks.clear()

    # -------------------------------------------------------------------------
    # Reset & Stats
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """重置检测器."""
        self._history.clear()
        self._last_values.clear()
        self._alarm_states.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息."""
        type_counts = {}
        severity_counts = {}

        for anomaly in self._anomalies:
            type_name = anomaly.anomaly_type.name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

            sev_name = anomaly.severity.name
            severity_counts[sev_name] = severity_counts.get(sev_name, 0) + 1

        return {
            'total_anomalies': len(self._anomalies),
            'unacknowledged': sum(1 for a in self._anomalies if not a.acknowledged),
            'by_type': type_counts,
            'by_severity': severity_counts,
            'rules_count': len(self._threshold_rules) + len(self._rate_rules),
            'active_alarms': sum(self._alarm_states.values()),
        }
