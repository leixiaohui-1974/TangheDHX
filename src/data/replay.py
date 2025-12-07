# -*- coding: utf-8 -*-
"""
History Replay Module.

历史数据回放模块，提供：
- 数据回放控制
- 多速度播放
- 事件同步
- 状态恢复
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable, Iterator
import numpy as np

from .storage import TimeSeriesStorage, DataPoint

logger = logging.getLogger(__name__)


class ReplayMode(Enum):
    """回放模式."""
    REALTIME = auto()      # 实时回放
    FAST = auto()          # 快速回放
    STEP = auto()          # 单步回放
    CONTINUOUS = auto()    # 连续回放


class ReplayState(Enum):
    """回放状态."""
    STOPPED = auto()
    PLAYING = auto()
    PAUSED = auto()
    FINISHED = auto()


@dataclass
class ReplayEvent:
    """回放事件."""
    timestamp: float
    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplaySession:
    """回放会话."""
    session_id: str
    start_time: float
    end_time: float
    current_time: float = 0.0
    speed: float = 1.0
    mode: ReplayMode = ReplayMode.REALTIME
    state: ReplayState = ReplayState.STOPPED
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def progress(self) -> float:
        """回放进度 (0-1)."""
        duration = self.end_time - self.start_time
        if duration <= 0:
            return 0.0
        return (self.current_time - self.start_time) / duration

    @property
    def remaining_time(self) -> float:
        """剩余时间 (秒)."""
        return self.end_time - self.current_time


class HistoryReplay:
    """
    历史数据回放器.

    提供历史数据的回放功能。
    """

    def __init__(self, storage: TimeSeriesStorage):
        """
        初始化回放器.

        Args:
            storage: 时序数据存储
        """
        self.storage = storage

        # 当前会话
        self._session: Optional[ReplaySession] = None

        # 回放数据缓存
        self._cached_data: Dict[str, List[DataPoint]] = {}

        # 事件列表
        self._events: List[ReplayEvent] = []

        # 回调
        self._data_callbacks: List[Callable[[str, DataPoint], None]] = []
        self._event_callbacks: List[Callable[[ReplayEvent], None]] = []
        self._state_callbacks: List[Callable[[ReplayState], None]] = []

        # 数据指针
        self._data_indices: Dict[str, int] = {}

        # 会话计数器
        self._session_counter = 0

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    def create_session(
        self,
        start_time: float,
        end_time: float,
        series_names: Optional[List[str]] = None,
        mode: ReplayMode = ReplayMode.REALTIME,
        speed: float = 1.0,
    ) -> ReplaySession:
        """
        创建回放会话.

        Args:
            start_time: 开始时间戳
            end_time: 结束时间戳
            series_names: 要回放的序列名称
            mode: 回放模式
            speed: 回放速度

        Returns:
            ReplaySession
        """
        self._session_counter += 1
        session_id = f"replay_{self._session_counter:04d}"

        self._session = ReplaySession(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            current_time=start_time,
            speed=speed,
            mode=mode,
        )

        # 加载数据
        names = series_names or self.storage.list_series()
        for name in names:
            points = self.storage.read(name, start_time, end_time)
            self._cached_data[name] = sorted(points, key=lambda p: p.timestamp)
            self._data_indices[name] = 0

        logger.info(f"Created replay session {session_id}: {start_time} -> {end_time}")
        return self._session

    def get_session(self) -> Optional[ReplaySession]:
        """获取当前会话."""
        return self._session

    # -------------------------------------------------------------------------
    # Playback Control
    # -------------------------------------------------------------------------

    def play(self) -> None:
        """开始/继续播放."""
        if self._session is None:
            logger.warning("No replay session")
            return

        self._session.state = ReplayState.PLAYING
        self._notify_state_change(ReplayState.PLAYING)
        logger.info(f"Replay started: {self._session.session_id}")

    def pause(self) -> None:
        """暂停播放."""
        if self._session and self._session.state == ReplayState.PLAYING:
            self._session.state = ReplayState.PAUSED
            self._notify_state_change(ReplayState.PAUSED)

    def stop(self) -> None:
        """停止播放."""
        if self._session:
            self._session.state = ReplayState.STOPPED
            self._session.current_time = self._session.start_time
            self._reset_indices()
            self._notify_state_change(ReplayState.STOPPED)

    def seek(self, timestamp: float) -> None:
        """
        跳转到指定时间.

        Args:
            timestamp: 目标时间戳
        """
        if self._session is None:
            return

        # 限制在有效范围内
        timestamp = max(self._session.start_time, min(timestamp, self._session.end_time))
        self._session.current_time = timestamp

        # 重置数据指针
        for name, points in self._cached_data.items():
            # 二分查找
            idx = self._binary_search(points, timestamp)
            self._data_indices[name] = idx

        logger.debug(f"Seek to {timestamp}")

    def set_speed(self, speed: float) -> None:
        """设置播放速度."""
        if self._session:
            self._session.speed = max(0.1, min(speed, 100.0))

    def set_mode(self, mode: ReplayMode) -> None:
        """设置播放模式."""
        if self._session:
            self._session.mode = mode

    # -------------------------------------------------------------------------
    # Playback Execution
    # -------------------------------------------------------------------------

    def step(self, dt: float = 0.1) -> Dict[str, Any]:
        """
        执行一步回放.

        Args:
            dt: 时间步长 (实际时间)

        Returns:
            回放数据
        """
        if self._session is None or self._session.state != ReplayState.PLAYING:
            return {'status': 'not_playing'}

        # 计算回放时间增量
        replay_dt = dt * self._session.speed

        # 更新当前时间
        new_time = self._session.current_time + replay_dt
        if new_time >= self._session.end_time:
            new_time = self._session.end_time
            self._session.state = ReplayState.FINISHED
            self._notify_state_change(ReplayState.FINISHED)

        # 收集这个时间段内的数据
        data_output = {}
        for name, points in self._cached_data.items():
            idx = self._data_indices[name]
            emitted = []

            while idx < len(points):
                point = points[idx]
                if point.timestamp <= new_time:
                    emitted.append(point)
                    self._notify_data(name, point)
                    idx += 1
                else:
                    break

            self._data_indices[name] = idx
            if emitted:
                data_output[name] = emitted

        # 处理事件
        events_output = []
        for event in self._events:
            if self._session.current_time < event.timestamp <= new_time:
                events_output.append(event)
                self._notify_event(event)

        self._session.current_time = new_time

        return {
            'status': 'ok',
            'current_time': new_time,
            'progress': self._session.progress,
            'data': data_output,
            'events': events_output,
        }

    def run_to_end(
        self,
        step_dt: float = 0.1,
        callback: Optional[Callable[[Dict], None]] = None,
    ) -> None:
        """
        运行到结束.

        Args:
            step_dt: 每步时间
            callback: 每步回调
        """
        self.play()

        while self._session and self._session.state == ReplayState.PLAYING:
            result = self.step(step_dt)
            if callback:
                callback(result)

            if self._session.mode == ReplayMode.STEP:
                break

    def iterate(
        self,
        step_dt: float = 0.1,
    ) -> Iterator[Dict[str, Any]]:
        """
        迭代回放.

        Yields:
            每步的回放数据
        """
        self.play()

        while self._session and self._session.state == ReplayState.PLAYING:
            result = self.step(step_dt)
            yield result

    # -------------------------------------------------------------------------
    # Events
    # -------------------------------------------------------------------------

    def add_event(self, event: ReplayEvent) -> None:
        """添加回放事件."""
        self._events.append(event)
        self._events.sort(key=lambda e: e.timestamp)

    def add_events_from_anomalies(self, anomalies: List[Any]) -> None:
        """从异常记录添加事件."""
        for anomaly in anomalies:
            event = ReplayEvent(
                timestamp=anomaly.timestamp,
                event_type='anomaly',
                data={
                    'anomaly_id': anomaly.anomaly_id,
                    'type': anomaly.anomaly_type.name,
                    'severity': anomaly.severity.name,
                    'message': anomaly.message,
                },
            )
            self.add_event(event)

    def clear_events(self) -> None:
        """清除事件."""
        self._events.clear()

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def on_data(self, callback: Callable[[str, DataPoint], None]) -> None:
        """注册数据回调."""
        self._data_callbacks.append(callback)

    def on_event(self, callback: Callable[[ReplayEvent], None]) -> None:
        """注册事件回调."""
        self._event_callbacks.append(callback)

    def on_state_change(self, callback: Callable[[ReplayState], None]) -> None:
        """注册状态变化回调."""
        self._state_callbacks.append(callback)

    def _notify_data(self, series_name: str, point: DataPoint) -> None:
        """通知数据回调."""
        for callback in self._data_callbacks:
            try:
                callback(series_name, point)
            except Exception as e:
                logger.error(f"Data callback error: {e}")

    def _notify_event(self, event: ReplayEvent) -> None:
        """通知事件回调."""
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    def _notify_state_change(self, state: ReplayState) -> None:
        """通知状态变化."""
        for callback in self._state_callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"State callback error: {e}")

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _binary_search(self, points: List[DataPoint], timestamp: float) -> int:
        """二分查找时间戳位置."""
        left, right = 0, len(points)
        while left < right:
            mid = (left + right) // 2
            if points[mid].timestamp < timestamp:
                left = mid + 1
            else:
                right = mid
        return left

    def _reset_indices(self) -> None:
        """重置数据指针."""
        for name in self._data_indices:
            self._data_indices[name] = 0

    def get_data_at_time(
        self,
        timestamp: float,
        series_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        获取指定时间点的数据.

        Args:
            timestamp: 时间戳
            series_names: 序列名称

        Returns:
            {序列名: 值} 字典
        """
        result = {}
        names = series_names or list(self._cached_data.keys())

        for name in names:
            points = self._cached_data.get(name, [])
            if not points:
                continue

            # 找到最近的点
            idx = self._binary_search(points, timestamp)

            if idx == 0:
                result[name] = points[0].value
            elif idx >= len(points):
                result[name] = points[-1].value
            else:
                # 线性插值
                p1, p2 = points[idx - 1], points[idx]
                t1, t2 = p1.timestamp, p2.timestamp
                if t2 != t1:
                    ratio = (timestamp - t1) / (t2 - t1)
                    result[name] = p1.value + ratio * (p2.value - p1.value)
                else:
                    result[name] = p1.value

        return result

    def get_time_range(self) -> Optional[tuple]:
        """获取数据时间范围."""
        if not self._cached_data:
            return None

        min_time = float('inf')
        max_time = float('-inf')

        for points in self._cached_data.values():
            if points:
                min_time = min(min_time, points[0].timestamp)
                max_time = max(max_time, points[-1].timestamp)

        return (min_time, max_time) if min_time != float('inf') else None

    def get_summary(self) -> Dict[str, Any]:
        """获取回放摘要."""
        if self._session is None:
            return {'status': 'no_session'}

        return {
            'session_id': self._session.session_id,
            'state': self._session.state.name,
            'mode': self._session.mode.name,
            'speed': self._session.speed,
            'start_time': self._session.start_time,
            'end_time': self._session.end_time,
            'current_time': self._session.current_time,
            'progress': self._session.progress,
            'remaining_time': self._session.remaining_time,
            'series_count': len(self._cached_data),
            'events_count': len(self._events),
            'total_points': sum(len(p) for p in self._cached_data.values()),
        }
