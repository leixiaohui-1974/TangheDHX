# -*- coding: utf-8 -*-
"""
Time Series Data Storage.

时序数据存储模块，提供：
- 内存时序存储
- 数据压缩
- 数据导出/导入
- 数据查询
"""

import logging
import json
import gzip
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Iterator, Tuple, Callable
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DataPoint:
    """数据点."""
    timestamp: float          # Unix时间戳
    value: float              # 数值
    quality: int = 100        # 数据质量 (0-100)
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataPoint':
        return cls(**data)


@dataclass
class DataSeries:
    """数据序列."""
    name: str
    unit: str = ""
    description: str = ""
    points: List[DataPoint] = field(default_factory=list)

    # 元数据
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def add_point(self, timestamp: float, value: float, quality: int = 100) -> None:
        """添加数据点."""
        self.points.append(DataPoint(timestamp, value, quality))

    def get_values(self) -> np.ndarray:
        """获取所有值."""
        return np.array([p.value for p in self.points])

    def get_timestamps(self) -> np.ndarray:
        """获取所有时间戳."""
        return np.array([p.timestamp for p in self.points])

    def get_range(
        self,
        start: float,
        end: float
    ) -> List[DataPoint]:
        """获取时间范围内的数据."""
        return [p for p in self.points if start <= p.timestamp <= end]

    def get_latest(self, n: int = 1) -> List[DataPoint]:
        """获取最新的n个点."""
        return self.points[-n:]

    def downsample(self, factor: int) -> 'DataSeries':
        """降采样."""
        new_series = DataSeries(
            name=f"{self.name}_downsampled",
            unit=self.unit,
            description=self.description,
        )
        for i in range(0, len(self.points), factor):
            chunk = self.points[i:i+factor]
            if chunk:
                avg_value = np.mean([p.value for p in chunk])
                avg_time = np.mean([p.timestamp for p in chunk])
                new_series.add_point(avg_time, avg_value)
        return new_series

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'unit': self.unit,
            'description': self.description,
            'tags': self.tags,
            'created_at': self.created_at.isoformat(),
            'points': [p.to_dict() for p in self.points],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataSeries':
        series = cls(
            name=data['name'],
            unit=data.get('unit', ''),
            description=data.get('description', ''),
            tags=data.get('tags', {}),
        )
        series.created_at = datetime.fromisoformat(data['created_at'])
        series.points = [DataPoint.from_dict(p) for p in data.get('points', [])]
        return series


class TimeSeriesStorage:
    """
    时序数据存储.

    提供内存时序数据存储，支持:
    - 多数据序列管理
    - 自动老化
    - 数据压缩
    - 导出/导入
    """

    def __init__(
        self,
        max_points_per_series: int = 100000,
        retention_seconds: float = 86400,  # 24小时
    ):
        """
        初始化存储.

        Args:
            max_points_per_series: 每个序列最大点数
            retention_seconds: 数据保留时间
        """
        self.max_points = max_points_per_series
        self.retention = retention_seconds

        # 数据序列
        self._series: Dict[str, DataSeries] = {}

        # 快速缓冲区 (用于高频写入)
        self._buffers: Dict[str, deque] = {}
        self._buffer_size = 1000

        # 统计
        self._stats = {
            'total_points': 0,
            'series_count': 0,
            'write_count': 0,
            'read_count': 0,
        }

        # 写入回调
        self._write_callbacks: List[Callable[[str, DataPoint], None]] = []

    # -------------------------------------------------------------------------
    # Series Management
    # -------------------------------------------------------------------------

    def create_series(
        self,
        name: str,
        unit: str = "",
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> DataSeries:
        """创建数据序列."""
        if name in self._series:
            return self._series[name]

        series = DataSeries(
            name=name,
            unit=unit,
            description=description,
            tags=tags or {},
        )
        self._series[name] = series
        self._buffers[name] = deque(maxlen=self._buffer_size)
        self._stats['series_count'] += 1

        logger.info(f"Created series: {name}")
        return series

    def get_series(self, name: str) -> Optional[DataSeries]:
        """获取数据序列."""
        return self._series.get(name)

    def list_series(self) -> List[str]:
        """列出所有序列名称."""
        return list(self._series.keys())

    def delete_series(self, name: str) -> bool:
        """删除数据序列."""
        if name in self._series:
            del self._series[name]
            del self._buffers[name]
            self._stats['series_count'] -= 1
            return True
        return False

    # -------------------------------------------------------------------------
    # Data Writing
    # -------------------------------------------------------------------------

    def write(
        self,
        series_name: str,
        value: float,
        timestamp: Optional[float] = None,
        quality: int = 100,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        写入数据点.

        Args:
            series_name: 序列名称
            value: 数值
            timestamp: 时间戳 (默认当前时间)
            quality: 数据质量
            tags: 标签
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()

        # 自动创建序列
        if series_name not in self._series:
            self.create_series(series_name)

        # 创建数据点
        point = DataPoint(
            timestamp=timestamp,
            value=value,
            quality=quality,
            tags=tags or {},
        )

        # 写入缓冲区
        self._buffers[series_name].append(point)

        # 写入序列
        series = self._series[series_name]
        series.points.append(point)

        # 限制点数
        if len(series.points) > self.max_points:
            series.points = series.points[-self.max_points:]

        self._stats['total_points'] += 1
        self._stats['write_count'] += 1

        # 触发回调
        for callback in self._write_callbacks:
            try:
                callback(series_name, point)
            except Exception as e:
                logger.error(f"Write callback error: {e}")

    def write_batch(
        self,
        series_name: str,
        values: List[Tuple[float, float]],
    ) -> None:
        """
        批量写入数据.

        Args:
            series_name: 序列名称
            values: (timestamp, value) 列表
        """
        for timestamp, value in values:
            self.write(series_name, value, timestamp)

    def write_dict(
        self,
        data: Dict[str, float],
        timestamp: Optional[float] = None,
    ) -> None:
        """
        写入字典数据 (多个序列).

        Args:
            data: {序列名: 值} 字典
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()

        for series_name, value in data.items():
            self.write(series_name, value, timestamp)

    # -------------------------------------------------------------------------
    # Data Reading
    # -------------------------------------------------------------------------

    def read(
        self,
        series_name: str,
        start: Optional[float] = None,
        end: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[DataPoint]:
        """
        读取数据.

        Args:
            series_name: 序列名称
            start: 开始时间戳
            end: 结束时间戳
            limit: 最大点数

        Returns:
            数据点列表
        """
        self._stats['read_count'] += 1

        series = self._series.get(series_name)
        if not series:
            return []

        points = series.points

        # 时间过滤
        if start is not None:
            points = [p for p in points if p.timestamp >= start]
        if end is not None:
            points = [p for p in points if p.timestamp <= end]

        # 限制数量
        if limit is not None:
            points = points[-limit:]

        return points

    def read_latest(
        self,
        series_name: str,
        n: int = 1,
    ) -> List[DataPoint]:
        """读取最新数据."""
        series = self._series.get(series_name)
        if not series:
            return []
        return series.get_latest(n)

    def read_value(
        self,
        series_name: str,
        default: float = 0.0,
    ) -> float:
        """读取最新值."""
        points = self.read_latest(series_name, 1)
        return points[0].value if points else default

    def read_buffer(self, series_name: str) -> List[DataPoint]:
        """读取缓冲区数据."""
        buffer = self._buffers.get(series_name)
        return list(buffer) if buffer else []

    # -------------------------------------------------------------------------
    # Data Aggregation
    # -------------------------------------------------------------------------

    def aggregate(
        self,
        series_name: str,
        start: float,
        end: float,
        interval: float,
        func: str = 'mean',
    ) -> List[Tuple[float, float]]:
        """
        聚合数据.

        Args:
            series_name: 序列名称
            start: 开始时间
            end: 结束时间
            interval: 聚合间隔 (秒)
            func: 聚合函数 (mean, max, min, sum, count)

        Returns:
            [(timestamp, value), ...] 列表
        """
        points = self.read(series_name, start, end)
        if not points:
            return []

        result = []
        current_bucket = start
        bucket_values = []

        for point in points:
            while point.timestamp >= current_bucket + interval:
                if bucket_values:
                    agg_value = self._aggregate_values(bucket_values, func)
                    result.append((current_bucket, agg_value))
                    bucket_values = []
                current_bucket += interval

            bucket_values.append(point.value)

        # 最后一个桶
        if bucket_values:
            agg_value = self._aggregate_values(bucket_values, func)
            result.append((current_bucket, agg_value))

        return result

    def _aggregate_values(self, values: List[float], func: str) -> float:
        """聚合值."""
        if not values:
            return 0.0

        if func == 'mean':
            return float(np.mean(values))
        elif func == 'max':
            return float(np.max(values))
        elif func == 'min':
            return float(np.min(values))
        elif func == 'sum':
            return float(np.sum(values))
        elif func == 'count':
            return float(len(values))
        elif func == 'std':
            return float(np.std(values))
        else:
            return float(np.mean(values))

    # -------------------------------------------------------------------------
    # Data Maintenance
    # -------------------------------------------------------------------------

    def cleanup(self, before: Optional[float] = None) -> int:
        """
        清理老数据.

        Args:
            before: 删除此时间戳之前的数据

        Returns:
            删除的点数
        """
        if before is None:
            before = datetime.now().timestamp() - self.retention

        removed = 0
        for series in self._series.values():
            original = len(series.points)
            series.points = [p for p in series.points if p.timestamp >= before]
            removed += original - len(series.points)

        self._stats['total_points'] -= removed
        return removed

    def compact(self, series_name: str, factor: int = 10) -> None:
        """
        压缩数据 (降采样旧数据).

        Args:
            series_name: 序列名称
            factor: 压缩因子
        """
        series = self._series.get(series_name)
        if not series or len(series.points) < factor * 2:
            return

        # 保留最新的数据，压缩旧数据
        keep_recent = self._buffer_size
        old_points = series.points[:-keep_recent]
        recent_points = series.points[-keep_recent:]

        # 降采样旧数据
        compacted = []
        for i in range(0, len(old_points), factor):
            chunk = old_points[i:i+factor]
            if chunk:
                avg_value = np.mean([p.value for p in chunk])
                avg_time = np.mean([p.timestamp for p in chunk])
                min_quality = min(p.quality for p in chunk)
                compacted.append(DataPoint(avg_time, avg_value, min_quality))

        series.points = compacted + recent_points

    # -------------------------------------------------------------------------
    # Export / Import
    # -------------------------------------------------------------------------

    def export_json(
        self,
        filepath: str,
        series_names: Optional[List[str]] = None,
        compress: bool = True,
    ) -> None:
        """
        导出为JSON文件.

        Args:
            filepath: 文件路径
            series_names: 要导出的序列 (None=全部)
            compress: 是否压缩
        """
        data = {
            'exported_at': datetime.now().isoformat(),
            'series': {},
        }

        names = series_names or list(self._series.keys())
        for name in names:
            series = self._series.get(name)
            if series:
                data['series'][name] = series.to_dict()

        content = json.dumps(data, indent=2)

        if compress:
            with gzip.open(filepath + '.gz', 'wt', encoding='utf-8') as f:
                f.write(content)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

        logger.info(f"Exported {len(names)} series to {filepath}")

    def import_json(self, filepath: str) -> int:
        """
        从JSON文件导入.

        Args:
            filepath: 文件路径

        Returns:
            导入的序列数
        """
        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                content = f.read()
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

        data = json.loads(content)

        count = 0
        for name, series_data in data.get('series', {}).items():
            series = DataSeries.from_dict(series_data)
            self._series[name] = series
            self._buffers[name] = deque(maxlen=self._buffer_size)
            self._stats['series_count'] += 1
            self._stats['total_points'] += len(series.points)
            count += 1

        logger.info(f"Imported {count} series from {filepath}")
        return count

    # -------------------------------------------------------------------------
    # Callbacks & Statistics
    # -------------------------------------------------------------------------

    def add_write_callback(
        self,
        callback: Callable[[str, DataPoint], None]
    ) -> None:
        """添加写入回调."""
        self._write_callbacks.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息."""
        series_stats = {}
        for name, series in self._series.items():
            series_stats[name] = {
                'points': len(series.points),
                'unit': series.unit,
            }

        return {
            **self._stats,
            'series': series_stats,
        }

    def get_series_stats(self, series_name: str) -> Dict[str, Any]:
        """获取序列统计."""
        series = self._series.get(series_name)
        if not series:
            return {}

        values = series.get_values()
        return {
            'count': len(values),
            'mean': float(np.mean(values)) if len(values) > 0 else 0.0,
            'std': float(np.std(values)) if len(values) > 0 else 0.0,
            'min': float(np.min(values)) if len(values) > 0 else 0.0,
            'max': float(np.max(values)) if len(values) > 0 else 0.0,
            'latest': values[-1] if len(values) > 0 else None,
        }
