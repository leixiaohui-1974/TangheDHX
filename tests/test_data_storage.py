# -*- coding: utf-8 -*-
"""
Tests for Data Storage and Replay Module.

测试数据存储与回放模块，包括：
- DataPoint 数据点
- DataSeries 数据序列
- TimeSeriesStorage 时序存储
- 数据聚合、清理、压缩
- 导入/导出功能
- 回调机制
- HistoryReplay 历史回放
"""

import pytest
import time
import tempfile
import os
import json
import gzip
from datetime import datetime
import numpy as np

from src.data.storage import TimeSeriesStorage, DataPoint, DataSeries
from src.data.replay import (
    HistoryReplay, ReplayMode, ReplayState, ReplaySession, ReplayEvent
)


# =============================================================================
# DataPoint Tests
# =============================================================================

class TestDataPoint:
    """Tests for DataPoint."""

    def test_creation(self):
        """Test data point creation."""
        ts = time.time()
        point = DataPoint(timestamp=ts, value=42.0)

        assert point.timestamp == ts
        assert point.value == 42.0
        assert point.quality == 100  # Default

    def test_with_quality(self):
        """Test data point with quality."""
        point = DataPoint(timestamp=1000.0, value=10.0, quality=90)
        assert point.quality == 90

    def test_with_tags(self):
        """Test data point with tags."""
        tags = {'source': 'sensor_1', 'unit': 'celsius'}
        point = DataPoint(timestamp=1000.0, value=25.0, quality=100, tags=tags)

        assert point.tags['source'] == 'sensor_1'
        assert point.tags['unit'] == 'celsius'

    def test_to_dict(self):
        """Test data point serialization."""
        point = DataPoint(timestamp=1000.0, value=50.0, quality=100)
        d = point.to_dict()

        assert d['timestamp'] == 1000.0
        assert d['value'] == 50.0
        assert 'quality' in d

    def test_from_dict(self):
        """Test data point deserialization."""
        data = {
            'timestamp': 2000.0,
            'value': 75.5,
            'quality': 95,
            'tags': {'zone': 'A'}
        }
        point = DataPoint.from_dict(data)

        assert point.timestamp == 2000.0
        assert point.value == 75.5
        assert point.quality == 95
        assert point.tags['zone'] == 'A'

    def test_round_trip(self):
        """Test serialization round trip."""
        original = DataPoint(
            timestamp=1500.0,
            value=33.3,
            quality=88,
            tags={'type': 'flow'}
        )
        restored = DataPoint.from_dict(original.to_dict())

        assert restored.timestamp == original.timestamp
        assert restored.value == original.value
        assert restored.quality == original.quality
        assert restored.tags == original.tags


# =============================================================================
# DataSeries Tests
# =============================================================================

class TestDataSeries:
    """Tests for DataSeries."""

    def test_creation(self):
        """Test series creation."""
        series = DataSeries(name='flow', unit='m³/s')

        assert series.name == 'flow'
        assert series.unit == 'm³/s'
        assert len(series.points) == 0

    def test_creation_with_description(self):
        """Test series creation with description."""
        series = DataSeries(
            name='temperature',
            unit='°C',
            description='Zone A temperature sensor'
        )

        assert series.description == 'Zone A temperature sensor'

    def test_creation_with_tags(self):
        """Test series creation with tags."""
        tags = {'zone': 'A', 'sensor_type': 'RTD'}
        series = DataSeries(name='temp', tags=tags)

        assert series.tags['zone'] == 'A'
        assert series.tags['sensor_type'] == 'RTD'

    def test_add_point(self):
        """Test adding points."""
        series = DataSeries(name='test')
        series.add_point(1.0, 10.0)
        series.add_point(2.0, 20.0)

        assert len(series.points) == 2

    def test_add_point_with_quality(self):
        """Test adding points with quality."""
        series = DataSeries(name='test')
        series.add_point(1.0, 10.0, quality=90)

        assert series.points[0].quality == 90

    def test_get_values(self):
        """Test getting values."""
        series = DataSeries(name='test')
        series.add_point(1.0, 10.0)
        series.add_point(2.0, 20.0)
        series.add_point(3.0, 30.0)

        values = series.get_values()
        assert len(values) == 3
        assert values[0] == 10.0

    def test_get_timestamps(self):
        """Test getting timestamps."""
        series = DataSeries(name='test')
        series.add_point(100.0, 10.0)
        series.add_point(200.0, 20.0)
        series.add_point(300.0, 30.0)

        timestamps = series.get_timestamps()
        assert len(timestamps) == 3
        assert timestamps[0] == 100.0
        assert timestamps[2] == 300.0

    def test_get_range(self):
        """Test getting range."""
        series = DataSeries(name='test')
        for i in range(10):
            series.add_point(float(i), float(i * 10))

        points = series.get_range(2.0, 5.0)
        assert len(points) == 4  # Points 2, 3, 4, 5

    def test_get_range_empty(self):
        """Test getting range with no matching points."""
        series = DataSeries(name='test')
        series.add_point(1.0, 10.0)
        series.add_point(2.0, 20.0)

        points = series.get_range(10.0, 20.0)
        assert len(points) == 0

    def test_get_latest(self):
        """Test getting latest points."""
        series = DataSeries(name='test')
        for i in range(10):
            series.add_point(float(i), float(i))

        latest = series.get_latest(3)
        assert len(latest) == 3
        assert latest[-1].value == 9.0

    def test_get_latest_more_than_available(self):
        """Test getting latest when requesting more than available."""
        series = DataSeries(name='test')
        series.add_point(1.0, 10.0)
        series.add_point(2.0, 20.0)

        latest = series.get_latest(10)
        assert len(latest) == 2

    def test_downsample(self):
        """Test downsampling."""
        series = DataSeries(name='test')
        for i in range(100):
            series.add_point(float(i), float(i))

        downsampled = series.downsample(10)
        assert len(downsampled.points) == 10
        assert '_downsampled' in downsampled.name

    def test_downsample_partial(self):
        """Test downsampling with non-divisible length."""
        series = DataSeries(name='test')
        for i in range(25):
            series.add_point(float(i), float(i))

        downsampled = series.downsample(10)
        assert len(downsampled.points) == 3  # 25 / 10 = 2.5, rounds to 3

    def test_to_dict(self):
        """Test series serialization."""
        series = DataSeries(name='flow', unit='m³/s')
        series.add_point(1.0, 10.0)

        d = series.to_dict()
        assert d['name'] == 'flow'
        assert d['unit'] == 'm³/s'
        assert len(d['points']) == 1
        assert 'created_at' in d

    def test_from_dict(self):
        """Test series deserialization."""
        data = {
            'name': 'pressure',
            'unit': 'bar',
            'description': 'Main pressure sensor',
            'tags': {'location': 'inlet'},
            'created_at': '2024-01-15T10:30:00',
            'points': [
                {'timestamp': 1000.0, 'value': 5.5, 'quality': 100, 'tags': {}},
                {'timestamp': 1001.0, 'value': 5.6, 'quality': 100, 'tags': {}},
            ]
        }
        series = DataSeries.from_dict(data)

        assert series.name == 'pressure'
        assert series.unit == 'bar'
        assert len(series.points) == 2
        assert series.points[0].value == 5.5

    def test_round_trip(self):
        """Test serialization round trip."""
        original = DataSeries(
            name='velocity',
            unit='m/s',
            description='Flow velocity'
        )
        original.add_point(100.0, 2.5)
        original.add_point(101.0, 2.6)

        restored = DataSeries.from_dict(original.to_dict())

        assert restored.name == original.name
        assert restored.unit == original.unit
        assert len(restored.points) == len(original.points)


# =============================================================================
# TimeSeriesStorage Tests
# =============================================================================

class TestTimeSeriesStorage:
    """Tests for TimeSeriesStorage."""

    def test_initialization(self):
        """Test storage initialization."""
        storage = TimeSeriesStorage()
        assert len(storage.list_series()) == 0

    def test_initialization_with_params(self):
        """Test storage initialization with custom parameters."""
        storage = TimeSeriesStorage(
            max_points_per_series=5000,
            retention_seconds=3600
        )
        assert storage.max_points == 5000
        assert storage.retention == 3600

    def test_create_series(self):
        """Test creating a series."""
        storage = TimeSeriesStorage()
        series = storage.create_series('flow', 'm³/s', 'Flow measurement')

        assert 'flow' in storage.list_series()
        assert series.unit == 'm³/s'
        assert series.description == 'Flow measurement'

    def test_create_series_with_tags(self):
        """Test creating a series with tags."""
        storage = TimeSeriesStorage()
        tags = {'zone': 'A', 'type': 'primary'}
        series = storage.create_series('temp', tags=tags)

        assert series.tags['zone'] == 'A'

    def test_create_series_duplicate(self):
        """Test creating duplicate series returns existing."""
        storage = TimeSeriesStorage()
        series1 = storage.create_series('flow')
        series1.add_point(1.0, 10.0)

        series2 = storage.create_series('flow')

        # Should return existing series
        assert len(series2.points) == 1

    def test_get_series(self):
        """Test getting a series."""
        storage = TimeSeriesStorage()
        storage.create_series('test')

        series = storage.get_series('test')
        assert series is not None
        assert series.name == 'test'

    def test_get_series_nonexistent(self):
        """Test getting non-existent series."""
        storage = TimeSeriesStorage()

        series = storage.get_series('nonexistent')
        assert series is None

    def test_write_creates_series(self):
        """Test writing to non-existent series creates it."""
        storage = TimeSeriesStorage()
        ts = time.time()
        storage.write('new_series', 100.0, ts)

        assert 'new_series' in storage.list_series()

    def test_write_with_quality(self):
        """Test writing with quality."""
        storage = TimeSeriesStorage()
        storage.write('test', 50.0, 1000.0, quality=85)

        points = storage.read('test')
        assert points[0].quality == 85

    def test_write_with_tags(self):
        """Test writing with tags."""
        storage = TimeSeriesStorage()
        tags = {'source': 'manual'}
        storage.write('test', 50.0, 1000.0, tags=tags)

        points = storage.read('test')
        assert points[0].tags['source'] == 'manual'

    def test_write_auto_timestamp(self):
        """Test writing with auto timestamp."""
        storage = TimeSeriesStorage()
        before = time.time()
        storage.write('test', 100.0)
        after = time.time()

        points = storage.read('test')
        assert before <= points[0].timestamp <= after

    def test_write_read(self):
        """Test writing and reading data."""
        storage = TimeSeriesStorage()
        storage.create_series('temperature')

        ts = time.time()
        storage.write('temperature', 25.5, ts)
        storage.write('temperature', 26.0, ts + 1)

        points = storage.read('temperature', ts - 1, ts + 2)
        assert len(points) == 2

    def test_write_max_points_limit(self):
        """Test write respects max points limit."""
        storage = TimeSeriesStorage(max_points_per_series=10)

        for i in range(20):
            storage.write('test', float(i), float(i))

        points = storage.read('test')
        assert len(points) == 10

    def test_write_batch(self):
        """Test batch writing."""
        storage = TimeSeriesStorage()

        values = [(float(i), float(i * 10)) for i in range(10)]
        storage.write_batch('batch_test', values)

        points = storage.read('batch_test')
        assert len(points) == 10

    def test_write_dict(self):
        """Test dictionary writing (multiple series)."""
        storage = TimeSeriesStorage()

        data = {
            'temp': 25.0,
            'pressure': 1.5,
            'flow': 100.0
        }
        ts = 1000.0
        storage.write_dict(data, ts)

        assert storage.read_value('temp') == 25.0
        assert storage.read_value('pressure') == 1.5
        assert storage.read_value('flow') == 100.0

    def test_write_dict_auto_timestamp(self):
        """Test dictionary writing with auto timestamp."""
        storage = TimeSeriesStorage()

        data = {'a': 1.0, 'b': 2.0}
        storage.write_dict(data)

        assert 'a' in storage.list_series()
        assert 'b' in storage.list_series()

    def test_read_nonexistent(self):
        """Test reading non-existent series."""
        storage = TimeSeriesStorage()

        points = storage.read('nonexistent')
        assert len(points) == 0

    def test_read_with_limit(self):
        """Test reading with limit."""
        storage = TimeSeriesStorage()

        for i in range(100):
            storage.write('test', float(i), float(i))

        points = storage.read('test', limit=10)
        assert len(points) == 10

    def test_read_latest(self):
        """Test reading latest values."""
        storage = TimeSeriesStorage()

        for i in range(10):
            storage.write('test', float(i), float(i))

        latest = storage.read_latest('test', n=3)
        assert len(latest) == 3
        assert latest[-1].value == 9.0

    def test_read_latest_nonexistent(self):
        """Test reading latest from non-existent series."""
        storage = TimeSeriesStorage()

        latest = storage.read_latest('nonexistent')
        assert len(latest) == 0

    def test_read_value(self):
        """Test reading single value."""
        storage = TimeSeriesStorage()
        storage.write('test', 42.0, 1000.0)

        value = storage.read_value('test')
        assert value == 42.0

    def test_read_value_default(self):
        """Test reading value with default."""
        storage = TimeSeriesStorage()

        value = storage.read_value('nonexistent', default=99.0)
        assert value == 99.0

    def test_read_buffer(self):
        """Test reading buffer."""
        storage = TimeSeriesStorage()

        for i in range(5):
            storage.write('test', float(i), float(i))

        buffer = storage.read_buffer('test')
        assert len(buffer) == 5

    def test_read_buffer_nonexistent(self):
        """Test reading buffer for non-existent series."""
        storage = TimeSeriesStorage()

        buffer = storage.read_buffer('nonexistent')
        assert len(buffer) == 0

    def test_delete_series(self):
        """Test deleting a series."""
        storage = TimeSeriesStorage()
        storage.create_series('test')
        storage.write('test', 10.0, 1.0)

        result = storage.delete_series('test')
        assert result is True
        assert 'test' not in storage.list_series()

    def test_delete_series_nonexistent(self):
        """Test deleting non-existent series."""
        storage = TimeSeriesStorage()

        result = storage.delete_series('nonexistent')
        assert result is False


class TestTimeSeriesStorageAggregation:
    """Tests for TimeSeriesStorage aggregation methods."""

    def test_aggregate_mean(self):
        """Test mean aggregation."""
        storage = TimeSeriesStorage()

        # Create data: values 0-9 at times 0-9
        for i in range(10):
            storage.write('test', float(i), float(i))

        # Aggregate in intervals of 5
        result = storage.aggregate('test', 0.0, 10.0, 5.0, 'mean')

        assert len(result) == 2
        # First bucket (0-5): mean of 0,1,2,3,4 = 2.0
        assert abs(result[0][1] - 2.0) < 0.1

    def test_aggregate_max(self):
        """Test max aggregation."""
        storage = TimeSeriesStorage()

        for i in range(10):
            storage.write('test', float(i), float(i))

        result = storage.aggregate('test', 0.0, 10.0, 5.0, 'max')

        assert len(result) == 2
        assert result[0][1] == 4.0  # Max of 0-4

    def test_aggregate_min(self):
        """Test min aggregation."""
        storage = TimeSeriesStorage()

        for i in range(10):
            storage.write('test', float(i * 10), float(i))

        result = storage.aggregate('test', 0.0, 100.0, 50.0, 'min')

        assert result[0][1] == 0.0  # Min of first 5 values

    def test_aggregate_sum(self):
        """Test sum aggregation."""
        storage = TimeSeriesStorage()

        for i in range(10):
            storage.write('test', float(i), float(i))

        result = storage.aggregate('test', 0.0, 10.0, 5.0, 'sum')

        # First bucket: sum of 0+1+2+3+4 = 10
        assert result[0][1] == 10.0

    def test_aggregate_count(self):
        """Test count aggregation."""
        storage = TimeSeriesStorage()

        for i in range(10):
            storage.write('test', float(i), float(i))

        result = storage.aggregate('test', 0.0, 10.0, 5.0, 'count')

        assert result[0][1] == 5.0

    def test_aggregate_std(self):
        """Test std aggregation."""
        storage = TimeSeriesStorage()

        for i in range(10):
            storage.write('test', float(i), float(i))

        result = storage.aggregate('test', 0.0, 10.0, 5.0, 'std')

        assert len(result) == 2
        assert result[0][1] > 0  # Std should be positive

    def test_aggregate_unknown_func(self):
        """Test aggregation with unknown function (defaults to mean)."""
        storage = TimeSeriesStorage()

        for i in range(10):
            storage.write('test', float(i), float(i))

        result = storage.aggregate('test', 0.0, 10.0, 5.0, 'unknown')

        # Should default to mean
        assert len(result) == 2

    def test_aggregate_empty(self):
        """Test aggregation on empty series."""
        storage = TimeSeriesStorage()

        result = storage.aggregate('empty', 0.0, 10.0, 1.0)

        assert len(result) == 0

    def test_aggregate_empty_values(self):
        """Test _aggregate_values with empty list."""
        storage = TimeSeriesStorage()

        result = storage._aggregate_values([], 'mean')
        assert result == 0.0


class TestTimeSeriesStorageMaintenance:
    """Tests for TimeSeriesStorage maintenance methods."""

    def test_cleanup(self):
        """Test data cleanup."""
        storage = TimeSeriesStorage()

        # Write old data
        storage.write('test', 10.0, 1000.0)
        storage.write('test', 20.0, 2000.0)
        storage.write('test', 30.0, 3000.0)

        # Cleanup data before 2500
        removed = storage.cleanup(before=2500.0)

        assert removed == 2
        points = storage.read('test')
        assert len(points) == 1
        assert points[0].value == 30.0

    def test_cleanup_default_retention(self):
        """Test cleanup with default retention."""
        storage = TimeSeriesStorage(retention_seconds=60)

        now = time.time()
        storage.write('test', 10.0, now - 120)  # 2 minutes ago
        storage.write('test', 20.0, now)  # Now

        # Default cleanup uses retention_seconds
        removed = storage.cleanup()

        assert removed == 1

    def test_compact(self):
        """Test data compaction."""
        storage = TimeSeriesStorage()
        storage._buffer_size = 100  # Override for test

        # Write many points
        for i in range(2000):
            storage.write('test', float(i), float(i))

        original_len = len(storage.get_series('test').points)

        # Compact with factor 10
        storage.compact('test', factor=10)

        new_len = len(storage.get_series('test').points)
        assert new_len < original_len

    def test_compact_short_series(self):
        """Test compact on short series (no-op)."""
        storage = TimeSeriesStorage()

        storage.write('test', 10.0, 1.0)
        storage.write('test', 20.0, 2.0)

        # Should not error on short series
        storage.compact('test', factor=10)

    def test_compact_nonexistent(self):
        """Test compact on non-existent series."""
        storage = TimeSeriesStorage()

        # Should not error
        storage.compact('nonexistent', factor=10)


class TestTimeSeriesStorageExportImport:
    """Tests for TimeSeriesStorage export/import methods."""

    def test_export_json_uncompressed(self):
        """Test JSON export without compression."""
        storage = TimeSeriesStorage()
        storage.create_series('flow', 'm³/s')
        storage.write('flow', 100.0, 1000.0)
        storage.write('flow', 110.0, 1001.0)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            storage.export_json(filepath, compress=False)

            with open(filepath, 'r') as f:
                data = json.load(f)

            assert 'exported_at' in data
            assert 'series' in data
            assert 'flow' in data['series']
            assert len(data['series']['flow']['points']) == 2
        finally:
            os.unlink(filepath)

    def test_export_json_compressed(self):
        """Test JSON export with compression."""
        storage = TimeSeriesStorage()
        storage.create_series('temp')
        storage.write('temp', 25.0, 1000.0)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            storage.export_json(filepath, compress=True)

            # Compressed file should exist
            assert os.path.exists(filepath + '.gz')

            with gzip.open(filepath + '.gz', 'rt') as f:
                data = json.load(f)

            assert 'temp' in data['series']
        finally:
            if os.path.exists(filepath + '.gz'):
                os.unlink(filepath + '.gz')
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_export_json_selective(self):
        """Test exporting specific series."""
        storage = TimeSeriesStorage()
        storage.write('a', 1.0, 1.0)
        storage.write('b', 2.0, 1.0)
        storage.write('c', 3.0, 1.0)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            storage.export_json(filepath, series_names=['a', 'c'], compress=False)

            with open(filepath, 'r') as f:
                data = json.load(f)

            assert 'a' in data['series']
            assert 'b' not in data['series']
            assert 'c' in data['series']
        finally:
            os.unlink(filepath)

    def test_import_json_uncompressed(self):
        """Test JSON import without compression."""
        # Create and export data
        storage1 = TimeSeriesStorage()
        storage1.write('test', 42.0, 1000.0)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            storage1.export_json(filepath, compress=False)

            # Import into new storage
            storage2 = TimeSeriesStorage()
            count = storage2.import_json(filepath)

            assert count == 1
            assert 'test' in storage2.list_series()
            assert storage2.read_value('test') == 42.0
        finally:
            os.unlink(filepath)

    def test_import_json_compressed(self):
        """Test JSON import with compression."""
        storage1 = TimeSeriesStorage()
        storage1.write('compressed', 99.0, 2000.0)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            storage1.export_json(filepath, compress=True)

            storage2 = TimeSeriesStorage()
            count = storage2.import_json(filepath + '.gz')

            assert count == 1
            assert storage2.read_value('compressed') == 99.0
        finally:
            if os.path.exists(filepath + '.gz'):
                os.unlink(filepath + '.gz')


class TestTimeSeriesStorageCallbacks:
    """Tests for TimeSeriesStorage callback methods."""

    def test_add_write_callback(self):
        """Test adding write callback."""
        storage = TimeSeriesStorage()
        callback_data = []

        def callback(series_name, point):
            callback_data.append((series_name, point.value))

        storage.add_write_callback(callback)
        storage.write('test', 123.0, 1000.0)

        assert len(callback_data) == 1
        assert callback_data[0] == ('test', 123.0)

    def test_multiple_callbacks(self):
        """Test multiple write callbacks."""
        storage = TimeSeriesStorage()
        results = {'cb1': 0, 'cb2': 0}

        def cb1(name, point):
            results['cb1'] += 1

        def cb2(name, point):
            results['cb2'] += 1

        storage.add_write_callback(cb1)
        storage.add_write_callback(cb2)
        storage.write('test', 1.0, 1.0)

        assert results['cb1'] == 1
        assert results['cb2'] == 1

    def test_callback_error_handling(self):
        """Test that callback errors don't break writes."""
        storage = TimeSeriesStorage()

        def bad_callback(name, point):
            raise ValueError("Callback error")

        storage.add_write_callback(bad_callback)

        # Should not raise despite callback error
        storage.write('test', 1.0, 1.0)

        # Data should still be written
        assert storage.read_value('test') == 1.0


class TestTimeSeriesStorageStatistics:
    """Tests for TimeSeriesStorage statistics methods."""

    def test_get_stats(self):
        """Test getting storage stats."""
        storage = TimeSeriesStorage()
        storage.write('a', 1.0, 1.0)
        storage.write('a', 2.0, 2.0)
        storage.write('b', 3.0, 1.0)

        stats = storage.get_stats()

        assert stats['series_count'] == 2
        assert stats['write_count'] == 3
        assert 'series' in stats
        assert 'a' in stats['series']
        assert stats['series']['a']['points'] == 2

    def test_get_series_stats(self):
        """Test getting series statistics."""
        storage = TimeSeriesStorage()

        for i in range(10):
            storage.write('test', float(i), float(i))

        stats = storage.get_series_stats('test')

        assert stats['count'] == 10
        assert stats['mean'] == 4.5
        assert stats['min'] == 0.0
        assert stats['max'] == 9.0
        assert stats['latest'] == 9.0

    def test_get_series_stats_empty(self):
        """Test getting stats for empty series."""
        storage = TimeSeriesStorage()
        storage.create_series('empty')

        stats = storage.get_series_stats('empty')

        assert stats['count'] == 0
        assert stats['mean'] == 0.0

    def test_get_series_stats_nonexistent(self):
        """Test getting stats for non-existent series."""
        storage = TimeSeriesStorage()

        stats = storage.get_series_stats('nonexistent')

        assert stats == {}


# =============================================================================
# ReplaySession Tests
# =============================================================================

class TestReplaySession:
    """Tests for ReplaySession."""

    def test_creation(self):
        """Test session creation."""
        session = ReplaySession(
            session_id="test_001",
            start_time=0.0,
            end_time=100.0
        )

        assert session.session_id == "test_001"
        assert session.start_time == 0.0
        assert session.end_time == 100.0
        assert session.state == ReplayState.STOPPED

    def test_progress(self):
        """Test progress calculation."""
        session = ReplaySession(
            session_id="test",
            start_time=0.0,
            end_time=100.0,
            current_time=50.0
        )

        assert session.progress == 0.5

    def test_remaining_time(self):
        """Test remaining time calculation."""
        session = ReplaySession(
            session_id="test",
            start_time=0.0,
            end_time=100.0,
            current_time=30.0
        )

        assert session.remaining_time == 70.0


class TestReplayEvent:
    """Tests for ReplayEvent."""

    def test_creation(self):
        """Test event creation."""
        event = ReplayEvent(
            timestamp=50.0,
            event_type='anomaly',
            data={'severity': 'high'}
        )

        assert event.timestamp == 50.0
        assert event.event_type == 'anomaly'
        assert event.data['severity'] == 'high'


class TestHistoryReplay:
    """Tests for HistoryReplay."""

    def test_initialization(self):
        """Test replay initialization."""
        storage = TimeSeriesStorage()
        replay = HistoryReplay(storage)
        assert replay.storage == storage
        assert replay.get_session() is None

    def test_create_session(self):
        """Test creating replay session."""
        storage = TimeSeriesStorage()
        storage.create_series('flow')

        for i in range(100):
            storage.write('flow', float(i), float(i * 10))

        replay = HistoryReplay(storage)
        session = replay.create_session(0.0, 100.0)

        assert session is not None
        assert session.start_time == 0.0
        assert session.end_time == 100.0

    def test_play_pause_stop(self):
        """Test playback controls."""
        storage = TimeSeriesStorage()
        storage.create_series('test')
        storage.write('test', 1.0, 10.0)

        replay = HistoryReplay(storage)
        replay.create_session(0.0, 100.0)

        replay.play()
        assert replay._session.state == ReplayState.PLAYING

        replay.pause()
        assert replay._session.state == ReplayState.PAUSED

        replay.stop()
        assert replay._session.state == ReplayState.STOPPED

    def test_seek(self):
        """Test seeking to timestamp."""
        storage = TimeSeriesStorage()
        storage.create_series('test')
        storage.write('test', 1.0, 10.0)

        replay = HistoryReplay(storage)
        replay.create_session(0.0, 100.0)

        replay.seek(50.0)
        assert replay._session.current_time == 50.0

    def test_set_speed(self):
        """Test setting playback speed."""
        storage = TimeSeriesStorage()
        storage.create_series('test')

        replay = HistoryReplay(storage)
        replay.create_session(0.0, 100.0)

        replay.set_speed(2.0)
        assert replay._session.speed == 2.0

    def test_add_event(self):
        """Test adding events."""
        storage = TimeSeriesStorage()
        storage.create_series('test')

        replay = HistoryReplay(storage)
        replay.create_session(0.0, 100.0)

        event = ReplayEvent(timestamp=50.0, event_type='test')
        replay.add_event(event)

        assert len(replay._events) == 1

    def test_get_summary(self):
        """Test getting replay summary."""
        storage = TimeSeriesStorage()
        storage.create_series('test')

        replay = HistoryReplay(storage)
        replay.create_session(0.0, 100.0, speed=2.0)

        summary = replay.get_summary()

        assert summary['session_id'] is not None
        assert summary['speed'] == 2.0
        assert 'progress' in summary
