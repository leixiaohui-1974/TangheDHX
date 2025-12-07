# -*- coding: utf-8 -*-
"""
Tests for Data Storage and Replay Module.

测试数据存储与回放模块。
"""

import pytest
import time
from datetime import datetime

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

    def test_to_dict(self):
        """Test data point serialization."""
        point = DataPoint(timestamp=1000.0, value=50.0, quality=100)
        d = point.to_dict()

        assert d['timestamp'] == 1000.0
        assert d['value'] == 50.0
        assert 'quality' in d


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

    def test_add_point(self):
        """Test adding points."""
        series = DataSeries(name='test')
        series.add_point(1.0, 10.0)
        series.add_point(2.0, 20.0)

        assert len(series.points) == 2

    def test_get_values(self):
        """Test getting values."""
        series = DataSeries(name='test')
        series.add_point(1.0, 10.0)
        series.add_point(2.0, 20.0)
        series.add_point(3.0, 30.0)

        values = series.get_values()
        assert len(values) == 3
        assert values[0] == 10.0

    def test_get_range(self):
        """Test getting range."""
        series = DataSeries(name='test')
        for i in range(10):
            series.add_point(float(i), float(i * 10))

        points = series.get_range(2.0, 5.0)
        assert len(points) == 4  # Points 2, 3, 4, 5

    def test_get_latest(self):
        """Test getting latest points."""
        series = DataSeries(name='test')
        for i in range(10):
            series.add_point(float(i), float(i))

        latest = series.get_latest(3)
        assert len(latest) == 3
        assert latest[-1].value == 9.0

    def test_downsample(self):
        """Test downsampling."""
        series = DataSeries(name='test')
        for i in range(100):
            series.add_point(float(i), float(i))

        downsampled = series.downsample(10)
        assert len(downsampled.points) == 10

    def test_to_dict(self):
        """Test series serialization."""
        series = DataSeries(name='flow', unit='m³/s')
        series.add_point(1.0, 10.0)

        d = series.to_dict()
        assert d['name'] == 'flow'
        assert d['unit'] == 'm³/s'
        assert len(d['points']) == 1


# =============================================================================
# TimeSeriesStorage Tests
# =============================================================================

class TestTimeSeriesStorage:
    """Tests for TimeSeriesStorage."""

    def test_initialization(self):
        """Test storage initialization."""
        storage = TimeSeriesStorage()
        assert len(storage.list_series()) == 0

    def test_create_series(self):
        """Test creating a series."""
        storage = TimeSeriesStorage()
        storage.create_series('flow', 'm³/s')

        assert 'flow' in storage.list_series()

    def test_write_creates_series(self):
        """Test writing to non-existent series creates it."""
        storage = TimeSeriesStorage()
        ts = time.time()
        storage.write('new_series', 100.0, ts)

        assert 'new_series' in storage.list_series()

    def test_write_read(self):
        """Test writing and reading data."""
        storage = TimeSeriesStorage()
        storage.create_series('temperature')

        ts = time.time()
        storage.write('temperature', 25.5, ts)
        storage.write('temperature', 26.0, ts + 1)

        points = storage.read('temperature', ts - 1, ts + 2)
        assert len(points) == 2

    def test_delete_series(self):
        """Test deleting a series."""
        storage = TimeSeriesStorage()
        storage.create_series('test')
        storage.write('test', 10.0, 1.0)

        storage.delete_series('test')
        assert 'test' not in storage.list_series()


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
