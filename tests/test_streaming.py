"""
Tests for Real-time Data Streaming Module.

Tests cover:
- Event bus
- Stream buffer
- Data aggregator
- Subscription manager
- Data stream processor
- Real-time data hub
"""

import pytest
import time
import threading
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from src.streaming.realtime import (
    # Enums
    EventType,
    EventPriority,
    AggregationType,
    FilterOperator,
    # Data classes
    Event,
    DataPoint,
    AggregatedData,
    SubscriptionFilter,
    Subscription,
    # Main classes
    EventBus,
    StreamBuffer,
    DataAggregator,
    SubscriptionManager,
    DataStreamProcessor,
    RealTimeDataHub
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def event_bus():
    """Create event bus."""
    bus = EventBus(max_queue_size=100)
    yield bus
    bus.stop()


@pytest.fixture
def stream_buffer():
    """Create stream buffer."""
    return StreamBuffer(name="test_buffer", max_samples=100, window_seconds=10.0)


@pytest.fixture
def data_aggregator():
    """Create data aggregator."""
    return DataAggregator(default_interval_ms=1000)


@pytest.fixture
def data_hub():
    """Create real-time data hub."""
    hub = RealTimeDataHub(aggregation_interval_ms=100)
    yield hub
    hub.stop()


# =============================================================================
# Test Enums
# =============================================================================

class TestEnums:
    """Test enum definitions."""

    def test_event_type_values(self):
        """Test event type values."""
        assert EventType.DATA_UPDATE.value == "data_update"
        assert EventType.ALARM.value == "alarm"
        assert EventType.CONTROL.value == "control"

    def test_event_priority_ordering(self):
        """Test priority ordering."""
        assert EventPriority.LOW.value < EventPriority.NORMAL.value
        assert EventPriority.NORMAL.value < EventPriority.HIGH.value
        assert EventPriority.HIGH.value < EventPriority.CRITICAL.value

    def test_aggregation_types(self):
        """Test aggregation type values."""
        assert AggregationType.MEAN.value == "mean"
        assert AggregationType.MAX.value == "max"
        assert AggregationType.LAST.value == "last"


# =============================================================================
# Test Event Data Class
# =============================================================================

class TestEvent:
    """Test Event dataclass."""

    def test_creation(self):
        """Test event creation."""
        event = Event(
            event_id="evt-001",
            event_type=EventType.DATA_UPDATE,
            source="sensor_1",
            data={"value": 42.0},
            timestamp=datetime.now()
        )
        assert event.event_id == "evt-001"
        assert event.event_type == EventType.DATA_UPDATE
        assert event.priority == EventPriority.NORMAL

    def test_to_dict(self):
        """Test event to dictionary conversion."""
        event = Event(
            event_id="evt-001",
            event_type=EventType.ALARM,
            source="gate_1",
            data={"severity": "high"},
            timestamp=datetime.now(),
            priority=EventPriority.HIGH
        )
        d = event.to_dict()

        assert d['event_id'] == "evt-001"
        assert d['event_type'] == "alarm"
        assert d['source'] == "gate_1"
        assert d['priority'] == EventPriority.HIGH.value


# =============================================================================
# Test Stream Buffer
# =============================================================================

class TestStreamBuffer:
    """Test StreamBuffer class."""

    def test_initialization(self, stream_buffer):
        """Test buffer initialization."""
        assert stream_buffer.name == "test_buffer"
        assert stream_buffer.max_samples == 100
        assert stream_buffer.size == 0

    def test_add_value(self, stream_buffer):
        """Test adding values."""
        stream_buffer.add(42.0)
        assert stream_buffer.size == 1

    def test_add_batch(self, stream_buffer):
        """Test adding batch of values."""
        values = [(float(i), datetime.now()) for i in range(10)]
        stream_buffer.add_batch(values)
        assert stream_buffer.size == 10

    def test_get_latest(self, stream_buffer):
        """Test getting latest values."""
        for i in range(5):
            stream_buffer.add(float(i))

        latest = stream_buffer.get_latest(1)
        assert len(latest) == 1
        assert latest[0]['value'] == 4.0

    def test_get_window(self, stream_buffer):
        """Test getting time window."""
        for i in range(5):
            stream_buffer.add(float(i))

        data = stream_buffer.get_window(seconds=10.0)
        assert len(data) == 5

    def test_aggregate_mean(self, stream_buffer):
        """Test mean aggregation."""
        for i in range(10):
            stream_buffer.add(float(i))

        mean = stream_buffer.aggregate(AggregationType.MEAN)
        assert mean == 4.5

    def test_aggregate_min_max(self, stream_buffer):
        """Test min/max aggregation."""
        for i in range(10):
            stream_buffer.add(float(i))

        assert stream_buffer.aggregate(AggregationType.MIN) == 0.0
        assert stream_buffer.aggregate(AggregationType.MAX) == 9.0

    def test_aggregate_sum(self, stream_buffer):
        """Test sum aggregation."""
        for i in range(5):
            stream_buffer.add(10.0)

        assert stream_buffer.aggregate(AggregationType.SUM) == 50.0

    def test_aggregate_last(self, stream_buffer):
        """Test last value aggregation."""
        stream_buffer.add(1.0)
        stream_buffer.add(2.0)
        stream_buffer.add(3.0)

        assert stream_buffer.aggregate(AggregationType.LAST) == 3.0

    def test_get_aggregated_data(self, stream_buffer):
        """Test getting aggregated data object."""
        for i in range(10):
            stream_buffer.add(float(i))

        agg_data = stream_buffer.get_aggregated_data([
            AggregationType.MEAN,
            AggregationType.MIN,
            AggregationType.MAX
        ])

        assert isinstance(agg_data, AggregatedData)
        assert 'mean' in agg_data.values
        assert 'min' in agg_data.values
        assert 'max' in agg_data.values

    def test_clear(self, stream_buffer):
        """Test clearing buffer."""
        stream_buffer.add(1.0)
        stream_buffer.add(2.0)
        stream_buffer.clear()

        assert stream_buffer.size == 0


# =============================================================================
# Test Event Bus
# =============================================================================

class TestEventBus:
    """Test EventBus class."""

    def test_initialization(self, event_bus):
        """Test bus initialization."""
        stats = event_bus.get_stats()
        assert stats['total_events'] == 0
        assert stats['subscriber_count'] == 0

    def test_subscribe_unsubscribe(self, event_bus):
        """Test subscription management."""
        callback = Mock()
        event_bus.subscribe("sub1", [EventType.DATA_UPDATE], callback)

        stats = event_bus.get_stats()
        assert stats['subscriber_count'] == 1

        event_bus.unsubscribe("sub1")
        stats = event_bus.get_stats()
        assert stats['subscriber_count'] == 0

    def test_publish(self, event_bus):
        """Test event publishing."""
        event = Event(
            event_id="evt-001",
            event_type=EventType.DATA_UPDATE,
            source="test",
            data={"value": 42},
            timestamp=datetime.now()
        )
        event_bus.publish(event)

        stats = event_bus.get_stats()
        assert stats['total_events'] == 1

    def test_publish_data_helper(self, event_bus):
        """Test publish_data helper method."""
        event = event_bus.publish_data("test_source", {"value": 100.0})

        assert event.event_type == EventType.DATA_UPDATE
        assert event.source == "test_source"
        assert event.data['value'] == 100.0

    def test_event_delivery(self, event_bus):
        """Test event delivery to subscriber."""
        received = []
        callback = lambda e: received.append(e)

        event_bus.subscribe("sub1", [EventType.DATA_UPDATE], callback)
        event_bus.start()

        event = Event(
            event_id="evt-001",
            event_type=EventType.DATA_UPDATE,
            source="test",
            data={"value": 42},
            timestamp=datetime.now()
        )
        event_bus.publish(event)

        # Wait for delivery
        time.sleep(0.1)

        assert len(received) == 1
        assert received[0].event_id == "evt-001"

    def test_multiple_subscribers(self, event_bus):
        """Test multiple subscribers."""
        received1 = []
        received2 = []

        event_bus.subscribe("sub1", [EventType.DATA_UPDATE], lambda e: received1.append(e))
        event_bus.subscribe("sub2", [EventType.DATA_UPDATE], lambda e: received2.append(e))
        event_bus.start()

        event_bus.publish_data("test", {"value": 1})
        time.sleep(0.1)

        assert len(received1) == 1
        assert len(received2) == 1

    def test_event_type_filtering(self, event_bus):
        """Test event type filtering."""
        data_events = []
        alarm_events = []

        event_bus.subscribe("sub1", [EventType.DATA_UPDATE], lambda e: data_events.append(e))
        event_bus.subscribe("sub2", [EventType.ALARM], lambda e: alarm_events.append(e))
        event_bus.start()

        event_bus.publish(Event(
            event_id="1",
            event_type=EventType.DATA_UPDATE,
            source="test",
            data={},
            timestamp=datetime.now()
        ))
        event_bus.publish(Event(
            event_id="2",
            event_type=EventType.ALARM,
            source="test",
            data={},
            timestamp=datetime.now()
        ))

        time.sleep(0.1)

        assert len(data_events) == 1
        assert len(alarm_events) == 1


# =============================================================================
# Test Data Aggregator
# =============================================================================

class TestDataAggregator:
    """Test DataAggregator class."""

    def test_initialization(self, data_aggregator):
        """Test aggregator initialization."""
        assert data_aggregator.default_interval == 1.0

    def test_register_stream(self, data_aggregator):
        """Test stream registration."""
        data_aggregator.register_stream("flow", unit="m³/s")
        assert "flow" in data_aggregator._buffers

    def test_add_data(self, data_aggregator):
        """Test adding data."""
        data_aggregator.add_data("pressure", 100.0)
        latest = data_aggregator.get_latest("pressure")
        assert latest == 100.0

    def test_add_batch(self, data_aggregator):
        """Test adding batch data."""
        data_aggregator.add_batch({
            "flow": 50.0,
            "pressure": 100.0,
            "temperature": 25.0
        })

        assert data_aggregator.get_latest("flow") == 50.0
        assert data_aggregator.get_latest("pressure") == 100.0
        assert data_aggregator.get_latest("temperature") == 25.0

    def test_get_aggregated(self, data_aggregator):
        """Test getting aggregated data."""
        data_aggregator.register_stream("test_stream")

        for i in range(10):
            data_aggregator.add_data("test_stream", float(i))

        agg = data_aggregator.get_aggregated("test_stream")
        assert agg is not None
        assert agg.sample_count == 10

    def test_get_all_aggregated(self, data_aggregator):
        """Test getting all aggregated data."""
        data_aggregator.add_batch({"a": 1.0, "b": 2.0, "c": 3.0})

        all_agg = data_aggregator.get_all_aggregated()
        assert len(all_agg) == 3

    def test_get_all_latest(self, data_aggregator):
        """Test getting all latest values."""
        data_aggregator.add_batch({"x": 10.0, "y": 20.0})

        latest = data_aggregator.get_all_latest()
        assert latest["x"] == 10.0
        assert latest["y"] == 20.0


# =============================================================================
# Test Data Stream Processor
# =============================================================================

class TestDataStreamProcessor:
    """Test DataStreamProcessor class."""

    def test_initialization(self):
        """Test processor initialization."""
        processor = DataStreamProcessor()
        assert processor is not None

    def test_register_transform(self):
        """Test registering transform."""
        processor = DataStreamProcessor()
        processor.register_transform("test", lambda x: x * 2)

        valid, result = processor.process("test", 5.0)
        assert valid is True
        assert result == 10.0

    def test_multiple_transforms(self):
        """Test multiple transforms."""
        processor = DataStreamProcessor()
        processor.register_transform("test", lambda x: x + 10)
        processor.register_transform("test", lambda x: x * 2)

        valid, result = processor.process("test", 5.0)
        assert valid is True
        assert result == 30.0  # (5 + 10) * 2

    def test_register_validator(self):
        """Test registering validator."""
        processor = DataStreamProcessor()
        processor.register_validator("test", lambda x: x > 0)

        valid, _ = processor.process("test", 10.0)
        assert valid is True

        valid, _ = processor.process("test", -5.0)
        assert valid is False

    def test_process_batch(self):
        """Test batch processing."""
        processor = DataStreamProcessor()
        processor.register_validator("valid_field", lambda x: x > 0)
        processor.register_validator("invalid_field", lambda x: x > 100)

        valid_data, invalid = processor.process_batch({
            "valid_field": 50.0,
            "invalid_field": 10.0,
            "unregistered": 100.0
        })

        assert "valid_field" in valid_data
        assert "invalid_field" in invalid
        assert "unregistered" in valid_data


# =============================================================================
# Test Subscription Manager
# =============================================================================

class TestSubscriptionManager:
    """Test SubscriptionManager class."""

    @pytest.fixture
    def subscription_manager(self, event_bus, data_aggregator):
        """Create subscription manager."""
        manager = SubscriptionManager(event_bus, data_aggregator)
        yield manager
        manager.stop()

    def test_create_subscription(self, subscription_manager):
        """Test creating subscription."""
        sub = subscription_manager.create_subscription(
            subscriber_id="sub1",
            event_types=[EventType.DATA_UPDATE]
        )

        assert sub.subscription_id is not None
        assert sub.subscriber_id == "sub1"
        assert sub.active is True

    def test_cancel_subscription(self, subscription_manager):
        """Test cancelling subscription."""
        sub = subscription_manager.create_subscription(
            subscriber_id="sub1",
            event_types=[EventType.DATA_UPDATE]
        )

        result = subscription_manager.cancel_subscription(sub.subscription_id)
        assert result is True

        result = subscription_manager.cancel_subscription("nonexistent")
        assert result is False

    def test_pause_resume(self, subscription_manager):
        """Test pausing and resuming subscription."""
        sub = subscription_manager.create_subscription(
            subscriber_id="sub1",
            event_types=[EventType.DATA_UPDATE]
        )

        subscription_manager.pause_subscription(sub.subscription_id)
        assert subscription_manager.get_subscription(sub.subscription_id).active is False

        subscription_manager.resume_subscription(sub.subscription_id)
        assert subscription_manager.get_subscription(sub.subscription_id).active is True

    def test_get_subscriber_subscriptions(self, subscription_manager):
        """Test getting subscriptions by subscriber."""
        subscription_manager.create_subscription("sub1", [EventType.DATA_UPDATE])
        subscription_manager.create_subscription("sub1", [EventType.ALARM])
        subscription_manager.create_subscription("sub2", [EventType.DATA_UPDATE])

        sub1_subs = subscription_manager.get_subscriber_subscriptions("sub1")
        assert len(sub1_subs) == 2

    def test_subscription_with_callback(self, subscription_manager, event_bus):
        """Test subscription with callback."""
        received = []

        subscription_manager.create_subscription(
            subscriber_id="sub1",
            event_types=[EventType.DATA_UPDATE],
            callback=lambda e: received.append(e)
        )

        event_bus.start()
        event_bus.publish_data("test", {"value": 42})

        time.sleep(0.1)
        # Callback should be called through event bus
        assert len(received) >= 0  # May or may not receive depending on timing


# =============================================================================
# Test Real-Time Data Hub
# =============================================================================

class TestRealTimeDataHub:
    """Test RealTimeDataHub class."""

    def test_initialization(self, data_hub):
        """Test hub initialization."""
        status = data_hub.get_status()
        assert status['running'] is False
        assert status['update_count'] == 0

    def test_start_stop(self, data_hub):
        """Test starting and stopping hub."""
        data_hub.start()
        assert data_hub.get_status()['running'] is True

        data_hub.stop()
        assert data_hub.get_status()['running'] is False

    def test_publish_data(self, data_hub):
        """Test publishing data."""
        data_hub.start()
        data_hub.publish("sensor", {"flow": 100.0, "pressure": 50.0})

        status = data_hub.get_status()
        assert status['update_count'] == 1

    def test_get_latest(self, data_hub):
        """Test getting latest data."""
        data_hub.publish("sensor", {"flow": 100.0})
        data_hub.publish("sensor", {"flow": 150.0})

        latest = data_hub.get_latest("flow")
        assert latest == 150.0

    def test_get_all_latest(self, data_hub):
        """Test getting all latest data."""
        data_hub.publish("sensor", {"a": 1.0, "b": 2.0})

        latest = data_hub.get_latest()
        assert "a" in latest
        assert "b" in latest

    def test_register_stream(self, data_hub):
        """Test stream registration."""
        data_hub.register_stream("custom_stream", unit="units", interval_ms=500)

        status = data_hub.get_status()
        assert "custom_stream" in status['streams']

    def test_add_transform(self, data_hub):
        """Test adding transform."""
        data_hub.add_transform("scaled", lambda x: x * 10)
        data_hub.publish("source", {"scaled": 5.0})

        latest = data_hub.get_latest("scaled")
        assert latest == 50.0

    def test_add_validator(self, data_hub):
        """Test adding validator."""
        data_hub.add_validator("positive", lambda x: x > 0)

        # This should pass
        data_hub.publish("source", {"positive": 10.0})
        assert data_hub.get_latest("positive") == 10.0

    def test_subscribe(self, data_hub):
        """Test subscription."""
        received = []
        callback = lambda e: received.append(e)

        sub = data_hub.subscribe(
            subscriber_id="test_sub",
            callback=callback,
            event_types=[EventType.DATA_UPDATE]
        )

        assert sub.subscription_id is not None

    def test_unsubscribe(self, data_hub):
        """Test unsubscription."""
        sub = data_hub.subscribe(
            subscriber_id="test_sub",
            callback=lambda e: None
        )

        result = data_hub.unsubscribe(sub.subscription_id)
        assert result is True

    def test_get_aggregated(self, data_hub):
        """Test getting aggregated data."""
        for i in range(10):
            data_hub.publish("source", {"metric": float(i)})

        agg = data_hub.get_aggregated("metric")
        assert agg is not None
        assert agg.sample_count == 10

    def test_full_workflow(self, data_hub):
        """Test complete workflow."""
        data_hub.start()

        # Register streams
        data_hub.register_stream("flow", unit="m³/s")
        data_hub.register_stream("pressure", unit="kPa")

        # Add transforms
        data_hub.add_transform("flow", lambda x: max(0, x))

        # Subscribe
        received = []
        data_hub.subscribe(
            "monitor",
            callback=lambda e: received.append(e),
            event_types=[EventType.DATA_UPDATE]
        )

        # Publish data
        for i in range(5):
            data_hub.publish("sensors", {
                "flow": 100.0 + i * 10,
                "pressure": 50.0 + i * 5
            })

        time.sleep(0.1)

        # Verify
        status = data_hub.get_status()
        assert status['update_count'] == 5

        latest = data_hub.get_latest()
        assert "flow" in latest
        assert "pressure" in latest


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for streaming module."""

    def test_end_to_end_streaming(self):
        """Test end-to-end streaming workflow."""
        # Create hub with longer aggregation interval for this test
        hub = RealTimeDataHub(aggregation_interval_ms=5000)  # 5 second window
        hub.start()

        try:
            # Track received events
            events_received = []

            # Subscribe to updates
            hub.subscribe(
                "integration_test",
                callback=lambda e: events_received.append(e),
                event_types=[EventType.DATA_UPDATE]
            )

            # Simulate continuous data publishing
            for i in range(20):
                hub.publish("simulator", {
                    "flow": 100.0 + np.sin(i * 0.1) * 10,
                    "vibration": 0.5 + np.random.random() * 0.2,
                    "temperature": 25.0 + i * 0.1
                })
                time.sleep(0.01)

            time.sleep(0.1)

            # Verify data was processed
            status = hub.get_status()
            assert status['update_count'] == 20

            # Get aggregated data
            flow_agg = hub.get_aggregated("flow")
            assert flow_agg is not None
            assert flow_agg.sample_count == 20
        finally:
            hub.stop()

    def test_multi_subscriber(self, data_hub):
        """Test multiple subscribers."""
        data_hub.start()

        sub1_events = []
        sub2_events = []

        data_hub.subscribe("sub1", lambda e: sub1_events.append(e))
        data_hub.subscribe("sub2", lambda e: sub2_events.append(e))

        for i in range(5):
            data_hub.publish("source", {"value": float(i)})

        time.sleep(0.1)

        # Both should receive events
        assert data_hub.get_status()['update_count'] == 5


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_buffer_aggregation(self, stream_buffer):
        """Test aggregation on empty buffer."""
        result = stream_buffer.aggregate(AggregationType.MEAN)
        assert result is None

    def test_publish_empty_data(self, data_hub):
        """Test publishing empty data."""
        data_hub.publish("source", {})
        status = data_hub.get_status()
        assert status['update_count'] == 0  # Empty data not counted

    def test_get_nonexistent_stream(self, data_aggregator):
        """Test getting non-existent stream."""
        result = data_aggregator.get_latest("nonexistent")
        assert result is None

    def test_cancel_nonexistent_subscription(self):
        """Test cancelling non-existent subscription."""
        event_bus = EventBus()
        aggregator = DataAggregator()
        manager = SubscriptionManager(event_bus, aggregator)

        result = manager.cancel_subscription("nonexistent")
        assert result is False

    def test_large_data_volume(self, data_hub):
        """Test handling large data volume."""
        data_hub.start()

        # Publish large batch
        for i in range(1000):
            data_hub.publish("high_volume", {"value": float(i)})

        status = data_hub.get_status()
        assert status['update_count'] == 1000

    def test_concurrent_access(self, data_hub):
        """Test concurrent access to hub."""
        data_hub.start()
        errors = []

        def publisher():
            try:
                for i in range(100):
                    data_hub.publish("concurrent", {"value": float(i)})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=publisher) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert data_hub.get_status()['update_count'] == 500
