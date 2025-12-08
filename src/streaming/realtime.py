"""
Real-time Data Streaming System.

Provides event-driven data streaming with:
- Publish/Subscribe pattern
- Data aggregation and downsampling
- Subscription management with filtering
- Efficient real-time updates
"""

import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class EventType(Enum):
    """Types of events in the system."""
    DATA_UPDATE = "data_update"
    STATE_CHANGE = "state_change"
    ALARM = "alarm"
    ALERT = "alert"
    CONTROL = "control"
    MAINTENANCE = "maintenance"
    SYSTEM = "system"
    HEARTBEAT = "heartbeat"


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class AggregationType(Enum):
    """Data aggregation methods."""
    LAST = "last"           # Most recent value
    MEAN = "mean"           # Average
    MIN = "min"             # Minimum
    MAX = "max"             # Maximum
    SUM = "sum"             # Sum
    COUNT = "count"         # Count of values
    MEDIAN = "median"       # Median value
    FIRST = "first"         # First value in window


class FilterOperator(Enum):
    """Filter operators for subscriptions."""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_OR_EQUAL = "gte"
    LESS_OR_EQUAL = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Event:
    """Event object for pub/sub system."""
    event_id: str
    event_type: EventType
    source: str
    data: Dict[str, Any]
    timestamp: datetime
    priority: EventPriority = EventPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'source': self.source,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority.value,
            'metadata': self.metadata
        }


@dataclass
class DataPoint:
    """Single data point with timestamp."""
    name: str
    value: float
    timestamp: datetime
    unit: str = ""
    quality: float = 1.0  # 0-1 quality indicator

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'unit': self.unit,
            'quality': self.quality
        }


@dataclass
class AggregatedData:
    """Aggregated data over a time window."""
    name: str
    values: Dict[str, float]  # aggregation_type -> value
    sample_count: int
    start_time: datetime
    end_time: datetime
    unit: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'values': self.values,
            'sample_count': self.sample_count,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'unit': self.unit
        }


@dataclass
class SubscriptionFilter:
    """Filter for subscription data."""
    field: str
    operator: FilterOperator
    value: Any


@dataclass
class Subscription:
    """Subscription configuration."""
    subscription_id: str
    subscriber_id: str
    event_types: List[EventType]
    sources: Optional[List[str]] = None
    data_fields: Optional[List[str]] = None
    filters: List[SubscriptionFilter] = field(default_factory=list)
    min_priority: EventPriority = EventPriority.LOW
    aggregation: Optional[AggregationType] = None
    aggregation_interval_ms: int = 1000
    callback: Optional[Callable[[Event], None]] = None
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


# =============================================================================
# Event Bus
# =============================================================================

class EventBus:
    """
    Central event bus for publish/subscribe pattern.

    Thread-safe implementation with priority-based delivery.
    """

    def __init__(self, max_queue_size: int = 10000):
        """
        Initialize event bus.

        Args:
            max_queue_size: Maximum events in queue
        """
        self._lock = threading.RLock()
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = {}
        self._type_subscribers: Dict[EventType, List[str]] = {t: [] for t in EventType}
        self._event_queue: deque = deque(maxlen=max_queue_size)
        self._event_count: int = 0
        self._running: bool = False
        self._dispatch_thread: Optional[threading.Thread] = None

        logger.info("EventBus initialized with queue size %d", max_queue_size)

    def start(self) -> None:
        """Start event dispatch thread."""
        if self._running:
            return

        self._running = True
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            daemon=True,
            name="EventBusDispatcher"
        )
        self._dispatch_thread.start()
        logger.info("EventBus dispatcher started")

    def stop(self) -> None:
        """Stop event dispatch thread."""
        self._running = False
        if self._dispatch_thread:
            self._dispatch_thread.join(timeout=2.0)
        logger.info("EventBus dispatcher stopped")

    def subscribe(
        self,
        subscriber_id: str,
        event_types: List[EventType],
        callback: Callable[[Event], None]
    ) -> None:
        """
        Subscribe to events.

        Args:
            subscriber_id: Unique subscriber identifier
            event_types: Event types to subscribe to
            callback: Function to call with events
        """
        with self._lock:
            self._subscribers[subscriber_id] = [callback]

            for event_type in event_types:
                if subscriber_id not in self._type_subscribers[event_type]:
                    self._type_subscribers[event_type].append(subscriber_id)

        logger.debug("Subscriber %s registered for %s", subscriber_id, event_types)

    def unsubscribe(self, subscriber_id: str) -> None:
        """Unsubscribe from all events."""
        with self._lock:
            if subscriber_id in self._subscribers:
                del self._subscribers[subscriber_id]

            for subscribers in self._type_subscribers.values():
                if subscriber_id in subscribers:
                    subscribers.remove(subscriber_id)

        logger.debug("Subscriber %s unregistered", subscriber_id)

    def publish(self, event: Event) -> None:
        """
        Publish an event.

        Args:
            event: Event to publish
        """
        with self._lock:
            self._event_queue.append(event)
            self._event_count += 1

    def publish_data(
        self,
        source: str,
        data: Dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL
    ) -> Event:
        """
        Publish data update event.

        Args:
            source: Event source
            data: Data payload
            priority: Event priority

        Returns:
            Created event
        """
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=EventType.DATA_UPDATE,
            source=source,
            data=data,
            timestamp=datetime.now(),
            priority=priority
        )
        self.publish(event)
        return event

    def _dispatch_loop(self) -> None:
        """Main dispatch loop."""
        while self._running:
            try:
                self._dispatch_pending()
                time.sleep(0.001)  # 1ms polling
            except Exception as e:
                logger.error("Error in dispatch loop: %s", e)

    def _dispatch_pending(self) -> None:
        """Dispatch pending events."""
        while True:
            event = None
            with self._lock:
                if self._event_queue:
                    event = self._event_queue.popleft()

            if event is None:
                break

            self._dispatch_event(event)

    def _dispatch_event(self, event: Event) -> None:
        """Dispatch single event to subscribers."""
        with self._lock:
            subscriber_ids = self._type_subscribers.get(event.event_type, [])

        for subscriber_id in subscriber_ids:
            with self._lock:
                callbacks = self._subscribers.get(subscriber_id, [])

            for callback in callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(
                        "Error in subscriber %s callback: %s",
                        subscriber_id, e
                    )

    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        with self._lock:
            return {
                'total_events': self._event_count,
                'queue_size': len(self._event_queue),
                'subscriber_count': len(self._subscribers),
                'running': self._running
            }


# =============================================================================
# Stream Buffer
# =============================================================================

class StreamBuffer:
    """
    Buffer for streaming data with time-based windowing.
    """

    def __init__(
        self,
        name: str,
        max_samples: int = 10000,
        window_seconds: float = 60.0
    ):
        """
        Initialize stream buffer.

        Args:
            name: Buffer name
            max_samples: Maximum samples to keep
            window_seconds: Time window for aggregations
        """
        self.name = name
        self.max_samples = max_samples
        self.window_seconds = window_seconds

        self._data: deque = deque(maxlen=max_samples)
        self._lock = threading.Lock()

        logger.debug("StreamBuffer '%s' initialized", name)

    def add(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Add a value to the buffer."""
        if timestamp is None:
            timestamp = datetime.now()

        with self._lock:
            self._data.append({
                'value': value,
                'timestamp': timestamp
            })

    def add_batch(self, values: List[Tuple[float, datetime]]) -> None:
        """Add multiple values."""
        with self._lock:
            for value, timestamp in values:
                self._data.append({
                    'value': value,
                    'timestamp': timestamp
                })

    def get_window(self, seconds: Optional[float] = None) -> List[Dict]:
        """Get data within time window."""
        if seconds is None:
            seconds = self.window_seconds

        cutoff = datetime.now() - timedelta(seconds=seconds)

        with self._lock:
            return [d for d in self._data if d['timestamp'] >= cutoff]

    def get_latest(self, n: int = 1) -> List[Dict]:
        """Get n most recent values."""
        with self._lock:
            return list(self._data)[-n:]

    def aggregate(
        self,
        aggregation_type: AggregationType,
        seconds: Optional[float] = None
    ) -> Optional[float]:
        """Aggregate values over time window."""
        data = self.get_window(seconds)
        if not data:
            return None

        values = [d['value'] for d in data]

        if aggregation_type == AggregationType.LAST:
            return values[-1]
        elif aggregation_type == AggregationType.FIRST:
            return values[0]
        elif aggregation_type == AggregationType.MEAN:
            return float(np.mean(values))
        elif aggregation_type == AggregationType.MIN:
            return float(np.min(values))
        elif aggregation_type == AggregationType.MAX:
            return float(np.max(values))
        elif aggregation_type == AggregationType.SUM:
            return float(np.sum(values))
        elif aggregation_type == AggregationType.COUNT:
            return float(len(values))
        elif aggregation_type == AggregationType.MEDIAN:
            return float(np.median(values))

        return None

    def get_aggregated_data(
        self,
        aggregations: List[AggregationType],
        seconds: Optional[float] = None
    ) -> AggregatedData:
        """Get multiple aggregations at once."""
        data = self.get_window(seconds)

        values = {}
        for agg_type in aggregations:
            result = self.aggregate(agg_type, seconds)
            if result is not None:
                values[agg_type.value] = result

        timestamps = [d['timestamp'] for d in data] if data else [datetime.now()]

        return AggregatedData(
            name=self.name,
            values=values,
            sample_count=len(data),
            start_time=min(timestamps),
            end_time=max(timestamps)
        )

    def clear(self) -> None:
        """Clear buffer."""
        with self._lock:
            self._data.clear()

    @property
    def size(self) -> int:
        """Get current buffer size."""
        with self._lock:
            return len(self._data)


# =============================================================================
# Data Aggregator
# =============================================================================

class DataAggregator:
    """
    Aggregates streaming data with configurable intervals.
    """

    def __init__(
        self,
        default_interval_ms: int = 1000,
        default_aggregations: Optional[List[AggregationType]] = None
    ):
        """
        Initialize data aggregator.

        Args:
            default_interval_ms: Default aggregation interval in ms
            default_aggregations: Default aggregation types
        """
        self.default_interval = default_interval_ms / 1000.0
        self.default_aggregations = default_aggregations or [
            AggregationType.LAST,
            AggregationType.MEAN,
            AggregationType.MIN,
            AggregationType.MAX
        ]

        self._buffers: Dict[str, StreamBuffer] = {}
        self._config: Dict[str, Dict] = {}
        self._lock = threading.RLock()  # Reentrant lock to allow nested calls

        logger.info("DataAggregator initialized with %dms interval", default_interval_ms)

    def register_stream(
        self,
        name: str,
        interval_ms: Optional[int] = None,
        aggregations: Optional[List[AggregationType]] = None,
        unit: str = ""
    ) -> None:
        """Register a data stream for aggregation."""
        interval = (interval_ms / 1000.0) if interval_ms else self.default_interval

        with self._lock:
            self._buffers[name] = StreamBuffer(
                name=name,
                window_seconds=interval * 10  # Keep 10x interval
            )
            self._config[name] = {
                'interval': interval,
                'aggregations': aggregations or self.default_aggregations,
                'unit': unit,
                'last_aggregation': datetime.now()
            }

        logger.debug("Stream '%s' registered", name)

    def add_data(self, name: str, value: float, timestamp: Optional[datetime] = None) -> None:
        """Add data point to stream."""
        with self._lock:
            if name not in self._buffers:
                self.register_stream(name)

            self._buffers[name].add(value, timestamp)

    def add_batch(self, data: Dict[str, float], timestamp: Optional[datetime] = None) -> None:
        """Add multiple data points at once."""
        if timestamp is None:
            timestamp = datetime.now()

        for name, value in data.items():
            self.add_data(name, value, timestamp)

    def get_aggregated(
        self,
        name: str,
        aggregations: Optional[List[AggregationType]] = None
    ) -> Optional[AggregatedData]:
        """Get aggregated data for a stream."""
        with self._lock:
            if name not in self._buffers:
                return None

            buffer = self._buffers[name]
            config = self._config[name]
            aggs = aggregations or config['aggregations']

            agg_data = buffer.get_aggregated_data(aggs, config['interval'])
            agg_data.unit = config['unit']

            return agg_data

    def get_all_aggregated(self) -> Dict[str, AggregatedData]:
        """Get aggregated data for all streams."""
        results = {}
        with self._lock:
            names = list(self._buffers.keys())

        for name in names:
            agg_data = self.get_aggregated(name)
            if agg_data:
                results[name] = agg_data

        return results

    def get_latest(self, name: str) -> Optional[float]:
        """Get latest value for a stream."""
        with self._lock:
            if name not in self._buffers:
                return None
            data = self._buffers[name].get_latest(1)
            return data[0]['value'] if data else None

    def get_all_latest(self) -> Dict[str, float]:
        """Get latest values for all streams."""
        results = {}
        with self._lock:
            for name, buffer in self._buffers.items():
                data = buffer.get_latest(1)
                if data:
                    results[name] = data[0]['value']
        return results


# =============================================================================
# Subscription Manager
# =============================================================================

class SubscriptionManager:
    """
    Manages data subscriptions with filtering and aggregation.
    """

    def __init__(self, event_bus: EventBus, aggregator: DataAggregator):
        """
        Initialize subscription manager.

        Args:
            event_bus: Event bus for event delivery
            aggregator: Data aggregator for data processing
        """
        self.event_bus = event_bus
        self.aggregator = aggregator

        self._subscriptions: Dict[str, Subscription] = {}
        self._lock = threading.Lock()
        self._running = False
        self._push_thread: Optional[threading.Thread] = None

        logger.info("SubscriptionManager initialized")

    def start(self) -> None:
        """Start subscription push thread."""
        if self._running:
            return

        self._running = True
        self._push_thread = threading.Thread(
            target=self._push_loop,
            daemon=True,
            name="SubscriptionPush"
        )
        self._push_thread.start()
        logger.info("Subscription push thread started")

    def stop(self) -> None:
        """Stop subscription push thread."""
        self._running = False
        if self._push_thread:
            self._push_thread.join(timeout=2.0)
        logger.info("Subscription push thread stopped")

    def create_subscription(
        self,
        subscriber_id: str,
        event_types: List[EventType],
        sources: Optional[List[str]] = None,
        data_fields: Optional[List[str]] = None,
        filters: Optional[List[SubscriptionFilter]] = None,
        aggregation: Optional[AggregationType] = None,
        aggregation_interval_ms: int = 1000,
        callback: Optional[Callable[[Event], None]] = None
    ) -> Subscription:
        """
        Create a new subscription.

        Args:
            subscriber_id: Unique subscriber ID
            event_types: Event types to subscribe to
            sources: Optional source filters
            data_fields: Optional data field filters
            filters: Optional value filters
            aggregation: Aggregation type for data
            aggregation_interval_ms: Aggregation interval
            callback: Callback function for events

        Returns:
            Created subscription
        """
        subscription_id = str(uuid.uuid4())

        subscription = Subscription(
            subscription_id=subscription_id,
            subscriber_id=subscriber_id,
            event_types=event_types,
            sources=sources,
            data_fields=data_fields,
            filters=filters or [],
            aggregation=aggregation,
            aggregation_interval_ms=aggregation_interval_ms,
            callback=callback
        )

        with self._lock:
            self._subscriptions[subscription_id] = subscription

        # Register with event bus if callback provided
        if callback:
            self.event_bus.subscribe(
                subscriber_id,
                event_types,
                lambda e: self._handle_event(subscription_id, e)
            )

        logger.info("Subscription created: %s for %s", subscription_id, subscriber_id)
        return subscription

    def cancel_subscription(self, subscription_id: str) -> bool:
        """Cancel a subscription."""
        with self._lock:
            if subscription_id not in self._subscriptions:
                return False

            subscription = self._subscriptions[subscription_id]
            del self._subscriptions[subscription_id]

        self.event_bus.unsubscribe(subscription.subscriber_id)
        logger.info("Subscription cancelled: %s", subscription_id)
        return True

    def pause_subscription(self, subscription_id: str) -> bool:
        """Pause a subscription."""
        with self._lock:
            if subscription_id in self._subscriptions:
                self._subscriptions[subscription_id].active = False
                return True
        return False

    def resume_subscription(self, subscription_id: str) -> bool:
        """Resume a paused subscription."""
        with self._lock:
            if subscription_id in self._subscriptions:
                self._subscriptions[subscription_id].active = True
                return True
        return False

    def get_subscription(self, subscription_id: str) -> Optional[Subscription]:
        """Get subscription by ID."""
        with self._lock:
            return self._subscriptions.get(subscription_id)

    def get_subscriber_subscriptions(self, subscriber_id: str) -> List[Subscription]:
        """Get all subscriptions for a subscriber."""
        with self._lock:
            return [
                s for s in self._subscriptions.values()
                if s.subscriber_id == subscriber_id
            ]

    def _handle_event(self, subscription_id: str, event: Event) -> None:
        """Handle event for subscription."""
        with self._lock:
            subscription = self._subscriptions.get(subscription_id)
            if not subscription or not subscription.active:
                return

        # Check priority filter
        if event.priority.value < subscription.min_priority.value:
            return

        # Check source filter
        if subscription.sources and event.source not in subscription.sources:
            return

        # Apply filters
        if not self._apply_filters(event, subscription.filters):
            return

        # Filter data fields if specified
        if subscription.data_fields:
            filtered_data = {
                k: v for k, v in event.data.items()
                if k in subscription.data_fields
            }
            event.data = filtered_data

        # Call callback
        if subscription.callback:
            try:
                subscription.callback(event)
            except Exception as e:
                logger.error("Error in subscription callback: %s", e)

    def _apply_filters(
        self,
        event: Event,
        filters: List[SubscriptionFilter]
    ) -> bool:
        """Apply filters to event."""
        for f in filters:
            value = event.data.get(f.field)
            if value is None:
                continue

            if f.operator == FilterOperator.EQUALS:
                if value != f.value:
                    return False
            elif f.operator == FilterOperator.NOT_EQUALS:
                if value == f.value:
                    return False
            elif f.operator == FilterOperator.GREATER_THAN:
                if not (value > f.value):
                    return False
            elif f.operator == FilterOperator.LESS_THAN:
                if not (value < f.value):
                    return False
            elif f.operator == FilterOperator.GREATER_OR_EQUAL:
                if not (value >= f.value):
                    return False
            elif f.operator == FilterOperator.LESS_OR_EQUAL:
                if not (value <= f.value):
                    return False
            elif f.operator == FilterOperator.IN:
                if value not in f.value:
                    return False
            elif f.operator == FilterOperator.NOT_IN:
                if value in f.value:
                    return False

        return True

    def _push_loop(self) -> None:
        """Push aggregated data to subscribers."""
        last_push: Dict[str, datetime] = {}

        while self._running:
            try:
                now = datetime.now()

                with self._lock:
                    subscriptions = list(self._subscriptions.values())

                for sub in subscriptions:
                    if not sub.active or not sub.aggregation:
                        continue

                    interval = timedelta(milliseconds=sub.aggregation_interval_ms)
                    last = last_push.get(sub.subscription_id, datetime.min)

                    if now - last >= interval:
                        self._push_aggregated_data(sub)
                        last_push[sub.subscription_id] = now

                time.sleep(0.01)  # 10ms polling

            except Exception as e:
                logger.error("Error in push loop: %s", e)

    def _push_aggregated_data(self, subscription: Subscription) -> None:
        """Push aggregated data for subscription."""
        if not subscription.callback:
            return

        # Get aggregated data
        all_data = self.aggregator.get_all_aggregated()

        # Filter by data fields
        if subscription.data_fields:
            all_data = {
                k: v for k, v in all_data.items()
                if k in subscription.data_fields
            }

        if not all_data:
            return

        # Create event with aggregated data
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=EventType.DATA_UPDATE,
            source="aggregator",
            data={
                name: agg.to_dict()
                for name, agg in all_data.items()
            },
            timestamp=datetime.now(),
            metadata={'aggregated': True}
        )

        try:
            subscription.callback(event)
        except Exception as e:
            logger.error("Error pushing aggregated data: %s", e)


# =============================================================================
# Data Stream Processor
# =============================================================================

class DataStreamProcessor:
    """
    Processes incoming data streams with transformations.
    """

    def __init__(self):
        """Initialize data stream processor."""
        self._transforms: Dict[str, List[Callable]] = {}
        self._validators: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()

        logger.info("DataStreamProcessor initialized")

    def register_transform(
        self,
        stream_name: str,
        transform: Callable[[float], float]
    ) -> None:
        """Register a transform function for a stream."""
        with self._lock:
            if stream_name not in self._transforms:
                self._transforms[stream_name] = []
            self._transforms[stream_name].append(transform)

    def register_validator(
        self,
        stream_name: str,
        validator: Callable[[float], bool]
    ) -> None:
        """Register a validator function for a stream."""
        with self._lock:
            if stream_name not in self._validators:
                self._validators[stream_name] = []
            self._validators[stream_name].append(validator)

    def process(
        self,
        stream_name: str,
        value: float
    ) -> Tuple[bool, float]:
        """
        Process a value through transforms and validators.

        Args:
            stream_name: Stream name
            value: Input value

        Returns:
            Tuple of (valid, transformed_value)
        """
        # Apply transforms
        with self._lock:
            transforms = self._transforms.get(stream_name, [])

        transformed = value
        for transform in transforms:
            try:
                transformed = transform(transformed)
            except Exception as e:
                logger.error("Transform error for %s: %s", stream_name, e)

        # Apply validators
        with self._lock:
            validators = self._validators.get(stream_name, [])

        for validator in validators:
            try:
                if not validator(transformed):
                    return False, transformed
            except Exception as e:
                logger.error("Validator error for %s: %s", stream_name, e)
                return False, transformed

        return True, transformed

    def process_batch(
        self,
        data: Dict[str, float]
    ) -> Tuple[Dict[str, float], List[str]]:
        """
        Process batch of values.

        Returns:
            Tuple of (valid_data, invalid_fields)
        """
        valid_data = {}
        invalid_fields = []

        for name, value in data.items():
            is_valid, processed = self.process(name, value)
            if is_valid:
                valid_data[name] = processed
            else:
                invalid_fields.append(name)

        return valid_data, invalid_fields


# =============================================================================
# Real-Time Data Hub (Main Interface)
# =============================================================================

class RealTimeDataHub:
    """
    Central hub for real-time data streaming.

    Combines event bus, aggregation, and subscription management.
    """

    def __init__(
        self,
        aggregation_interval_ms: int = 1000,
        max_queue_size: int = 10000
    ):
        """
        Initialize real-time data hub.

        Args:
            aggregation_interval_ms: Default aggregation interval
            max_queue_size: Maximum event queue size
        """
        self.event_bus = EventBus(max_queue_size)
        self.aggregator = DataAggregator(aggregation_interval_ms)
        self.subscription_manager = SubscriptionManager(self.event_bus, self.aggregator)
        self.processor = DataStreamProcessor()

        self._running = False
        self._update_count = 0

        logger.info("RealTimeDataHub initialized")

    def start(self) -> None:
        """Start the data hub."""
        if self._running:
            return

        self._running = True
        self.event_bus.start()
        self.subscription_manager.start()

        logger.info("RealTimeDataHub started")

    def stop(self) -> None:
        """Stop the data hub."""
        self._running = False
        self.subscription_manager.stop()
        self.event_bus.stop()

        logger.info("RealTimeDataHub stopped")

    def publish(
        self,
        source: str,
        data: Dict[str, float],
        event_type: EventType = EventType.DATA_UPDATE,
        priority: EventPriority = EventPriority.NORMAL
    ) -> None:
        """
        Publish data to the hub.

        Args:
            source: Data source identifier
            data: Data payload
            event_type: Event type
            priority: Event priority
        """
        # Process data
        valid_data, invalid = self.processor.process_batch(data)

        if invalid:
            logger.warning("Invalid data fields: %s", invalid)

        if not valid_data:
            return

        # Add to aggregator
        self.aggregator.add_batch(valid_data)

        # Publish event
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            source=source,
            data=valid_data,
            timestamp=datetime.now(),
            priority=priority
        )
        self.event_bus.publish(event)

        self._update_count += 1

    def subscribe(
        self,
        subscriber_id: str,
        callback: Callable[[Event], None],
        event_types: Optional[List[EventType]] = None,
        data_fields: Optional[List[str]] = None,
        aggregation: Optional[AggregationType] = None,
        interval_ms: int = 1000
    ) -> Subscription:
        """
        Subscribe to data updates.

        Args:
            subscriber_id: Unique subscriber ID
            callback: Callback for events
            event_types: Event types to subscribe to
            data_fields: Data fields to include
            aggregation: Aggregation type
            interval_ms: Aggregation interval

        Returns:
            Subscription object
        """
        return self.subscription_manager.create_subscription(
            subscriber_id=subscriber_id,
            event_types=event_types or [EventType.DATA_UPDATE],
            data_fields=data_fields,
            aggregation=aggregation,
            aggregation_interval_ms=interval_ms,
            callback=callback
        )

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from data updates."""
        return self.subscription_manager.cancel_subscription(subscription_id)

    def get_latest(self, field: Optional[str] = None) -> Any:
        """Get latest data."""
        if field:
            return self.aggregator.get_latest(field)
        return self.aggregator.get_all_latest()

    def get_aggregated(
        self,
        field: Optional[str] = None
    ) -> Any:
        """Get aggregated data."""
        if field:
            return self.aggregator.get_aggregated(field)
        return self.aggregator.get_all_aggregated()

    def register_stream(
        self,
        name: str,
        unit: str = "",
        interval_ms: Optional[int] = None
    ) -> None:
        """Register a data stream."""
        self.aggregator.register_stream(name, interval_ms, unit=unit)

    def add_transform(
        self,
        stream_name: str,
        transform: Callable[[float], float]
    ) -> None:
        """Add transform to a stream."""
        self.processor.register_transform(stream_name, transform)

    def add_validator(
        self,
        stream_name: str,
        validator: Callable[[float], bool]
    ) -> None:
        """Add validator to a stream."""
        self.processor.register_validator(stream_name, validator)

    def get_status(self) -> Dict[str, Any]:
        """Get hub status."""
        return {
            'running': self._running,
            'update_count': self._update_count,
            'event_bus': self.event_bus.get_stats(),
            'streams': list(self.aggregator._buffers.keys()),
            'subscriptions': len(self.subscription_manager._subscriptions)
        }
