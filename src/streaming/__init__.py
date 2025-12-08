"""
Real-time Data Streaming Module.

Provides:
- Event-driven data publishing
- Subscription management
- Data aggregation and filtering
- Real-time update distribution
"""

from src.streaming.realtime import (
    Event,
    EventType,
    EventPriority,
    EventBus,
    DataAggregator,
    AggregationType,
    SubscriptionManager,
    Subscription,
    DataStreamProcessor,
    StreamBuffer,
    RealTimeDataHub
)

__all__ = [
    'Event',
    'EventType',
    'EventPriority',
    'EventBus',
    'DataAggregator',
    'AggregationType',
    'SubscriptionManager',
    'Subscription',
    'DataStreamProcessor',
    'StreamBuffer',
    'RealTimeDataHub'
]
