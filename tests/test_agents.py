# -*- coding: utf-8 -*-
"""
Tests for Hierarchical Distributed Agent System.

测试分层分布式智能体系统。
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.agents.base import (
    BaseAgent, AgentState, AgentRole, MessageType, Priority,
    AgentMessage, AgentCapability, AgentContract
)
from src.agents.communication import MessageBus, AgentNetwork, PrioritizedMessage


# =============================================================================
# Concrete Agent for Testing
# =============================================================================

class ConcreteAgent(BaseAgent):
    """Concrete implementation for testing."""

    def __init__(self, agent_id: str, role: AgentRole = AgentRole.OBSERVER):
        super().__init__(agent_id, role)
        self.perceive_called = False
        self.decide_called = False
        self.act_called = False

    def perceive(self):
        self.perceive_called = True
        return {'test_key': 'test_value'}

    def decide(self):
        self.decide_called = True
        return [{'action': 'test_action'}]

    def act(self, decisions):
        self.act_called = True
        return [{'result': 'success'}]


# =============================================================================
# AgentMessage Tests
# =============================================================================

class TestAgentMessage:
    """Tests for AgentMessage."""

    def test_message_creation(self):
        """Test message creation with defaults."""
        msg = AgentMessage()
        assert msg.msg_id is not None
        assert msg.sender_id == ""
        assert msg.receiver_id == ""
        assert msg.msg_type == MessageType.STATUS
        assert msg.priority == Priority.NORMAL

    def test_message_with_payload(self):
        """Test message with payload."""
        msg = AgentMessage(
            sender_id="agent_1",
            receiver_id="agent_2",
            msg_type=MessageType.COMMAND,
            priority=Priority.HIGH,
            payload={'command': 'open_gate', 'value': 0.5}
        )
        assert msg.sender_id == "agent_1"
        assert msg.receiver_id == "agent_2"
        assert msg.payload['command'] == 'open_gate'

    def test_message_to_dict(self):
        """Test message serialization."""
        msg = AgentMessage(
            sender_id="test",
            msg_type=MessageType.ALERT,
            payload={'alert': 'high_vibration'}
        )
        d = msg.to_dict()
        assert d['sender_id'] == "test"
        assert d['msg_type'] == 'ALERT'
        assert 'timestamp' in d

    def test_message_from_dict(self):
        """Test message deserialization."""
        data = {
            'msg_id': 'test123',
            'sender_id': 'agent_1',
            'receiver_id': 'agent_2',
            'msg_type': 'COMMAND',
            'priority': 'URGENT',
            'payload': {'key': 'value'},
            'timestamp': datetime.now().isoformat(),
        }
        msg = AgentMessage.from_dict(data)
        assert msg.msg_id == 'test123'
        assert msg.msg_type == MessageType.COMMAND
        assert msg.priority == Priority.URGENT


# =============================================================================
# BaseAgent Tests
# =============================================================================

class TestBaseAgent:
    """Tests for BaseAgent."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = ConcreteAgent("test_agent", AgentRole.OBSERVER)
        assert agent.agent_id == "test_agent"
        assert agent.role == AgentRole.OBSERVER
        assert agent.state == AgentState.INITIALIZING

    def test_start_stop(self):
        """Test agent lifecycle."""
        agent = ConcreteAgent("test")
        agent.start()
        assert agent.state == AgentState.RUNNING

        agent.stop()
        assert agent.state == AgentState.SHUTDOWN

    def test_pause_resume(self):
        """Test agent pause/resume."""
        agent = ConcreteAgent("test")
        agent.start()
        agent.pause()
        assert agent.state == AgentState.PAUSED

        agent.resume()
        assert agent.state == AgentState.RUNNING

    def test_update_cycle(self):
        """Test BDI update cycle."""
        agent = ConcreteAgent("test")
        agent.start()

        result = agent.update(0.1)
        assert result['status'] == 'ok'
        assert agent.perceive_called
        assert agent.decide_called
        assert agent.act_called

    def test_update_when_not_running(self):
        """Test update returns inactive when not running."""
        agent = ConcreteAgent("test")
        result = agent.update(0.1)
        assert result['status'] == 'inactive'

    def test_belief_management(self):
        """Test belief get/set."""
        agent = ConcreteAgent("test")
        agent.set_belief('temperature', 25.0)
        assert agent.get_belief('temperature') == 25.0
        assert agent.get_belief('nonexistent', 'default') == 'default'

    def test_desire_management(self):
        """Test desire add/remove."""
        agent = ConcreteAgent("test")
        agent.add_desire('maintain_flow', 0.8)
        assert 'maintain_flow' in agent._desires
        assert agent._desires['maintain_flow'] == 0.8

        agent.remove_desire('maintain_flow')
        assert 'maintain_flow' not in agent._desires

    def test_intention_management(self):
        """Test intention add/clear."""
        agent = ConcreteAgent("test")
        agent.add_intention('open_gate', {'gate': 0, 'value': 0.5})
        assert len(agent._intentions) == 1

        agent.clear_intentions()
        assert len(agent._intentions) == 0

    def test_message_sending(self):
        """Test message sending."""
        agent = ConcreteAgent("sender")
        msg = AgentMessage(
            receiver_id="receiver",
            msg_type=MessageType.COMMAND,
            payload={'test': True}
        )
        agent.send_message(msg)
        assert len(agent._outbox) == 1
        assert agent._outbox[0].sender_id == "sender"

    def test_message_receiving(self):
        """Test message receiving."""
        agent = ConcreteAgent("receiver")
        msg = AgentMessage(
            sender_id="sender",
            receiver_id="receiver",
            msg_type=MessageType.STATUS,
            ttl=5
        )
        agent.receive_message(msg)
        assert len(agent._inbox) == 1
        assert msg.ttl == 4  # Decremented

    def test_message_ttl_expired(self):
        """Test expired message is dropped."""
        agent = ConcreteAgent("receiver")
        msg = AgentMessage(ttl=0)
        agent.receive_message(msg)
        assert len(agent._inbox) == 0

    def test_child_management(self):
        """Test child agent management."""
        parent = ConcreteAgent("parent", AgentRole.COORDINATOR)
        child = ConcreteAgent("child", AgentRole.DEVICE_CONTROLLER)

        parent.add_child(child)
        assert "child" in parent.children
        assert child.parent_id == "parent"

        removed = parent.remove_child("child")
        assert removed == child
        assert "child" not in parent.children

    def test_reset(self):
        """Test agent reset."""
        agent = ConcreteAgent("test")
        agent.set_belief('key', 'value')
        agent.add_desire('goal', 1.0)
        agent.add_intention('action', {})

        agent.reset()
        assert len(agent._beliefs) == 0
        assert len(agent._desires) == 0
        assert len(agent._intentions) == 0
        assert agent.state == AgentState.IDLE

    def test_get_status(self):
        """Test status retrieval."""
        agent = ConcreteAgent("test")
        status = agent.get_status()

        assert status['agent_id'] == "test"
        assert 'state' in status
        assert 'stats' in status

    def test_get_diagnostics(self):
        """Test diagnostics retrieval."""
        agent = ConcreteAgent("test")
        agent.set_belief('sensor_value', 42)

        diag = agent.get_diagnostics()
        assert 'beliefs' in diag
        assert diag['beliefs']['sensor_value'] == 42


# =============================================================================
# MessageBus Tests
# =============================================================================

class TestMessageBus:
    """Tests for MessageBus."""

    def test_initialization(self):
        """Test message bus initialization."""
        bus = MessageBus()
        assert bus._message_queue.empty()
        assert len(bus._agents) == 0

    def test_agent_registration(self):
        """Test agent registration."""
        bus = MessageBus()
        agent = ConcreteAgent("test")

        bus.register_agent(agent)
        assert "test" in bus._agents
        assert bus.get_agent("test") == agent

    def test_agent_unregistration(self):
        """Test agent unregistration."""
        bus = MessageBus()
        agent = ConcreteAgent("test")

        bus.register_agent(agent)
        bus.unregister_agent("test")
        assert "test" not in bus._agents

    def test_send_message(self):
        """Test message sending."""
        bus = MessageBus()
        msg = AgentMessage(sender_id="a", receiver_id="b")

        result = bus.send(msg)
        assert result is True
        assert not bus._message_queue.empty()

    def test_message_priority(self):
        """Test message priority ordering."""
        bus = MessageBus()

        low = AgentMessage(priority=Priority.LOW)
        high = AgentMessage(priority=Priority.HIGH)
        normal = AgentMessage(priority=Priority.NORMAL)

        bus.send(low)
        bus.send(high)
        bus.send(normal)

        # High priority should come out first
        wrapped = bus._message_queue.get()
        assert wrapped.message.priority == Priority.HIGH

    def test_subscribe_publish(self):
        """Test topic subscription and publishing."""
        bus = MessageBus()
        agent1 = ConcreteAgent("agent1")
        agent2 = ConcreteAgent("agent2")

        bus.register_agent(agent1)
        bus.register_agent(agent2)

        bus.subscribe("agent1", "alerts")
        bus.subscribe("agent2", "alerts")

        msg = AgentMessage(sender_id="system", payload={'alert': 'test'})
        count = bus.publish("alerts", msg)

        assert count == 2

    def test_broadcast(self):
        """Test message broadcasting."""
        bus = MessageBus()
        agent1 = ConcreteAgent("agent1")
        agent2 = ConcreteAgent("agent2")

        bus.register_agent(agent1)
        bus.register_agent(agent2)

        msg = AgentMessage(sender_id="system")
        count = bus.broadcast(msg)

        assert count == 2

    def test_message_filter(self):
        """Test message filtering."""
        bus = MessageBus()

        # Filter that blocks all messages
        bus.add_filter(lambda m: False)

        msg = AgentMessage()
        result = bus.send(msg)

        assert result is False
        assert bus._stats['dropped'] == 1

    def test_process_messages(self):
        """Test message processing."""
        bus = MessageBus()
        agent = ConcreteAgent("receiver")
        bus.register_agent(agent)

        msg = AgentMessage(sender_id="sender", receiver_id="receiver")
        bus.send(msg)

        processed = bus.process_messages()
        assert processed == 1
        assert len(agent._inbox) == 1

    def test_get_stats(self):
        """Test statistics retrieval."""
        bus = MessageBus()
        msg = AgentMessage()
        bus.send(msg)

        stats = bus.get_stats()
        assert stats['total_messages'] == 1
        assert 'queue_size' in stats

    def test_history(self):
        """Test message history."""
        bus = MessageBus()
        bus.enable_history(True)

        msg = AgentMessage(msg_type=MessageType.ALERT)
        bus.send(msg)

        history = bus.get_history(limit=10)
        assert len(history) == 1

        filtered = bus.get_history(msg_type=MessageType.COMMAND)
        assert len(filtered) == 0


# =============================================================================
# AgentNetwork Tests
# =============================================================================

class TestAgentNetwork:
    """Tests for AgentNetwork."""

    def test_initialization(self):
        """Test network initialization."""
        network = AgentNetwork()
        assert not network._running
        assert len(network._agents) == 0

    def test_add_agent(self):
        """Test adding agent to network."""
        network = AgentNetwork()
        agent = ConcreteAgent("test")

        network.add_agent(agent)
        assert "test" in network._agents

    def test_add_agent_with_parent(self):
        """Test adding agent with parent."""
        network = AgentNetwork()
        parent = ConcreteAgent("parent")
        child = ConcreteAgent("child")

        network.add_agent(parent)
        network.add_agent(child, parent_id="parent")

        assert child.parent_id == "parent"
        assert "child" in network._hierarchy["parent"]

    def test_remove_agent(self):
        """Test removing agent from network."""
        network = AgentNetwork()
        agent = ConcreteAgent("test")

        network.add_agent(agent)
        network.remove_agent("test")

        assert "test" not in network._agents

    def test_get_network_status(self):
        """Test network status retrieval."""
        network = AgentNetwork()
        agent = ConcreteAgent("test")
        network.add_agent(agent)

        status = network.get_network_status()
        assert status['agents_count'] == 1
        assert 'message_bus' in status

    def test_get_hierarchy_tree(self):
        """Test hierarchy tree retrieval."""
        network = AgentNetwork()
        root = ConcreteAgent("root", AgentRole.COORDINATOR)
        child = ConcreteAgent("child", AgentRole.DEVICE_CONTROLLER)

        network.add_agent(root)
        network._root = root
        network.add_agent(child, parent_id="root")

        tree = network.get_hierarchy_tree()
        assert tree['id'] == "root"
        assert len(tree['children']) == 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestAgentIntegration:
    """Integration tests for agent system."""

    def test_message_flow(self):
        """Test message flow between agents."""
        bus = MessageBus()

        sender = ConcreteAgent("sender")
        receiver = ConcreteAgent("receiver")

        bus.register_agent(sender)
        bus.register_agent(receiver)

        # Sender sends message
        msg = AgentMessage(
            receiver_id="receiver",
            msg_type=MessageType.COMMAND,
            payload={'action': 'test'}
        )
        sender.send_message(msg)

        # Collect and process
        bus.collect_outgoing()
        bus.process_messages()

        # Receiver should have message
        assert len(receiver._inbox) == 1
        assert receiver._inbox[0].payload['action'] == 'test'

    def test_hierarchical_update(self):
        """Test hierarchical agent update."""
        parent = ConcreteAgent("parent", AgentRole.ZONE_MANAGER)
        child1 = ConcreteAgent("child1", AgentRole.DEVICE_CONTROLLER)
        child2 = ConcreteAgent("child2", AgentRole.DEVICE_CONTROLLER)

        parent.add_child(child1)
        parent.add_child(child2)

        parent.start()
        result = parent.update(0.1)

        assert result['status'] == 'ok'
        assert 'children' in result
        assert 'child1' in result['children']
        assert 'child2' in result['children']
