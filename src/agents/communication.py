# -*- coding: utf-8 -*-
"""
Agent Communication System.

智能体通信系统，提供：
- 消息总线 (Message Bus)
- 智能体网络管理 (Agent Network)
- 消息路由 (Message Routing)
- 订阅发布机制 (Pub/Sub)
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Callable
from queue import PriorityQueue
import threading

from .base import BaseAgent, AgentMessage, MessageType, Priority

logger = logging.getLogger(__name__)


@dataclass(order=True)
class PrioritizedMessage:
    """优先级消息包装."""
    priority: int
    timestamp: float = field(compare=False)
    message: AgentMessage = field(compare=False)


class MessageBus:
    """
    消息总线.

    负责智能体间消息的路由和传递。
    支持:
    - 点对点消息
    - 广播消息
    - 主题订阅
    - 优先级队列
    """

    def __init__(self):
        # 消息队列
        self._message_queue: PriorityQueue = PriorityQueue()

        # 注册的智能体
        self._agents: Dict[str, BaseAgent] = {}

        # 主题订阅
        self._subscriptions: Dict[str, Set[str]] = defaultdict(set)

        # 消息统计
        self._stats = {
            'total_messages': 0,
            'delivered': 0,
            'dropped': 0,
            'broadcast': 0,
        }

        # 消息历史 (可选，用于调试)
        self._history: List[AgentMessage] = []
        self._history_enabled = False
        self._max_history = 1000

        # 消息过滤器
        self._filters: List[Callable[[AgentMessage], bool]] = []

        # 运行状态
        self._running = False
        self._lock = threading.Lock()

    # -------------------------------------------------------------------------
    # Agent Registration
    # -------------------------------------------------------------------------

    def register_agent(self, agent: BaseAgent) -> None:
        """注册智能体."""
        with self._lock:
            self._agents[agent.agent_id] = agent
            logger.info(f"Agent {agent.agent_id} registered to message bus")

    def unregister_agent(self, agent_id: str) -> None:
        """注销智能体."""
        with self._lock:
            if agent_id in self._agents:
                del self._agents[agent_id]
                # 清理订阅
                for topic in self._subscriptions:
                    self._subscriptions[topic].discard(agent_id)
                logger.info(f"Agent {agent_id} unregistered from message bus")

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """获取智能体."""
        return self._agents.get(agent_id)

    # -------------------------------------------------------------------------
    # Message Sending
    # -------------------------------------------------------------------------

    def send(self, message: AgentMessage) -> bool:
        """
        发送消息.

        Args:
            message: 要发送的消息

        Returns:
            是否成功入队
        """
        self._stats['total_messages'] += 1

        # 应用过滤器
        for filter_func in self._filters:
            if not filter_func(message):
                self._stats['dropped'] += 1
                return False

        # 记录历史
        if self._history_enabled:
            self._history.append(message)
            if len(self._history) > self._max_history:
                self._history.pop(0)

        # 加入优先级队列
        priority = -message.priority.value  # 负数使高优先级先出队
        wrapped = PrioritizedMessage(
            priority=priority,
            timestamp=datetime.now().timestamp(),
            message=message,
        )
        self._message_queue.put(wrapped)

        return True

    def broadcast(self, message: AgentMessage) -> int:
        """
        广播消息给所有智能体.

        Args:
            message: 要广播的消息

        Returns:
            接收者数量
        """
        count = 0
        for agent_id in self._agents:
            if agent_id != message.sender_id:
                msg_copy = AgentMessage(
                    msg_id=message.msg_id,
                    timestamp=message.timestamp,
                    sender_id=message.sender_id,
                    receiver_id=agent_id,
                    msg_type=message.msg_type,
                    priority=message.priority,
                    payload=message.payload.copy(),
                    correlation_id=message.correlation_id,
                    ttl=message.ttl,
                )
                self.send(msg_copy)
                count += 1

        self._stats['broadcast'] += 1
        return count

    # -------------------------------------------------------------------------
    # Topic Subscription
    # -------------------------------------------------------------------------

    def subscribe(self, agent_id: str, topic: str) -> None:
        """订阅主题."""
        self._subscriptions[topic].add(agent_id)
        logger.debug(f"Agent {agent_id} subscribed to topic {topic}")

    def unsubscribe(self, agent_id: str, topic: str) -> None:
        """取消订阅."""
        self._subscriptions[topic].discard(agent_id)

    def publish(self, topic: str, message: AgentMessage) -> int:
        """
        发布消息到主题.

        Args:
            topic: 主题名称
            message: 消息

        Returns:
            接收者数量
        """
        subscribers = self._subscriptions.get(topic, set())
        count = 0

        for subscriber_id in subscribers:
            if subscriber_id != message.sender_id:
                msg_copy = AgentMessage(
                    msg_id=message.msg_id,
                    timestamp=message.timestamp,
                    sender_id=message.sender_id,
                    receiver_id=subscriber_id,
                    msg_type=message.msg_type,
                    priority=message.priority,
                    payload={**message.payload, '_topic': topic},
                    correlation_id=message.correlation_id,
                    ttl=message.ttl,
                )
                self.send(msg_copy)
                count += 1

        return count

    # -------------------------------------------------------------------------
    # Message Processing
    # -------------------------------------------------------------------------

    def process_messages(self, max_messages: int = 100) -> int:
        """
        处理消息队列中的消息.

        Args:
            max_messages: 最大处理消息数

        Returns:
            处理的消息数量
        """
        processed = 0

        while not self._message_queue.empty() and processed < max_messages:
            try:
                wrapped = self._message_queue.get_nowait()
                message = wrapped.message
                delivered = self._deliver_message(message)

                if delivered:
                    self._stats['delivered'] += 1
                else:
                    self._stats['dropped'] += 1

                processed += 1

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                break

        return processed

    def _deliver_message(self, message: AgentMessage) -> bool:
        """传递消息到目标智能体."""
        receiver_id = message.receiver_id

        # 空接收者 = 广播 (应该已经被展开)
        if not receiver_id:
            return False

        # 查找接收者
        receiver = self._agents.get(receiver_id)
        if not receiver:
            logger.warning(f"Receiver {receiver_id} not found")
            return False

        # 传递消息
        try:
            receiver.receive_message(message)
            return True
        except Exception as e:
            logger.error(f"Error delivering to {receiver_id}: {e}")
            return False

    def collect_outgoing(self) -> None:
        """收集所有智能体的待发送消息."""
        for agent in self._agents.values():
            messages = agent.get_outgoing_messages()
            for msg in messages:
                if not msg.receiver_id:
                    # 广播
                    self.broadcast(msg)
                else:
                    self.send(msg)

    # -------------------------------------------------------------------------
    # Filtering
    # -------------------------------------------------------------------------

    def add_filter(self, filter_func: Callable[[AgentMessage], bool]) -> None:
        """添加消息过滤器."""
        self._filters.append(filter_func)

    def clear_filters(self) -> None:
        """清除所有过滤器."""
        self._filters.clear()

    # -------------------------------------------------------------------------
    # Statistics & Monitoring
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息."""
        return {
            **self._stats,
            'queue_size': self._message_queue.qsize(),
            'registered_agents': len(self._agents),
            'subscriptions': {k: len(v) for k, v in self._subscriptions.items()},
        }

    def enable_history(self, enabled: bool = True) -> None:
        """启用/禁用消息历史."""
        self._history_enabled = enabled

    def get_history(
        self,
        limit: int = 100,
        msg_type: Optional[MessageType] = None
    ) -> List[AgentMessage]:
        """获取消息历史."""
        history = self._history[-limit:]
        if msg_type:
            history = [m for m in history if m.msg_type == msg_type]
        return history

    def clear(self) -> None:
        """清空消息队列和历史."""
        while not self._message_queue.empty():
            try:
                self._message_queue.get_nowait()
            except:
                break
        self._history.clear()


class AgentNetwork:
    """
    智能体网络.

    管理智能体的层级结构和通信。
    """

    def __init__(self, model: Optional['TangheSiphonModel'] = None):
        self.model = model

        # 消息总线
        self.message_bus = MessageBus()

        # 智能体注册表
        self._agents: Dict[str, BaseAgent] = {}

        # 层级结构
        self._hierarchy: Dict[str, List[str]] = defaultdict(list)

        # 根节点 (协调者)
        self._root: Optional[BaseAgent] = None

        # 运行状态
        self._running = False
        self._update_count = 0

    # -------------------------------------------------------------------------
    # Network Setup
    # -------------------------------------------------------------------------

    def create_standard_network(self) -> 'AgentNetwork':
        """
        创建标准的三层智能体网络.

        结构:
        - 1个中央协调者
        - 1个区域管理者 (管理所有3个闸门)
        - 3个闸门控制器
        """
        from .coordinator import CentralCoordinator
        from .zone_manager import ZoneManager
        from .device_controller import GateAgent

        # 创建协调者
        coordinator = CentralCoordinator(
            agent_id="coordinator",
            model=self.model,
        )
        self.add_agent(coordinator)
        self._root = coordinator

        # 创建区域管理者
        zone = ZoneManager(
            agent_id="zone_main",
            zone_name="主控制区",
            gates=[0, 1, 2],
            model=self.model,
        )
        self.add_agent(zone, parent_id="coordinator")

        # 创建闸门控制器
        for i in range(3):
            gate_agent = GateAgent(
                agent_id=f"gate_{i}",
                gate_index=i,
                model=self.model,
            )
            self.add_agent(gate_agent, parent_id="zone_main")

        logger.info("Standard agent network created")
        return self

    def add_agent(
        self,
        agent: BaseAgent,
        parent_id: Optional[str] = None
    ) -> None:
        """添加智能体到网络."""
        # 注册到网络
        self._agents[agent.agent_id] = agent

        # 注册到消息总线
        self.message_bus.register_agent(agent)

        # 设置层级关系
        if parent_id:
            parent = self._agents.get(parent_id)
            if parent:
                parent.add_child(agent)
                self._hierarchy[parent_id].append(agent.agent_id)

        logger.info(f"Agent {agent.agent_id} added to network")

    def remove_agent(self, agent_id: str) -> None:
        """从网络移除智能体."""
        agent = self._agents.pop(agent_id, None)
        if agent:
            self.message_bus.unregister_agent(agent_id)

            # 移除层级关系
            for parent_id, children in self._hierarchy.items():
                if agent_id in children:
                    children.remove(agent_id)

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """获取智能体."""
        return self._agents.get(agent_id)

    # -------------------------------------------------------------------------
    # Network Control
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """启动网络."""
        self._running = True

        # 启动所有智能体
        for agent in self._agents.values():
            if agent.parent_id is None:  # 从根节点开始
                agent.start()

        logger.info("Agent network started")

    def stop(self) -> None:
        """停止网络."""
        self._running = False

        # 停止所有智能体
        for agent in self._agents.values():
            if agent.parent_id is None:
                agent.stop()

        logger.info("Agent network stopped")

    def update(self, dt: float) -> Dict[str, Any]:
        """
        更新网络中的所有智能体.

        Args:
            dt: 时间步长

        Returns:
            更新结果
        """
        if not self._running:
            return {'status': 'not_running'}

        results = {}

        # 1. 更新智能体 (从根节点开始，递归更新子节点)
        if self._root:
            results['coordinator'] = self._root.update(dt)

        # 2. 收集发出的消息
        self.message_bus.collect_outgoing()

        # 3. 处理消息
        processed = self.message_bus.process_messages()

        results['messages_processed'] = processed
        results['update_count'] = self._update_count
        self._update_count += 1

        return results

    def reset(self) -> None:
        """重置网络."""
        for agent in self._agents.values():
            agent.reset()
        self.message_bus.clear()
        self._update_count = 0

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def get_network_status(self) -> Dict[str, Any]:
        """获取网络状态."""
        return {
            'running': self._running,
            'agents_count': len(self._agents),
            'hierarchy': dict(self._hierarchy),
            'message_bus': self.message_bus.get_stats(),
            'update_count': self._update_count,
        }

    def get_all_agent_status(self) -> Dict[str, Dict]:
        """获取所有智能体状态."""
        return {
            agent_id: agent.get_status()
            for agent_id, agent in self._agents.items()
        }

    def get_hierarchy_tree(self) -> Dict[str, Any]:
        """获取层级树."""
        def build_tree(agent_id: str) -> Dict[str, Any]:
            agent = self._agents.get(agent_id)
            if not agent:
                return {}

            return {
                'id': agent_id,
                'role': agent.role.name,
                'state': agent.state.name,
                'children': [
                    build_tree(child_id)
                    for child_id in self._hierarchy.get(agent_id, [])
                ],
            }

        if self._root:
            return build_tree(self._root.agent_id)
        return {}

    # -------------------------------------------------------------------------
    # Global Control
    # -------------------------------------------------------------------------

    def set_global_target_flow(self, target_flow: float) -> None:
        """设置全局目标流量."""
        if self._root:
            from .coordinator import CentralCoordinator
            if isinstance(self._root, CentralCoordinator):
                self._root.set_global_objective('total_flow', target_flow)

    def broadcast_emergency(self, reason: str) -> None:
        """广播紧急消息."""
        msg = AgentMessage(
            sender_id="network",
            msg_type=MessageType.EMERGENCY,
            priority=Priority.EMERGENCY,
            payload={'reason': reason},
        )
        self.message_bus.broadcast(msg)

    def inject_fault(self, gate_index: int, fault_type: str) -> None:
        """注入故障 (用于测试)."""
        if self.model:
            self.model.inject_fault(gate_index, fault_type)

            # 通知相关智能体
            msg = AgentMessage(
                sender_id="network",
                msg_type=MessageType.ALERT,
                priority=Priority.HIGH,
                payload={
                    'alert_type': 'fault_injection',
                    'gate': gate_index,
                    'fault_type': fault_type,
                },
            )
            self.message_bus.broadcast(msg)
