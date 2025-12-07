# -*- coding: utf-8 -*-
"""
Base Agent Classes for Hierarchical Distributed System.

提供智能体的基础类定义，包括：
- 智能体状态管理
- 消息通信协议
- 生命周期管理
- 决策框架
"""

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable, Set
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Agent State Management
# =============================================================================

class AgentState(Enum):
    """智能体状态枚举."""
    INITIALIZING = auto()   # 初始化中
    IDLE = auto()           # 空闲
    RUNNING = auto()        # 运行中
    PAUSED = auto()         # 暂停
    ERROR = auto()          # 错误
    SHUTDOWN = auto()       # 关闭


class AgentRole(Enum):
    """智能体角色枚举."""
    COORDINATOR = auto()    # 中央协调者
    ZONE_MANAGER = auto()   # 区域管理者
    DEVICE_CONTROLLER = auto()  # 设备控制器
    SENSOR = auto()         # 传感器代理
    OBSERVER = auto()       # 观察者


class MessageType(Enum):
    """消息类型枚举."""
    # 控制消息
    COMMAND = auto()        # 控制指令
    SETPOINT = auto()       # 设定值
    CONSTRAINT = auto()     # 约束条件

    # 状态消息
    STATUS = auto()         # 状态报告
    MEASUREMENT = auto()    # 测量数据
    DIAGNOSTIC = auto()     # 诊断信息

    # 协调消息
    REQUEST = auto()        # 请求
    RESPONSE = auto()       # 响应
    NEGOTIATION = auto()    # 协商
    AGREEMENT = auto()      # 协议达成

    # 系统消息
    HEARTBEAT = auto()      # 心跳
    ALERT = auto()          # 告警
    EMERGENCY = auto()      # 紧急
    SHUTDOWN = auto()       # 关闭


class Priority(Enum):
    """消息优先级."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    EMERGENCY = 5


# =============================================================================
# Message Protocol
# =============================================================================

@dataclass
class AgentMessage:
    """智能体间通信消息."""
    # 消息标识
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)

    # 路由信息
    sender_id: str = ""
    receiver_id: str = ""  # 空字符串表示广播

    # 消息内容
    msg_type: MessageType = MessageType.STATUS
    priority: Priority = Priority.NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)

    # 追踪
    correlation_id: Optional[str] = None  # 关联消息ID
    ttl: int = 10  # 生存时间(跳数)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典."""
        return {
            'msg_id': self.msg_id,
            'timestamp': self.timestamp.isoformat(),
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'msg_type': self.msg_type.name,
            'priority': self.priority.name,
            'payload': self.payload,
            'correlation_id': self.correlation_id,
            'ttl': self.ttl,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """从字典创建."""
        return cls(
            msg_id=data.get('msg_id', ''),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now(),
            sender_id=data.get('sender_id', ''),
            receiver_id=data.get('receiver_id', ''),
            msg_type=MessageType[data.get('msg_type', 'STATUS')],
            priority=Priority[data.get('priority', 'NORMAL')],
            payload=data.get('payload', {}),
            correlation_id=data.get('correlation_id'),
            ttl=data.get('ttl', 10),
        )


# =============================================================================
# Base Agent
# =============================================================================

class BaseAgent(ABC):
    """
    智能体基类.

    所有智能体必须继承此类，实现核心方法：
    - perceive(): 感知环境
    - decide(): 决策
    - act(): 执行动作
    - communicate(): 通信
    """

    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        parent_id: Optional[str] = None,
    ):
        """
        初始化智能体.

        Args:
            agent_id: 唯一标识符
            role: 智能体角色
            parent_id: 父智能体ID (用于层级结构)
        """
        self.agent_id = agent_id
        self.role = role
        self.parent_id = parent_id

        # 状态
        self._state = AgentState.INITIALIZING
        self._last_update = datetime.now()

        # 子智能体
        self._children: Dict[str, 'BaseAgent'] = {}

        # 消息队列
        self._inbox: List[AgentMessage] = []
        self._outbox: List[AgentMessage] = []

        # 信念-愿望-意图 (BDI)
        self._beliefs: Dict[str, Any] = {}      # 环境信念
        self._desires: Dict[str, float] = {}    # 目标愿望
        self._intentions: List[Dict] = []       # 当前意图

        # 消息处理器
        self._message_handlers: Dict[MessageType, Callable] = {}
        self._register_default_handlers()

        # 性能统计
        self._stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'decisions_made': 0,
            'actions_taken': 0,
            'errors': 0,
        }

        logger.info(f"Agent {agent_id} ({role.name}) initialized")

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def state(self) -> AgentState:
        """获取当前状态."""
        return self._state

    @state.setter
    def state(self, value: AgentState) -> None:
        """设置状态."""
        old_state = self._state
        self._state = value
        logger.debug(f"Agent {self.agent_id}: {old_state.name} -> {value.name}")

    @property
    def is_active(self) -> bool:
        """是否处于活动状态."""
        return self._state in [AgentState.RUNNING, AgentState.IDLE]

    @property
    def children(self) -> Dict[str, 'BaseAgent']:
        """获取子智能体."""
        return self._children.copy()

    # -------------------------------------------------------------------------
    # Core Agent Loop (BDI Cycle)
    # -------------------------------------------------------------------------

    def update(self, dt: float) -> Dict[str, Any]:
        """
        智能体主循环 (BDI循环).

        Args:
            dt: 时间步长 [s]

        Returns:
            更新结果
        """
        if self._state != AgentState.RUNNING:
            return {'status': 'inactive', 'state': self._state.name}

        try:
            # 1. 感知 (Perceive) - 更新信念
            perceptions = self.perceive()
            self._update_beliefs(perceptions)

            # 2. 处理消息 - 更新信念
            self._process_inbox()

            # 3. 决策 (Decide) - 生成意图
            decisions = self.decide()
            self._stats['decisions_made'] += 1

            # 4. 执行 (Act) - 执行意图
            actions = self.act(decisions)
            self._stats['actions_taken'] += 1

            # 5. 通信 (Communicate) - 发送消息
            self._process_outbox()

            # 6. 更新子智能体
            child_results = {}
            for child_id, child in self._children.items():
                child_results[child_id] = child.update(dt)

            self._last_update = datetime.now()

            return {
                'status': 'ok',
                'perceptions': perceptions,
                'decisions': decisions,
                'actions': actions,
                'children': child_results,
            }

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Agent {self.agent_id} error: {e}")
            self._state = AgentState.ERROR
            return {'status': 'error', 'error': str(e)}

    @abstractmethod
    def perceive(self) -> Dict[str, Any]:
        """
        感知环境.

        Returns:
            感知结果字典
        """
        pass

    @abstractmethod
    def decide(self) -> List[Dict[str, Any]]:
        """
        决策过程.

        Returns:
            决策列表
        """
        pass

    @abstractmethod
    def act(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        执行动作.

        Args:
            decisions: 决策列表

        Returns:
            执行结果
        """
        pass

    # -------------------------------------------------------------------------
    # Belief Management
    # -------------------------------------------------------------------------

    def _update_beliefs(self, perceptions: Dict[str, Any]) -> None:
        """更新信念."""
        for key, value in perceptions.items():
            self._beliefs[key] = {
                'value': value,
                'timestamp': datetime.now(),
                'confidence': 1.0,
            }

    def get_belief(self, key: str, default: Any = None) -> Any:
        """获取信念值."""
        if key in self._beliefs:
            return self._beliefs[key]['value']
        return default

    def set_belief(self, key: str, value: Any, confidence: float = 1.0) -> None:
        """设置信念."""
        self._beliefs[key] = {
            'value': value,
            'timestamp': datetime.now(),
            'confidence': confidence,
        }

    # -------------------------------------------------------------------------
    # Desire & Intention Management
    # -------------------------------------------------------------------------

    def add_desire(self, goal: str, priority: float) -> None:
        """添加愿望/目标."""
        self._desires[goal] = priority

    def remove_desire(self, goal: str) -> None:
        """移除愿望."""
        self._desires.pop(goal, None)

    def add_intention(self, action: str, params: Dict[str, Any]) -> None:
        """添加意图."""
        self._intentions.append({
            'action': action,
            'params': params,
            'created': datetime.now(),
        })

    def clear_intentions(self) -> None:
        """清空意图."""
        self._intentions.clear()

    # -------------------------------------------------------------------------
    # Communication
    # -------------------------------------------------------------------------

    def send_message(self, message: AgentMessage) -> None:
        """发送消息."""
        message.sender_id = self.agent_id
        self._outbox.append(message)
        self._stats['messages_sent'] += 1

    def receive_message(self, message: AgentMessage) -> None:
        """接收消息."""
        if message.ttl <= 0:
            return  # 消息过期
        message.ttl -= 1
        self._inbox.append(message)
        self._stats['messages_received'] += 1

    def _process_inbox(self) -> None:
        """处理收件箱."""
        # 按优先级排序
        self._inbox.sort(key=lambda m: m.priority.value, reverse=True)

        while self._inbox:
            msg = self._inbox.pop(0)
            handler = self._message_handlers.get(msg.msg_type)
            if handler:
                try:
                    handler(msg)
                except Exception as e:
                    logger.error(f"Message handler error: {e}")

    def _process_outbox(self) -> None:
        """处理发件箱 (由消息总线调用)."""
        # 消息将由外部消息总线处理
        pass

    def get_outgoing_messages(self) -> List[AgentMessage]:
        """获取并清空待发送消息."""
        messages = self._outbox.copy()
        self._outbox.clear()
        return messages

    def _register_default_handlers(self) -> None:
        """注册默认消息处理器."""
        self._message_handlers[MessageType.HEARTBEAT] = self._handle_heartbeat
        self._message_handlers[MessageType.STATUS] = self._handle_status
        self._message_handlers[MessageType.EMERGENCY] = self._handle_emergency

    def _handle_heartbeat(self, msg: AgentMessage) -> None:
        """处理心跳消息."""
        # 回复心跳
        response = AgentMessage(
            receiver_id=msg.sender_id,
            msg_type=MessageType.HEARTBEAT,
            correlation_id=msg.msg_id,
            payload={'alive': True, 'state': self._state.name},
        )
        self.send_message(response)

    def _handle_status(self, msg: AgentMessage) -> None:
        """处理状态请求."""
        response = AgentMessage(
            receiver_id=msg.sender_id,
            msg_type=MessageType.STATUS,
            correlation_id=msg.msg_id,
            payload=self.get_status(),
        )
        self.send_message(response)

    def _handle_emergency(self, msg: AgentMessage) -> None:
        """处理紧急消息."""
        logger.warning(f"Agent {self.agent_id} received EMERGENCY: {msg.payload}")
        # 转发给子智能体
        for child in self._children.values():
            child.receive_message(msg)

    def register_handler(
        self,
        msg_type: MessageType,
        handler: Callable[[AgentMessage], None]
    ) -> None:
        """注册消息处理器."""
        self._message_handlers[msg_type] = handler

    # -------------------------------------------------------------------------
    # Hierarchy Management
    # -------------------------------------------------------------------------

    def add_child(self, child: 'BaseAgent') -> None:
        """添加子智能体."""
        child.parent_id = self.agent_id
        self._children[child.agent_id] = child
        logger.info(f"Agent {self.agent_id} added child {child.agent_id}")

    def remove_child(self, child_id: str) -> Optional['BaseAgent']:
        """移除子智能体."""
        return self._children.pop(child_id, None)

    def get_child(self, child_id: str) -> Optional['BaseAgent']:
        """获取子智能体."""
        return self._children.get(child_id)

    # -------------------------------------------------------------------------
    # Lifecycle Management
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """启动智能体."""
        self._state = AgentState.RUNNING
        logger.info(f"Agent {self.agent_id} started")

        # 启动子智能体
        for child in self._children.values():
            child.start()

    def stop(self) -> None:
        """停止智能体."""
        self._state = AgentState.SHUTDOWN
        logger.info(f"Agent {self.agent_id} stopped")

        # 停止子智能体
        for child in self._children.values():
            child.stop()

    def pause(self) -> None:
        """暂停智能体."""
        self._state = AgentState.PAUSED

    def resume(self) -> None:
        """恢复智能体."""
        if self._state == AgentState.PAUSED:
            self._state = AgentState.RUNNING

    def reset(self) -> None:
        """重置智能体."""
        self._beliefs.clear()
        self._desires.clear()
        self._intentions.clear()
        self._inbox.clear()
        self._outbox.clear()
        self._state = AgentState.IDLE

        for child in self._children.values():
            child.reset()

    # -------------------------------------------------------------------------
    # Status & Diagnostics
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """获取智能体状态."""
        return {
            'agent_id': self.agent_id,
            'role': self.role.name,
            'state': self._state.name,
            'parent_id': self.parent_id,
            'children': list(self._children.keys()),
            'beliefs_count': len(self._beliefs),
            'desires_count': len(self._desires),
            'intentions_count': len(self._intentions),
            'inbox_size': len(self._inbox),
            'outbox_size': len(self._outbox),
            'stats': self._stats.copy(),
            'last_update': self._last_update.isoformat(),
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """获取诊断信息."""
        return {
            'status': self.get_status(),
            'beliefs': {k: v['value'] for k, v in self._beliefs.items()},
            'desires': self._desires.copy(),
            'intentions': self._intentions.copy(),
        }


# =============================================================================
# Utility Classes
# =============================================================================

@dataclass
class AgentCapability:
    """智能体能力描述."""
    name: str
    description: str
    parameters: Dict[str, type] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentContract:
    """智能体间协议/合同."""
    contract_id: str
    parties: List[str]  # 参与方agent_id
    terms: Dict[str, Any]
    created: datetime = field(default_factory=datetime.now)
    expires: Optional[datetime] = None
    status: str = "active"
