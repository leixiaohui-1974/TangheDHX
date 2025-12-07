# -*- coding: utf-8 -*-
"""
Central Coordinator Agent.

中央协调层智能体，负责：
- 全局目标分解与分配
- 区域间协调与冲突解决
- 整体性能优化
- 紧急情况处理
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .base import (
    BaseAgent, AgentState, AgentRole, AgentMessage,
    MessageType, Priority
)

logger = logging.getLogger(__name__)


@dataclass
class GlobalObjective:
    """全局目标定义."""
    objective_id: str
    name: str
    target_value: float
    current_value: float = 0.0
    weight: float = 1.0
    tolerance: float = 0.05  # 允许偏差
    priority: Priority = Priority.NORMAL

    @property
    def error(self) -> float:
        """计算误差."""
        return abs(self.target_value - self.current_value)

    @property
    def is_satisfied(self) -> bool:
        """是否满足目标."""
        return self.error <= self.target_value * self.tolerance


@dataclass
class ResourceAllocation:
    """资源分配."""
    zone_id: str
    flow_quota: float      # 流量配额 [m³/s]
    gate_assignment: List[int]  # 分配的闸门
    priority: int = 1


class CentralCoordinator(BaseAgent):
    """
    中央协调者智能体.

    职责：
    1. 全局目标管理 - 接收上级调度指令，分解为子目标
    2. 资源协调 - 在区域间分配流量配额和闸门资源
    3. 冲突解决 - 处理区域间的资源冲突
    4. 性能监控 - 监控整体系统性能
    5. 应急响应 - 处理紧急情况
    """

    def __init__(
        self,
        agent_id: str = "coordinator",
        model: Optional['TangheSiphonModel'] = None,
    ):
        super().__init__(agent_id, AgentRole.COORDINATOR)

        self.model = model

        # 全局目标
        self._global_objectives: Dict[str, GlobalObjective] = {}

        # 资源分配
        self._allocations: Dict[str, ResourceAllocation] = {}

        # 区域状态缓存
        self._zone_states: Dict[str, Dict[str, Any]] = {}

        # 协调参数
        self._coordination_interval = 1.0  # 协调周期 [s]
        self._last_coordination = 0.0

        # 冲突历史
        self._conflicts: List[Dict[str, Any]] = []

        # 性能指标
        self._performance_history: List[Dict[str, float]] = []

        # 注册消息处理器
        self._register_coordinator_handlers()

        # 默认目标
        self._setup_default_objectives()

    def _register_coordinator_handlers(self) -> None:
        """注册协调者特有的消息处理器."""
        self.register_handler(MessageType.REQUEST, self._handle_request)
        self.register_handler(MessageType.MEASUREMENT, self._handle_measurement)
        self.register_handler(MessageType.ALERT, self._handle_alert)
        self.register_handler(MessageType.NEGOTIATION, self._handle_negotiation)

    def _setup_default_objectives(self) -> None:
        """设置默认目标."""
        self._global_objectives['total_flow'] = GlobalObjective(
            objective_id='total_flow',
            name='总流量目标',
            target_value=100.0,
            weight=1.0,
            priority=Priority.HIGH,
        )
        self._global_objectives['vibration_limit'] = GlobalObjective(
            objective_id='vibration_limit',
            name='振动限制',
            target_value=0.5,  # g
            weight=0.5,
            priority=Priority.HIGH,
        )
        self._global_objectives['efficiency'] = GlobalObjective(
            objective_id='efficiency',
            name='效率目标',
            target_value=0.9,
            weight=0.3,
            priority=Priority.NORMAL,
        )

    # -------------------------------------------------------------------------
    # Core BDI Methods
    # -------------------------------------------------------------------------

    def perceive(self) -> Dict[str, Any]:
        """感知全局状态."""
        perceptions = {}

        # 从模型获取状态
        if self.model:
            perceptions['total_flow'] = float(np.sum(self.model.flow_rates))
            perceptions['max_vibration'] = float(np.max(self.model.vibration_accel))
            perceptions['gate_openings'] = self.model.gate_openings.tolist()
            perceptions['head_difference'] = (
                self.model.head_upstream - self.model.head_downstream
            )

        # 聚合区域状态
        for zone_id, zone_state in self._zone_states.items():
            perceptions[f'zone_{zone_id}_status'] = zone_state

        return perceptions

    def decide(self) -> List[Dict[str, Any]]:
        """全局决策."""
        decisions = []

        # 1. 评估全局目标
        objective_status = self._evaluate_objectives()

        # 2. 检测和解决冲突
        conflicts = self._detect_conflicts()
        if conflicts:
            resolutions = self._resolve_conflicts(conflicts)
            decisions.extend(resolutions)

        # 3. 资源重分配决策
        if self._should_reallocate():
            allocation_decisions = self._compute_allocation()
            decisions.extend(allocation_decisions)

        # 4. 处理异常情况
        anomalies = self._detect_anomalies()
        if anomalies:
            anomaly_decisions = self._handle_anomalies(anomalies)
            decisions.extend(anomaly_decisions)

        return decisions

    def act(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行协调动作."""
        results = []

        for decision in decisions:
            action_type = decision.get('action')

            if action_type == 'reallocate':
                result = self._execute_reallocation(decision)
            elif action_type == 'resolve_conflict':
                result = self._execute_conflict_resolution(decision)
            elif action_type == 'emergency_response':
                result = self._execute_emergency_response(decision)
            elif action_type == 'adjust_objective':
                result = self._execute_objective_adjustment(decision)
            else:
                result = {'status': 'unknown_action', 'decision': decision}

            results.append(result)

        return results

    # -------------------------------------------------------------------------
    # Objective Management
    # -------------------------------------------------------------------------

    def set_global_objective(
        self,
        objective_id: str,
        target_value: float,
        weight: float = 1.0,
        priority: Priority = Priority.NORMAL,
    ) -> None:
        """设置全局目标."""
        if objective_id in self._global_objectives:
            obj = self._global_objectives[objective_id]
            obj.target_value = target_value
            obj.weight = weight
            obj.priority = priority
        else:
            self._global_objectives[objective_id] = GlobalObjective(
                objective_id=objective_id,
                name=objective_id,
                target_value=target_value,
                weight=weight,
                priority=priority,
            )

        # 通知区域管理者
        self._broadcast_objective_update(objective_id)

    def _evaluate_objectives(self) -> Dict[str, Dict[str, Any]]:
        """评估全局目标完成情况."""
        status = {}

        for obj_id, obj in self._global_objectives.items():
            # 从信念中获取当前值
            current = self.get_belief(obj_id, obj.current_value)
            obj.current_value = current

            status[obj_id] = {
                'target': obj.target_value,
                'current': current,
                'error': obj.error,
                'satisfied': obj.is_satisfied,
                'priority': obj.priority.name,
            }

        return status

    def _broadcast_objective_update(self, objective_id: str) -> None:
        """广播目标更新."""
        obj = self._global_objectives.get(objective_id)
        if not obj:
            return

        msg = AgentMessage(
            msg_type=MessageType.SETPOINT,
            priority=obj.priority,
            payload={
                'objective_id': objective_id,
                'target_value': obj.target_value,
                'weight': obj.weight,
            },
        )

        # 发送给所有区域管理者
        for child in self._children.values():
            msg.receiver_id = child.agent_id
            self.send_message(msg)

    # -------------------------------------------------------------------------
    # Resource Allocation
    # -------------------------------------------------------------------------

    def _should_reallocate(self) -> bool:
        """判断是否需要重新分配资源."""
        # 检查目标是否满足
        for obj in self._global_objectives.values():
            if not obj.is_satisfied and obj.priority.value >= Priority.HIGH.value:
                return True

        # 检查是否有区域请求
        for zone_state in self._zone_states.values():
            if zone_state.get('needs_reallocation', False):
                return True

        return False

    def _compute_allocation(self) -> List[Dict[str, Any]]:
        """计算资源分配."""
        decisions = []

        # 获取总流量目标
        total_flow_obj = self._global_objectives.get('total_flow')
        if not total_flow_obj:
            return decisions

        target_flow = total_flow_obj.target_value
        num_zones = max(1, len(self._children))

        # 简单均分策略 (可扩展为更复杂的优化算法)
        base_allocation = target_flow / num_zones

        for zone_id in self._children.keys():
            # 考虑区域能力和状态
            zone_state = self._zone_states.get(zone_id, {})
            capacity = zone_state.get('capacity', base_allocation * 1.5)
            current_load = zone_state.get('current_flow', 0.0)

            # 计算配额
            quota = min(base_allocation, capacity)

            # 生成分配决策
            decisions.append({
                'action': 'reallocate',
                'zone_id': zone_id,
                'flow_quota': quota,
                'reason': 'periodic_reallocation',
            })

            # 更新分配记录
            self._allocations[zone_id] = ResourceAllocation(
                zone_id=zone_id,
                flow_quota=quota,
                gate_assignment=zone_state.get('gates', []),
            )

        return decisions

    def _execute_reallocation(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """执行资源重分配."""
        zone_id = decision['zone_id']
        quota = decision['flow_quota']

        # 发送配额更新消息
        msg = AgentMessage(
            receiver_id=zone_id,
            msg_type=MessageType.COMMAND,
            priority=Priority.HIGH,
            payload={
                'command': 'set_quota',
                'flow_quota': quota,
            },
        )
        self.send_message(msg)

        return {
            'status': 'ok',
            'zone_id': zone_id,
            'new_quota': quota,
        }

    # -------------------------------------------------------------------------
    # Conflict Resolution
    # -------------------------------------------------------------------------

    def _detect_conflicts(self) -> List[Dict[str, Any]]:
        """检测资源冲突."""
        conflicts = []

        # 检测流量配额冲突
        total_allocated = sum(a.flow_quota for a in self._allocations.values())
        total_target = self._global_objectives.get('total_flow', GlobalObjective(
            'total_flow', 'Total Flow', 100.0
        )).target_value

        if total_allocated > total_target * 1.1:
            conflicts.append({
                'type': 'over_allocation',
                'total_allocated': total_allocated,
                'target': total_target,
                'severity': 'medium',
            })

        # 检测闸门分配冲突
        assigned_gates: Dict[int, List[str]] = {}
        for zone_id, alloc in self._allocations.items():
            for gate in alloc.gate_assignment:
                if gate not in assigned_gates:
                    assigned_gates[gate] = []
                assigned_gates[gate].append(zone_id)

        for gate, zones in assigned_gates.items():
            if len(zones) > 1:
                conflicts.append({
                    'type': 'gate_conflict',
                    'gate': gate,
                    'zones': zones,
                    'severity': 'high',
                })

        return conflicts

    def _resolve_conflicts(
        self,
        conflicts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """解决冲突."""
        decisions = []

        for conflict in conflicts:
            conflict_type = conflict['type']

            if conflict_type == 'over_allocation':
                # 按比例削减配额
                ratio = conflict['target'] / conflict['total_allocated']
                for zone_id, alloc in self._allocations.items():
                    decisions.append({
                        'action': 'resolve_conflict',
                        'conflict_type': conflict_type,
                        'zone_id': zone_id,
                        'new_quota': alloc.flow_quota * ratio,
                    })

            elif conflict_type == 'gate_conflict':
                # 按优先级分配闸门
                zones = conflict['zones']
                gate = conflict['gate']
                # 简单策略：分配给第一个区域
                winner = zones[0]
                for zone_id in zones[1:]:
                    decisions.append({
                        'action': 'resolve_conflict',
                        'conflict_type': conflict_type,
                        'zone_id': zone_id,
                        'release_gate': gate,
                    })

            # 记录冲突
            conflict['resolved_at'] = datetime.now().isoformat()
            self._conflicts.append(conflict)

        return decisions

    def _execute_conflict_resolution(
        self,
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行冲突解决."""
        zone_id = decision.get('zone_id')

        if 'new_quota' in decision:
            # 更新配额
            if zone_id in self._allocations:
                self._allocations[zone_id].flow_quota = decision['new_quota']

            msg = AgentMessage(
                receiver_id=zone_id,
                msg_type=MessageType.COMMAND,
                payload={
                    'command': 'update_quota',
                    'quota': decision['new_quota'],
                    'reason': 'conflict_resolution',
                },
            )
            self.send_message(msg)

        if 'release_gate' in decision:
            # 释放闸门
            msg = AgentMessage(
                receiver_id=zone_id,
                msg_type=MessageType.COMMAND,
                payload={
                    'command': 'release_gate',
                    'gate': decision['release_gate'],
                },
            )
            self.send_message(msg)

        return {'status': 'ok', 'decision': decision}

    # -------------------------------------------------------------------------
    # Anomaly Detection & Emergency Response
    # -------------------------------------------------------------------------

    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """检测异常."""
        anomalies = []

        # 振动异常
        max_vib = self.get_belief('max_vibration', 0.0)
        if max_vib > 0.7:  # 临界振动
            anomalies.append({
                'type': 'high_vibration',
                'value': max_vib,
                'threshold': 0.7,
                'severity': 'critical' if max_vib > 0.9 else 'warning',
            })

        # 流量偏差
        target_flow = self._global_objectives.get(
            'total_flow', GlobalObjective('', '', 100.0)
        ).target_value
        current_flow = self.get_belief('total_flow', target_flow)
        flow_error = abs(current_flow - target_flow) / max(target_flow, 1.0)

        if flow_error > 0.2:  # 20%偏差
            anomalies.append({
                'type': 'flow_deviation',
                'current': current_flow,
                'target': target_flow,
                'error_percent': flow_error * 100,
                'severity': 'warning',
            })

        return anomalies

    def _handle_anomalies(
        self,
        anomalies: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """处理异常."""
        decisions = []

        for anomaly in anomalies:
            anomaly_type = anomaly['type']

            if anomaly_type == 'high_vibration':
                # 降低流量以减少振动
                decisions.append({
                    'action': 'emergency_response',
                    'response_type': 'reduce_flow',
                    'factor': 0.8,  # 降低20%
                    'reason': anomaly,
                })

            elif anomaly_type == 'flow_deviation':
                # 调整配额
                decisions.append({
                    'action': 'adjust_objective',
                    'objective_id': 'total_flow',
                    'adjustment': 'recompute',
                    'reason': anomaly,
                })

        return decisions

    def _execute_emergency_response(
        self,
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行紧急响应."""
        response_type = decision.get('response_type')

        if response_type == 'reduce_flow':
            factor = decision.get('factor', 0.8)

            # 广播紧急减流指令
            msg = AgentMessage(
                msg_type=MessageType.EMERGENCY,
                priority=Priority.EMERGENCY,
                payload={
                    'command': 'reduce_flow',
                    'factor': factor,
                    'reason': decision.get('reason', {}),
                },
            )

            for child in self._children.values():
                msg.receiver_id = child.agent_id
                self.send_message(msg)

            return {'status': 'ok', 'response_type': response_type}

        return {'status': 'unknown_response', 'decision': decision}

    def _execute_objective_adjustment(
        self,
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行目标调整."""
        obj_id = decision.get('objective_id')
        if obj_id and decision.get('adjustment') == 'recompute':
            self._broadcast_objective_update(obj_id)

        return {'status': 'ok', 'decision': decision}

    # -------------------------------------------------------------------------
    # Message Handlers
    # -------------------------------------------------------------------------

    def _handle_request(self, msg: AgentMessage) -> None:
        """处理请求消息."""
        request_type = msg.payload.get('request_type')

        if request_type == 'quota_increase':
            # 处理配额增加请求
            zone_id = msg.sender_id
            requested = msg.payload.get('requested_quota', 0.0)

            # 简单策略：如果有余量则批准
            total_allocated = sum(a.flow_quota for a in self._allocations.values())
            target = self._global_objectives.get('total_flow', GlobalObjective(
                '', '', 100.0
            )).target_value
            available = target - total_allocated

            approved = min(requested, available * 0.5)  # 最多批准50%余量

            response = AgentMessage(
                receiver_id=zone_id,
                msg_type=MessageType.RESPONSE,
                correlation_id=msg.msg_id,
                payload={
                    'request_type': request_type,
                    'approved': approved > 0,
                    'approved_quota': approved,
                },
            )
            self.send_message(response)

    def _handle_measurement(self, msg: AgentMessage) -> None:
        """处理测量数据消息."""
        zone_id = msg.sender_id
        self._zone_states[zone_id] = msg.payload

    def _handle_alert(self, msg: AgentMessage) -> None:
        """处理告警消息."""
        alert_type = msg.payload.get('alert_type')
        severity = msg.payload.get('severity', 'info')

        logger.warning(f"Alert from {msg.sender_id}: {alert_type} ({severity})")

        # 记录并可能触发紧急响应
        if severity == 'critical':
            emergency_msg = AgentMessage(
                msg_type=MessageType.EMERGENCY,
                priority=Priority.EMERGENCY,
                payload=msg.payload,
            )
            self._handle_emergency(emergency_msg)

    def _handle_negotiation(self, msg: AgentMessage) -> None:
        """处理协商消息."""
        # 区域间协商的仲裁
        negotiation_type = msg.payload.get('negotiation_type')

        if negotiation_type == 'resource_sharing':
            # 资源共享协商
            parties = msg.payload.get('parties', [])
            resource = msg.payload.get('resource')

            # 简单仲裁：按当前负载比例分配
            total_load = 0
            loads = {}
            for party in parties:
                state = self._zone_states.get(party, {})
                load = state.get('current_flow', 0.0)
                loads[party] = load
                total_load += load

            # 发送仲裁结果
            for party in parties:
                share = loads[party] / max(total_load, 1.0)
                response = AgentMessage(
                    receiver_id=party,
                    msg_type=MessageType.AGREEMENT,
                    correlation_id=msg.msg_id,
                    payload={
                        'resource': resource,
                        'share': share,
                    },
                )
                self.send_message(response)

    # -------------------------------------------------------------------------
    # Performance Monitoring
    # -------------------------------------------------------------------------

    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标."""
        return {
            'objectives': self._evaluate_objectives(),
            'allocations': {
                k: {'quota': v.flow_quota, 'gates': v.gate_assignment}
                for k, v in self._allocations.items()
            },
            'conflicts_count': len(self._conflicts),
            'zones_count': len(self._children),
            'stats': self._stats.copy(),
        }
