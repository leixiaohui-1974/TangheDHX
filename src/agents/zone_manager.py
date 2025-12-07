# -*- coding: utf-8 -*-
"""
Zone Manager Agent.

区域管理层智能体，负责：
- 区域内设备协调
- 子目标分解与执行
- 本地优化
- 状态聚合与上报
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
class ZoneObjective:
    """区域目标."""
    objective_id: str
    name: str
    target_value: float
    current_value: float = 0.0
    weight: float = 1.0
    source: str = "coordinator"  # 目标来源


@dataclass
class DeviceState:
    """设备状态."""
    device_id: str
    device_type: str
    status: str = "normal"
    last_reading: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)


class ZoneManager(BaseAgent):
    """
    区域管理者智能体.

    职责：
    1. 子目标管理 - 接收协调者目标，分解为设备级任务
    2. 设备协调 - 协调区域内多个设备控制器
    3. 本地优化 - 在约束内优化区域性能
    4. 状态聚合 - 聚合设备状态并上报
    5. 故障隔离 - 处理区域内故障
    """

    def __init__(
        self,
        agent_id: str,
        zone_name: str,
        gates: List[int],
        model: Optional['TangheSiphonModel'] = None,
    ):
        """
        初始化区域管理者.

        Args:
            agent_id: 智能体ID
            zone_name: 区域名称
            gates: 管理的闸门索引列表
            model: 物理模型引用
        """
        super().__init__(agent_id, AgentRole.ZONE_MANAGER)

        self.zone_name = zone_name
        self.gates = gates
        self.model = model

        # 区域目标
        self._zone_objectives: Dict[str, ZoneObjective] = {}

        # 资源配额
        self._flow_quota: float = 100.0 / 3  # 默认均分

        # 设备状态缓存
        self._device_states: Dict[str, DeviceState] = {}

        # 本地控制参数
        self._local_setpoints: Dict[int, float] = {g: 0.0 for g in gates}

        # 优化状态
        self._optimization_enabled = True
        self._last_optimization = datetime.now()

        # 故障记录
        self._faults: List[Dict[str, Any]] = []

        # 注册消息处理器
        self._register_zone_handlers()

    def _register_zone_handlers(self) -> None:
        """注册区域管理者消息处理器."""
        self.register_handler(MessageType.COMMAND, self._handle_command)
        self.register_handler(MessageType.SETPOINT, self._handle_setpoint)
        self.register_handler(MessageType.MEASUREMENT, self._handle_measurement)
        self.register_handler(MessageType.DIAGNOSTIC, self._handle_diagnostic)

    # -------------------------------------------------------------------------
    # Core BDI Methods
    # -------------------------------------------------------------------------

    def perceive(self) -> Dict[str, Any]:
        """感知区域状态."""
        perceptions = {}

        if self.model:
            # 区域流量
            zone_flow = sum(self.model.flow_rates[g] for g in self.gates)
            perceptions['zone_flow'] = zone_flow

            # 区域振动
            zone_vibration = max(self.model.vibration_accel[g] for g in self.gates)
            perceptions['zone_vibration'] = zone_vibration

            # 闸门状态
            for g in self.gates:
                perceptions[f'gate_{g}_opening'] = self.model.gate_openings[g]
                perceptions[f'gate_{g}_flow'] = self.model.flow_rates[g]
                perceptions[f'gate_{g}_vibration'] = self.model.vibration_accel[g]

            # 水头
            perceptions['head_upstream'] = self.model.head_upstream
            perceptions['head_downstream'] = self.model.head_downstream

        # 聚合子设备状态
        for device_id, state in self._device_states.items():
            perceptions[f'device_{device_id}'] = state.status

        return perceptions

    def decide(self) -> List[Dict[str, Any]]:
        """区域级决策."""
        decisions = []

        # 1. 评估区域目标
        objective_gaps = self._evaluate_objectives()

        # 2. 本地优化决策
        if self._optimization_enabled and objective_gaps:
            optimization_decisions = self._optimize_locally(objective_gaps)
            decisions.extend(optimization_decisions)

        # 3. 分配子任务给设备
        task_allocations = self._allocate_tasks()
        decisions.extend(task_allocations)

        # 4. 故障处理决策
        fault_decisions = self._handle_local_faults()
        decisions.extend(fault_decisions)

        return decisions

    def act(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行区域动作."""
        results = []

        for decision in decisions:
            action_type = decision.get('action')

            if action_type == 'set_gate_opening':
                result = self._execute_gate_control(decision)
            elif action_type == 'allocate_task':
                result = self._execute_task_allocation(decision)
            elif action_type == 'report_status':
                result = self._execute_status_report(decision)
            elif action_type == 'handle_fault':
                result = self._execute_fault_handling(decision)
            else:
                result = {'status': 'unknown_action'}

            results.append(result)

        return results

    # -------------------------------------------------------------------------
    # Objective Management
    # -------------------------------------------------------------------------

    def _evaluate_objectives(self) -> Dict[str, float]:
        """评估目标差距."""
        gaps = {}

        for obj_id, obj in self._zone_objectives.items():
            current = self.get_belief(obj_id, obj.current_value)
            obj.current_value = current
            gap = obj.target_value - current
            if abs(gap) > obj.target_value * 0.05:  # 5%容差
                gaps[obj_id] = gap

        return gaps

    def set_zone_objective(
        self,
        objective_id: str,
        target_value: float,
        weight: float = 1.0,
    ) -> None:
        """设置区域目标."""
        self._zone_objectives[objective_id] = ZoneObjective(
            objective_id=objective_id,
            name=objective_id,
            target_value=target_value,
            weight=weight,
        )

    # -------------------------------------------------------------------------
    # Local Optimization
    # -------------------------------------------------------------------------

    def _optimize_locally(
        self,
        objective_gaps: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """本地优化."""
        decisions = []

        # 流量优化
        if 'zone_flow' in objective_gaps:
            flow_gap = objective_gaps['zone_flow']
            decisions.extend(self._optimize_flow(flow_gap))

        # 振动优化
        zone_vib = self.get_belief('zone_vibration', 0.0)
        if zone_vib > 0.3:  # 振动阈值
            decisions.extend(self._optimize_vibration(zone_vib))

        return decisions

    def _optimize_flow(self, flow_gap: float) -> List[Dict[str, Any]]:
        """优化流量."""
        decisions = []

        # 根据流量差距调整闸门
        adjustment_per_gate = flow_gap / max(len(self.gates), 1)

        for gate in self.gates:
            current_opening = self.get_belief(f'gate_{gate}_opening', 0.0)

            # 简单比例调整
            if flow_gap > 0:  # 需要增加流量
                new_opening = min(5.0, current_opening + 0.1)
            else:  # 需要减少流量
                new_opening = max(0.0, current_opening - 0.1)

            if abs(new_opening - current_opening) > 0.01:
                decisions.append({
                    'action': 'set_gate_opening',
                    'gate': gate,
                    'opening': new_opening,
                    'reason': 'flow_optimization',
                })

        return decisions

    def _optimize_vibration(self, current_vib: float) -> List[Dict[str, Any]]:
        """优化振动."""
        decisions = []

        # 找到振动最大的闸门
        max_vib_gate = None
        max_vib = 0.0

        for gate in self.gates:
            vib = self.get_belief(f'gate_{gate}_vibration', 0.0)
            if vib > max_vib:
                max_vib = vib
                max_vib_gate = gate

        if max_vib_gate is not None and max_vib > 0.3:
            current_opening = self.get_belief(f'gate_{max_vib_gate}_opening', 0.0)

            # 微调开度以避开共振
            decisions.append({
                'action': 'set_gate_opening',
                'gate': max_vib_gate,
                'opening': current_opening * 0.95,  # 略微减小
                'reason': 'vibration_optimization',
            })

        return decisions

    # -------------------------------------------------------------------------
    # Task Allocation
    # -------------------------------------------------------------------------

    def _allocate_tasks(self) -> List[Dict[str, Any]]:
        """分配任务给设备控制器."""
        decisions = []

        # 计算每个闸门的目标流量
        zone_flow_target = self._zone_objectives.get(
            'zone_flow',
            ZoneObjective('zone_flow', 'Zone Flow', self._flow_quota)
        ).target_value

        flow_per_gate = zone_flow_target / max(len(self.gates), 1)

        for gate in self.gates:
            device_id = f"gate_{gate}"
            if device_id in self._children:
                decisions.append({
                    'action': 'allocate_task',
                    'device_id': device_id,
                    'task': 'maintain_flow',
                    'target_flow': flow_per_gate,
                })

        return decisions

    def _execute_task_allocation(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务分配."""
        device_id = decision.get('device_id')
        task = decision.get('task')
        target = decision.get('target_flow', 0.0)

        msg = AgentMessage(
            receiver_id=device_id,
            msg_type=MessageType.COMMAND,
            payload={
                'task': task,
                'target_flow': target,
            },
        )
        self.send_message(msg)

        return {'status': 'ok', 'device_id': device_id, 'task': task}

    # -------------------------------------------------------------------------
    # Gate Control
    # -------------------------------------------------------------------------

    def _execute_gate_control(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """执行闸门控制."""
        gate = decision.get('gate')
        opening = decision.get('opening')

        if self.model and gate in self.gates:
            # 直接控制模型 (在实际系统中会发送给设备控制器)
            self.model.gate_openings[gate] = opening
            self._local_setpoints[gate] = opening

            return {
                'status': 'ok',
                'gate': gate,
                'opening': opening,
            }

        return {'status': 'error', 'reason': 'invalid_gate'}

    # -------------------------------------------------------------------------
    # Fault Handling
    # -------------------------------------------------------------------------

    def _handle_local_faults(self) -> List[Dict[str, Any]]:
        """处理本地故障."""
        decisions = []

        for gate in self.gates:
            # 检测闸门故障
            if self.model and self.model.gate_stuck[gate]:
                fault = {
                    'type': 'gate_stuck',
                    'gate': gate,
                    'detected_at': datetime.now().isoformat(),
                }

                if fault not in self._faults:
                    self._faults.append(fault)

                    # 上报故障
                    decisions.append({
                        'action': 'handle_fault',
                        'fault': fault,
                        'response': 'redistribute_load',
                    })

        return decisions

    def _execute_fault_handling(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """执行故障处理."""
        fault = decision.get('fault', {})
        response = decision.get('response')

        if response == 'redistribute_load':
            faulty_gate = fault.get('gate')

            # 将负载重新分配给其他闸门
            working_gates = [g for g in self.gates if g != faulty_gate]

            if working_gates:
                # 增加其他闸门的开度
                for gate in working_gates:
                    current = self.get_belief(f'gate_{gate}_opening', 0.0)
                    factor = len(self.gates) / len(working_gates)
                    new_opening = min(5.0, current * factor)
                    self._local_setpoints[gate] = new_opening

                    if self.model:
                        self.model.gate_openings[gate] = new_opening

        # 上报给协调者
        alert_msg = AgentMessage(
            receiver_id=self.parent_id or "",
            msg_type=MessageType.ALERT,
            priority=Priority.HIGH,
            payload={
                'alert_type': 'fault',
                'fault': fault,
                'zone_id': self.agent_id,
                'severity': 'high',
            },
        )
        self.send_message(alert_msg)

        return {'status': 'ok', 'fault_handled': fault}

    # -------------------------------------------------------------------------
    # Status Reporting
    # -------------------------------------------------------------------------

    def _execute_status_report(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """执行状态上报."""
        status = self._collect_zone_status()

        msg = AgentMessage(
            receiver_id=self.parent_id or "",
            msg_type=MessageType.MEASUREMENT,
            payload=status,
        )
        self.send_message(msg)

        return {'status': 'ok', 'reported': status}

    def _collect_zone_status(self) -> Dict[str, Any]:
        """收集区域状态."""
        return {
            'zone_id': self.agent_id,
            'zone_name': self.zone_name,
            'gates': self.gates,
            'current_flow': self.get_belief('zone_flow', 0.0),
            'max_vibration': self.get_belief('zone_vibration', 0.0),
            'flow_quota': self._flow_quota,
            'setpoints': self._local_setpoints.copy(),
            'faults': len(self._faults),
            'objectives_met': all(
                abs(obj.target_value - obj.current_value) < obj.target_value * 0.1
                for obj in self._zone_objectives.values()
            ),
        }

    # -------------------------------------------------------------------------
    # Message Handlers
    # -------------------------------------------------------------------------

    def _handle_command(self, msg: AgentMessage) -> None:
        """处理命令消息."""
        command = msg.payload.get('command')

        if command == 'set_quota':
            self._flow_quota = msg.payload.get('flow_quota', self._flow_quota)
            self.set_zone_objective('zone_flow', self._flow_quota)
            logger.info(f"Zone {self.agent_id} quota updated: {self._flow_quota}")

        elif command == 'update_quota':
            self._flow_quota = msg.payload.get('quota', self._flow_quota)
            self.set_zone_objective('zone_flow', self._flow_quota)

        elif command == 'release_gate':
            gate = msg.payload.get('gate')
            if gate in self.gates:
                self.gates.remove(gate)
                logger.info(f"Zone {self.agent_id} released gate {gate}")

        elif command == 'reduce_flow':
            factor = msg.payload.get('factor', 0.8)
            self._flow_quota *= factor
            self.set_zone_objective('zone_flow', self._flow_quota)
            logger.warning(f"Zone {self.agent_id} reducing flow by {1-factor:.0%}")

    def _handle_setpoint(self, msg: AgentMessage) -> None:
        """处理设定值消息."""
        obj_id = msg.payload.get('objective_id')
        target = msg.payload.get('target_value')
        weight = msg.payload.get('weight', 1.0)

        if obj_id and target is not None:
            self.set_zone_objective(obj_id, target, weight)

    def _handle_measurement(self, msg: AgentMessage) -> None:
        """处理测量消息 (来自设备控制器)."""
        device_id = msg.sender_id
        self._device_states[device_id] = DeviceState(
            device_id=device_id,
            device_type=msg.payload.get('device_type', 'unknown'),
            status=msg.payload.get('status', 'unknown'),
            last_reading=msg.payload.get('reading', 0.0),
        )

    def _handle_diagnostic(self, msg: AgentMessage) -> None:
        """处理诊断消息."""
        device_id = msg.sender_id
        diagnostic = msg.payload

        # 检查是否有故障
        if diagnostic.get('fault_detected'):
            fault = {
                'type': diagnostic.get('fault_type'),
                'device_id': device_id,
                'details': diagnostic,
                'detected_at': datetime.now().isoformat(),
            }
            self._faults.append(fault)

    # -------------------------------------------------------------------------
    # Negotiation
    # -------------------------------------------------------------------------

    def request_quota_increase(self, amount: float) -> None:
        """请求增加配额."""
        msg = AgentMessage(
            receiver_id=self.parent_id or "",
            msg_type=MessageType.REQUEST,
            priority=Priority.NORMAL,
            payload={
                'request_type': 'quota_increase',
                'current_quota': self._flow_quota,
                'requested_quota': amount,
                'reason': 'capacity_available',
            },
        )
        self.send_message(msg)

    def negotiate_with_peer(
        self,
        peer_id: str,
        resource: str,
        request: str
    ) -> None:
        """与对等区域协商."""
        msg = AgentMessage(
            receiver_id=self.parent_id or "",  # 通过协调者
            msg_type=MessageType.NEGOTIATION,
            payload={
                'negotiation_type': 'resource_sharing',
                'parties': [self.agent_id, peer_id],
                'resource': resource,
                'request': request,
            },
        )
        self.send_message(msg)
