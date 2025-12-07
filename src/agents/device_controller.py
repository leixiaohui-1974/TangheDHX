# -*- coding: utf-8 -*-
"""
Device Controller Agent.

设备控制层智能体，负责：
- 单一设备控制
- 实时反馈控制
- 故障检测与上报
- 状态监控
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

from .base import (
    BaseAgent, AgentState, AgentRole, AgentMessage,
    MessageType, Priority
)

logger = logging.getLogger(__name__)


@dataclass
class ControlState:
    """控制状态."""
    setpoint: float = 0.0
    measured: float = 0.0
    error: float = 0.0
    output: float = 0.0
    last_update: datetime = None

    def __post_init__(self):
        if self.last_update is None:
            self.last_update = datetime.now()


class DeviceController(BaseAgent):
    """
    设备控制器基类.

    职责：
    1. 设备级控制 - 执行底层控制算法
    2. 实时反馈 - 处理传感器反馈
    3. 故障检测 - 检测设备级故障
    4. 状态上报 - 向区域管理者上报状态
    """

    def __init__(
        self,
        agent_id: str,
        device_type: str,
        model: Optional['TangheSiphonModel'] = None,
    ):
        super().__init__(agent_id, AgentRole.DEVICE_CONTROLLER)

        self.device_type = device_type
        self.model = model

        # 控制状态
        self._control_state = ControlState()

        # PID参数
        self._kp = 1.0
        self._ki = 0.1
        self._kd = 0.05
        self._integral = 0.0
        self._last_error = 0.0

        # 约束
        self._output_min = 0.0
        self._output_max = 5.0
        self._rate_limit = 0.5  # 最大变化率

        # 故障检测
        self._fault_thresholds: Dict[str, float] = {}
        self._fault_history: List[Dict] = []

        # 注册消息处理器
        self._register_device_handlers()

    def _register_device_handlers(self) -> None:
        """注册设备控制器消息处理器."""
        self.register_handler(MessageType.COMMAND, self._handle_command)
        self.register_handler(MessageType.SETPOINT, self._handle_setpoint)

    # -------------------------------------------------------------------------
    # Core BDI Methods
    # -------------------------------------------------------------------------

    def perceive(self) -> Dict[str, Any]:
        """感知设备状态."""
        return {
            'measured': self._control_state.measured,
            'setpoint': self._control_state.setpoint,
            'output': self._control_state.output,
            'error': self._control_state.error,
        }

    def decide(self) -> List[Dict[str, Any]]:
        """控制决策."""
        decisions = []

        # 计算控制输出
        output = self._compute_control()

        decisions.append({
            'action': 'apply_control',
            'output': output,
        })

        # 故障检测
        fault = self._detect_fault()
        if fault:
            decisions.append({
                'action': 'report_fault',
                'fault': fault,
            })

        return decisions

    def act(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行控制动作."""
        results = []

        for decision in decisions:
            action = decision.get('action')

            if action == 'apply_control':
                result = self._apply_control(decision['output'])
            elif action == 'report_fault':
                result = self._report_fault(decision['fault'])
            else:
                result = {'status': 'unknown_action'}

            results.append(result)

        return results

    # -------------------------------------------------------------------------
    # Control Algorithm
    # -------------------------------------------------------------------------

    def _compute_control(self) -> float:
        """计算PID控制输出."""
        error = self._control_state.setpoint - self._control_state.measured
        self._control_state.error = error

        # P项
        p_term = self._kp * error

        # I项 (带抗饱和)
        self._integral += error
        self._integral = np.clip(
            self._integral,
            -self._output_max / self._ki,
            self._output_max / self._ki
        )
        i_term = self._ki * self._integral

        # D项
        d_term = self._kd * (error - self._last_error)
        self._last_error = error

        # 总输出
        output = p_term + i_term + d_term

        # 输出限幅
        output = np.clip(output, self._output_min, self._output_max)

        # 速率限制
        current_output = self._control_state.output
        delta = output - current_output
        if abs(delta) > self._rate_limit:
            output = current_output + np.sign(delta) * self._rate_limit

        return output

    def _apply_control(self, output: float) -> Dict[str, Any]:
        """应用控制输出."""
        self._control_state.output = output
        self._control_state.last_update = datetime.now()

        return {
            'status': 'ok',
            'output': output,
        }

    # -------------------------------------------------------------------------
    # Fault Detection
    # -------------------------------------------------------------------------

    def _detect_fault(self) -> Optional[Dict[str, Any]]:
        """检测故障."""
        # 检测控制误差过大
        if abs(self._control_state.error) > self._fault_thresholds.get('error', 10.0):
            return {
                'type': 'control_error',
                'value': self._control_state.error,
                'threshold': self._fault_thresholds.get('error', 10.0),
            }

        # 检测输出饱和
        if (self._control_state.output >= self._output_max or
            self._control_state.output <= self._output_min):
            saturation_time = self._fault_thresholds.get('saturation_time', 10.0)
            # 简化：直接检测饱和
            return {
                'type': 'output_saturation',
                'value': self._control_state.output,
                'limit': self._output_max if self._control_state.output >= self._output_max else self._output_min,
            }

        return None

    def _report_fault(self, fault: Dict[str, Any]) -> Dict[str, Any]:
        """上报故障."""
        self._fault_history.append({
            **fault,
            'timestamp': datetime.now().isoformat(),
        })

        msg = AgentMessage(
            receiver_id=self.parent_id or "",
            msg_type=MessageType.DIAGNOSTIC,
            priority=Priority.HIGH,
            payload={
                'device_id': self.agent_id,
                'device_type': self.device_type,
                'fault_detected': True,
                'fault_type': fault.get('type'),
                'details': fault,
            },
        )
        self.send_message(msg)

        return {'status': 'ok', 'fault_reported': fault}

    # -------------------------------------------------------------------------
    # Message Handlers
    # -------------------------------------------------------------------------

    def _handle_command(self, msg: AgentMessage) -> None:
        """处理命令消息."""
        task = msg.payload.get('task')

        if task == 'maintain_flow':
            # 设置流量目标
            target_flow = msg.payload.get('target_flow', 0.0)
            self._control_state.setpoint = target_flow

        elif task == 'stop':
            self._control_state.setpoint = 0.0

    def _handle_setpoint(self, msg: AgentMessage) -> None:
        """处理设定值消息."""
        setpoint = msg.payload.get('setpoint')
        if setpoint is not None:
            self._control_state.setpoint = setpoint

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def set_pid_gains(self, kp: float, ki: float, kd: float) -> None:
        """设置PID参数."""
        self._kp = kp
        self._ki = ki
        self._kd = kd

    def set_output_limits(self, min_val: float, max_val: float) -> None:
        """设置输出限制."""
        self._output_min = min_val
        self._output_max = max_val

    def set_rate_limit(self, rate: float) -> None:
        """设置速率限制."""
        self._rate_limit = rate

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """获取设备状态."""
        base_status = super().get_status()
        base_status.update({
            'device_type': self.device_type,
            'control_state': {
                'setpoint': self._control_state.setpoint,
                'measured': self._control_state.measured,
                'error': self._control_state.error,
                'output': self._control_state.output,
            },
            'pid_gains': {
                'kp': self._kp,
                'ki': self._ki,
                'kd': self._kd,
            },
            'faults_count': len(self._fault_history),
        })
        return base_status


class GateAgent(DeviceController):
    """
    闸门控制智能体.

    专门用于控制单个闸门的智能体。
    """

    def __init__(
        self,
        agent_id: str,
        gate_index: int,
        model: Optional['TangheSiphonModel'] = None,
    ):
        super().__init__(agent_id, device_type='gate', model=model)

        self.gate_index = gate_index

        # 闸门特有参数
        self._max_opening = 5.0  # 最大开度 [m]
        self._min_opening = 0.0
        self._max_velocity = 0.1  # 最大开闭速度 [m/s]

        # 振动监控
        self._vibration_limit = 0.5  # g
        self._resonance_avoidance = True

        # 设置故障阈值
        self._fault_thresholds = {
            'error': 5.0,  # 流量误差阈值
            'vibration': 0.7,  # 振动阈值
            'position_error': 0.1,  # 位置误差阈值
        }

    def perceive(self) -> Dict[str, Any]:
        """感知闸门状态."""
        perceptions = super().perceive()

        if self.model:
            # 闸门开度
            opening = self.model.gate_openings[self.gate_index]
            perceptions['opening'] = opening

            # 流量
            flow = self.model.flow_rates[self.gate_index]
            perceptions['flow'] = flow
            self._control_state.measured = flow

            # 振动
            vibration = self.model.vibration_accel[self.gate_index]
            perceptions['vibration'] = vibration

            # 故障状态
            perceptions['stuck'] = self.model.gate_stuck[self.gate_index]

        return perceptions

    def decide(self) -> List[Dict[str, Any]]:
        """闸门控制决策."""
        decisions = []

        # 基本控制决策
        base_decisions = super().decide()
        decisions.extend(base_decisions)

        # 振动检查
        vibration = self.get_belief('vibration', 0.0)
        if vibration > self._vibration_limit:
            decisions.append({
                'action': 'reduce_vibration',
                'current_vibration': vibration,
            })

        # 共振规避
        if self._resonance_avoidance:
            resonance_action = self._check_resonance()
            if resonance_action:
                decisions.append(resonance_action)

        return decisions

    def act(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行闸门动作."""
        results = []

        for decision in decisions:
            action = decision.get('action')

            if action == 'apply_control':
                result = self._apply_gate_control(decision['output'])
            elif action == 'reduce_vibration':
                result = self._reduce_vibration()
            elif action == 'avoid_resonance':
                result = self._avoid_resonance(decision)
            elif action == 'report_fault':
                result = self._report_fault(decision['fault'])
            else:
                result = {'status': 'unknown_action'}

            results.append(result)

        return results

    def _apply_gate_control(self, target_flow: float) -> Dict[str, Any]:
        """应用闸门控制."""
        if not self.model:
            return {'status': 'error', 'reason': 'no_model'}

        # 流量到开度的转换 (简化)
        # 实际应该使用流量方程求解
        current_opening = self.model.gate_openings[self.gate_index]
        current_flow = self._control_state.measured

        if current_flow > 0.1:
            # 估算需要的开度
            ratio = target_flow / current_flow
            target_opening = current_opening * ratio
        else:
            # 没有流量时，根据目标流量估算
            target_opening = target_flow / 50.0  # 简化估算

        # 限制开度范围
        target_opening = np.clip(target_opening, self._min_opening, self._max_opening)

        # 限制变化速率
        delta = target_opening - current_opening
        max_delta = self._max_velocity * 0.1  # 假设0.1秒更新周期
        if abs(delta) > max_delta:
            target_opening = current_opening + np.sign(delta) * max_delta

        # 应用到模型
        self.model.gate_openings[self.gate_index] = target_opening
        self._control_state.output = target_opening

        return {
            'status': 'ok',
            'gate': self.gate_index,
            'opening': target_opening,
        }

    def _reduce_vibration(self) -> Dict[str, Any]:
        """减少振动."""
        if not self.model:
            return {'status': 'error'}

        current_opening = self.model.gate_openings[self.gate_index]

        # 略微调整开度以改变流态
        adjustment = 0.05 * (1 if np.random.random() > 0.5 else -1)
        new_opening = np.clip(
            current_opening + adjustment,
            self._min_opening,
            self._max_opening
        )

        self.model.gate_openings[self.gate_index] = new_opening

        return {
            'status': 'ok',
            'action': 'vibration_reduction',
            'old_opening': current_opening,
            'new_opening': new_opening,
        }

    def _check_resonance(self) -> Optional[Dict[str, Any]]:
        """检查共振风险."""
        if not self.model:
            return None

        flow = self.model.flow_rates[self.gate_index]
        vibration = self.model.vibration_accel[self.gate_index]

        # 简化的共振检测
        resonance_flow_range = (70, 100)  # 共振流量范围

        if resonance_flow_range[0] <= flow <= resonance_flow_range[1]:
            if vibration > 0.2:  # 有振动迹象
                return {
                    'action': 'avoid_resonance',
                    'current_flow': flow,
                    'target_flow': resonance_flow_range[1] + 10 if flow > 85 else resonance_flow_range[0] - 10,
                }

        return None

    def _avoid_resonance(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """避开共振."""
        target_flow = decision.get('target_flow')

        # 修改设定值以避开共振区
        self._control_state.setpoint = target_flow

        # 上报给区域管理者
        msg = AgentMessage(
            receiver_id=self.parent_id or "",
            msg_type=MessageType.ALERT,
            payload={
                'alert_type': 'resonance_avoidance',
                'gate': self.gate_index,
                'old_target': decision.get('current_flow'),
                'new_target': target_flow,
            },
        )
        self.send_message(msg)

        return {
            'status': 'ok',
            'action': 'resonance_avoidance',
            'new_target': target_flow,
        }

    def _detect_fault(self) -> Optional[Dict[str, Any]]:
        """检测闸门故障."""
        # 基类故障检测
        fault = super()._detect_fault()
        if fault:
            return fault

        if self.model:
            # 检测卡住
            if self.model.gate_stuck[self.gate_index]:
                return {
                    'type': 'gate_stuck',
                    'gate': self.gate_index,
                }

            # 检测振动过高
            vibration = self.model.vibration_accel[self.gate_index]
            if vibration > self._fault_thresholds.get('vibration', 0.7):
                return {
                    'type': 'high_vibration',
                    'gate': self.gate_index,
                    'value': vibration,
                    'threshold': self._fault_thresholds['vibration'],
                }

        return None

    def get_status(self) -> Dict[str, Any]:
        """获取闸门状态."""
        status = super().get_status()
        status.update({
            'gate_index': self.gate_index,
            'opening': self.get_belief('opening', 0.0),
            'flow': self.get_belief('flow', 0.0),
            'vibration': self.get_belief('vibration', 0.0),
            'stuck': self.get_belief('stuck', False),
        })
        return status
