# -*- coding: utf-8 -*-
"""
Tanghe Inverted Siphon Web Application.

This module provides the Flask web application for monitoring and
controlling the digital twin simulation with full scenario-aware control.

Features:
- Real-time simulation monitoring
- Hierarchical agent system control
- Time-series data storage and analysis
- Anomaly detection and alerting
- History replay functionality
"""

import logging
import threading
import time
import numpy as np
from typing import Any, Dict, Optional, Tuple, List

from flask import Flask, jsonify, render_template, request

from src.config import get_config
from src.simulation.physics import TangheSiphonModel
from src.simulation.sensors import ADCPSensor, VibrationSensor
from src.simulation.actuators import GateController
from src.simulation.sensors_extended import SensorNetwork
from src.simulation.actuators_extended import ActuatorNetwork as ExtActuatorNetwork
from src.control.mpc import SpectralMPC
from src.control.local import LocalController
from src.control.manager import ScenarioManager
from src.control.integrated_controller import IntegratedController, ScenarioType
from src.control.scenario_advanced import AdvancedScenarioManager
from src.control.model_calibration import IDZModelCalibrator
from src.control.state_evaluation import RealTimeStateEvaluator
from src.control.state_prediction import RealTimeStatePredictor
from src.agents.communication import AgentNetwork
from src.data.storage import TimeSeriesStorage, DataPoint
from src.data.analysis import DataAnalyzer
from src.data.anomaly import AnomalyDetector, ThresholdRule, RateRule, Severity
from src.data.replay import HistoryReplay, ReplayMode
from src.data.governance import DataGovernanceEngine
from src.data.assimilation import DataAssimilationEngine
from src.maintenance.predictive_maintenance import (
    PredictiveMaintenanceSystem,
    EquipmentInfo,
    WeibullDegradation,
    ExponentialDegradation
)
from src.analytics.reporting import (
    AnalyticsEngine,
    ReportType,
    ReportFormat,
    AlarmSeverity,
    AlarmCategory
)
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimulationState:
    """
    Thread-safe container for simulation state.

    Uses a lock to protect all mutable state from race conditions.
    Now includes integrated controller with scenario-aware adaptation.
    """

    def __init__(self, use_integrated: bool = True) -> None:
        """
        Initialize simulation state with thread lock.

        Args:
            use_integrated: Use integrated controller with scenario adaptation
        """
        self._lock = threading.RLock()
        self._running = True
        self._target_flow = get_config().simulation.default_target_flow
        self._use_integrated = use_integrated

        # Initialize physics model
        self.model = TangheSiphonModel()

        # Sensors
        self.sensors = {
            'adcp': [ADCPSensor(self.model, i) for i in range(3)],
            'vib': [VibrationSensor(self.model, i) for i in range(3)]
        }
        self.actuator = GateController(self.model)

        # Control systems
        if use_integrated:
            # Use new integrated controller with scenario adaptation
            self.integrated = IntegratedController(self.model)
            self.integrated.set_target_flow(self._target_flow)
            logger.info("Using IntegratedController with scenario adaptation")
        else:
            # Legacy control path
            self.mpc = SpectralMPC(self.model)
            self.local_ctrl = LocalController(self.model)
            self.scenario_mgr = ScenarioManager(self.model)
            logger.info("Using legacy control system")

        # Diagnostics storage
        self._last_diagnostics: Dict[str, Any] = {}

        # Agent network for hierarchical control
        self.agent_network = AgentNetwork(self.model)
        self.agent_network.create_standard_network()

        # Data storage for time-series
        self.data_storage = TimeSeriesStorage()
        self._init_data_series()

        # Data analyzer
        self.data_analyzer = DataAnalyzer()

        # Anomaly detector
        self.anomaly_detector = AnomalyDetector()
        self._init_anomaly_rules()

        # History replay
        self.replay = HistoryReplay(self.data_storage)

        # =====================================================================
        # New Extended Features
        # =====================================================================

        # Extended sensor network
        self.sensor_network = SensorNetwork(self.model)

        # Extended actuator network
        self.ext_actuator_network = ExtActuatorNetwork(self.model)

        # Data governance engine
        self.data_governance = DataGovernanceEngine()

        # Data assimilation engine
        self.data_assimilation = DataAssimilationEngine(state_dim=12)

        # Model calibration (IDZ parameter update)
        self.model_calibrator = IDZModelCalibrator()

        # Real-time state evaluation
        self.state_evaluator = RealTimeStateEvaluator()

        # Real-time state prediction
        self.state_predictor = RealTimeStatePredictor(self.model)

        # Predictive maintenance system
        self.maintenance_system = PredictiveMaintenanceSystem(
            failure_threshold=30.0,
            warning_threshold=50.0,
            planning_horizon_days=30
        )
        self._init_maintenance_equipment()

        # Analytics and reporting engine
        self.analytics = AnalyticsEngine(history_hours=720)  # 30 days history

        logger.info("SimulationState initialized with full feature set including extended modules")

    def _init_maintenance_equipment(self) -> None:
        """Initialize equipment for predictive maintenance monitoring."""
        installation_date = datetime(2020, 1, 1)

        # Register gates
        for i in range(3):
            gate_info = EquipmentInfo(
                equipment_id=f"gate_{i}",
                equipment_type="gate",
                installation_date=installation_date,
                manufacturer="Tanghe Engineering",
                model="TG-600",
                expected_lifetime_hours=50000.0,
                maintenance_interval_hours=2000.0,
                replacement_cost=15000.0,
                downtime_cost_per_hour=800.0
            )
            self.maintenance_system.register_equipment(
                gate_info,
                WeibullDegradation(shape=2.5, scale=50000.0)
            )

        # Register sensors
        sensor_types = ['flow', 'pressure', 'velocity', 'water_level']
        for i, sensor_type in enumerate(sensor_types):
            sensor_info = EquipmentInfo(
                equipment_id=f"sensor_{sensor_type}",
                equipment_type="sensor",
                installation_date=installation_date,
                manufacturer="Tanghe Instruments",
                model=f"TS-{sensor_type.upper()}",
                expected_lifetime_hours=30000.0,
                maintenance_interval_hours=1000.0,
                replacement_cost=2000.0,
                downtime_cost_per_hour=200.0
            )
            self.maintenance_system.register_equipment(
                sensor_info,
                ExponentialDegradation(failure_rate=0.00003, degradation_rate=0.001)
            )

    def _init_data_series(self) -> None:
        """Initialize data series for storage."""
        # Flow and hydraulic series
        self.data_storage.create_series('total_flow', 'm³/s')
        self.data_storage.create_series('head_upstream', 'm')
        self.data_storage.create_series('head_downstream', 'm')
        self.data_storage.create_series('head_diff', 'm')

        # Gate series
        for i in range(3):
            self.data_storage.create_series(f'gate_{i}_opening', '%')
            self.data_storage.create_series(f'gate_{i}_flow', 'm³/s')
            self.data_storage.create_series(f'gate_{i}_velocity', 'm/s')
            self.data_storage.create_series(f'gate_{i}_vibration', 'mm/s')

        # Control series
        self.data_storage.create_series('target_flow', 'm³/s')
        self.data_storage.create_series('flow_error', 'm³/s')

    def _init_anomaly_rules(self) -> None:
        """Initialize anomaly detection rules."""
        # Flow anomalies
        self.anomaly_detector.add_threshold_rule(ThresholdRule(
            name='flow_high',
            series_name='total_flow',
            low=0,
            high=450,
        ))

        # Head difference anomalies
        self.anomaly_detector.add_threshold_rule(ThresholdRule(
            name='head_diff_extreme',
            series_name='head_diff',
            low=-5,
            high=10,
        ))

        # Vibration anomalies for each gate
        for i in range(3):
            self.anomaly_detector.add_threshold_rule(ThresholdRule(
                name=f'vib_gate_{i}',
                series_name=f'gate_{i}_vibration',
                low=0,
                high=50,
                high_high=80,
            ))

            # Rate of change rule
            self.anomaly_detector.add_rate_rule(RateRule(
                name=f'vib_rate_gate_{i}',
                series_name=f'gate_{i}_vibration',
                max_rate=20,
                window=5.0,
            ))

    @property
    def running(self) -> bool:
        """Get running state (thread-safe)."""
        with self._lock:
            return self._running

    @running.setter
    def running(self, value: bool) -> None:
        """Set running state (thread-safe)."""
        with self._lock:
            self._running = value
            logger.info("Simulation %s", "started" if value else "paused")

    @property
    def target_flow(self) -> float:
        """Get target flow (thread-safe)."""
        with self._lock:
            return self._target_flow

    @target_flow.setter
    def target_flow(self, value: float) -> None:
        """Set target flow with validation (thread-safe)."""
        with self._lock:
            if value < 0:
                raise ValueError("Target flow must be non-negative")
            if value > 500:  # Reasonable maximum
                raise ValueError("Target flow exceeds maximum (500 m³/s)")
            self._target_flow = value
            if self._use_integrated:
                self.integrated.set_target_flow(value)
            logger.info("Target flow set to %.2f m³/s", value)

    def get_state(self) -> Dict[str, Any]:
        """Get current system state (thread-safe)."""
        with self._lock:
            state = self.model.get_state()
            state['target_flow'] = self._target_flow
            state['simulation_running'] = self._running

            if self._use_integrated:
                state['active_scenario'] = self.integrated.get_current_scenario().name
                state['scenario_config'] = {
                    'description': self.integrated.get_current_config().description,
                    'target_modifier': self.integrated.get_current_config().target_flow_modifier,
                    'use_pid': self.integrated.get_current_config().use_pid_backup,
                }
            else:
                state['active_scenario'] = self.scenario_mgr.active_scenario

            return state

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostics (thread-safe)."""
        with self._lock:
            return self._last_diagnostics.copy()

    def get_history(self, n: int = 100) -> List[Dict[str, Any]]:
        """Get control history (thread-safe)."""
        with self._lock:
            if self._use_integrated:
                return self.integrated.get_history(n)
            return []

    def set_scenario(self, scenario_id: str) -> None:
        """Set active scenario (thread-safe)."""
        with self._lock:
            if self._use_integrated:
                # Find scenario type by name
                try:
                    scenario_type = ScenarioType[scenario_id]
                    self.integrated.force_scenario(scenario_type)
                except KeyError:
                    # Try advanced scenario manager
                    self.integrated._scenario_mgr.set_scenario(scenario_id)
            else:
                self.scenario_mgr.set_scenario(scenario_id)

    def step(self, dt: float) -> None:
        """Execute one simulation step (thread-safe)."""
        with self._lock:
            if self._use_integrated:
                # Integrated controller handles everything
                self._last_diagnostics = self.integrated.update(dt)

                # Get outputs and apply to actuators
                outputs = self._last_diagnostics['control']['outputs']
                for i, cmd in enumerate(outputs):
                    self.actuator.set_opening(i, cmd)

                # Physics step
                self.model.step(self.actuator.get_target_openings(), dt)
            else:
                # Legacy control path
                self.scenario_mgr.update()

                current_openings = self.model.gate_openings
                head_diff = self.model.head_upstream - self.model.head_downstream

                mpc_targets = self.mpc.get_target_openings(
                    self._target_flow, current_openings, head_diff
                )
                final_cmds = self.local_ctrl.update(mpc_targets, dt)

                for i in range(3):
                    self.actuator.set_opening(i, final_cmds[i])

                self.model.step(self.actuator.get_target_openings(), dt)

            # Record data to storage
            self._record_data()

            # Check for anomalies
            self._check_anomalies()

    def _record_data(self) -> None:
        """Record current state to data storage."""
        ts = time.time()
        state = self.model.get_state()

        # Record flow and hydraulic data
        self.data_storage.write('total_flow', ts, state['total_flow'])
        self.data_storage.write('head_upstream', ts, state['head_upstream'])
        self.data_storage.write('head_downstream', ts, state['head_downstream'])
        self.data_storage.write('head_diff', ts,
                                state['head_upstream'] - state['head_downstream'])

        # Record gate data
        for i in range(3):
            self.data_storage.write(f'gate_{i}_opening', ts,
                                    state['gate_openings'][i] * 100)
            self.data_storage.write(f'gate_{i}_flow', ts,
                                    state['gate_flows'][i])
            self.data_storage.write(f'gate_{i}_velocity', ts,
                                    state['velocities'][i])
            self.data_storage.write(f'gate_{i}_vibration', ts,
                                    state['vibrations'][i])

        # Record control data
        self.data_storage.write('target_flow', ts, self._target_flow)
        self.data_storage.write('flow_error', ts,
                                abs(state['total_flow'] - self._target_flow))

    def _check_anomalies(self) -> None:
        """Check for anomalies in current data."""
        anomalies = self.anomaly_detector.check_all()
        if anomalies:
            for anomaly in anomalies:
                logger.warning(f"Anomaly detected: {anomaly.message}")

    def reset(self) -> None:
        """Reset simulation to initial state (thread-safe)."""
        with self._lock:
            self.model.reset()
            if self._use_integrated:
                self.integrated.reset()
            else:
                self.mpc.reset()
                self.local_ctrl.reset()
                self.scenario_mgr.reset()
            self._last_diagnostics = {}

            # Reset agent network
            self.agent_network.reset()

            # Clear anomaly history
            self.anomaly_detector.clear_history()

            logger.info("Simulation reset to initial state")


# Global simulation state
sim_state: Optional[SimulationState] = None


def create_app() -> Flask:
    """
    Create and configure the Flask application.

    Returns:
        Configured Flask application instance.
    """
    app = Flask(
        __name__,
        template_folder='src/web/templates',
        static_folder='src/web/static'
    )

    # Error handlers
    @app.errorhandler(400)
    def bad_request(error: Exception) -> Tuple[Dict[str, Any], int]:
        logger.warning("Bad request: %s", error)
        return jsonify({'error': 'Bad request', 'message': str(error)}), 400

    @app.errorhandler(404)
    def not_found(error: Exception) -> Tuple[Dict[str, Any], int]:
        return jsonify({'error': 'Not found'}), 404

    @app.errorhandler(500)
    def internal_error(error: Exception) -> Tuple[Dict[str, Any], int]:
        logger.error("Internal error: %s", error, exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

    # Routes
    @app.route('/')
    def index():
        """Render main dashboard."""
        return render_template('index.html')

    @app.route('/api/state')
    def get_state():
        """Get current simulation state."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500
        try:
            return jsonify(sim_state.get_state())
        except Exception as e:
            logger.error("Error getting state: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/diagnostics')
    def get_diagnostics():
        """Get detailed control diagnostics."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500
        try:
            return jsonify(sim_state.get_diagnostics())
        except Exception as e:
            logger.error("Error getting diagnostics: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/history')
    def get_history():
        """Get control history."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500
        try:
            n = request.args.get('n', 100, type=int)
            return jsonify({'history': sim_state.get_history(n)})
        except Exception as e:
            logger.error("Error getting history: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/control', methods=['POST'])
    def set_control():
        """Set control parameters."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            data = request.json
            if data is None:
                return jsonify({'error': 'No JSON data provided'}), 400

            if 'target_flow' in data:
                try:
                    flow = float(data['target_flow'])
                    sim_state.target_flow = flow
                except ValueError as e:
                    return jsonify({'error': str(e)}), 400

            if 'scenario' in data:
                try:
                    sim_state.set_scenario(data['scenario'])
                except ValueError as e:
                    return jsonify({'error': str(e)}), 400

            if 'running' in data:
                sim_state.running = bool(data['running'])

            return jsonify({'status': 'ok', 'state': sim_state.get_state()})

        except Exception as e:
            logger.error("Error in set_control: %s", e, exc_info=True)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/reset', methods=['POST'])
    def reset_simulation():
        """Reset simulation to initial state."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500
        try:
            sim_state.reset()
            return jsonify({'status': 'ok', 'message': 'Simulation reset'})
        except Exception as e:
            logger.error("Error resetting simulation: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/scenarios')
    def get_scenarios():
        """Get available scenarios with configurations."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            if sim_state._use_integrated:
                return jsonify({
                    'scenarios': sim_state.integrated.get_scenario_configs(),
                    'current': sim_state.integrated.get_current_scenario().name,
                })
            else:
                return jsonify({
                    'scenarios': ScenarioManager.SCENARIOS
                })
        except Exception as e:
            logger.error("Error getting scenarios: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/config')
    def get_app_config():
        """Get current configuration (read-only view)."""
        cfg = get_config()
        return jsonify({
            'physics': {
                'num_gates': cfg.physics.num_gates,
                'gate_width': cfg.physics.gate_width,
                'structural_frequency': cfg.physics.structural_frequency,
            },
            'control': {
                'alpha': cfg.control.alpha_flow_tracking,
                'beta': cfg.control.beta_action_penalty,
                'gamma': cfg.control.gamma_spectral_avoidance,
            },
            'simulation': {
                'dt': cfg.simulation.dt,
                'default_target_flow': cfg.simulation.default_target_flow,
            }
        })

    @app.route('/api/mpc_config')
    def get_mpc_config():
        """Get current MPC configuration."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            diag = sim_state.get_diagnostics()
            if 'mpc_config' in diag:
                return jsonify({
                    'mpc': diag['mpc_config'],
                    'constraints': diag.get('constraints', {}),
                    'adaptation': diag.get('adaptation', {}),
                })
            return jsonify({'error': 'MPC config not available'}), 404
        except Exception as e:
            logger.error("Error getting MPC config: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/inject_fault', methods=['POST'])
    def inject_fault():
        """Inject fault for testing."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            data = request.json
            if data is None:
                return jsonify({'error': 'No JSON data provided'}), 400

            gate = data.get('gate', 0)
            fault_type = data.get('type', 'stuck')

            sim_state.model.inject_fault(gate, fault_type)
            return jsonify({
                'status': 'ok',
                'message': f'Fault {fault_type} injected on gate {gate}'
            })
        except Exception as e:
            logger.error("Error injecting fault: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/performance')
    def get_performance():
        """Get performance metrics."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            history = sim_state.get_history(100)
            if not history:
                return jsonify({'error': 'No history available'}), 404

            # Calculate metrics
            flow_errors = [h['performance']['flow_tracking_error'] for h in history]
            max_vibs = [h['performance']['max_vibration'] for h in history]
            in_resonance = [h['performance']['in_resonance_band'] for h in history]

            return jsonify({
                'flow_tracking': {
                    'mean_error': sum(flow_errors) / len(flow_errors),
                    'max_error': max(flow_errors),
                    'min_error': min(flow_errors),
                },
                'vibration': {
                    'mean_max': sum(max_vibs) / len(max_vibs),
                    'peak': max(max_vibs),
                },
                'resonance': {
                    'time_in_band_pct': 100.0 * sum(in_resonance) / len(in_resonance),
                },
                'sample_count': len(history),
            })
        except Exception as e:
            logger.error("Error getting performance: %s", e)
            return jsonify({'error': str(e)}), 500

    # =========================================================================
    # Agent Network API Endpoints
    # =========================================================================

    @app.route('/api/agents')
    def get_agents():
        """Get all agents status."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            return jsonify({
                'network': sim_state.agent_network.get_network_status(),
                'agents': sim_state.agent_network.get_all_agent_status(),
            })
        except Exception as e:
            logger.error("Error getting agents: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/agents/hierarchy')
    def get_agent_hierarchy():
        """Get agent hierarchy tree."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            return jsonify({
                'hierarchy': sim_state.agent_network.get_hierarchy_tree(),
            })
        except Exception as e:
            logger.error("Error getting agent hierarchy: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/agents/control', methods=['POST'])
    def control_agents():
        """Control agent network."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            data = request.json
            if data is None:
                return jsonify({'error': 'No JSON data provided'}), 400

            action = data.get('action', '')

            if action == 'start':
                sim_state.agent_network.start()
            elif action == 'stop':
                sim_state.agent_network.stop()
            elif action == 'reset':
                sim_state.agent_network.reset()
            elif action == 'set_target':
                target = float(data.get('target_flow', 150))
                sim_state.agent_network.set_global_target_flow(target)
            elif action == 'emergency':
                reason = data.get('reason', 'Manual emergency')
                sim_state.agent_network.broadcast_emergency(reason)
            else:
                return jsonify({'error': f'Unknown action: {action}'}), 400

            return jsonify({
                'status': 'ok',
                'action': action,
                'network': sim_state.agent_network.get_network_status(),
            })
        except Exception as e:
            logger.error("Error controlling agents: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/agents/<agent_id>')
    def get_agent_detail(agent_id: str):
        """Get specific agent details."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            agent = sim_state.agent_network.get_agent(agent_id)
            if agent is None:
                return jsonify({'error': f'Agent {agent_id} not found'}), 404

            return jsonify({
                'status': agent.get_status(),
                'diagnostics': agent.get_diagnostics(),
            })
        except Exception as e:
            logger.error("Error getting agent detail: %s", e)
            return jsonify({'error': str(e)}), 500

    # =========================================================================
    # Data Storage API Endpoints
    # =========================================================================

    @app.route('/api/data/series')
    def get_data_series():
        """Get list of available data series."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            series_list = sim_state.data_storage.list_series()
            series_info = {}
            for name in series_list:
                info = sim_state.data_storage.get_series_info(name)
                if info:
                    series_info[name] = info

            return jsonify({
                'series': series_list,
                'info': series_info,
            })
        except Exception as e:
            logger.error("Error getting data series: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/data/query')
    def query_data():
        """Query time-series data."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            series_name = request.args.get('series', '')
            start_time = request.args.get('start', type=float)
            end_time = request.args.get('end', type=float)
            limit = request.args.get('limit', 1000, type=int)

            if not series_name:
                return jsonify({'error': 'Series name required'}), 400

            # Default to last hour if no time range specified
            if end_time is None:
                end_time = time.time()
            if start_time is None:
                start_time = end_time - 3600

            points = sim_state.data_storage.read(
                series_name, start_time, end_time, limit
            )

            return jsonify({
                'series': series_name,
                'start': start_time,
                'end': end_time,
                'count': len(points),
                'data': [
                    {'timestamp': p.timestamp, 'value': p.value, 'quality': p.quality}
                    for p in points
                ],
            })
        except Exception as e:
            logger.error("Error querying data: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/data/latest')
    def get_latest_data():
        """Get latest data point for each series."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            series_names = request.args.getlist('series')
            if not series_names:
                series_names = sim_state.data_storage.list_series()

            latest = {}
            for name in series_names:
                point = sim_state.data_storage.read_latest(name)
                if point:
                    latest[name] = {
                        'timestamp': point.timestamp,
                        'value': point.value,
                        'quality': point.quality,
                    }

            return jsonify({'latest': latest})
        except Exception as e:
            logger.error("Error getting latest data: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/data/analysis')
    def analyze_data():
        """Analyze data series."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            series_name = request.args.get('series', '')
            analysis_type = request.args.get('type', 'stats')
            window = request.args.get('window', 300, type=float)

            if not series_name:
                return jsonify({'error': 'Series name required'}), 400

            end_time = time.time()
            start_time = end_time - window

            result = {}
            if analysis_type == 'stats':
                result = sim_state.data_analyzer.basic_stats(
                    series_name, start_time, end_time
                )
            elif analysis_type == 'trend':
                result = sim_state.data_analyzer.analyze_trend(
                    series_name, start_time, end_time
                )
            elif analysis_type == 'outliers':
                outliers = sim_state.data_analyzer.outlier_detection(
                    series_name, start_time, end_time
                )
                result = {
                    'count': len(outliers),
                    'outliers': [
                        {'timestamp': p.timestamp, 'value': p.value}
                        for p in outliers[:100]  # Limit output
                    ]
                }
            else:
                return jsonify({'error': f'Unknown analysis type: {analysis_type}'}), 400

            return jsonify({
                'series': series_name,
                'type': analysis_type,
                'window': window,
                'result': result,
            })
        except Exception as e:
            logger.error("Error analyzing data: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/data/export', methods=['POST'])
    def export_data():
        """Export data to JSON."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            data = request.json or {}
            series_names = data.get('series', None)
            start_time = data.get('start', None)
            end_time = data.get('end', None)

            export_data = sim_state.data_storage.export_json(
                series_names, start_time, end_time
            )

            return jsonify(export_data)
        except Exception as e:
            logger.error("Error exporting data: %s", e)
            return jsonify({'error': str(e)}), 500

    # =========================================================================
    # Anomaly Detection API Endpoints
    # =========================================================================

    @app.route('/api/anomalies')
    def get_anomalies():
        """Get detected anomalies."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            limit = request.args.get('limit', 100, type=int)
            severity = request.args.get('severity', None)

            anomalies = sim_state.anomaly_detector.get_history(limit)

            if severity:
                try:
                    sev = Severity[severity.upper()]
                    anomalies = [a for a in anomalies if a.severity == sev]
                except KeyError:
                    pass

            return jsonify({
                'count': len(anomalies),
                'anomalies': [
                    {
                        'anomaly_id': a.anomaly_id,
                        'timestamp': a.timestamp,
                        'type': a.anomaly_type.name,
                        'severity': a.severity.name,
                        'series': a.series_name,
                        'value': a.value,
                        'message': a.message,
                        'acknowledged': a.acknowledged,
                    }
                    for a in anomalies
                ],
            })
        except Exception as e:
            logger.error("Error getting anomalies: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/anomalies/rules')
    def get_anomaly_rules():
        """Get configured anomaly rules."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            rules = sim_state.anomaly_detector.get_rules()
            return jsonify({
                'count': len(rules),
                'rules': [
                    {
                        'rule_id': r.rule_id,
                        'series': r.series_name,
                        'type': type(r).__name__,
                        'severity': r.severity.name,
                        'enabled': r.enabled,
                    }
                    for r in rules
                ],
            })
        except Exception as e:
            logger.error("Error getting anomaly rules: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/anomalies/acknowledge', methods=['POST'])
    def acknowledge_anomaly():
        """Acknowledge an anomaly."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            data = request.json
            if data is None:
                return jsonify({'error': 'No JSON data provided'}), 400

            anomaly_id = data.get('anomaly_id', '')
            if not anomaly_id:
                return jsonify({'error': 'Anomaly ID required'}), 400

            success = sim_state.anomaly_detector.acknowledge(anomaly_id)
            if success:
                return jsonify({'status': 'ok', 'anomaly_id': anomaly_id})
            return jsonify({'error': 'Anomaly not found'}), 404
        except Exception as e:
            logger.error("Error acknowledging anomaly: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/anomalies/stats')
    def get_anomaly_stats():
        """Get anomaly statistics."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            stats = sim_state.anomaly_detector.get_statistics()
            return jsonify(stats)
        except Exception as e:
            logger.error("Error getting anomaly stats: %s", e)
            return jsonify({'error': str(e)}), 500

    # =========================================================================
    # History Replay API Endpoints
    # =========================================================================

    @app.route('/api/replay/create', methods=['POST'])
    def create_replay():
        """Create a replay session."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            data = request.json or {}
            start_time = data.get('start', time.time() - 3600)
            end_time = data.get('end', time.time())
            series = data.get('series', None)
            speed = data.get('speed', 1.0)
            mode = data.get('mode', 'REALTIME')

            try:
                replay_mode = ReplayMode[mode.upper()]
            except KeyError:
                replay_mode = ReplayMode.REALTIME

            session = sim_state.replay.create_session(
                start_time=start_time,
                end_time=end_time,
                series_names=series,
                mode=replay_mode,
                speed=speed,
            )

            return jsonify({
                'status': 'ok',
                'session_id': session.session_id,
                'start_time': session.start_time,
                'end_time': session.end_time,
            })
        except Exception as e:
            logger.error("Error creating replay: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/replay/control', methods=['POST'])
    def control_replay():
        """Control replay playback."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            data = request.json
            if data is None:
                return jsonify({'error': 'No JSON data provided'}), 400

            action = data.get('action', '')

            if action == 'play':
                sim_state.replay.play()
            elif action == 'pause':
                sim_state.replay.pause()
            elif action == 'stop':
                sim_state.replay.stop()
            elif action == 'seek':
                timestamp = float(data.get('timestamp', 0))
                sim_state.replay.seek(timestamp)
            elif action == 'speed':
                speed = float(data.get('speed', 1.0))
                sim_state.replay.set_speed(speed)
            else:
                return jsonify({'error': f'Unknown action: {action}'}), 400

            return jsonify({
                'status': 'ok',
                'action': action,
                'summary': sim_state.replay.get_summary(),
            })
        except Exception as e:
            logger.error("Error controlling replay: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/replay/status')
    def get_replay_status():
        """Get replay session status."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            return jsonify(sim_state.replay.get_summary())
        except Exception as e:
            logger.error("Error getting replay status: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/replay/step', methods=['POST'])
    def step_replay():
        """Execute one replay step."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            data = request.json or {}
            dt = data.get('dt', 0.1)

            result = sim_state.replay.step(dt)

            # Convert data points to serializable format
            if 'data' in result:
                serialized_data = {}
                for series_name, points in result['data'].items():
                    serialized_data[series_name] = [
                        {'timestamp': p.timestamp, 'value': p.value}
                        for p in points
                    ]
                result['data'] = serialized_data

            # Convert events
            if 'events' in result:
                result['events'] = [
                    {
                        'timestamp': e.timestamp,
                        'type': e.event_type,
                        'data': e.data,
                    }
                    for e in result['events']
                ]

            return jsonify(result)
        except Exception as e:
            logger.error("Error stepping replay: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/replay/data')
    def get_replay_data():
        """Get data at specific replay time."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            timestamp = request.args.get('timestamp', type=float)
            series = request.args.getlist('series')

            if timestamp is None:
                session = sim_state.replay.get_session()
                if session:
                    timestamp = session.current_time
                else:
                    timestamp = time.time()

            data = sim_state.replay.get_data_at_time(
                timestamp,
                series if series else None
            )

            return jsonify({
                'timestamp': timestamp,
                'data': data,
            })
        except Exception as e:
            logger.error("Error getting replay data: %s", e)
            return jsonify({'error': str(e)}), 500

    # =========================================================================
    # System Info API Endpoint
    # =========================================================================

    @app.route('/api/system')
    def get_system_info():
        """Get comprehensive system information."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            return jsonify({
                'simulation': {
                    'running': sim_state.running,
                    'target_flow': sim_state.target_flow,
                },
                'agents': {
                    'count': len(sim_state.agent_network._agents),
                    'running': sim_state.agent_network._running,
                },
                'data': {
                    'series_count': len(sim_state.data_storage.list_series()),
                    'storage_info': sim_state.data_storage.get_storage_stats(),
                },
                'anomalies': sim_state.anomaly_detector.get_statistics(),
                'replay': sim_state.replay.get_summary(),
            })
        except Exception as e:
            logger.error("Error getting system info: %s", e)
            return jsonify({'error': str(e)}), 500

    # =========================================================================
    # Extended Sensor Network API Endpoints
    # =========================================================================

    @app.route('/api/sensors/extended')
    def get_extended_sensors():
        """Get readings from extended sensor network."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            readings = sim_state.sensor_network.sample_all()
            # Convert SensorReading objects to dicts
            serialized = {'timestamp': readings['timestamp']}

            for key, value in readings.items():
                if key == 'timestamp':
                    continue
                if hasattr(value, 'value'):
                    serialized[key] = {
                        'value': value.value,
                        'quality': value.quality,
                        'status': value.status.value if hasattr(value.status, 'value') else str(value.status),
                        'unit': value.unit
                    }
                elif isinstance(value, list):
                    serialized[key] = [
                        {'value': r.value, 'quality': r.quality, 'unit': r.unit}
                        for r in value if hasattr(r, 'value')
                    ]
                elif isinstance(value, dict):
                    serialized[key] = {
                        k: {'value': v.value, 'quality': v.quality, 'unit': v.unit}
                        for k, v in value.items() if hasattr(v, 'value')
                    }
                else:
                    serialized[key] = value

            return jsonify(serialized)
        except Exception as e:
            logger.error("Error getting extended sensors: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/sensors/health')
    def get_sensor_health():
        """Get sensor health status."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            health = sim_state.sensor_network.get_health_status()
            # Convert health objects to dicts
            serialized = {}
            for key, value in health.items():
                if hasattr(value, 'status'):
                    serialized[key] = {
                        'status': value.status.value,
                        'signal_quality': value.signal_quality,
                        'fault_count': value.fault_count
                    }
                elif isinstance(value, list):
                    serialized[key] = [
                        {'status': h.status.value, 'fault_count': h.fault_count}
                        for h in value if hasattr(h, 'status')
                    ]
            return jsonify({'health': serialized})
        except Exception as e:
            logger.error("Error getting sensor health: %s", e)
            return jsonify({'error': str(e)}), 500

    # =========================================================================
    # Extended Actuator Network API Endpoints
    # =========================================================================

    @app.route('/api/actuators/extended')
    def get_extended_actuators():
        """Get extended actuator status."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            health = sim_state.ext_actuator_network.get_all_health()
            serialized = {
                'emergency_stop_active': health['emergency_stop_active'],
                'gates': []
            }
            for h in health['gates']:
                serialized['gates'].append({
                    'status': h.status.value,
                    'fault_type': h.fault_type.value,
                    'operating_hours': h.operating_hours,
                    'cycle_count': h.cycle_count,
                    'temperature': h.temperature,
                    'power_consumption': h.power_consumption,
                    'wear_level': h.wear_level
                })
            return jsonify(serialized)
        except Exception as e:
            logger.error("Error getting extended actuators: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/actuators/command', methods=['POST'])
    def command_extended_actuators():
        """Command extended actuators."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            data = request.json
            if data is None:
                return jsonify({'error': 'No JSON data provided'}), 400

            import numpy as np

            if 'targets' in data:
                targets = np.array(data['targets'])
                sim_state.ext_actuator_network.command_gates(targets)

            if 'emergency_stop' in data and data['emergency_stop']:
                sim_state.ext_actuator_network.emergency_stop()

            if 'reset_emergency' in data and data['reset_emergency']:
                sim_state.ext_actuator_network.reset_emergency_stop()

            return jsonify({'status': 'ok'})
        except Exception as e:
            logger.error("Error commanding actuators: %s", e)
            return jsonify({'error': str(e)}), 500

    # =========================================================================
    # Data Governance API Endpoints
    # =========================================================================

    @app.route('/api/governance/process', methods=['POST'])
    def process_data_governance():
        """Process data through governance pipeline."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            data = request.json
            if data is None:
                return jsonify({'error': 'No JSON data provided'}), 400

            processed, quality, status = sim_state.data_governance.process(
                data,
                source=data.get('source', 'api'),
                timestamp=time.time()
            )

            return jsonify({
                'processed_data': processed,
                'quality_score': {
                    'overall': quality.overall,
                    'completeness': quality.completeness,
                    'accuracy': quality.accuracy,
                    'consistency': quality.consistency,
                    'issues': quality.issues
                },
                'status': status.value
            })
        except Exception as e:
            logger.error("Error in data governance: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/governance/report')
    def get_governance_report():
        """Get data governance report."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            report = sim_state.data_governance.get_governance_report()
            return jsonify(report)
        except Exception as e:
            logger.error("Error getting governance report: %s", e)
            return jsonify({'error': str(e)}), 500

    # =========================================================================
    # Data Assimilation API Endpoints
    # =========================================================================

    @app.route('/api/assimilation/run', methods=['POST'])
    def run_assimilation():
        """Run data assimilation with observations."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            data = request.json
            if data is None:
                return jsonify({'error': 'No JSON data provided'}), 400

            observations = data.get('observations', {})
            error_std = data.get('error_std', 0.1)

            result = sim_state.data_assimilation.assimilate(
                sim_state.model,
                observations,
                error_std=error_std
            )

            return jsonify({
                'status': 'ok',
                'method': result['method'],
                'timestamp': result['timestamp']
            })
        except Exception as e:
            logger.error("Error in data assimilation: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/assimilation/uncertainty')
    def get_assimilation_uncertainty():
        """Get current state uncertainty from assimilation."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            uncertainty = sim_state.data_assimilation.get_state_uncertainty()
            return jsonify({
                'uncertainty': uncertainty.tolist()
            })
        except Exception as e:
            logger.error("Error getting uncertainty: %s", e)
            return jsonify({'error': str(e)}), 500

    # =========================================================================
    # Model Calibration API Endpoints
    # =========================================================================

    @app.route('/api/calibration/update', methods=['POST'])
    def update_calibration():
        """Update model calibration with high-fidelity data."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            data = request.json
            if data is None:
                return jsonify({'error': 'No JSON data provided'}), 400

            hifi_state = data.get('hifi_state', sim_state.model.get_state())
            idz_state = data.get('idz_state', sim_state.model.get_state())

            result = sim_state.model_calibrator.update_from_high_fidelity(
                hifi_state, idz_state, timestamp=sim_state.model.time
            )

            return jsonify({
                'parameters': result.parameters,
                'residual': result.residual,
                'status': result.convergence_status.value,
                'iterations': result.iterations
            })
        except Exception as e:
            logger.error("Error updating calibration: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/calibration/status')
    def get_calibration_status():
        """Get current calibration status."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            summary = sim_state.model_calibrator.get_calibration_summary()
            return jsonify(summary)
        except Exception as e:
            logger.error("Error getting calibration status: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/calibration/apply', methods=['POST'])
    def apply_calibration():
        """Apply calibrated parameters to model."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            sim_state.model_calibrator.apply_to_model(sim_state.model)
            return jsonify({'status': 'ok', 'message': 'Parameters applied'})
        except Exception as e:
            logger.error("Error applying calibration: %s", e)
            return jsonify({'error': str(e)}), 500

    # =========================================================================
    # State Evaluation API Endpoints
    # =========================================================================

    @app.route('/api/evaluation')
    def get_state_evaluation():
        """Get real-time state evaluation."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            target_flow = request.args.get('target_flow', type=float)
            evaluation = sim_state.state_evaluator.evaluate(
                sim_state.model,
                target_flow=target_flow or sim_state.target_flow
            )

            return jsonify({
                'performance': {
                    'overall': evaluation.performance.overall,
                    'level': evaluation.performance.level.value,
                    'by_objective': evaluation.performance.by_objective
                },
                'deviations': [
                    {
                        'objective': d.objective.name,
                        'current_value': d.current_value,
                        'target_value': d.objective.target_value,
                        'deviation': d.deviation,
                        'within_tolerance': d.within_tolerance
                    }
                    for d in evaluation.deviations
                ],
                'alarms': evaluation.alarms,
                'recommendations': evaluation.recommendations,
                'timestamp': evaluation.timestamp
            })
        except Exception as e:
            logger.error("Error in state evaluation: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/evaluation/objectives')
    def get_control_objectives():
        """Get control objectives configuration."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            return jsonify(sim_state.state_evaluator.get_objective_summary())
        except Exception as e:
            logger.error("Error getting objectives: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/evaluation/objectives', methods=['POST'])
    def update_control_objective():
        """Update a control objective."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            data = request.json
            if data is None:
                return jsonify({'error': 'No JSON data provided'}), 400

            name = data.get('name')
            target = data.get('target')

            if name and target is not None:
                sim_state.state_evaluator.objective_manager.update_target(name, target)

            return jsonify({'status': 'ok'})
        except Exception as e:
            logger.error("Error updating objective: %s", e)
            return jsonify({'error': str(e)}), 500

    # =========================================================================
    # State Prediction API Endpoints
    # =========================================================================

    @app.route('/api/prediction')
    def get_state_prediction():
        """Get state prediction."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            horizon = request.args.get('horizon', 60.0, type=float)
            result = sim_state.state_predictor.predict(horizon_seconds=horizon)

            physics = result['physics_prediction']
            return jsonify({
                'method': physics.method.value,
                'horizon': physics.horizon.value,
                'predictions': {
                    'times': physics.prediction_times.tolist(),
                    'total_flow': physics.predicted_state['total_flow'].tolist(),
                },
                'uncertainty': physics.uncertainty,
                'trend_analyses': {
                    k: {
                        'current_value': v.current_value,
                        'trend_direction': v.trend_direction,
                        'predicted_value': v.predicted_value,
                        'confidence': v.confidence
                    }
                    for k, v in result['trend_analyses'].items()
                },
                'active_alerts': [
                    {
                        'type': a.alert_type,
                        'severity': a.severity,
                        'variable': a.variable,
                        'message': a.message,
                        'time_to_event': a.time_to_event
                    }
                    for a in result['active_alerts']
                ]
            })
        except Exception as e:
            logger.error("Error in state prediction: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/prediction/whatif', methods=['POST'])
    def what_if_analysis():
        """Perform what-if analysis."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            import numpy as np

            data = request.json
            if data is None:
                return jsonify({'error': 'No JSON data provided'}), 400

            targets = np.array(data.get('target_openings', [1.0, 1.0, 1.0]))
            horizon = data.get('horizon', 60.0)

            analysis = sim_state.state_predictor.what_if_analysis(
                target_openings=targets,
                horizon_seconds=horizon
            )

            return jsonify(analysis)
        except Exception as e:
            logger.error("Error in what-if analysis: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/prediction/alerts')
    def get_predictive_alerts():
        """Get active predictive alerts."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            alerts = sim_state.state_predictor.alert_engine.get_active_alerts()
            return jsonify({
                'count': len(alerts),
                'alerts': [
                    {
                        'type': a.alert_type,
                        'severity': a.severity,
                        'variable': a.variable,
                        'current_value': a.current_value,
                        'predicted_value': a.predicted_value,
                        'threshold': a.threshold,
                        'time_to_event': a.time_to_event,
                        'message': a.message
                    }
                    for a in alerts
                ]
            })
        except Exception as e:
            logger.error("Error getting alerts: %s", e)
            return jsonify({'error': str(e)}), 500

    # =========================================================================
    # Predictive Maintenance API Endpoints
    # =========================================================================

    @app.route('/api/maintenance/status')
    def get_maintenance_status():
        """Get overall predictive maintenance system status."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            status = sim_state.maintenance_system.get_system_status()
            return jsonify(status)
        except Exception as e:
            logger.error("Error getting maintenance status: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/maintenance/equipment')
    def get_equipment_list():
        """Get list of monitored equipment with health status."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            status = sim_state.maintenance_system.get_system_status()
            equipment = []
            for eq_id, eq_status in status.get('equipment_status', {}).items():
                equipment.append({
                    'equipment_id': eq_id,
                    'health': eq_status['health'],
                    'status': eq_status['status'],
                    'operating_hours': eq_status['operating_hours'],
                    'rul_hours': eq_status['rul_hours'],
                    'failure_probability': eq_status['failure_probability']
                })
            return jsonify({
                'count': len(equipment),
                'equipment': equipment
            })
        except Exception as e:
            logger.error("Error getting equipment list: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/maintenance/equipment/<equipment_id>')
    def get_equipment_details(equipment_id):
        """Get detailed report for specific equipment."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            report = sim_state.maintenance_system.get_equipment_report(equipment_id)
            if report is None:
                return jsonify({'error': f'Equipment {equipment_id} not found'}), 404
            return jsonify(report)
        except Exception as e:
            logger.error("Error getting equipment details: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/maintenance/predictions')
    def get_failure_predictions():
        """Get failure predictions for all equipment."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            predictions = sim_state.maintenance_system.predict_failures()
            result = {}
            for eq_id, eq_predictions in predictions.items():
                result[eq_id] = [
                    {
                        'failure_mode': p.failure_mode.value,
                        'probability': p.probability,
                        'time_to_failure_hours': p.time_to_failure_hours,
                        'confidence': p.confidence,
                        'contributing_factors': p.contributing_factors,
                        'recommended_actions': p.recommended_actions
                    }
                    for p in eq_predictions
                ]
            return jsonify({
                'equipment_count': len(result),
                'predictions': result
            })
        except Exception as e:
            logger.error("Error getting failure predictions: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/maintenance/plan')
    def get_maintenance_plan():
        """Get optimized maintenance plan."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            plan = sim_state.maintenance_system.generate_maintenance_plan()
            actions = [
                {
                    'equipment_id': a.equipment_id,
                    'action_type': a.action_type.value,
                    'priority': a.priority.name,
                    'description': a.description,
                    'estimated_duration_hours': a.estimated_duration_hours,
                    'estimated_cost': a.estimated_cost,
                    'due_date': a.due_date.isoformat(),
                    'risk_if_delayed': a.risk_if_delayed
                }
                for a in plan
            ]
            cost_summary = sim_state.maintenance_system.scheduler.get_cost_summary()
            return jsonify({
                'plan': actions,
                'summary': cost_summary
            })
        except Exception as e:
            logger.error("Error generating maintenance plan: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/maintenance/update', methods=['POST'])
    def update_maintenance_data():
        """Update maintenance system with new sensor data."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            data = request.get_json() or {}
            timestamp = data.get('timestamp', time.time() / 3600)  # Default to current hour

            # Get current state from physics model
            state = sim_state.model.get_state()

            # Build sensor data for maintenance update
            sensor_data = {}
            for i in range(3):
                sensor_data[f'gate_{i}'] = {
                    'vibration': state['vibrations'][i] if i < len(state['vibrations']) else 0.0,
                    'temperature': 25.0 + np.random.normal(0, 2),  # Simulated
                    'performance_metric': 90.0 - state['vibrations'][i] * 5
                }

            # Add sensors
            sensor_data['sensor_flow'] = {'performance_metric': 95.0}
            sensor_data['sensor_pressure'] = {'performance_metric': 95.0}
            sensor_data['sensor_velocity'] = {'performance_metric': 95.0}
            sensor_data['sensor_water_level'] = {'performance_metric': 95.0}

            result = sim_state.maintenance_system.update(timestamp, sensor_data)
            return jsonify(result)
        except Exception as e:
            logger.error("Error updating maintenance data: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/maintenance/record', methods=['POST'])
    def record_maintenance_action():
        """Record that maintenance was performed on equipment."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            data = request.get_json() or {}
            equipment_id = data.get('equipment_id')

            if not equipment_id:
                return jsonify({'error': 'equipment_id is required'}), 400

            if equipment_id not in sim_state.maintenance_system._monitors:
                return jsonify({'error': f'Equipment {equipment_id} not found'}), 404

            monitor = sim_state.maintenance_system._monitors[equipment_id]
            monitor.record_maintenance(datetime.now())

            return jsonify({
                'status': 'success',
                'message': f'Maintenance recorded for {equipment_id}',
                'new_health': monitor._current_health.overall
            })
        except Exception as e:
            logger.error("Error recording maintenance: %s", e)
            return jsonify({'error': str(e)}), 500

    # =========================================================================
    # Analytics and Reporting API Endpoints
    # =========================================================================

    @app.route('/api/analytics/dashboard')
    def get_analytics_dashboard():
        """Get analytics dashboard data."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            dashboard = sim_state.analytics.get_dashboard_data()
            return jsonify(dashboard)
        except Exception as e:
            logger.error("Error getting dashboard data: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/analytics/kpis')
    def get_all_kpis():
        """Get all current KPI values."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            kpis = sim_state.analytics.kpi_tracker.get_all_current()
            return jsonify({
                'count': len(kpis),
                'kpis': {
                    name: {
                        'value': v.value,
                        'target': v.target,
                        'deviation_pct': v.deviation_pct,
                        'status': v.status,
                        'trend': v.trend
                    }
                    for name, v in kpis.items()
                }
            })
        except Exception as e:
            logger.error("Error getting KPIs: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/analytics/kpis/alerts')
    def get_kpi_alerts():
        """Get KPIs in warning or critical status."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            alerts = sim_state.analytics.kpi_tracker.get_alerts()
            return jsonify({
                'count': len(alerts),
                'alerts': [
                    {
                        'name': a.name,
                        'value': a.value,
                        'target': a.target,
                        'status': a.status,
                        'trend': a.trend
                    }
                    for a in alerts
                ]
            })
        except Exception as e:
            logger.error("Error getting KPI alerts: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/analytics/alarms')
    def get_analytics_alarms():
        """Get alarm statistics and active alarms."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            stats = sim_state.analytics.alarm_manager.get_statistics(24)
            active = sim_state.analytics.alarm_manager.get_active()
            return jsonify({
                'statistics': stats,
                'active_alarms': [
                    {
                        'event_id': a.event_id,
                        'timestamp': a.timestamp.isoformat(),
                        'severity': a.severity.name,
                        'category': a.category.name,
                        'source': a.source,
                        'message': a.message,
                        'acknowledged': a.acknowledged
                    }
                    for a in active
                ]
            })
        except Exception as e:
            logger.error("Error getting alarms: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/analytics/alarms/acknowledge', methods=['POST'])
    def acknowledge_alarm():
        """Acknowledge an alarm."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            data = request.get_json() or {}
            event_id = data.get('event_id')
            acknowledged_by = data.get('acknowledged_by', 'api_user')

            if not event_id:
                return jsonify({'error': 'event_id is required'}), 400

            result = sim_state.analytics.alarm_manager.acknowledge(event_id, acknowledged_by)
            if result:
                return jsonify({'status': 'success', 'message': f'Alarm {event_id} acknowledged'})
            else:
                return jsonify({'error': f'Alarm {event_id} not found'}), 404
        except Exception as e:
            logger.error("Error acknowledging alarm: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/analytics/alarms/clear', methods=['POST'])
    def clear_alarm():
        """Clear an alarm."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            data = request.get_json() or {}
            event_id = data.get('event_id')

            if not event_id:
                return jsonify({'error': 'event_id is required'}), 400

            result = sim_state.analytics.alarm_manager.clear(event_id)
            if result:
                return jsonify({'status': 'success', 'message': f'Alarm {event_id} cleared'})
            else:
                return jsonify({'error': f'Alarm {event_id} not found'}), 404
        except Exception as e:
            logger.error("Error clearing alarm: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/analytics/report')
    def generate_report():
        """Generate analytics report."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            report_type_str = request.args.get('type', 'daily')
            format_str = request.args.get('format', 'json')

            # Map strings to enums
            type_map = {
                'daily': ReportType.DAILY,
                'weekly': ReportType.WEEKLY,
                'monthly': ReportType.MONTHLY,
                'performance': ReportType.PERFORMANCE
            }
            format_map = {
                'json': ReportFormat.JSON,
                'text': ReportFormat.TEXT,
                'html': ReportFormat.HTML
            }

            report_type = type_map.get(report_type_str, ReportType.DAILY)
            report_format = format_map.get(format_str, ReportFormat.JSON)

            report = sim_state.analytics.generate_report(report_type, report_format)

            if report_format == ReportFormat.HTML:
                return report, 200, {'Content-Type': 'text/html'}
            elif report_format == ReportFormat.TEXT:
                return report, 200, {'Content-Type': 'text/plain'}
            else:
                return report, 200, {'Content-Type': 'application/json'}
        except Exception as e:
            logger.error("Error generating report: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/analytics/statistics')
    def get_operation_statistics():
        """Get operation statistics summary."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            hours = request.args.get('hours', 24, type=int)
            summaries = sim_state.analytics.statistics.get_all_summaries(hours)

            return jsonify({
                'period_hours': hours,
                'uptime_pct': sim_state.analytics.statistics.get_uptime_pct(),
                'metrics': {
                    name: {
                        'count': s.count,
                        'mean': s.mean,
                        'std': s.std,
                        'min': s.min,
                        'max': s.max,
                        'trend': s.trend
                    }
                    for name, s in summaries.items()
                }
            })
        except Exception as e:
            logger.error("Error getting statistics: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/analytics/update', methods=['POST'])
    def update_analytics():
        """Update analytics with current simulation data."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            # Get current state from physics model
            state = sim_state.model.get_state()
            target_flow = sim_state.target_flow

            # Build analytics data
            total_flow = float(np.sum(state['flows']))
            max_vibration = float(np.max(state['vibrations']))
            flow_error = abs(total_flow - target_flow) / target_flow * 100 if target_flow > 0 else 0

            analytics_data = {
                'total_flow': total_flow,
                'target_flow': target_flow,
                'flow_error': flow_error,
                'max_vibration': max_vibration
            }

            result = sim_state.analytics.update(analytics_data)
            return jsonify(result)
        except Exception as e:
            logger.error("Error updating analytics: %s", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/analytics/status')
    def get_analytics_status():
        """Get analytics engine status."""
        if sim_state is None:
            return jsonify({'error': 'Simulation not initialized'}), 500

        try:
            status = sim_state.analytics.get_status()
            return jsonify(status)
        except Exception as e:
            logger.error("Error getting analytics status: %s", e)
            return jsonify({'error': str(e)}), 500

    return app


def simulation_loop(state: SimulationState) -> None:
    """
    Main simulation loop running in a separate thread.

    Args:
        state: Shared simulation state object.
    """
    cfg = get_config()
    dt = cfg.simulation.dt

    logger.info("Simulation loop started (dt=%.2fs)", dt)

    while True:
        try:
            if not state.running:
                time.sleep(0.1)
                continue

            start_time = time.time()

            # Execute simulation step
            state.step(dt)

            # Real-time pacing
            elapsed = time.time() - start_time
            sleep_time = max(0, dt - elapsed)
            time.sleep(sleep_time)

        except Exception as e:
            logger.error("Error in simulation loop: %s", e, exc_info=True)
            time.sleep(1.0)  # Prevent tight error loop


def main() -> None:
    """Main entry point."""
    global sim_state

    # Initialize simulation state with integrated controller
    sim_state = SimulationState(use_integrated=True)

    # Start simulation thread
    sim_thread = threading.Thread(
        target=simulation_loop,
        args=(sim_state,),
        daemon=True
    )
    sim_thread.start()

    # Create and run Flask app
    app = create_app()
    cfg = get_config().web
    app.run(host=cfg.host, port=cfg.port, debug=cfg.debug)


if __name__ == '__main__':
    main()
