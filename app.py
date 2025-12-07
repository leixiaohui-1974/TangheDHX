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
from typing import Any, Dict, Optional, Tuple, List

from flask import Flask, jsonify, render_template, request

from src.config import get_config
from src.simulation.physics import TangheSiphonModel
from src.simulation.sensors import ADCPSensor, VibrationSensor
from src.simulation.actuators import GateController
from src.control.mpc import SpectralMPC
from src.control.local import LocalController
from src.control.manager import ScenarioManager
from src.control.integrated_controller import IntegratedController, ScenarioType
from src.control.scenario_advanced import AdvancedScenarioManager
from src.agents.communication import AgentNetwork
from src.data.storage import TimeSeriesStorage, DataPoint
from src.data.analysis import DataAnalyzer
from src.data.anomaly import AnomalyDetector, ThresholdRule, RateRule, Severity
from src.data.replay import HistoryReplay, ReplayMode

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
        self.data_analyzer = DataAnalyzer(self.data_storage)

        # Anomaly detector
        self.anomaly_detector = AnomalyDetector(self.data_storage)
        self._init_anomaly_rules()

        # History replay
        self.replay = HistoryReplay(self.data_storage)

        logger.info("SimulationState initialized with full feature set")

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
        self.anomaly_detector.add_rule(ThresholdRule(
            rule_id='flow_high',
            series_name='total_flow',
            min_value=0,
            max_value=450,
            severity=Severity.WARNING,
            message='Total flow exceeds normal range'
        ))

        # Head difference anomalies
        self.anomaly_detector.add_rule(ThresholdRule(
            rule_id='head_diff_extreme',
            series_name='head_diff',
            min_value=-5,
            max_value=10,
            severity=Severity.WARNING,
            message='Head difference abnormal'
        ))

        # Vibration anomalies for each gate
        for i in range(3):
            self.anomaly_detector.add_rule(ThresholdRule(
                rule_id=f'vib_gate_{i}',
                series_name=f'gate_{i}_vibration',
                min_value=0,
                max_value=50,
                severity=Severity.CRITICAL,
                message=f'Gate {i} vibration exceeds safety limit'
            ))

            # Rate of change rule
            self.anomaly_detector.add_rule(RateRule(
                rule_id=f'vib_rate_gate_{i}',
                series_name=f'gate_{i}_vibration',
                max_rate=20,
                window=5.0,
                severity=Severity.WARNING,
                message=f'Gate {i} vibration changing rapidly'
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
