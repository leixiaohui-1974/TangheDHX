# -*- coding: utf-8 -*-
"""
Tanghe Inverted Siphon Web Application.

This module provides the Flask web application for monitoring and
controlling the digital twin simulation with full scenario-aware control.
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

        logger.info("SimulationState initialized")

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
