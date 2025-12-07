# -*- coding: utf-8 -*-
"""
Tanghe Inverted Siphon Web Application.

This module provides the Flask web application for monitoring and
controlling the digital twin simulation.
"""

import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

from flask import Flask, jsonify, render_template, request

from src.config import get_config
from src.simulation.physics import TangheSiphonModel
from src.simulation.sensors import ADCPSensor, VibrationSensor
from src.simulation.actuators import GateController
from src.control.mpc import SpectralMPC
from src.control.local import LocalController
from src.control.manager import ScenarioManager

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
    """

    def __init__(self) -> None:
        """Initialize simulation state with thread lock."""
        self._lock = threading.RLock()
        self._running = True
        self._target_flow = get_config().simulation.default_target_flow

        # Initialize components
        self.model = TangheSiphonModel()
        self.sensors = {
            'adcp': [ADCPSensor(self.model, i) for i in range(3)],
            'vib': [VibrationSensor(self.model, i) for i in range(3)]
        }
        self.actuator = GateController(self.model)
        self.mpc = SpectralMPC(self.model)
        self.local_ctrl = LocalController(self.model)
        self.scenario_mgr = ScenarioManager(self.model)

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
            logger.info("Target flow set to %.2f m³/s", value)

    def get_state(self) -> Dict[str, Any]:
        """Get current system state (thread-safe)."""
        with self._lock:
            state = self.model.get_state()
            state['target_flow'] = self._target_flow
            state['active_scenario'] = self.scenario_mgr.active_scenario
            state['simulation_running'] = self._running
            return state

    def set_scenario(self, scenario_id: str) -> None:
        """Set active scenario (thread-safe)."""
        with self._lock:
            self.scenario_mgr.set_scenario(scenario_id)

    def step(self, dt: float) -> None:
        """Execute one simulation step (thread-safe)."""
        with self._lock:
            # 1. Scenario Update
            self.scenario_mgr.update()

            # 2. Control Step
            current_openings = self.model.gate_openings
            head_diff = self.model.head_upstream - self.model.head_downstream

            # MPC (High Level)
            mpc_targets = self.mpc.get_target_openings(
                self._target_flow, current_openings, head_diff
            )

            # Local Control (Low Level)
            final_cmds = self.local_ctrl.update(mpc_targets, dt)

            # 3. Actuation
            for i in range(3):
                self.actuator.set_opening(i, final_cmds[i])

            # 4. Physics Step
            self.model.step(self.actuator.get_target_openings(), dt)

    def reset(self) -> None:
        """Reset simulation to initial state (thread-safe)."""
        with self._lock:
            self.model.reset()
            self.mpc.reset()
            self.local_ctrl.reset()
            self.scenario_mgr.reset()
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
        """Get available scenarios."""
        return jsonify({
            'scenarios': ScenarioManager.SCENARIOS
        })

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

    # Initialize simulation state
    sim_state = SimulationState()

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
