from flask import Flask, render_template, jsonify, request
import threading
import time
import numpy as np

from src.simulation.physics import TangheSiphonModel
from src.simulation.sensors import ADCPSensor, VibrationSensor
from src.simulation.actuators import GateController
from src.control.mpc import SpectralMPC
from src.control.local import LocalController
from src.control.manager import ScenarioManager
from src.control.supervisor import Supervisor

app = Flask(__name__, template_folder='src/web/templates', static_folder='src/web/static')

# --- Global System State ---
model = TangheSiphonModel()
sensors = {
    'adcp': [ADCPSensor(model, i) for i in range(3)],
    'vib': [VibrationSensor(model, i) for i in range(3)]
}
actuator = GateController(model)
mpc = SpectralMPC(model)
local_ctrl = LocalController(model)
scenario_mgr = ScenarioManager(model)
supervisor = Supervisor(model)

# Simulation Loop Control
SIM_RUNNING = True
target_flow = 100.0 # m3/s

def simulation_loop():
    global target_flow, SIM_RUNNING
    dt = 0.1

    while True:
        if not SIM_RUNNING:
            time.sleep(0.1)
            continue

        start_time = time.time()

        # 1. Scenario & Supervisor Update
        scenario_mgr.update()
        supervisor.update()

        # Auto-pilot flow override (if enabled)
        # (Not fully hooked up to override target_flow yet, kept simple for now)

        # 2. Control Step
        # a. Get Sensor Data
        current_openings = model.gate_openings
        # Effective head for flow calculation vs Measured Head?
        # MPC should use measured head (upstream - downstream).
        # Trash rack loss might not be directly measured unless differential sensor exists.
        # Report says "Delta H > 15cm" is monitored.
        # So we pass effective head or head diff to MPC?
        # If blockage, MPC should know capacity is reduced?
        # For now, pass physical head diff.
        head_diff = model.head_upstream - model.head_downstream

        # b. MPC (High Level)
        # If ROBUST mode, maybe MPC changes parameters?
        # For now, standard MPC.
        mpc_targets = mpc.get_target_openings(target_flow, current_openings, head_diff)

        # c. Local Control (Low Level)
        final_cmds = local_ctrl.update(mpc_targets, dt)

        # 3. Actuation
        for i in range(3):
            actuator.set_opening(i, final_cmds[i])

        # 4. Physics Step
        model.step(actuator.get_target_openings(), dt)

        # Real-time pacing
        elapsed = time.time() - start_time
        sleep_time = max(0, dt - elapsed)
        time.sleep(sleep_time)

# Start Simulation Thread
sim_thread = threading.Thread(target=simulation_loop, daemon=True)
sim_thread.start()

# --- API Endpoints ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state')
def get_state():
    state = model.get_state()
    # Add Control Info
    state['target_flow'] = target_flow
    state['active_scenario'] = scenario_mgr.active_scenario
    state['supervisor'] = supervisor.get_status()
    return jsonify(state)

@app.route('/api/control', methods=['POST'])
def set_control():
    global target_flow
    data = request.json
    if 'target_flow' in data:
        target_flow = float(data['target_flow'])
    if 'scenario' in data:
        scenario_mgr.set_scenario(data['scenario'])
    if 'inject_fault' in data:
        fault = data['inject_fault']
        # Helper for UI triggered faults
        if fault == 'trash_rack':
            model.inject_fault(0, 'trash_rack', 0.5)
        elif fault == 'leakage':
            model.inject_fault(0, 'leakage')
        elif fault == 'clear':
            model.inject_fault(0, 'clear')
            supervisor.alarms = [] # Clear alarms
            supervisor.set_mode('NORMAL')

    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
