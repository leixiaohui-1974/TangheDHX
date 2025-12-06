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

        # 1. Scenario Update
        scenario_mgr.update()

        # 2. Control Step
        # a. Get Sensor Data (simulated)
        current_openings = model.gate_openings
        head_diff = model.head_upstream - model.head_downstream

        # b. MPC (High Level) - Run every 1.0s (simulated by counter or just run every step for smoothness in demo)
        # In real life MPC is slow. Here we run it every step but with 'last_target' memory it should be fine.
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
        # For smoother web visual, we might run faster than real time or sync.
        # Let's run roughly real-time.
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
    return jsonify(state)

@app.route('/api/control', methods=['POST'])
def set_control():
    global target_flow
    data = request.json
    if 'target_flow' in data:
        target_flow = float(data['target_flow'])
    if 'scenario' in data:
        scenario_mgr.set_scenario(data['scenario'])
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
