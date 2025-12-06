import time
import numpy as np

class Supervisor:
    """
    Autonomous Operation Supervisor (The "Brain").
    Manages System Modes, Monitors Health, Interlocks, and Scenario Identification.
    """

    MODES = ['NORMAL', 'ROBUST', 'EMERGENCY']

    def __init__(self, model):
        self.model = model
        self.mode = 'NORMAL'
        self.alarms = []
        self.events = []

        # Interlock Thresholds
        self.TRASH_RACK_LIMIT = 0.3 # meters head loss

        # Auto-Pilot State
        self.auto_pilot = False

        # Scenario ID
        self.detected_scenario = "Unknown"

    def update(self):
        """
        Main Supervisor Loop.
        """
        self._check_interlocks()
        self._identify_scenario()
        self._manage_autopilot()

    def _check_interlocks(self):
        # 1. FSI Interlock (Strategy 3): Trash Rack Blockage
        loss = self.model.trash_rack_loss
        if loss > self.TRASH_RACK_LIMIT and self.mode == 'NORMAL':
            self.trigger_alarm("Trash Rack Blockage Detected! High Head Loss.")
            self.set_mode('ROBUST')

        # 2. Resonance Persistent Check
        for i in range(3):
            if self.model.vibration_accel[i] > 0.4:
                self.trigger_alarm(f"Critical Vibration on Gate {i+1}!")

    def _identify_scenario(self):
        """
        Infer the current operational scenario from sensor data.
        """
        # Gather Data
        flows = self.model.flow_rates
        vibs = self.model.vibration_accel
        openings = self.model.gate_openings
        freqs = self.model.vortex_freqs
        loss = self.model.trash_rack_loss

        # Logic Tree

        # Faults First
        if loss > self.TRASH_RACK_LIMIT:
            self.detected_scenario = "S4.1 Trash Rack Blockage"
            return

        # S4.3 Leakage: Closed gate but has flow
        for i in range(3):
            if openings[i] < 0.05 and flows[i] > 1.0:
                self.detected_scenario = "S4.3 Seal Leakage"
                return

        # S2.1 Resonance: High Vib & Freq match
        is_resonance = False
        for i in range(3):
            if vibs[i] > 0.2 and 2.5 < freqs[i] < 3.1:
                is_resonance = True
        if is_resonance:
            self.detected_scenario = "S2.1 Resonance Zone"
            return

        # S2.3 Flapping: High Vib but Low Freq/Flow? Or Specific Condition
        # My physics model sets high vib for flapping.
        for i in range(3):
            if 0.01 < openings[i] < 0.1 and vibs[i] > 0.1:
                self.detected_scenario = "S2.3 Bottom Edge Flapping"
                return

        # Normal Operation Analysis
        # Check Symmetry
        flow_std = np.std(flows)
        if flow_std < 1.0:
            self.detected_scenario = "S1.1 Normal (Balanced)"
        else:
            self.detected_scenario = "S1.2 Normal (Unbalanced)"

    def _manage_autopilot(self):
        if not self.auto_pilot:
            return
        t = time.time()
        # Period 60s
        # flow_demand = 100 + 40 * np.sin(2 * np.pi * t / 60.0)
        pass

    def set_mode(self, new_mode):
        if new_mode in self.MODES:
            self.mode = new_mode
            self.events.append(f"System Mode switched to {new_mode}")
            print(f"SUPERVISOR: Mode -> {new_mode}")

    def trigger_alarm(self, msg):
        if msg not in self.alarms:
            self.alarms.append(msg)
            self.events.append(f"ALARM: {msg}")
            print(f"SUPERVISOR ALARM: {msg}")

    def get_status(self):
        return {
            'mode': self.mode,
            'alarms': self.alarms,
            'latest_event': self.events[-1] if self.events else "",
            'detected_scenario': self.detected_scenario
        }
