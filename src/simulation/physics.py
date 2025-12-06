import numpy as np
import time

class TangheSiphonModel:
    """
    Physical Digital Twin of Tanghe Inverted Siphon.

    Attributes:
        num_gates (int): 3
        width (float): 6.0m
        gate_opening (np.array): Current opening height [m] for each gate.
        flow_rates (np.array): Current flow rate [m^3/s].
        velocities (np.array): Current local velocity [m/s].
        vibrations (np.array): Current vibration acceleration [g].
        frequencies (np.array): Current vortex shedding frequency [Hz].
    """

    def __init__(self):
        self.num_gates = 3
        self.width = 6.0  # meters
        self.gate_openings = np.array([0.0, 0.0, 0.0]) # 0 to 5.0m

        # Environmental conditions
        self.head_upstream = 10.0 # meters
        self.head_downstream = 8.0 # meters

        # Physics Constants
        # Calibrated to match report: v=2.6 m/s => f=2.8 Hz (Resonance)
        # f = k * v => k = 2.8 / 2.6 = 1.077
        self.k_freq = 1.077
        self.f_struct = 2.8 # Hz (Rod natural frequency)
        self.damping_ratio = 0.02

        # State
        self.flow_rates = np.zeros(3)
        self.velocities = np.zeros(3)
        self.vortex_freqs = np.zeros(3)
        self.vibration_accel = np.zeros(3) # in g
        self.time = 0.0

        # Failure Injection
        self.gate_stuck = [False, False, False]
        self.gate_noise = [0.0, 0.0, 0.0]

    def step(self, target_openings, dt=0.1):
        """
        Advance simulation by dt seconds.
        """
        self.time += dt

        # 1. Actuator Dynamics (Gate movement)
        max_speed = 0.05 # m/s
        for i in range(self.num_gates):
            if self.gate_stuck[i]:
                continue # No movement

            error = target_openings[i] - self.gate_openings[i]
            move = np.clip(error, -max_speed * dt, max_speed * dt)
            self.gate_openings[i] += move

            # Mechanical Limits
            self.gate_openings[i] = np.clip(self.gate_openings[i], 0.0, 5.0)

        # 2. Hydraulics
        # Simplified Model: Q proportional to Opening * sqrt(Head)
        # But we need to match the Report's Velocity logic where v ~ Q.
        # Let's assume v_local = C_v * Opening * sqrt(H) / Opening? No.
        # Let's align with Report:
        # S1.1: Q=33.3 -> v=2.0.
        # S1.1 implies Opening ~ 3.0m? (Just a guess)
        # Let's define: Q = Cd * B * e * sqrt(2gdH)
        # And v_local = Q / A_eff.  (A_eff = 16.6 m^2 from analysis)

        A_eff = 16.6
        Cd = 0.7
        g = 9.81
        delta_H = max(0, self.head_upstream - self.head_downstream)

        for i in range(self.num_gates):
            # Calculate Flow
            # Special case: if e is very small, leakage or small flow
            q = Cd * self.width * self.gate_openings[i] * np.sqrt(2 * g * delta_H)
            self.flow_rates[i] = q

            # Calculate Velocity (Report definition)
            # Avoid division by zero if using Q/A approach
            self.velocities[i] = q / A_eff

        # 3. Vortex Dynamics (FIV)
        for i in range(self.num_gates):
            v = self.velocities[i]
            if v < 0.1:
                f_s = 0.0
                amp = 0.0
            else:
                f_s = self.k_freq * v

                # Lock-in Check
                # 0.9 fn < fs < 1.1 fn
                ratio = f_s / self.f_struct
                is_lock_in = 0.9 < ratio < 1.1

                # Base vibration (random noise + flow induced)
                # Amplitude grows with v^2 usually
                base_amp = 0.01 * (v**2)

                if is_lock_in:
                    # Resonance amplification 5-10x
                    amp = base_amp * 8.0
                else:
                    amp = base_amp

            self.vortex_freqs[i] = f_s
            self.vibration_accel[i] = amp

    def get_state(self):
        return {
            'timestamp': self.time,
            'openings': self.gate_openings.tolist(),
            'flows': self.flow_rates.tolist(),
            'velocities': self.velocities.tolist(),
            'frequencies': self.vortex_freqs.tolist(),
            'vibrations': self.vibration_accel.tolist(),
            'total_flow': np.sum(self.flow_rates)
        }

    def inject_fault(self, gate_index, fault_type):
        if fault_type == 'stuck':
            self.gate_stuck[gate_index] = True
        elif fault_type == 'clear':
            self.gate_stuck[gate_index] = False
