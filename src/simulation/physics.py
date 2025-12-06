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

        # Trash Rack State (Head Loss in meters)
        self.trash_rack_loss = 0.0

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
        self.seal_leakage = [False, False, False] # S4.3

    def step(self, target_openings, dt=0.1):
        """
        Advance simulation by dt seconds.
        """
        self.time += dt

        # 1. Actuator Dynamics (Gate movement)
        # Check if we need BOOSTED speed (Fast Transit)
        # We need to know if "Fast Transit" is active.
        # Ideally, `LocalController` sends a command for speed too.
        # But `step` only takes `target_openings`.
        # Hack: We can infer urgency, or we can assume `target_openings` is just position.
        # But `LocalController` is separate.
        # Let's assume the passed `target_openings` are strictly position.
        # To support "Fast Transit" simulation, let's allow `max_speed` to be higher
        # if the ERROR is large and we are in the resonance zone?
        # Or better: Just allow high speed generally.
        max_speed = 0.50 # m/s - High speed capability for Fast Transit

        for i in range(self.num_gates):
            if self.gate_stuck[i]:
                continue # No movement

            error = target_openings[i] - self.gate_openings[i]

            # Use 'Fast Transit' if error is large enough to cross zones?
            # Or just use max_speed always. Real actuators have limits.
            # Normal operation speed: 0.05. Fast: 0.20.
            # We can simulate this: If error > 0.5m, assume Fast Transit triggered?
            # No, let's keep it simple: 0.50 m/s max capability.
            # Control logic (step size in targets) dictates actual speed if slow.
            # But here targets jump.

            move = np.clip(error, -max_speed * dt, max_speed * dt)
            self.gate_openings[i] += move

            # Mechanical Limits
            self.gate_openings[i] = np.clip(self.gate_openings[i], 0.0, 5.0)

        # 2. Hydraulics
        A_eff = 16.6
        Cd = 0.7
        g = 9.81

        # Effective Head Difference = Upstream - Downstream - TrashRackLoss
        delta_H = max(0, self.head_upstream - self.head_downstream - self.trash_rack_loss)

        for i in range(self.num_gates):
            # S4.3 Seal Leakage
            leakage_flow = 0.0
            if self.seal_leakage[i] and self.gate_openings[i] < 0.05:
                # Simulate leakage flow (equivalent to small opening)
                leakage_flow = 2.0 # m3/s fixed leakage

            # Calculate Flow
            q = Cd * self.width * self.gate_openings[i] * np.sqrt(2 * g * delta_H)
            self.flow_rates[i] = q + leakage_flow

            # Calculate Velocity (Report definition)
            self.velocities[i] = self.flow_rates[i] / A_eff

        # 3. Vortex Dynamics (FIV)
        for i in range(self.num_gates):
            v = self.velocities[i]
            e = self.gate_openings[i]

            # S2.3 Bottom Edge Flapping (Small opening instability)
            # If opening is small (< 10cm) but non-zero, and significant head exists
            is_flapping = 0.01 < e < 0.10 and delta_H > 1.0

            if v < 0.1 and not is_flapping:
                f_s = 0.0
                amp = 0.0
            else:
                f_s = self.k_freq * v

                # Lock-in Check
                ratio = f_s / self.f_struct if self.f_struct > 0 else 0
                is_lock_in = 0.9 < ratio < 1.1

                # Base vibration
                # Adjusted coefficient to 0.004 to keep high-velocity non-resonant vib reasonable
                base_amp = 0.004 * (v**2)

                if is_lock_in:
                    # Resonance amplification
                    amp = base_amp * 8.0
                elif is_flapping:
                    # Flapping vibration (S2.3) - High amplitude, random frequency
                    amp = 0.2 # 0.2g constant banging
                    f_s = 10.0 # High freq noise
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
            'total_flow': np.sum(self.flow_rates),
            'trash_rack_loss': self.trash_rack_loss
        }

    def inject_fault(self, gate_index, fault_type, value=None):
        if fault_type == 'stuck':
            self.gate_stuck[gate_index] = True
        elif fault_type == 'clear':
            self.gate_stuck[gate_index] = False
            self.seal_leakage[gate_index] = False
            self.trash_rack_loss = 0.0
        elif fault_type == 'leakage':
            self.seal_leakage[gate_index] = True
        elif fault_type == 'trash_rack':
            self.trash_rack_loss = value if value is not None else 0.5
