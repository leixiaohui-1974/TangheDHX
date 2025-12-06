import unittest
import numpy as np
import time
from src.simulation.physics import TangheSiphonModel
from src.control.mpc import SpectralMPC
from src.control.manager import ScenarioManager

class TestTangheSystem(unittest.TestCase):

    def test_physics_resonance(self):
        """Test if resonance logic triggers at expected velocity."""
        model = TangheSiphonModel()

        # We calibrated v=2.6 -> f=2.8.
        # Let's force v=2.6
        # To get v=2.6, we need specific opening.

        # Calculate K
        Cd = 0.7
        B = 6.0
        g = 9.81
        dH = 2.0
        A_eff = 16.6
        K = Cd * B * np.sqrt(2 * g * dH) / A_eff
        # K = 1.58

        # To get v=2.6, e = 2.6 / 1.58 = 1.64m.

        target_e = 1.64
        # FORCE the opening directly to bypass actuator delay
        model.gate_openings = np.array([target_e, target_e, target_e])

        # Run 1 step to update physics
        model.step(model.gate_openings, dt=0.1)

        v = model.velocities[0]
        f = model.vortex_freqs[0]
        vib = model.vibration_accel[0]

        print(f"DEBUG: e={model.gate_openings[0]:.2f}, v={v:.2f}, f={f:.2f}, vib={vib:.4f}")

        # Check resonance frequency
        # f should be close to 2.8
        self.assertTrue(2.7 < f < 2.9, f"Frequency {f} not in resonance range")

        # Check vibration amplification (Lock-in)
        # Base amp = 0.01 * v^2 = 0.01 * 2.6^2 = 0.067
        # Resonance amp = 8x = 0.54
        self.assertTrue(vib > 0.4, f"Vibration {vib} should be amplified")

    def test_mpc_avoidance(self):
        """Test if MPC avoids the resonance opening."""
        model = TangheSiphonModel()
        mpc = SpectralMPC(model)

        # Target flow that WOULD require v=2.6 if symmetric.
        # v=2.6 -> Q = v * A_eff = 2.6 * 16.6 = 43.16 per gate.
        # Total Q = 130.

        target_flow = 130.0
        head_diff = 2.0

        # Initial state: closed
        current_openings = np.array([0.5, 0.5, 0.5]) # small flow

        # Run MPC
        targets = mpc.get_target_openings(target_flow, current_openings, head_diff)

        # Check predictions
        # Using the same physics prediction logic as MPC
        Cd = 0.7
        g = 9.81
        A_eff = 16.6
        width = 6.0

        flows = []
        velocities = []
        for e in targets:
            q = Cd * width * e * np.sqrt(2 * g * head_diff)
            v = q / A_eff
            flows.append(q)
            velocities.append(v)

        total_flow = sum(flows)
        print(f"DEBUG MPC: Targets={targets}, Velocities={velocities}, TotalQ={total_flow}")

        # Check asymmetry: std dev of openings should be > 0.1
        self.assertTrue(np.std(targets) > 0.05, "MPC should choose asymmetric openings to avoid resonance")

        # Check if total flow is close to target (it might sacrifice exact flow for safety,
        # or maybe it finds a safe combination)
        # With alpha=1.0 and gamma=10.0, it prioritizes safety/flow trade-off.
        # It should be reasonably close.
        self.assertTrue(abs(total_flow - target_flow) < 5.0, f"Flow {total_flow} too far from {target_flow}")

    def test_scenario_fault(self):
        """Test Fault Injection logic."""
        model = TangheSiphonModel()
        mgr = ScenarioManager(model)

        mgr.set_scenario('S4.2')
        model.time = 0

        # Run for 15s
        for _ in range(150):
            model.step(np.array([1.0, 1.0, 1.0]), dt=0.1)
            mgr.update()

        self.assertTrue(model.gate_stuck[1], "Gate 2 should be stuck after 10s in S4.2")

if __name__ == '__main__':
    unittest.main()
