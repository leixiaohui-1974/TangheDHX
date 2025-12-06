import unittest
import numpy as np
from src.simulation.physics import TangheSiphonModel
from src.control.supervisor import Supervisor

class TestAdvancedFeatures(unittest.TestCase):

    def test_trash_rack_interlock(self):
        """Test FSI Interlock: High Head Loss -> ROBUST Mode"""
        model = TangheSiphonModel()
        supervisor = Supervisor(model)

        # 1. Normal State
        self.assertEqual(supervisor.mode, 'NORMAL')

        # 2. Inject Fault (High Head Loss)
        model.trash_rack_loss = 0.5 # > 0.3 threshold

        # 3. Update Supervisor
        supervisor.update()

        # 4. Check Mode Switch
        self.assertEqual(supervisor.mode, 'ROBUST')
        self.assertTrue("Trash Rack Blockage" in supervisor.alarms[0])

        # 5. Check Physics Impact (Flow reduction)
        # e=1.0. Head=10, Tail=8, Loss=0.5 -> EffHead=1.5
        model.gate_openings = np.array([1.0, 1.0, 1.0])
        model.step(model.gate_openings)

        # Compare with 0 loss
        flow_with_loss = model.flow_rates[0]

        model.trash_rack_loss = 0.0
        model.step(model.gate_openings)
        flow_no_loss = model.flow_rates[0]

        print(f"DEBUG: Flow Loss={flow_with_loss:.2f}, NoLoss={flow_no_loss:.2f}")
        self.assertTrue(flow_with_loss < flow_no_loss)

    def test_small_opening_instability(self):
        """Test S2.3: Small opening creates flapping vibration"""
        model = TangheSiphonModel()

        # Set small opening (e.g. 0.05m)
        model.gate_openings = np.array([0.05, 0.05, 0.05])

        # Run step
        model.step(model.gate_openings)

        # Should trigger flapping amplitude (0.2g)
        # Normal flow vib would be tiny at this low speed
        vib = model.vibration_accel[0]
        print(f"DEBUG: Flapping Vib={vib:.4f}")

        self.assertTrue(vib > 0.15, "Should have high flapping vibration")

    def test_leakage(self):
        """Test S4.3: Seal Leakage when closed"""
        model = TangheSiphonModel()

        # Gates closed
        model.gate_openings = np.array([0.0, 0.0, 0.0])
        model.inject_fault(0, 'leakage')

        model.step(model.gate_openings)

        # Gate 0 should have flow despite 0 opening
        self.assertTrue(model.flow_rates[0] > 0.1, "Leakage should exist")
        self.assertTrue(model.flow_rates[1] < 0.001, "Normal gate should handle 0 flow")

if __name__ == '__main__':
    unittest.main()
