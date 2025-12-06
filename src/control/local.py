import numpy as np

class LocalController:
    """
    Local Controller (PLC level).
    Handles fast-loop logic: pass-through and dithering.
    """

    def __init__(self, model):
        self.model = model
        # Resonance frequency range to avoid (in terms of Opening/Velocity?)
        # Ideally, we monitor vibration.
        self.dithering_active = [False, False, False]
        self.dithering_phase = [0.0, 0.0, 0.0]

    def update(self, mpc_targets, dt):
        """
        Adjusts MPC targets based on local conditions.
        Returns final actuator commands.
        """
        final_cmds = np.copy(mpc_targets)

        # 1. Pass-through Logic (Active Jumping)
        # If we are commanding a change that crosses the resonance zone, do it FAST.
        # But `physics.py` limits speed. The PLC basically just sets the target.
        # The 'Logic' is more about NOT STOPPING in the zone.
        # Since MPC already avoids the zone in steady state, the transition is the issue.
        # Here we just pass the target.
        # But if MPC fails and asks for a value inside the zone (due to constraints),
        # Local controller could override?
        # For now, we trust MPC's "Blacklist".

        # 2. Dithering (Micro-perturbation)
        # If vibration is high, add sine wave to opening.

        for i in range(3):
            accel = self.model.vibration_accel[i]

            # Threshold for Dithering trigger
            if accel > 0.15: # 0.15g is quite high
                self.dithering_active[i] = True
            elif accel < 0.05:
                self.dithering_active[i] = False

            if self.dithering_active[i]:
                # Add +/- 2cm sine wave at 0.5 Hz
                amp = 0.02
                omega = 2 * np.pi * 0.5
                self.dithering_phase[i] += omega * dt
                perturbation = amp * np.sin(self.dithering_phase[i])
                final_cmds[i] += perturbation

        return final_cmds
