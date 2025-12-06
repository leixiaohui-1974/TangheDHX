import numpy as np

class LocalController:
    """
    Local Controller (PLC level).
    Handles fast-loop logic: "Fast Transit" (Switching Control), dithering, and safety jumps.
    """

    def __init__(self, model):
        self.model = model
        self.dithering_active = [False, False, False]
        self.dithering_phase = [0.0, 0.0, 0.0]

        # Resonance Jump Logic
        self.resonance_timer = [0.0, 0.0, 0.0]
        self.JUMP_TIMEOUT = 2.0 # Seconds before forced jump

        # Fast Transit Logic
        # Define Resonance Zone in terms of Opening (approx)
        # Based on calibration: v=2.6 -> f=2.8.
        # v ~ 1.58 * e.
        # e_res = 2.6 / 1.58 = 1.64m.
        # Let's define zone as +/- 10% around 1.64m: [1.45, 1.85]
        self.RES_ZONE_LOW = 1.45
        self.RES_ZONE_HIGH = 1.85

        # Override Flags
        self.fast_transit_active = [False, False, False]

    def update(self, mpc_targets, dt):
        """
        Adjusts MPC targets based on local conditions.
        Returns final actuator commands.
        """
        final_cmds = np.copy(mpc_targets)
        current_openings = self.model.gate_openings

        for i in range(3):
            # --- Fast Transit (Switching Control) ---
            # If we need to cross the resonance zone, do it at MAX speed.
            # In simulation, "speed" is limited in physics step.
            # However, the 'target' dictates the direction.
            # If we are in or about to cross zone, we KEEP target far away.
            # But the key is preventing the MPC from putting an INTERMEDIATE target inside the zone.
            # (Which MPC shouldn't do anyway due to cost function).
            # The issue is physical travel time.

            # If current is below zone and target is above (or vice versa):
            curr = current_openings[i]
            targ = mpc_targets[i]

            crossing_up = (curr < self.RES_ZONE_LOW and targ > self.RES_ZONE_HIGH)
            crossing_down = (curr > self.RES_ZONE_HIGH and targ < self.RES_ZONE_LOW)
            in_zone = (self.RES_ZONE_LOW <= curr <= self.RES_ZONE_HIGH)

            if crossing_up or crossing_down or in_zone:
                self.fast_transit_active[i] = True
            else:
                self.fast_transit_active[i] = False

            # If fast transit is active, we don't change the TARGET (MPC set it correctly).
            # But we might disable dithering or other fine controls.
            # And in the Real World, we would send a "Speed=MAX" command to VFD.
            # In this Sim, we can't change speed here easily without hacking Physics.
            # But we CAN suppress "Safety Jump" if we are already transiting intentionally.

            if self.fast_transit_active[i]:
                # Suppress other logic, just go.
                pass
            else:
                # --- Resonance Safety Jump (Strategy 2 Refined) ---
                # Only if NOT intentionally transiting.
                accel = self.model.vibration_accel[i]
                if accel > 0.2:
                    self.resonance_timer[i] += dt
                else:
                    self.resonance_timer[i] = 0.0

                if self.resonance_timer[i] > self.JUMP_TIMEOUT:
                    final_cmds[i] += 0.2
                    self.resonance_timer[i] = 0.0

                # --- Dithering ---
                if accel > 0.15:
                    self.dithering_active[i] = True
                elif accel < 0.05:
                    self.dithering_active[i] = False

                if self.dithering_active[i]:
                    amp = 0.02
                    omega = 2 * np.pi * 0.5
                    self.dithering_phase[i] += omega * dt
                    perturbation = amp * np.sin(self.dithering_phase[i])
                    final_cmds[i] += perturbation

        return final_cmds

    def get_speed_multiplier(self, gate_index):
        """
        Called by Physics engine (if hooked up) to boost speed.
        """
        if self.fast_transit_active[gate_index]:
            return 5.0 # 5x speed boost for Fast Transit
        return 1.0
