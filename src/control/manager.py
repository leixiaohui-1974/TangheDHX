import time
import numpy as np

class ScenarioManager:
    """
    Orchestrates the simulation scenarios.
    """

    def __init__(self, model):
        self.model = model
        self.active_scenario = None
        self.scenario_start_time = 0

    def set_scenario(self, scenario_id):
        self.active_scenario = scenario_id
        self.scenario_start_time = self.model.time
        print(f"Scenario {scenario_id} started at {self.model.time:.2f}s")

        # Reset faults
        self.model.gate_stuck = [False, False, False]

        # Apply Scenario Initial Conditions
        if scenario_id == 'S1.1':
            # Normal Operation
            self.model.head_upstream = 10.0

        elif scenario_id == 'S2.1':
            # Resonance Zone Crossing
            # We want to force the system to traverse the resonance zone.
            self.model.head_upstream = 10.0

        elif scenario_id == 'S4.1':
            # Trash Rack Blockage (High Head Loss / Low Flow for same opening)
            # We can model this by reducing effective head or Cd
            # For simplicity, we just lower upstream head to simulate loss
            self.model.head_upstream = 8.5

        elif scenario_id == 'S4.2':
            # Gate Stuck
            # We will trigger the stickiness after a few seconds in update()
            pass

    def update(self):
        """
        Dynamic scenario updates (events happening over time).
        """
        elapsed = self.model.time - self.scenario_start_time

        if self.active_scenario == 'S4.2':
            # At T+10s, Gate 2 gets stuck
            if elapsed > 10.0 and not self.model.gate_stuck[1]:
                print("Injecting Fault: Gate 2 Stuck!")
                self.model.inject_fault(1, 'stuck')
