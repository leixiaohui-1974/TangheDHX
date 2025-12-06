import numpy as np

class GateController:
    """
    Interface to the gate actuators.
    """
    def __init__(self, model):
        self.model = model
        self.target_openings = np.copy(model.gate_openings)

    def set_opening(self, gate_index, value):
        self.target_openings[gate_index] = value

    def get_target_openings(self):
        return self.target_openings
