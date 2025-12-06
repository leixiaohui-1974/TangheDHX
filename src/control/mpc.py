import numpy as np
from scipy.optimize import minimize

class SpectralMPC:
    """
    Spectral Model Predictive Control (Spectral-MPC).
    Optimizes gate openings to meet flow demand while avoiding resonance frequencies.
    """

    def __init__(self, model):
        self.model = model
        self.width = model.width
        self.k_freq = model.k_freq
        self.f_struct = model.f_struct

        # Weights
        self.alpha = 1.0   # Flow tracking
        self.beta = 0.1    # Action penalty (minimize movement)
        self.gamma = 10.0  # Spectral avoidance

        self.last_target_openings = np.array([0.0, 0.0, 0.0])

    def get_target_openings(self, target_flow, current_openings, current_head_diff):
        """
        Solve optimization problem to find optimal openings.
        """

        # Initial guess: previous target
        x0 = current_openings

        # Constraints: 0 <= e <= 5.0
        bounds = [(0.0, 5.0) for _ in range(3)]

        # Physics helper for the optimizer (replicates physics.py logic essentially)
        # We assume the controller has an internal model of the physics
        def predict_physics(openings):
            # Hydraulics
            Cd = 0.7
            g = 9.81
            A_eff = 16.6

            flows = []
            velocities = []
            for e in openings:
                q = Cd * self.width * e * np.sqrt(2 * g * current_head_diff)
                v = q / A_eff
                flows.append(q)
                velocities.append(v)

            return np.array(flows), np.array(velocities)

        def cost_function(x):
            flows, velocities = predict_physics(x)

            # 1. Flow Tracking
            total_flow = np.sum(flows)
            J_flow = self.alpha * (total_flow - target_flow)**2

            # 2. Action Penalty (vs current position or previous target?
            # Usually vs current position to minimize movement effort)
            J_action = self.beta * np.sum((x - current_openings)**2)

            # 3. Spectral Potential
            J_spectral = 0
            epsilon = 0.1
            for v in velocities:
                # Predicted Shedding Freq
                if v < 0.1:
                    fs = 0
                else:
                    fs = self.k_freq * v

                # Potential Function: 1 / (dist^2 + eps)
                # Distance to resonance
                dist = abs(fs - self.f_struct)
                psi = 1.0 / (dist**2 + epsilon)
                J_spectral += psi

            J_spectral = self.gamma * J_spectral

            return J_flow + J_action + J_spectral

        # Run Optimization
        res = minimize(cost_function, x0, bounds=bounds, method='SLSQP')

        if res.success:
            self.last_target_openings = res.x
            return res.x
        else:
            # Fallback
            return current_openings
