import numpy as np
from scipy.optimize import minimize

class SpectralMPC:
    """
    Spectral Model Predictive Control (Spectral-MPC).
    Optimizes gate openings to meet flow demand while avoiding resonance frequencies
    and enforcing de-coherence (asymmetry) to prevent structural shaking.
    """

    def __init__(self, model):
        self.model = model
        self.width = model.width
        self.k_freq = model.k_freq
        self.f_struct = model.f_struct

        # Weights
        self.alpha = 1.0   # Flow tracking
        self.beta = 0.1    # Action penalty (minimize movement)
        self.gamma = 10.0  # Spectral avoidance (Resonance)
        self.delta = 5.0   # De-coherence (Frequency spread)

        self.last_target_openings = np.array([0.0, 0.0, 0.0])

    def get_target_openings(self, target_flow, current_openings, current_head_diff):
        """
        Solve optimization problem to find optimal openings.
        """

        # Initial guess: Use previous target to encourage stability (Sticky Control)
        # unless it's way off (e.g. start)
        x0 = self.last_target_openings
        if np.sum(x0) == 0:
            x0 = current_openings

        # Constraints: 0 <= e <= 5.0
        bounds = [(0.0, 5.0) for _ in range(3)]

        # Physics helper for the optimizer
        def predict_physics(openings):
            # Hydraulics
            Cd = 0.7
            g = 9.81
            A_eff = 16.6

            flows = []
            velocities = []
            frequencies = []

            for e in openings:
                q = Cd * self.width * e * np.sqrt(2 * g * current_head_diff)
                v = q / A_eff

                if v < 0.1:
                    fs = 0.0
                else:
                    fs = self.k_freq * v

                flows.append(q)
                velocities.append(v)
                frequencies.append(fs)

            return np.array(flows), np.array(velocities), np.array(frequencies)

        def cost_function(x):
            flows, velocities, freqs = predict_physics(x)

            # 1. Flow Tracking
            total_flow = np.sum(flows)
            J_flow = self.alpha * (total_flow - target_flow)**2

            # 2. Action Penalty
            J_action = self.beta * np.sum((x - current_openings)**2)

            # 3. Spectral Potential (Avoid Resonance)
            J_spectral = 0
            epsilon = 0.1
            for fs in freqs:
                dist = abs(fs - self.f_struct)
                psi = 1.0 / (dist**2 + epsilon)
                J_spectral += psi
            J_spectral = self.gamma * J_spectral

            # 4. De-coherence (Maximize spread between frequencies)
            # Strategy 1 from report: "Asymmetric De-coherence Scheduling"
            # We want to PENALIZE if frequencies are close to each other.
            # J_decoh = Sum( 1 / (|fi - fj|^2 + eps) ) for i != j
            J_decoh = 0
            for i in range(3):
                for j in range(i + 1, 3):
                    diff = abs(freqs[i] - freqs[j])
                    # If diff is small, penalty is large.
                    # We only care if they are "significant" frequencies (i.e., flow > 0)
                    if freqs[i] > 0.5 and freqs[j] > 0.5:
                         J_decoh += 1.0 / (diff**2 + 0.05)

            J_decoh = self.delta * J_decoh

            return J_flow + J_action + J_spectral + J_decoh

        # Run Optimization
        # Use SLQSP for constrained optimization
        res = minimize(cost_function, x0, bounds=bounds, method='SLSQP')

        if res.success:
            self.last_target_openings = res.x
            return res.x
        else:
            return current_openings
