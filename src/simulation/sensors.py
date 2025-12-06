import numpy as np

class ADCPSensor:
    """
    Acoustic Doppler Current Profiler (ADCP).
    Measures local flow velocity.
    """
    def __init__(self, model, gate_index):
        self.model = model
        self.gate_index = gate_index
        self.noise_std = 0.05 # m/s

    def read(self):
        """Returns measured velocity with noise."""
        true_v = self.model.velocities[self.gate_index]
        return true_v + np.random.normal(0, self.noise_std)

class VibrationSensor:
    """
    MEMS Vibration Sensor.
    Measures acceleration and provides spectral analysis.
    """
    def __init__(self, model, gate_index):
        self.model = model
        self.gate_index = gate_index
        self.noise_std = 0.005 # g

    def read_accel(self):
        """Returns peak acceleration (g)."""
        true_a = self.model.vibration_accel[self.gate_index]
        return max(0, true_a + np.random.normal(0, self.noise_std))

    def get_spectrum_peak(self):
        """Returns the dominant frequency (Hz)."""
        # In a real system, this would do FFT on a time series.
        # Here we cheat and look at the physics model's vortex frequency.
        true_f = self.model.vortex_freqs[self.gate_index]
        if true_f == 0:
            return 0.0
        return true_f + np.random.normal(0, 0.05) # Hz noise
