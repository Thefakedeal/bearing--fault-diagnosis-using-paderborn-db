import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

class SignalProcessor:
    def __init__(self, signal=np.array([]), fs=64000):
        """
        :param signal: 1D numpy array (the raw data)
        :param fs: Sampling frequency (PU dataset vibration is 64kHz)
        """
        self.signal = signal
        self.fs = fs

    def get_time_features(self):
        """Calculates statistical 'Condition Indicators' in the time domain."""
        s = self.signal
        rms = np.sqrt(np.mean(s**2))
        peak = np.max(np.abs(s))
        
        return {
            'rms':          rms,
            'kurtosis':     kurtosis(s),
            'skewness':     skew(s),
            'crest_factor': peak / rms if rms > 0 else 0,
            'peak_to_peak': np.ptp(s),
            'std_dev':      np.std(s)
        }

    def get_all_features(self, prefix="vib"):
        """Returns a flattened dictionary with a prefix for ML labeling."""
        time_feats = self.get_time_features()
        # Add a prefix so you can distinguish between vib_rms and current_rms
        return {f"{prefix}_{k}": v for k, v in time_feats.items()}