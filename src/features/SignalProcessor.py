import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

class SignalProcessor:
    def __init__(self, signal, fs=64000):
        """
        Process vibration signals for feature extraction.

        :param signal: 1D numpy array, raw vibration signal
        :param fs: Sampling frequency (default: 64 kHz for Paderborn dataset)
        """
        self.signal = np.array(signal)
        self.fs = fs

    # -----------------------------
    # Time-domain features
    # -----------------------------
    def get_time_features(self, prefix="time"):
        s = self.signal
        rms = np.sqrt(np.mean(s**2))
        peak = np.max(np.abs(s))

        features = {
            f"{prefix}_rms": rms,
            f"{prefix}_kurtosis": kurtosis(s),
            f"{prefix}_skewness": skew(s),
            f"{prefix}_crest_factor": peak / rms if rms > 0 else 0,
            f"{prefix}_peak_to_peak": np.ptp(s),
            f"{prefix}_std_dev": np.std(s)
        }
        return features

    # -----------------------------
    # Frequency-domain features
    # -----------------------------
    def get_frequency_features(self, prefix="freq"):
        s = self.signal
        n = len(s)
        fft_vals = np.fft.fft(s)
        fft_vals = np.abs(fft_vals[:n//2])  # positive frequency magnitudes only

        peak = np.max(fft_vals)
        rms = np.sqrt(np.mean(fft_vals**2))

        features = {
            f"{prefix}_rms": rms,
            f"{prefix}_kurtosis": kurtosis(fft_vals),
            f"{prefix}_skewness": skew(fft_vals),
            f"{prefix}_crest_factor": peak / rms if rms > 0 else 0,
            f"{prefix}_peak_to_peak": np.ptp(fft_vals),
            f"{prefix}_std_dev": np.std(fft_vals)
        }
        return features

    # -----------------------------
    # Combine all features
    # -----------------------------
    def get_all_features(self, prefix="vibration"):
        """
        Returns all features in a single dictionary ready for DataFrame.
        """
        features = self.get_time_features()
        features.update(self.get_frequency_features())
        features = {f"{prefix}_{k}": v for k, v in features.items()}
        return features
