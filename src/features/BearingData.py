import scipy.io as sio
import numpy as np
import pandas as pd
import os

class BearingData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path).replace('.mat', '')
        
        # Load and automatically find the main key
        mat = sio.loadmat(file_path)
        keys = [k for k in mat.keys() if not k.startswith('__')]
        self.raw_struct = mat[keys[0]][0, 0]
        
        # Metadata and Signals
        self.metadata = self._parse_filename()
        self.speed_rpm = self._decode_speed(self.metadata['speed_rpm_code'])
        self.torque_nm = self._decode_torque(self.metadata['load_torque_code'])
        self.radial_force_n = self._decode_force(self.metadata['radial_force_code'])
        
        self.signals = self._extract_signals()

    def _parse_filename(self):
        parts = self.file_name.split('_')
        return {
            'speed_rpm_code': parts[0],
            'load_torque_code': parts[1],
            'radial_force_code': parts[2],
            'bearing_id': parts[3],
            'label': self._get_label(parts[3])
        }

    def _get_label(self, b_id):
        if b_id.startswith("K00"):
            return 0  # Healthy
    
        elif b_id.startswith("KA"):
            return 1  # Artificial Outer Race
        
        elif b_id.startswith("KI"):
            return 2  # Artificial Inner Race
        
        elif b_id.startswith("KB"):
            return 3  # Artificial Other Damage
        
        else:
            return -1  # Unknown / real damage (handle separately)

    def _decode_speed(self, code):
        speed_map = {
            "N15": 1500,
            "N09": 900,
            "N03": 300
        }
        return speed_map.get(code, None)


    def _decode_torque(self, code):
        torque_map = {
            "M01": 0.1,
            "M07": 0.7,
            "M15": 1.5
        }
        return torque_map.get(code, None)


    def _decode_force(self, code):
        force_map = {
            "F04": 400,
            "F10": 1000,
            "F16": 1600
        }
        return force_map.get(code, None)

    def _extract_signals(self):
        """Iterates through 'Y' to find signals and get the FULL array."""
        signals = {}
        y_data = self.raw_struct['Y']
        for i in range(y_data.shape[1]):
            item = y_data[0, i]
            name = item['Name'][0]
            
            # --- THE CRITICAL FIX ---
            # Remove [0, 0] to get the entire array of 256k points
            data = item['Data'].flatten() 
            # ------------------------
            
            signals[name] = data
        return signals

# Usage Verification
# data = BearingData('N09_M07_F10_K001_10.mat')
# vib_signal = data.signals['vibration_1']

# print(f"Signal Name: vibration_1")
# print(f"Total Points: {len(vib_signal)}") # This should now show 256608
# print(f"First 5 Points: {vib_signal[:5]}")