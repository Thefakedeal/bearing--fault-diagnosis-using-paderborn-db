import os
import pandas as pd
from tqdm import tqdm
from src.features.BearingData import BearingData
from src.features.SignalProcessor import SignalProcessor

def load_all_matfiles(root_dir="../dataset/paderborn-db",channels=None,signal_processor=SignalProcessor):
    """
    Recursively load all .mat files in dataset and extract features and signals.
    """
    all_data = []
    skipped_file_count = 0
    for subdir, _, files in os.walk(root_dir):
        mat_files = [f for f in files if f.endswith('.mat')]
        for f in tqdm(mat_files, desc=f"Processing {subdir}"):
            full_path = os.path.join(subdir, f)
            try:
                bd = BearingData(full_path)
                
                # Metadata
                row = {}
                row['bearing_id'] = bd.metadata['bearing_id']
                row['torque_nm'] = bd.torque_nm
                row['speed_rpm'] = bd.speed_rpm
                row['radial_force_n'] = bd.radial_force_n
                row['label'] = bd.metadata['label']

                # Time-domain features per channel
                for ch_name, signal in bd.signals.items():
                    if channels is not None and ch_name not in channels:
                        continue
                    sp = signal_processor(signal)
                    feats = sp.get_all_features(prefix=ch_name)
                    row.update(feats)

                all_data.append(row)

            except Exception as e:
                print(f"Error loading {full_path}: {e}")
                skipped_file_count += 1
                continue
    print(f"Skipped {skipped_file_count} files.")
    df = pd.DataFrame(all_data)
    return df

if __name__ == "__main__":
    df = load_all_matfiles()
    print(df.head())