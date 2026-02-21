import os
import numpy as np
from tqdm import tqdm
from src.features.BearingData import BearingData

def load_all_matfiles(root_dir="../dataset/paderborn-db", window_size = 2048, overlap = 0.5, channels=None):
    """
    Recursively load all .mat files in dataset and extract features and raw signals.
    """

    stride = int(window_size * (1 - overlap))
    X = []
    y = []
    skipped_file_count = 0

    for subdir, _, files in os.walk(root_dir):
        mat_files = [f for f in files if f.endswith('.mat')]
        for f in tqdm(mat_files, desc=f"Processing {subdir}"):
            full_path = os.path.join(subdir, f)
            try:
                bd = BearingData(full_path)
                label = bd.metadata['label']

                for ch_name, signal in bd.signals.items():
                    if channels is not None and ch_name not in channels:
                        continue
                        
                    signal = np.array(signal)

                    # Segmentation
                    for start in range(0, len(signal) - window_size, stride):
                        window = signal[start: start + window_size]
                        std = np.std(window)
                        if std > 0:
                            window = (window - np.mean(window)) / std
                        else:
                            continue
                            
                        X.append(window)
                        y.append(label)
                
            except Exception as e:
                print(f"Error loading {full_path}: {e}")
                skipped_file_count += 1
                continue
    
    X = np.array(X)
    y = np.array(y)
    X = X[..., np.newaxis]
    print(f"Skipped {skipped_file_count} files.")
    print(f"Final shape: {X.shape}")
    return X, y

if __name__ == "__main__":
    df = load_all_matfiles()
    print(df.head())