import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

from src.features.BearingData import BearingData

def get_folders(root_dir):

    folders = []
    for item in os.listdir(root_dir):
        full_path = os.path.join(root_dir, item)
        if os.path.isdir(full_path):
            folders.append(full_path)
    return folders

def split_folders(root_dir, text_size = 0.2, random_state = 42):
    folders = get_folders(root_dir)

    labels = []
    for folder in folders:
        name = os.path.basename(folder)
        labels.append(0 if name.startswith("K00") else 1)

    train_folders, test_folders = train_test_split(
        folders,
        test_size = text_size, 
        random_state = random_state,
        shuffle = True,
        stratify = labels
    )
    return train_folders, test_folders

def create_windows(
        folder_list,
        window_size = 2048,
        overlap = 0.5, 
        ch_name = "vibration_1"
):
    #print("Creating Window...\n")
    stride = int(window_size * (1 - overlap))
    X = []
    y = []

    for folder in folder_list:
        folder_name = os.path.basename(folder)
        if folder_name.startswith("K00"):
            label = 0 # Healthy
            print("This bearing is healthy")
        else:
            label = 1 # Faulty
            print("This bearing is faulty")
        
        for file in os.listdir(folder):
            if not file.endswith(".mat"):
                continue
            
            full_path = os.path.join(folder, file)
            #print("Opening file: ", full_path)

            try:
                bd = BearingData(full_path)
            except Exception as e:
                #print("Failed file: ", full_path)
                #print("Error: ", e)
                continue

            if ch_name not in bd.signals:
                continue
            signal = np.array(bd.signals[ch_name])
            
            for start in range(0, len(signal) - window_size, stride):
                window = signal[start:start + window_size]
                std = np.std(window)
                if std == 0:
                    continue
                window = (window - np.mean(window)) / std 
                X.append(window)
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    X = X[..., np.newaxis] 
    return X, y

def load_dataset(root_dir, window_size = 2048, overlap = 0.5):
    train_folders, test_folders =  split_folders(root_dir)
    X_train, y_train = create_windows(train_folders, window_size, overlap)
    X_test, y_test = create_windows(test_folders, window_size, overlap)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_dataset("dataset")
    print("Train shape", X_train.shape)
    print("Test shpae", X_test.shape)