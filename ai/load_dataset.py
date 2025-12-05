import os
import numpy as np
import pandas as pd
from utils import normalize_landmarks

def process_dataset():
    """Process the ASL alphabet dataset from CSV files and extract hand landmarks."""
    dataset_dir = "ASLTrainingData"
    
    all_coords = []
    all_labels = []

    # Get all CSV files
    csv_files = [f for f in os.listdir(dataset_dir) if f.endswith(".csv") and f.startswith("ASL_Data_")]
    
    if not csv_files:
        print(f"No CSV files found in {dataset_dir}")
        return

    print(f"Found {len(csv_files)} CSV files.")

    for filename in sorted(csv_files):
        file_path = os.path.join(dataset_dir, filename)
        print(f"Processing {filename}...")
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        # Expected columns based on the CSV header provided by user
        # Time,Label,Handedness,Palm_X,Palm_Y,Palm_Z,Wrist_X,Wrist_Y,Wrist_Z,...
        
        # We need to extract X, Y, Z for all joints.
        # The joints seem to be: Palm, Wrist, ThumbMetacarpal, ThumbProximal, ThumbDistal, ThumbTip, ...
        # Total 22 joints: Palm + Wrist + 5 fingers * 4 joints/finger = 2 + 20 = 22
        
        # Let's identify coordinate columns
        coord_cols = [c for c in df.columns if c.endswith("_X") or c.endswith("_Y") or c.endswith("_Z")]
        
        # Sort columns to ensure X, Y, Z order for each joint
        # The CSV header order seems to be Joint_X, Joint_Y, Joint_Z.
        # We can just trust the order if we select them carefully.
        
        # Let's verify we have 66 coordinate columns (22 joints * 3)
        if len(coord_cols) != 66:
            print(f"Warning: Expected 66 coordinate columns, found {len(coord_cols)} in {filename}. Skipping.")
            continue

        # Extract coordinates and labels
        for index, row in df.iterrows():
            label = row['Label']
            
            # Extract coordinates as a flat array
            coords_flat = row[coord_cols].values.astype(np.float32)
            
            # Reshape to (22, 3) for normalization
            coords = coords_flat.reshape(-1, 3)
            
            # Normalize landmarks
            coords = normalize_landmarks(coords)
            
            # Flatten back to (66,)
            flat = coords.flatten()
            
            all_coords.append(flat)
            all_labels.append(label)

    # Save for training
    os.makedirs("processed_data", exist_ok=True)
    np.save("processed_data/debug_coords.npy", np.array(all_coords))
    np.save("processed_data/debug_labels.npy", np.array(all_labels))

    print("Done! Extracted:", len(all_coords), "samples")


if __name__ == "__main__":
    process_dataset()
