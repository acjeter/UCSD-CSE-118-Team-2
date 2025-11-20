import os
import numpy as np
import cv2
import kagglehub
from utils import normalize_landmarks, extract_landmarks_from_image

# Download latest version
dataset_path = kagglehub.dataset_download("grassknoted/asl-alphabet")

print("Path to dataset files:", dataset_path)


def rotate_image(image, angle):
    """Rotate image by a given angle (in degrees) around the center."""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated


def process_dataset():
    """Process the ASL alphabet dataset and extract hand landmarks."""
    full_dataset_path = os.path.join(dataset_path, "asl_alphabet_train", "asl_alphabet_train")
    
    all_coords = []
    all_labels = []

    # Only keep the 26 letters A–Z
    valid_labels = set(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

    for label in sorted(os.listdir(full_dataset_path)):
        # Skip unwanted folders like "del", "space", "nothing"
        if label not in valid_labels:
            print(f"Skipping non-letter class: {label}")
            continue

        label_dir = os.path.join(full_dataset_path, label)
        if not os.path.isdir(label_dir):
            continue

        print(f"Processing label: {label}")

        count = 0
        for filename in os.listdir(label_dir):
            if count >= 50:
                break
            
            img_path = os.path.join(label_dir, filename)
            
            # Read the image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Apply random rotation between -20° and +20°
            random_angle = np.random.uniform(-20, 20)
            rotated_img = rotate_image(img, random_angle)
            
            # Extract landmarks from the rotated image
            coords = extract_landmarks_from_image(rotated_img)
            if coords is None:
                # Mediapipe couldn't detect a hand, skip silently
                continue
            
            # Normalize landmarks (includes roll normalization)
            coords = normalize_landmarks(coords)
            flat = coords.flatten()  # 21 landmarks × 3 coords = 63 values
            all_coords.append(flat)
            all_labels.append(label)

            count += 1

    # Save for later training
    os.makedirs("processed_data", exist_ok=True)
    np.save("processed_data/debug_coords.npy", all_coords)
    np.save("processed_data/debug_labels.npy", all_labels)

    print("Done! Extracted:", len(all_coords), "samples")


if __name__ == "__main__":
    process_dataset()
