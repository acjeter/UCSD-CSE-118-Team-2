#THIS FILE IS FOR RUNNING RANDOM SCRIPTS FROM TERMINAL
#Dataset path - # Open the folder - open /Users/pranavsubash/.cache/kagglehub/datasets/grassknoted/asl-alphabet/versions/1/asl_alphabet_train/asl_alphabet_train/

import numpy as np
import os
import cv2
import torch
from utils import extract_landmarks, normalize_landmarks
from load_dataset import dataset_path
from model import ASLModel

# ============================================================
# CURRENT SCRIPT: Test predictions on rotated dataset images
# ============================================================

def predict_from_image(img, classes, model):
    """Run prediction on an image array (not path)"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Use mediapipe directly
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    
    results = hands.process(img_rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    # Extract landmarks
    hand_landmarks = results.multi_hand_landmarks[0]
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z])
    coords = np.array(coords)
    
    # Normalize and predict
    coords = normalize_landmarks(coords)
    x = torch.tensor(coords.flatten(), dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        output = model(x)
        pred_id = torch.argmax(output, dim=1).item()
    
    return classes[pred_id]

# Load model once
classes = np.load("label_classes.npy")
model = ASLModel(num_classes=len(classes))
model.load_state_dict(torch.load("asl_model.pth", map_location='cpu'))
model.eval()

# Get dataset path
full_dataset_path = os.path.join(dataset_path, "asl_alphabet_train", "asl_alphabet_train")

# Get all valid letters
valid_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
valid_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

# Collect all images from all letters
all_images = []
for letter in valid_labels:
    letter_dir = os.path.join(full_dataset_path, letter)
    if not os.path.isdir(letter_dir):
        continue
    
    images = [f for f in os.listdir(letter_dir) if any(f.endswith(ext) for ext in valid_extensions)]
    for img in images[:5]:  # Take first 5 from each letter
        all_images.append((letter, os.path.join(letter_dir, img)))

# Pick 3 random images
np.random.seed(42)  # For reproducibility
random_indices = np.random.choice(len(all_images), size=3, replace=False)
selected_images = [all_images[i] for i in random_indices]

print("="*60)
print("TESTING PREDICTIONS ON ROTATED DATASET IMAGES")
print("="*60)

rotation_angles = [0, 90, 180, 270]

for img_idx, (true_label, img_path) in enumerate(selected_images, 1):
    print(f"\n{'='*60}")
    print(f"Image {img_idx}: {os.path.basename(img_path)} (True label: {true_label})")
    print(f"{'='*60}")
    
    # Load original image
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load image!")
        continue
    
    for angle in rotation_angles:
        # Rotate image
        if angle == 0:
            rotated = img
        elif angle == 90:
            rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Predict
        predicted = predict_from_image(rotated, classes, model)
        
        if predicted is None:
            status = "✗"
            result = "No hand detected"
        else:
            status = "✓" if predicted == true_label else "✗"
            result = f"Predicted: {predicted}"
        
        print(f"  {angle:3d}° rotation: {status} {result}")

print("\n" + "="*60)
print("DONE!")
print("="*60)


# ============================================================
# OLD SCRIPTS (COMMENTED OUT FOR FUTURE USE)
# ============================================================

# # -------- Script 1: Compare coordinates between your image and training set --------
# # Get your image coordinates
# print("="*60)
# print("EXTRACTING COORDINATES FROM YOUR IMAGE (data3.jpg)")
# print("="*60)
# your_coords = extract_landmarks("/Users/pranavsubash/Downloads/data3.jpg")
# if your_coords is None:
#     print("No hand detected in your image!")
# else:
#     print("\nRaw coordinates (21 landmarks x 3 coords):")
#     print(your_coords)
#     print(f"\nShape: {your_coords.shape}")
#     
#     your_coords_normalized = normalize_landmarks(your_coords)
#     print("\nNormalized coordinates:")
#     print(your_coords_normalized)
#     
#     print("\nFlattened (what the model sees):")
#     print(your_coords_normalized.flatten())
#     print(f"Shape: {your_coords_normalized.flatten().shape}")

# # Get training set image coordinates
# print("\n" + "="*60)
# print("EXTRACTING COORDINATES FROM TRAINING SET (Letter H)")
# print("="*60)

# # Reuse the dataset_path from load_dataset.py (already downloaded)
# full_dataset_path = os.path.join(dataset_path, "asl_alphabet_train", "asl_alphabet_train")
# h_dir = os.path.join(full_dataset_path, "H")

# # Get valid image files only
# all_files = sorted(os.listdir(h_dir))
# valid_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
# h_images = [f for f in all_files if any(f.endswith(ext) for ext in valid_extensions)]

# print(f"Found {len(h_images)} images in training set for letter H")

# # Try to find an image with a detectable hand
# training_coords = None
# training_image_name = None
# for img_file in h_images[:10]:  # Try first 10 images
#     h_image_path = os.path.join(h_dir, img_file)
#     print(f"Trying: {img_file}...", end=" ")
#     training_coords = extract_landmarks(h_image_path)
#     if training_coords is not None:
#         training_image_name = img_file
#         print("✓ Hand detected!")
#         break
#     else:
#         print("✗ No hand detected")

# if training_coords is None:
#     print("\nCouldn't find a valid training image with detectable hand!")
# else:
#     print(f"\nUsing training image: {training_image_name}")
#     print("\nRaw coordinates (21 landmarks x 3 coords):")
#     print(training_coords)
#     print(f"\nShape: {training_coords.shape}")
#     
#     training_coords_normalized = normalize_landmarks(training_coords)
#     print("\nNormalized coordinates:")
#     print(training_coords_normalized)
#     
#     print("\nFlattened (what the model sees):")
#     print(training_coords_normalized.flatten())
#     print(f"Shape: {training_coords_normalized.flatten().shape}")

# # Compare
# if your_coords is not None and training_coords is not None:
#     print("\n" + "="*60)
#     print("COMPARISON")
#     print("="*60)
#     your_flat = normalize_landmarks(your_coords).flatten()
#     training_flat = normalize_landmarks(training_coords).flatten()
#     
#     diff = your_flat - training_flat
#     print(f"Mean absolute difference: {np.mean(np.abs(diff)):.6f}")
#     print(f"Max absolute difference: {np.max(np.abs(diff)):.6f}")
#     print(f"Euclidean distance: {np.linalg.norm(diff):.6f}")


