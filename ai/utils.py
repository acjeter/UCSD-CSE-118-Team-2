import numpy as np
import cv2
import mediapipe as mp
from torch.utils.data import Dataset

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)


def extract_landmarks_from_image(image):
    """Run mediapipe on an image array and return raw (x,y,z) coords"""
    if image is None:
        return None

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None  # no hand detected

    # Grab the first detected hand
    hand_landmarks = results.multi_hand_landmarks[0]

    coords = []
    for lm in hand_landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z])

    return np.array(coords)  # shape (21, 3)


def extract_landmarks(image_path):
    """Run mediapipe on an image path and return raw (x,y,z) coords"""
    img = cv2.imread(image_path)
    return extract_landmarks_from_image(img)

import numpy as np

def normalize_roll(coords):
    """
    Apply in-plane roll normalization.
    Handles both MediaPipe (21 joints) and OpenXR (26 joints).
    """
    coords = coords.copy()
    num_joints = coords.shape[0]

    # Define indices for 22-joint schema (Palm + Wrist + 4*5)
    wrist_idx = 1
    mcp_indices = [3, 7, 11, 15, 19]  # ThumbProximal, IndexProximal, MiddleProximal, RingProximal, LittleProximal

    # ---- 1. Compute palm center from MCP joints ----
    wrist = coords[wrist_idx][:2]  # (x, y)

    mcp_points = coords[mcp_indices, :2]  # shape (4 or 5, 2)
    palm_center = mcp_points.mean(axis=0)  # average (x, y)

    # Vector from wrist → palm center
    v = palm_center - wrist
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return coords
    v /= norm

    # ---- 2. Compute angle of this vector relative to vertical ----
    angle = np.arctan2(v[0], v[1])

    # ---- 3. Build rotation matrix to undo roll ----
    cos_t = np.cos(-angle)
    sin_t = np.sin(-angle)
    R = np.array([[cos_t, -sin_t],
                  [sin_t,  cos_t]])

    # ---- 4. Rotate all (x,y) around the wrist ----
    for i in range(num_joints):
        xy = coords[i][:2] - wrist
        xy_rot = R @ xy
        coords[i][0], coords[i][1] = xy_rot[0], xy_rot[1]

    return coords


def normalize_landmarks(coords):
    """
    coords: numpy array of shape (N, 3) where N is 21 or 26
    returns: normalized numpy array of same shape
    """
    coords = coords.copy()
    num_joints = coords.shape[0]

    # Step 1: apply roll normalization
    coords = normalize_roll(coords)

    # Define indices
    # Define indices for 22-joint schema
    wrist_idx = 1
    middle_tip_idx = 13  # MiddleTip

    # Step 2: translate wrist to origin
    wrist = coords[wrist_idx]
    coords -= wrist

    # Step 3: compute hand size using wrist → middle_finger_tip
    middle_tip = coords[middle_tip_idx]
    hand_size = np.linalg.norm(middle_tip)

    # avoid divide-by-zero edge cases
    if hand_size < 1e-6:
        hand_size = 1e-6

    # Step 4: scale the entire hand by hand_size
    coords /= hand_size

    return coords
    return coords


class ASLDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
