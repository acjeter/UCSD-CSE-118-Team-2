import numpy as np
import cv2
import mediapipe as mp

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
    Apply in-plane roll normalization to 21×3 landmark coords.
    Uses wrist → palm-center as the orientation vector.
    """
    coords = coords.copy()

    # ---- 1. Compute palm center from MCP joints ----
    wrist = coords[0][:2]  # (x, y)

    mcp_indices = [5, 9, 13, 17]  # index, middle, ring, pinky MCPs
    mcp_points = coords[mcp_indices, :2]  # shape (4, 2)
    palm_center = mcp_points.mean(axis=0)  # average (x, y)

    # Vector from wrist → palm center
    v = palm_center - wrist
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        # Degenerate case: don't rotate at all
        return coords
    v /= norm

    # ---- 2. Compute angle of this vector relative to vertical ----
    # We want v to point straight UP (0, 1).
    angle = np.arctan2(v[0], v[1])  # (x, y) swapped to measure from vertical

    # ---- 3. Build rotation matrix to undo roll ----
    cos_t = np.cos(-angle)
    sin_t = np.sin(-angle)
    R = np.array([[cos_t, -sin_t],
                  [sin_t,  cos_t]])

    # ---- 4. Rotate all (x,y) around the wrist ----
    for i in range(21):
        xy = coords[i][:2] - wrist
        xy_rot = R @ xy
        coords[i][0], coords[i][1] = xy_rot[0], xy_rot[1]

    return coords


def normalize_landmarks(coords_21x3):
    """
    coords_21x3: numpy array of shape (21,3)
    returns: normalized numpy array of shape (21,3)
    """

    coords = coords_21x3.copy()

    # Step 1: apply roll normalization
    # coords = normalize_roll(coords)

    # Step 2: translate wrist to origin
    wrist = coords[0]                # wrist is landmark 0
    coords -= wrist                  # subtract from all landmarks

    # Step 3: compute hand size using wrist → middle_finger_tip (landmark 12)
    middle_tip = coords[12]          # after translation, it's a vector from wrist
    hand_size = np.linalg.norm(middle_tip)

    # avoid divide-by-zero edge cases
    if hand_size < 1e-6:
        hand_size = 1e-6

    # Step 4: scale the entire hand by hand_size
    coords /= hand_size

    return coords
