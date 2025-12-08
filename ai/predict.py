import sys
import numpy as np
import torch
import torch.nn as nn

from utils import extract_landmarks, normalize_landmarks

from model import ASLModel

def predict(image_path):
    coords = extract_landmarks(image_path)

    if coords is None:
        print("No hand detected in image.")
        return

    coords = normalize_landmarks(coords)
    x = torch.tensor(coords.flatten(), dtype=torch.float32).unsqueeze(0)

    classes = np.load("label_classes.npy")

    model = ASLModel(num_classes=len(classes))
    model.load_state_dict(torch.load("asl_model.pth", map_location='cpu'))
    model.eval()

    with torch.no_grad():
        output = model(x)
        pred_id = torch.argmax(output, dim=1).item()

    print("Predicted:", classes[pred_id])


if __name__ == "__main__":
    predict(sys.argv[1])
