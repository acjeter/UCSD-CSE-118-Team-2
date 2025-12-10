import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from model import ASLModel

# Resolve paths relative to this file so the script
# works regardless of the current working directory.
BASE_DIR = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")

# Load the validation data
X_val = torch.tensor(
    np.load(os.path.join(PROCESSED_DIR, "X_val.npy")),
    dtype=torch.float32,
)
y_val = torch.tensor(
    np.load(os.path.join(PROCESSED_DIR, "y_val.npy")),
    dtype=torch.long,
)

# Load the label classes
classes = np.load(os.path.join(BASE_DIR, "label_classes.npy"))

# Load the trained model (use the best validation checkpoint)
model = ASLModel()
model.load_state_dict(
    torch.load(os.path.join(BASE_DIR, "asl_model_best.pth"), map_location="cpu")
)
model.eval()

print("="*60)
print("VALIDATION SET EVALUATION")
print("="*60)
print(f"Validation set size: {len(X_val)} samples")
print(f"Number of classes: {len(classes)}")
print(f"Classes: {', '.join(classes)}")
print()

# Run predictions
all_predictions = []
all_true_labels = []

with torch.no_grad():
    for i in range(len(X_val)):
        x = X_val[i].unsqueeze(0)  # Add batch dimension
        y = y_val[i].item()
        
        output = model(x)
        pred_id = torch.argmax(output, dim=1).item()
        
        all_predictions.append(pred_id)
        all_true_labels.append(y)

# Convert to numpy arrays
all_predictions = np.array(all_predictions)
all_true_labels = np.array(all_true_labels)

# Calculate overall accuracy
overall_accuracy = accuracy_score(all_true_labels, all_predictions)
print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
print()

# Per-class accuracy
print("="*60)
print("PER-CLASS ACCURACY")
print("="*60)
per_class_correct = {}
per_class_total = {}

for true_label, pred_label in zip(all_true_labels, all_predictions):
    class_name = classes[true_label]
    
    if class_name not in per_class_total:
        per_class_total[class_name] = 0
        per_class_correct[class_name] = 0
    
    per_class_total[class_name] += 1
    if true_label == pred_label:
        per_class_correct[class_name] += 1

# Sort by class name
for class_name in sorted(classes):
    if class_name in per_class_total:
        accuracy = per_class_correct[class_name] / per_class_total[class_name]
        correct = per_class_correct[class_name]
        total = per_class_total[class_name]
        print(f"{class_name}: {accuracy:.3f} ({correct}/{total})")

print()

# Confusion matrix
print("="*60)
print("CONFUSION MATRIX")
print("="*60)
cm = confusion_matrix(all_true_labels, all_predictions)

# Find most confused pairs
print("\nMost Common Misclassifications:")
print("-" * 60)
misclassifications = []
for i in range(len(classes)):
    for j in range(len(classes)):
        if i != j and cm[i][j] > 0:
            misclassifications.append((cm[i][j], classes[i], classes[j]))

# Sort by count (descending)
misclassifications.sort(reverse=True)

# Show top 10
for count, true_class, pred_class in misclassifications[:10]:
    print(f"{true_class} → {pred_class}: {count} times")

print()

# Classification report
print("="*60)
print("DETAILED CLASSIFICATION REPORT")
print("="*60)
print(classification_report(all_true_labels, all_predictions, target_names=classes, digits=3))

print("="*60)
print("VALIDATION COMPLETE")
print("="*60)
