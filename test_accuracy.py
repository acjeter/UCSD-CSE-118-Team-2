import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from ai.model import ASLModel

REPO_DIR = os.path.dirname(__file__)
AI_DIR = os.path.join(REPO_DIR, "ai")
PROCESSED_DIR = os.path.join(AI_DIR, "processed_data")

# Load the test data
print("Loading test data...")
X_test = torch.tensor(
    np.load(os.path.join(PROCESSED_DIR, "X_test.npy")), dtype=torch.float32
)
y_test = torch.tensor(
    np.load(os.path.join(PROCESSED_DIR, "y_test.npy")), dtype=torch.long
)

# Load the label classes
classes = np.load(os.path.join(AI_DIR, "label_classes.npy"))

# Load the trained model (use the best validation checkpoint)
print("Loading trained model...")
model = ASLModel()
model.load_state_dict(
    torch.load(os.path.join(AI_DIR, "asl_model_best.pth"), map_location="cpu")
)
model.eval()

print("="*60)
print("TEST SET EVALUATION")
print("="*60)
print(f"Test set size: {len(X_test)} samples")
print(f"Number of classes: {len(classes)}")
print()

# Run predictions
all_predictions = []
all_true_labels = []

with torch.no_grad():
    for i in range(len(X_test)):
        x = X_test[i].unsqueeze(0)  # Add batch dimension
        y = y_test[i].item()
        
        output = model(x)
        pred_id = torch.argmax(output, dim=1).item()
        
        all_predictions.append(pred_id)
        all_true_labels.append(y)

# Convert to numpy arrays
all_predictions = np.array(all_predictions)
all_true_labels = np.array(all_true_labels)

# Calculate overall accuracy
overall_accuracy = accuracy_score(all_true_labels, all_predictions)
print(f"Overall Test Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
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
print("CONFUSION MATRIX (Top Misclassifications)")
print("="*60)
cm = confusion_matrix(all_true_labels, all_predictions)

# Find most confused pairs
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
print("="*60)
print("TEST EVALUATION COMPLETE")
print("="*60)
