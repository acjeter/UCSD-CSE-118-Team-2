import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils import ASLDataset


BASE_DIR = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")


coords = np.load(os.path.join(PROCESSED_DIR, "debug_coords.npy"))  # shape: (N, 66)
labels = np.load(os.path.join(PROCESSED_DIR, "debug_labels.npy"))


label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# # Convert to torch tensors
X = torch.tensor(coords, dtype=torch.float32)
y = torch.tensor(labels_encoded, dtype=torch.long)

# First split: separate out test set (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# Second split: split remaining into train (70%) and validation (15%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
)  # 0.176 * 0.85 ≈ 0.15 of total

# Save all splits inside ai/processed_data
np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train.numpy())
np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train.numpy())
np.save(os.path.join(PROCESSED_DIR, "X_val.npy"), X_val.numpy())
np.save(os.path.join(PROCESSED_DIR, "y_val.npy"), y_val.numpy())
np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test.numpy())
np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test.numpy())

print(f"Data split sizes:")
print(f"  Training:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"  Test:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
print()


# Make a Dataset + DataLoader


train_dataset = ASLDataset(X_train, y_train)
val_dataset   = ASLDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


from model import ASLModel

model = ASLModel()

# Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop with early stopping
EPOCHS = 100
best_val_acc = 0.0
patience = 15  # Stop if no improvement for 15 epochs
patience_counter = 0

for epoch in range(EPOCHS):
    total_loss = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()          
        outputs = model(batch_X)       
        loss = criterion(outputs, batch_y)  
        loss.backward()                
        optimizer.step()               

        total_loss += loss.item()
    
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, dim=1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    val_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Val Acc: {val_acc:.3f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "asl_model_best.pth")
        print(f"  → New best model saved! (Val Acc: {val_acc:.3f})")
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= patience:
        print(f"\nEarly stopping triggered after {epoch+1} epochs")
        print(f"Best validation accuracy: {best_val_acc:.3f}")
        break


# Save final model and labels into the ai/ directory
torch.save(model.state_dict(), os.path.join(BASE_DIR, "asl_model.pth"))
np.save(os.path.join(BASE_DIR, "label_classes.npy"), label_encoder.classes_)

print("\nTraining complete!")
print(f"Best model saved to: asl_model_best.pth (Val Acc: {best_val_acc:.3f})")
print(f"Final model saved to: asl_model.pth")
