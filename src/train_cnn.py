import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from src.models import CNNModel


CNN_OUTPUT_DIR = os.path.join("outputs", "cnn")
CNN_MODEL_DIR = os.path.join("outputs", "trained_models", "cnn")
CNN_MODEL_PATH = os.path.join(CNN_MODEL_DIR, "cnn_model.pth")


def save_confusion_matrix(y_true, y_pred, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Stroke", "Stroke"],
        yticklabels=["Non-Stroke", "Stroke"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("CNN Confusion Matrix (Test Set)")
    plt.figtext(
        0.5,
        -0.02,
        f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | F1 Score: {f1:.4f} | Recall: {recall:.4f}",
        ha="center",
        fontsize=10,
    )
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), bbox_inches="tight")
    plt.close()


def save_accuracy_plot(train_accuracies, val_accuracies, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    epochs = range(1, len(train_accuracies) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_accuracies, marker="o", label="Train Accuracy")
    plt.plot(epochs, val_accuracies, marker="o", label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Accuracy vs Epoch")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_vs_epoch.png"))
    plt.close()


# Dataset
class EEGDataset(Dataset):
    def __init__(self, root):
        self.data = []

        for label_name in ['stroke', 'non_stroke']:
            label = 1 if label_name == 'stroke' else 0
            folder = os.path.join(root, label_name)

            for file in os.listdir(folder):
                if file.endswith('.npy'):
                    self.data.append((os.path.join(folder, file), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        x = np.load(path)

        # normalize
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x).view(-1)
            preds = (torch.sigmoid(output) > 0.5).float()

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)

    return accuracy, f1, recall, precision, all_labels, all_preds


# Load data
train_dataset = EEGDataset("data/topomap/train")
val_dataset = EEGDataset("data/topomap/val")
test_dataset = EEGDataset("data/topomap/test")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
train_accuracies = []
val_accuracies = []

for epoch in range(15):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x).view(-1)   # safer than squeeze
        loss = criterion(output, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_accuracy, train_f1, train_recall, train_precision, _, _ = evaluate(model, train_loader, device)
    val_accuracy, val_f1, val_recall, val_precision, _, _ = evaluate(model, val_loader, device)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(
        f"Epoch {epoch+1}, Loss: {total_loss:.4f}, "
        f"Train Accuracy: {train_accuracy:.4f}, "
        f"Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, Val F1: {val_f1:.4f}, Val Recall: {val_recall:.4f}"
    )

save_accuracy_plot(train_accuracies, val_accuracies, CNN_OUTPUT_DIR)
print(f"Accuracy plot saved in {CNN_OUTPUT_DIR}")

# Test
accuracy, f1, recall, precision, all_labels, all_preds = evaluate(model, test_loader, device)

print("Test Accuracy:", accuracy)
print("Precision:", precision)
print("F1 Score:", f1)
print("Recall:", recall)

save_confusion_matrix(all_labels, all_preds, CNN_OUTPUT_DIR)
print(f"Confusion matrix saved in {CNN_OUTPUT_DIR}")

# =========================
# SAVE MODEL
# =========================
os.makedirs(CNN_MODEL_DIR, exist_ok=True)
torch.save(model.state_dict(), CNN_MODEL_PATH)
print(f"Model saved as {CNN_MODEL_PATH}")
