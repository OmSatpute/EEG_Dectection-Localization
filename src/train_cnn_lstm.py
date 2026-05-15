import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from torch.utils.data import DataLoader, Dataset, random_split

from src.dataset_utils import numeric_key
from src.models import CNNLSTMModel


CNN_LSTM_OUTPUT_DIR = os.path.join("outputs", "cnn_lstm")
CNN_LSTM_MODEL_DIR = os.path.join("outputs", "trained_models", "cnn_lstm")
CNN_LSTM_MODEL_PATH = os.path.join(CNN_LSTM_MODEL_DIR, "cnn_lstm_model.pth")


class EEGSequenceDataset(Dataset):
    def __init__(self, root, sequence_length=32):
        self.samples = []
        self.sequence_length = sequence_length

        for label_name in ["stroke", "non_stroke"]:
            label = 1 if label_name == "stroke" else 0
            label_dir = os.path.join(root, label_name)

            if not os.path.isdir(label_dir):
                continue

            for subject in sorted(os.listdir(label_dir)):
                subject_dir = os.path.join(label_dir, subject)
                if not os.path.isdir(subject_dir):
                    continue

                files = [
                    os.path.join(subject_dir, file)
                    for file in os.listdir(subject_dir)
                    if file.endswith(".npy")
                ]

                if files:
                    self.samples.append((sorted(files, key=numeric_key), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        files, label = self.samples[idx]
        files = files[:self.sequence_length]

        frames = [np.load(path).astype(np.float32) for path in files]

        while len(frames) < self.sequence_length:
            frames.append(np.zeros_like(frames[0]))

        x = np.stack(frames)
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
    plt.title("CNN-LSTM Confusion Matrix (Test Set)")
    plt.figtext(
        0.5,
        -0.02,
        f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | F1 Score: {f1:.4f} | Recall: {recall:.4f}",
        ha="center",
        fontsize=10,
    )
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train CNN-LSTM on subject topomap sequences.")
    parser.add_argument("--data-root", default="data/topomap_sequence")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--output", default=CNN_LSTM_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_root = os.path.join(args.data_root, "train")
    val_root = os.path.join(args.data_root, "val")
    test_root = os.path.join(args.data_root, "test")

    if os.path.isdir(train_root) and os.path.isdir(test_root):
        train_dataset = EEGSequenceDataset(train_root, sequence_length=args.sequence_length)
        val_dataset = (
            EEGSequenceDataset(val_root, sequence_length=args.sequence_length)
            if os.path.isdir(val_root)
            else None
        )
        test_dataset = EEGSequenceDataset(test_root, sequence_length=args.sequence_length)
    else:
        dataset = EEGSequenceDataset(args.data_root, sequence_length=args.sequence_length)

        if len(dataset) < 4:
            print(
                "Warning: CNN-LSTM needs more subject sequences for a reliable train/test split. "
                f"Found only {len(dataset)} sequence(s). Training on all data without test metrics."
            )
            train_dataset = dataset
            val_dataset = None
            test_dataset = None
        else:
            test_size = max(1, int(len(dataset) * 0.2))
            val_size = max(1, int(len(dataset) * 0.1))
            train_size = len(dataset) - test_size - val_size
            train_dataset, val_dataset, test_dataset = random_split(
                dataset, [train_size, val_size, test_size]
            )

    if len(train_dataset) == 0:
        raise ValueError("No CNN-LSTM training sequences found.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size)
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )
    test_loader = (
        DataLoader(test_dataset, batch_size=args.batch_size)
        if test_dataset is not None and len(test_dataset) > 0
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNLSTMModel().to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x).view(-1)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if val_loader is not None:
            val_accuracy, val_f1, val_recall, val_precision, _, _ = evaluate(model, val_loader, device)
            print(
                f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, "
                f"Val F1: {val_f1:.4f}, Val Recall: {val_recall:.4f}"
            )
        else:
            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    if test_loader is not None:
        accuracy, f1, recall, precision, all_labels, all_preds = evaluate(model, test_loader, device)
        print("Test Accuracy:", accuracy)
        print("Precision:", precision)
        print("F1 Score:", f1)
        print("Recall:", recall)
        save_confusion_matrix(all_labels, all_preds, CNN_LSTM_OUTPUT_DIR)
        print(f"Confusion matrix saved in {CNN_LSTM_OUTPUT_DIR}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"CNN-LSTM model saved as {args.output}")


if __name__ == "__main__":
    main()
