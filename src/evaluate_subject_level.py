import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from src.models import CNNModel

def evaluate_subject_level(model_path, test_dir, device):
    model = CNNModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    true_labels = []
    pred_labels = []
    
    labels = ["non_stroke", "stroke"]
    for label_idx, label_name in enumerate(labels):
        label_dir = os.path.join(test_dir, label_name)
        if not os.path.exists(label_dir):
            continue
            
        for subject in os.listdir(label_dir):
            subject_dir = os.path.join(label_dir, subject)
            if not os.path.isdir(subject_dir):
                continue
                
            segment_files = [f for f in os.listdir(subject_dir) if f.endswith('.npy')]
            if len(segment_files) == 0:
                continue
                
            subject_preds = []
            
            for seg_file in segment_files:
                path = os.path.join(subject_dir, seg_file)
                x = np.load(path).astype(np.float32)
                
                # Normalize
                x = (x - x.min()) / (x.max() - x.min() + 1e-8)
                
                x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(x_tensor).view(-1)
                    pred = (torch.sigmoid(output) > 0.5).float().item()
                    subject_preds.append(pred)
            
            # Majority vote
            majority_vote = 1 if sum(subject_preds) / len(subject_preds) > 0.5 else 0
            
            true_labels.append(label_idx)
            pred_labels.append(majority_vote)
            
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    
    print("=== CNN Subject-Level Evaluation ===")
    print(f"Total Subjects Evaluated: {len(true_labels)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Save confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
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
    plt.title("CNN Confusion Matrix (Subject-Level Test Set)")
    plt.figtext(
        0.5,
        -0.02,
        f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | F1 Score: {f1:.4f} | Recall: {recall:.4f}",
        ha="center",
        fontsize=10,
    )
    
    output_dir = "outputs/cnn_subject_level"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved in {output_dir}")

if __name__ == "__main__":
    test_dir = "data/topomap_sequence/test"
    model_path = os.path.join("outputs", "trained_models", "cnn", "cnn_model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(model_path) and os.path.exists(test_dir):
        evaluate_subject_level(model_path, test_dir, device)
    else:
        print(f"Model path or test dir not found. Run CNN training and sequence building first.")
