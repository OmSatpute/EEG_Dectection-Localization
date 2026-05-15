import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import Counter
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, roc_auc_score
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
SVM_OUTPUT_DIR = ROOT_DIR / "outputs" / "svm"
SVM_MODEL_DIR = ROOT_DIR / "outputs" / "trained_models" / "svm"

def check_class_balance(y, name="Dataset"):
    unique, counts = np.unique(y, return_counts=True)
    
    print(f"\n{name} Class Distribution:")
    total = len(y)
    
    for cls, count in zip(unique, counts):
        percentage = (count / total) * 100
        print(f"Class {cls}: {count} samples ({percentage:.2f}%)")
    
    ratio = max(counts) / min(counts)
    print(f"Imbalance Ratio (max/min): {ratio:.2f}")
    
    if ratio > 1.5:
        print("Classes are imbalanced")
    else:
        print("Classes are balanced")

def main():
    SVM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SVM_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    try:
        X_train = np.load(DATA_DIR / "X_train_features.npy")
        X_test  = np.load(DATA_DIR / "X_test_features.npy")
        y_train = np.load(DATA_DIR / "y_train.npy")
        y_test  = np.load(DATA_DIR / "y_test.npy")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure that the feature (.npy) files are generated and in the data/ directory.")
        return

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    check_class_balance(y_train, "Train")
    check_class_balance(y_test, "Test")

    print("\nClass Distribution (Train):")
    print(np.unique(y_train, return_counts=True))

    print("\nClass Distribution (Test):")
    print(np.unique(y_test, return_counts=True))

    # Pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=f_classif)),
        ("svc", SVC(kernel='rbf'))
    ])

    # GridSearchCV
    param_grid = {
        "selector__k": [40, 60, 80], # k most imp features
        "svc__C": [1, 10], # boundary to separate classes
        "svc__gamma": [0.01, 0.1],
        "svc__class_weight": [None, 'balanced']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nRunning GridSearchCV...")
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    print("Best parameters:", grid.best_params_)

    # Cross validation
    scores = cross_val_score(model, X_train, y_train, cv=cv)

    print("\n=== CROSS VALIDATION ===")
    print("CV scores:", scores)
    print("Mean CV accuracy:", scores.mean())

    # Shuffle test
    y_test_shuffled = np.random.permutation(y_test)

    print("\n=== SHUFFLE TEST ===")
    print("Accuracy on shuffled labels:", model.score(X_test, y_test_shuffled))

    # Cross-validated ROC
    y_scores_cv = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method='decision_function'
    )

    fpr_cv, tpr_cv, _ = roc_curve(y_train, y_scores_cv)
    auc_cv = roc_auc_score(y_train, y_scores_cv)

    print("\n=== CROSS-VALIDATED ROC ===")
    print("CV AUC:", auc_cv)

    plt.figure()
    plt.plot(fpr_cv, tpr_cv)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Cross-validated ROC (AUC = {auc_cv:.4f})")
    plt.savefig(SVM_OUTPUT_DIR / 'cv_roc_curve.png')
    plt.close()

    # Feature Importance
    selector = model.named_steps["selector"]
    scores_k = selector.scores_
    top_idx = np.argsort(scores_k)[-10:]

    print("\nTotal features:", len(scores_k))

    num_features = len(scores_k)
    num_channels = 26
    num_bands = 4

    # Safety check
    assert num_features == num_channels * num_bands

    bands = ["Delta", "Theta", "Alpha", "Beta"]

    feature_names = []
    for ch in range(num_channels):
        for band in bands:
            feature_names.append(f"Channel{ch+1}_{band}")

    assert len(feature_names) == len(scores_k)

    top_features = [feature_names[i] for i in top_idx]

    print("\nTop 10 Important Features:")
    for name, score in zip(top_features, scores_k[top_idx]):
        print(f"{name}: {score:.4f}")

    print("\n=== INTERPRETATION ===")
    bands_only = [name.split("_")[1] for name in top_features]
    band_counts = Counter(bands_only)

    print("Band Importance:")
    for band, count in band_counts.items():
        print(f"{band}: {count} features")

    # Save model
    model_path = SVM_MODEL_DIR / "final_svm_model.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

    # Final Test Evaluation
    print("\n=== FINAL TEST EVALUATION ===")
    y_pred = model.predict(X_test)
    print("Test Accuracy:", model.score(X_test, y_test))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Stroke', 'Stroke'],
                yticklabels=['Non-Stroke', 'Stroke'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Test Set)')
    plt.savefig(SVM_OUTPUT_DIR / 'confusion_matrix.png', bbox_inches='tight')
    plt.close()

    # ROC curve
    y_scores = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc_score = roc_auc_score(y_test, y_scores)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC = {auc_score:.4f})")
    plt.savefig(SVM_OUTPUT_DIR / 'roc_curve.png')
    plt.close()
    
    print(f"Plots saved in {SVM_OUTPUT_DIR}")

if __name__ == "__main__":
    main()
