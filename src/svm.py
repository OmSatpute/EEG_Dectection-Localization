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
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, recall_score, roc_curve, roc_auc_score
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
SVM_OUTPUT_DIR = ROOT_DIR / "outputs" / "svm"
SVM_MODEL_DIR = ROOT_DIR / "outputs" / "trained_models" / "svm"


def build_model():
    """Build the SVM pipeline with best parameters found via GridSearchCV."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=f_classif, k=40)),
        ("svc", SVC(kernel='rbf', C=10, gamma=0.1, class_weight=None))
    ])
    return pipe


def load_data(x_train_path=None, x_test_path=None, y_train_path=None, y_test_path=None):
    """Load train and test data."""
    x_train_path = x_train_path or DATA_DIR / "X_train_features.npy"
    x_test_path = x_test_path or DATA_DIR / "X_test_features.npy"
    y_train_path = y_train_path or DATA_DIR / "y_train.npy"
    y_test_path = y_test_path or DATA_DIR / "y_test.npy"

    X_train = np.load(x_train_path)
    X_test = np.load(x_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate the model using cross-validation and test set."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n=== CROSS VALIDATION ===")
    scores = cross_val_score(model, X_train, y_train, cv=cv)
    print("CV scores:", scores)
    print("Mean CV accuracy:", scores.mean())
    
    print("\n=== CROSS-VALIDATED ROC ===")
    y_scores_cv = cross_val_predict(
        model, X_train, y_train, cv=cv, method='decision_function'
    )
    auc_cv = roc_auc_score(y_train, y_scores_cv)
    print("CV AUC:", auc_cv)
    
    print("\n=== SHUFFLE TEST ===")
    y_test_shuffled = np.random.permutation(y_test)
    print("Accuracy on shuffled labels:", model.score(X_test, y_test_shuffled))
    
    print("\n=== FINAL TEST EVALUATION ===")
    y_pred = model.predict(X_test)
    print("Test Accuracy:", model.score(X_test, y_test))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    y_scores_test = model.decision_function(X_test)
    return y_pred, y_scores_test


def plot_results(y_test, y_pred, y_scores):
    """Plot and save evaluation results."""
    SVM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Stroke', 'Stroke'],
                yticklabels=['Non-Stroke', 'Stroke'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Test Set)')
    plt.figtext(
        0.5, -0.02,
        f"Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f} | Recall: {recall:.4f}",
        ha="center",
        fontsize=10,
    )
    plt.savefig(SVM_OUTPUT_DIR / 'confusion_matrix.png', bbox_inches='tight')
    plt.close()

    # ROC Curve
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


def get_top_features(model):
    """Extract and interpret the top 10 features from the selector."""
    selector = model.named_steps["selector"]
    scores = selector.scores_
    top_idx = np.argsort(scores)[-10:]
    
    print("\nTotal features:", len(scores))
    
    num_channels = 26
    num_bands = 4
    bands = ["Delta", "Theta", "Alpha", "Beta"]
    
    feature_names = []
    for ch in range(num_channels):
        for band in bands:
            feature_names.append(f"Channel{ch+1}_{band}")
            
    top_features = [feature_names[i] for i in top_idx]
    
    print("\nTop 10 Important Features:")
    for name, score in zip(top_features, scores[top_idx]):
        print(f"{name}: {score:.4f}")
        
    print("\n=== INTERPRETATION ===")
    bands_only = [name.split("_")[1] for name in top_features]
    band_counts = Counter(bands_only)
    
    print("Band Importance:")
    for band, count in band_counts.items():
        print(f"{band}: {count} features")


def train_and_save(model, X_train, y_train, model_path=None):
    """Train the model and save it to disk."""
    SVM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = model_path or SVM_MODEL_DIR / "final_svm_model.pkl"

    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model


if __name__ == "__main__":
    print("Loading data...")
    try:
        X_train, X_test, y_train, y_test = load_data()
        
        print(f"Train shape: {X_train.shape}")
        print(f"Test shape: {X_test.shape}")
        
        print("\nBuilding model...")
        model = build_model()
        
        print("\nTraining model...")
        model = train_and_save(model, X_train, y_train)
        
        print("\nEvaluating model...")
        y_pred, y_scores_test = evaluate_model(model, X_train, y_train, X_test, y_test)
        
        get_top_features(model)
        plot_results(y_test, y_pred, y_scores_test)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure that the feature (.npy) files are in the same directory or update the paths in load_data().")
