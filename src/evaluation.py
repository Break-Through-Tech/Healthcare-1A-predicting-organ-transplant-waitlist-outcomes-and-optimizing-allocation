# ==========================================
# 1. Imports
# ==========================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.calibration import calibration_curve

# ==========================================
# 2. Numerical Metrics Function
# ==========================================

def evaluate_risk_model(y_true, y_prob, threshold=0.5):
    """
    Evaluates risk predictions for organ waitlist adverse outcomes.
    """
    # 1. Convert continuous probabilities to binary predictions
    y_pred = (y_prob >= threshold).astype(int)

    # 2. Compute metrics
    metrics = {
        # 1. Ranking & Discrimination (Evaluates y_prob across all thresholds)
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
        # 2. Probability Calibration (Evaluates how realistic the percentages are)
        "brier_score": brier_score_loss(y_true, y_prob),
        # 3. Decision-Level Performance (Evaluates y_pred at ONE specific cutoff)
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        # 4. F1-score
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        # 5. Metadata / Settings
        "threshold": threshold,
    }

    return metrics

# ==========================================
# 3. Calibration Visualizer Function
# ==========================================

def plot_calibration_curve(y_true, y_prob, n_bins=10, model_name="Risk Model"):
    """
    Plots a reliability curve to visually inspect model calibration.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label=model_name)
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Fraction of Positives")
    plt.title(f"Calibration Curve ({model_name})")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.show()

# ==========================================
# 4. Lift Calculation Function
# ==========================================

def compute_lift_at_k(y_true, y_prob, top_k=0.10):
    """
    Computes model lift for the top k fraction of predicted highest-risk patients.
    """
    # 1. Determine overall baseline event rate
    overall_rate = np.mean(y_true)
    if overall_rate == 0:
        return 0.0

    # 2. Number of patients in top k
    n_top = int(np.ceil(len(y_prob) * top_k))

    # 3. Get indices of top k highest predicted probabilities
    top_indices = np.argsort(y_prob)[::-1][:n_top]

    # 4. Calculate event rate in the top k group
    top_k_rate = np.mean(np.array(y_true)[top_indices])

    # 5. Lift = (rate in top group) / (overall rate)
    return top_k_rate / overall_rate
