# ==========================================
# 1. Imports
# ==========================================

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ==========================================
# 2. Input Validation Helper
# ==========================================

def _validate_inputs(y_true, y_prob):
    """
    Validates input arrays for empty values, dimension mismatches, and NaNs.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    if len(y_true) == 0 or len(y_prob) == 0:
        raise ValueError("Input arrays y_true and y_prob must not be empty.")

    if len(y_true) != len(y_prob):
        raise ValueError(
            f"Input length mismatch: y_true has {len(y_true)} elements, "
            f"while y_prob has {len(y_prob)} elements."
        )

    if np.isnan(y_true).any() or np.isnan(y_prob).any():
        raise ValueError("Inputs y_true and y_prob must not contain NaN values.")

    if not np.all((y_prob >= 0.0) & (y_prob <= 1.0)):
        raise ValueError("All predicted probabilities y_prob must fall between 0 and 1.")

    unique_vals = np.unique(y_true)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError("Target array y_true must contain only binary values (0 and 1).")

    return y_true, y_prob

# ==========================================
# 3. Lift Calculation Function
# ==========================================

def compute_lift_at_k(y_true, y_prob, top_k=0.10):
    """
    Computes model lift for the top k fraction of predicted highest-risk patients.
    """
    y_true, y_prob = _validate_inputs(y_true, y_prob)

    overall_rate = np.mean(y_true)
    if overall_rate == 0:
        return 0.0

    n_top = int(np.ceil(len(y_prob) * top_k))
    top_indices = np.argsort(y_prob)[::-1][:n_top]
    top_k_rate = np.mean(y_true[top_indices])

    return float(top_k_rate / overall_rate)

# ==========================================
# 4. Numerical Metrics Function
# ==========================================

def evaluate_risk_model(y_true, y_prob, threshold=0.5):
    """
    Evaluates risk predictions for organ waitlist adverse outcomes.
    """
    # 1. Validate inputs
    y_true, y_prob = _validate_inputs(y_true, y_prob)

    # 2. Convert continuous probabilities to binary predictions
    y_pred = (y_prob >= threshold).astype(int)

    # Compute ROC-AUC (identical to Concordance Index / C-index for binary classification)
    roc_auc = float(roc_auc_score(y_true, y_prob))

    # 3. Compute metrics
    metrics = {
        # 1. Ranking & Discrimination
        "roc_auc": roc_auc,
        "c_index": roc_auc,
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        # 2. Probability Calibration
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        # 3. Decision-Level Performance
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        # 4. F1-score
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        # 5. Risk Stratification
        "top_decile_lift": compute_lift_at_k(y_true, y_prob, top_k=0.10),
        # 6. Metadata / Settings
        "threshold": threshold,
    }

    return metrics

# ==========================================
# 5. Calibration Visualizer Function
# ==========================================

def plot_calibration_curve(y_true, y_prob, n_bins=10, model_name="Risk Model"):
    """
    Plots a reliability curve to visually inspect model calibration.
    """
    y_true, y_prob = _validate_inputs(y_true, y_prob)

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
