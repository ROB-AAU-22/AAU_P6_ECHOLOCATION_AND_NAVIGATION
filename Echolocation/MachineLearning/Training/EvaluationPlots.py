import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from MachineLearning.Training.TrainingConfig import PLOT_DPI, CLASSIFICATION_THRESHOLD
DPI = PLOT_DPI

def plot_precision_recall_curve(y_true, y_probs, save_path=None):
    #print(f"y_true: {y_true}")
    #print(f"y_probs: {y_probs}")
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    # calculate scores
    prec_score = precision_score(y_true, (y_probs > CLASSIFICATION_THRESHOLD).astype(int))
    rec_score = recall_score(y_true, (y_probs > CLASSIFICATION_THRESHOLD).astype(int))
    f1 = f1_score(y_true, (y_probs > CLASSIFICATION_THRESHOLD).astype(int))
    acc = accuracy_score(y_true, (y_probs > CLASSIFICATION_THRESHOLD).astype(int))
    plt.figure(figsize=(8, 6), dpi=DPI)
    plt.plot(recall, precision, label="Precision-Recall Curve", color="blue")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=DPI)
    plt.close()
    return prec_score, rec_score, f1, acc

def plot_confusion_matrix_all(y_true, y_probs, threshold=0.5, save_path=None):
    y_pred = (np.array(y_probs) > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Object", "Object"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Threshold = {threshold})")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=DPI)
    plt.close()
    return cm

def plot_roc_curve(y_true, y_probs, save_path=None):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6), dpi=DPI)
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color="darkorange")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.grid(True)
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=DPI)
    plt.close()
    
    # calculate best threshold
    #best_threshold_index = np.argmax(tpr - fpr)
    #best_threshold = thresholds[best_threshold_index]
    return roc_auc#, best_threshold
