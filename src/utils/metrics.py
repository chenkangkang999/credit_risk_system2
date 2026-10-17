from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)


def positive_probability(values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 2 and values.shape[1] == 2:
        values = values[:, 1]
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("预测概率必须是一维有限数值，或二分类的两列概率")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("预测概率必须在 0 到 1 之间")
    return values


@dataclass
class EvalResult:
    auc: float | None = None
    ks: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    log_loss: float = 0.0
    threshold: float = 0.5

    def to_dict(self):
        return asdict(self)

    @property
    def summary(self):
        auc = f"{self.auc:.4f}" if self.auc is not None else "不适用（单一类别）"
        return f"AUC: {auc} | KS: {self.ks:.4f} | F1: {self.f1:.4f}"


class Evaluator:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def evaluate(self, y_true, y_pred_proba=None, threshold=None, **kwargs):
        p = positive_probability(y_pred_proba if y_pred_proba is not None else kwargs["y_proba"])
        y = np.asarray(y_true)
        if not len(y) or len(y) != len(p) or not np.isin(y, [0, 1]).all():
            raise ValueError("真实标签必须为 0/1，且数量与预测概率一致")
        threshold = self.threshold if threshold is None else threshold
        if not 0 <= threshold <= 1:
            raise ValueError("分类阈值必须在 0 到 1 之间")
        pred = (p >= threshold).astype(int)
        return EvalResult(
            auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            ks=self.calculate_ks(y, p), accuracy=float(accuracy_score(y, pred)),
            precision=float(precision_score(y, pred, zero_division=0)),
            recall=float(recall_score(y, pred, zero_division=0)),
            f1=float(f1_score(y, pred, zero_division=0)),
            log_loss=float(log_loss(y, np.clip(p, 1e-10, 1-1e-10), labels=[0, 1])),
            threshold=float(threshold))

    @staticmethod
    def calculate_ks(y_true, y_pred_proba):
        if len(np.unique(y_true)) < 2:
            return 0.0
        fpr, tpr, _ = roc_curve(y_true, positive_probability(y_pred_proba))
        return float(np.max(np.abs(tpr - fpr)))

    @staticmethod
    def find_best_threshold(y_true, y_pred_proba, method="youden"):
        p = positive_probability(y_pred_proba)
        if len(np.unique(y_true)) < 2:
            return 0.5
        if method == "youden":
            fpr, tpr, thresholds = roc_curve(y_true, p)
            valid = np.isfinite(thresholds) & (thresholds <= 1)
            return float(thresholds[valid][np.argmax((tpr-fpr)[valid])])
        if method == "f1":
            precision, recall, thresholds = precision_recall_curve(y_true, p)
            scores = 2*precision*recall / (precision+recall+1e-10)
            return float(thresholds[np.argmax(scores[:-1])])
        if method == "fixed":
            return 0.5
        raise ValueError("阈值方法应为 youden、f1 或 fixed")

    @staticmethod
    def get_confusion_matrix(y_true, y_pred_proba, threshold=0.5):
        return confusion_matrix(y_true, positive_probability(y_pred_proba) >= threshold, labels=[0, 1])

    @staticmethod
    def get_roc_curve(y_true, y_pred_proba):
        return roc_curve(y_true, positive_probability(y_pred_proba))

    @staticmethod
    def get_pr_curve(y_true, y_pred_proba):
        return precision_recall_curve(y_true, positive_probability(y_pred_proba))

    @staticmethod
    def calculate_lift(y_true, y_pred_proba, n_bins=10):
        frame = pd.DataFrame({"target": y_true, "probability": positive_probability(y_pred_proba)})
        frame = frame.sort_values("probability", ascending=False).reset_index(drop=True)
        frame["bin"] = np.minimum(np.arange(len(frame))*n_bins//max(len(frame), 1)+1, n_bins)
        result = frame.groupby("bin")["target"].agg(["count", "mean"])
        result["lift"] = result["mean"] / max(float(frame.target.mean()), 1e-10)
        return result.reset_index()
