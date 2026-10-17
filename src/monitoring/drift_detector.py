"""
模型漂移检测器
==============
监控模型上线后数据分布是否发生变化（数据漂移）。

原理：
    对每个特征做 KS 检验（Kolmogorov-Smirnov test），
    比较"训练时的数据分布"和"新进来的数据分布"是否一致。
    如果 KS 统计量超过阈值，说明这个特征分布漂移了，
    模型可能需要重新训练。

要点：
    1. 数值特征用 KS 检验
    2. 类别特征用卡方检验（或分布差异）
    3. 返回漂移报告，可接入 Streamlit 监控仪表盘
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

from src.utils.logger import get_logger

logger = get_logger("monitoring.drift")


class DriftDetector:
    """
    数据漂移检测器。

    使用示例:
        >>> detector = DriftDetector(X_train)
        >>> drifts = detector.detect(X_new, threshold=0.1)
        >>> if drifts:
        ...     print(f"有 {len(drifts)} 个特征发生漂移，建议重新训练")
    """

    def __init__(self, baseline_data: pd.DataFrame):
        """
        参数:
            baseline_data: 基线数据（通常是训练集），后续新数据和它比较
        """
        self.baseline = baseline_data.copy()
        self._numerical_cols = baseline_data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self._categorical_cols = baseline_data.select_dtypes(
            exclude=[np.number]
        ).columns.tolist()

        logger.info(
            f"DriftDetector 初始化 - 数值特征 {len(self._numerical_cols)} 个, "
            f"类别特征 {len(self._categorical_cols)} 个"
        )

    def detect(
        self,
        new_data: pd.DataFrame,
        threshold: float = 0.1,
        p_threshold: float = 0.05,
    ) -> Dict[str, Dict]:
        """
        对每个特征做漂移检测。

        参数:
            new_data: 新进来的数据（线上数据）
            threshold: KS 统计量阈值，超过即认为漂移（默认 0.1）
            p_threshold: p 值阈值，小于即认为分布显著不同（默认 0.05）

        返回:
            drifts 字典，key 是发生漂移的列名，value 是统计信息：
            {
                "feature_name": {"ks_stat": 0.15, "p_value": 0.001, "severity": "medium"}
            }
        """
        drifts = {}

        # 数值特征：KS 检验
        for col in self._numerical_cols:
            if col not in new_data.columns:
                continue

            base_vals = self.baseline[col].dropna()
            new_vals = new_data[col].dropna()

            if len(base_vals) < 2 or len(new_vals) < 2:
                continue

            try:
                stat, p_value = ks_2samp(base_vals, new_vals)

                if stat > threshold:
                    drifts[col] = {
                        "type": "numerical",
                        "ks_stat": round(float(stat), 4),
                        "p_value": round(float(p_value), 6),
                        "severity": self._get_severity(stat),
                        "baseline_mean": round(float(base_vals.mean()), 4),
                        "new_mean": round(float(new_vals.mean()), 4),
                    }
            except Exception as e:
                logger.warning(f"数值特征 {col} 漂移检测失败: {e}")

        # 类别特征：分布差异检测
        for col in self._categorical_cols:
            if col not in new_data.columns:
                continue

            try:
                stat, p_value = self._categorical_drift(
                    self.baseline[col], new_data[col]
                )

                if stat > threshold:
                    drifts[col] = {
                        "type": "categorical",
                        "ks_stat": round(float(stat), 4),
                        "p_value": round(float(p_value), 6),
                        "severity": self._get_severity(stat),
                    }
            except Exception as e:
                logger.warning(f"类别特征 {col} 漂移检测失败: {e}")

        if drifts:
            logger.warning(
                f"检测到 {len(drifts)} 个特征发生漂移: {list(drifts.keys())}"
            )
        else:
            logger.info("未检测到明显数据漂移")

        return drifts

    def _categorical_drift(self, baseline: pd.Series, new: pd.Series):
        """类别特征漂移检测：用卡方检验"""
        # 构建列联表
        categories = sorted(set(baseline.dropna().unique()) | set(new.dropna().unique()))

        base_counts = baseline.value_counts().reindex(categories, fill_value=0)
        new_counts = new.value_counts().reindex(categories, fill_value=0)

        contingency = np.array([base_counts.values, new_counts.values])

        # 如果类别太少或者期望频数太小，跳过
        if contingency.shape[1] < 2 or contingency.min() < 1:
            return 0.0, 1.0

        chi2, p_value, dof, _ = chi2_contingency(contingency)
        # 用 Cramer's V 作为标准化的效应量（0~1之间）
        n = contingency.sum()
        min_dim = min(contingency.shape) - 1
        if min_dim == 0 or n == 0:
            return 0.0, 1.0
        cramer_v = np.sqrt(chi2 / (n * min_dim))

        return float(cramer_v), float(p_value)

    @staticmethod
    def _get_severity(stat: float) -> str:
        """根据统计量判断漂移严重程度"""
        if stat < 0.1:
            return "low"
        elif stat < 0.2:
            return "medium"
        else:
            return "high"

    def get_drift_report(self, new_data: pd.DataFrame, threshold: float = 0.1) -> str:
        """
        生成文本版漂移报告，方便直接展示或写入日志。
        """
        drifts = self.detect(new_data, threshold=threshold)

        if not drifts:
            return "✅ 所有特征分布稳定，未检测到数据漂移。"

        lines = [f"⚠️ 检测到 {len(drifts)} 个特征发生漂移：\n"]
        for col, info in drifts.items():
            severity_emoji = {"low": "🟡", "medium": "🟠", "high": "🔴"}.get(
                info["severity"], "⚪"
            )
            lines.append(
                f"{severity_emoji} {col} [{info['severity']}]  "
                f"KS={info['ks_stat']:.4f}, p={info['p_value']:.6f}"
            )
            if info["type"] == "numerical":
                lines.append(
                    f"    基线均值: {info['baseline_mean']} → "
                    f"新数据均值: {info['new_mean']}"
                )

        lines.append("\n建议：漂移严重的特征较多时，请考虑重新训练模型。")
        return "\n".join(lines)
