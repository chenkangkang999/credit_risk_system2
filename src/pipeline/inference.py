import numpy as np
import pandas as pd

from src.utils.metrics import positive_probability


def risk_level(probability, low=0.3, high=0.7):
    return "低风险" if probability < low else "中风险" if probability < high else "高风险"


def decision_for(probability, low=0.3, high=0.7):
    return "建议通过" if probability < low else "人工复核" if probability < high else "建议拒绝"


def monthly_payment(amount, interest_rate, years):
    months = int(years * 12)
    rate = interest_rate / 1200
    if months <= 0 or amount <= 0 or interest_rate < 0:
        raise ValueError("金额、期限应为正数，利率不能为负数")
    return amount / months if rate == 0 else amount * rate / (1-(1+rate)**(-months))


class InferenceService:
    def __init__(self, artifacts):
        if artifacts.get("meta", {}).get("schema_version") != 2:
            raise ValueError("训练产物版本不兼容，请重新训练")
        self.artifacts = artifacts
        self.models = artifacts["models"]

    def prepare_raw(self, raw_df):
        if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
            raise ValueError("请输入至少一条申请记录")
        raw = raw_df.copy()
        for col in self.artifacts["meta"]["raw_columns"]:
            if col not in raw:
                raw[col] = self.artifacts["input_defaults"][col]
        raw = raw[self.artifacts["meta"]["raw_columns"]]
        data_cfg = self.artifacts["preprocessor"].data_cfg
        for col in data_cfg.columns.numerical + data_cfg.columns.anonymous:
            if col in raw:
                converted = pd.to_numeric(raw[col], errors="coerce")
                if (raw[col].notna() & converted.isna()).any():
                    raise ValueError(f"{col} 应为数值")
                raw[col] = converted
        if np.isinf(raw.select_dtypes(include="number").to_numpy()).any():
            raise ValueError("输入不能包含无穷大")
        return raw

    def transform(self, raw_df):
        raw = self.prepare_raw(raw_df)
        X = self.artifacts["feature_builder"].build(raw)
        X = self.artifacts["preprocessor"].transform(X)
        selector = self.artifacts["feature_selector"]
        if selector is not None:
            X = selector.transform(X)
        X = X[self.artifacts["meta"]["feature_names"]]
        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("输入数据转换后仍有缺失，请检查字段")
        return X

    def predict_proba(self, raw_df, model_name="xgboost"):
        if model_name not in self.models:
            raise ValueError(f"当前训练产物不含 {model_name} 模型")
        return positive_probability(self.models[model_name].predict_proba(self.transform(raw_df)))
