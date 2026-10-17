"""
LightGBM 模型
=============
梯度提升树，训练快、效果好，适合风控场景。
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.base import BaseModel


class LGBModel(BaseModel):
    """LightGBM 二分类模型"""

    def __init__(self, config, name="lightgbm"):
        super().__init__(config, name)
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        params = self.config.params.to_dict()
        self.model = lgb.LGBMClassifier(**params)
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X_train, y_train, eval_set=eval_set)
        self._is_fitted = True
        self._feature_names = list(X_train.columns)
        return self

    def predict_proba(self, X):
        self._check_fitted()
        # 返回正类概率（第二列）
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 else proba

    def save(self, file_path):
        self.model.booster_.save_model(file_path)

    @classmethod
    def load(cls, file_path, config):
        instance = cls(config)
        instance.model = lgb.Booster(model_file=file_path)
        instance._is_fitted = True
        return instance
