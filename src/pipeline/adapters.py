import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.data.preprocessor import Preprocessor
from src.tuning.optimizer import OptunaOptimizer


class ProjectPreprocessor(Preprocessor):
    def _fit_imputers(self, df):
        self.empty_columns = [c for c in self.data_cfg.columns.numerical + self.data_cfg.columns.anonymous
                              if c in df.columns and df[c].isna().all()]
        for col in self.empty_columns:
            df[col] = 0.0
        super()._fit_imputers(df)

    def _apply_imputers(self, df):
        for col in self.empty_columns:
            if col in df:
                df[col] = df[col].fillna(0.0)
        return super()._apply_imputers(df)

    def _apply_scaler(self, df):
        method = self.feature_cfg.scaling.method
        if method == "none":
            return df
        choices = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
        if method not in choices:
            raise ValueError("不支持的标准化方式")
        if self._scaler is None:
            self.scaler_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            self._scaler = choices[method]()
            df[self.scaler_columns] = self._scaler.fit_transform(df[self.scaler_columns])
        else:
            df[self.scaler_columns] = self._scaler.transform(df[self.scaler_columns])
        return df


class CompatibleOptimizer(OptunaOptimizer):
    def optimize(self, *args, **kwargs):
        try:
            return super().optimize(*args, **kwargs)
        except RuntimeError:
            if not self.multi_objective or self.study is None or not self.study.best_trials:
                raise
            best = max(self.study.best_trials, key=lambda t: (t.values[0], -t.values[1]))
            self.best_params = best.params
            return dict(best.params)
