import copy
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.loader import DataLoader, DataSplitter
from src.features.project_builder import ProjectFeatureBuilder
from src.features.selector import FeatureSelector
from src.models.xgb_model import XGBModel
from src.models.lgb_model import LGBModel
from src.models.mlp_model import MLPModel
from src.models.ensemble import EnsembleModel
from src.pipeline.adapters import CompatibleOptimizer, ProjectPreprocessor
from src.utils.metrics import Evaluator, positive_probability
from src.utils.logger import get_logger

ROOT = Path(__file__).resolve().parents[2]
logger = get_logger("pipeline")


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def validate_training_data(df):
    required = ["loanAmnt", "term", "interestRate", "installment", "annualIncome",
                "dti", "revolUtil", "ficoRangeLow", "ficoRangeHigh", "isDefault"]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError("数据缺少字段：" + "、".join(missing))
    if len(df) < 50 or not df["isDefault"].isin([0, 1]).all() or df["isDefault"].nunique() != 2:
        raise ValueError("训练数据至少 50 行，标签 isDefault 必须为 0/1 且同时包含两类")
    if df["isDefault"].value_counts().min() < 10:
        raise ValueError("每个标签类别至少需要 10 条样本")
    return df


class TrainingPipeline:
    def __init__(self, configs):
        self.configs = configs
        self.data_cfg = copy.deepcopy(configs.get("data"))
        self.feature_cfg = copy.deepcopy(configs.get("feature"))
        self.model_cfg = copy.deepcopy(configs.get("model"))
        self.project_cfg = copy.deepcopy(configs.get("project"))
        for key, value in list(self.data_cfg.paths.items()):
            if key.endswith("_dir"):
                self.data_cfg.paths[key] = str(project_path(value))
        self.model_cfg.persistence.save_dir = str(project_path(self.model_cfg.persistence.save_dir))
        self.models = {}
        self.eval_results = {}
        self.state = {}
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.xgb_model = self.mlp_model = self.ensemble = None
        self.feature_selector = None
        self.tuning_history = []
        self.tuning_metric = self.model_cfg.xgboost.tuning.metric

    def run(self, use_sample=False, skip_mlp=False, skip_tuning=True, progress_callback=None,
            model_names=None, data=None, quick=False):
        start = time.time()
        np.random.seed(42)
        random.seed(42)
        import torch
        torch.manual_seed(42)
        torch.set_num_threads(2)
        names = list(model_names) if model_names is not None else [
            n for n in ["xgboost", "lightgbm", "mlp"] if self.model_cfg[n].enabled]
        if skip_mlp:
            names = [n for n in names if n != "mlp"]
        if not names or set(names) - {"xgboost", "lightgbm", "mlp"}:
            raise ValueError("请选择 XGBoost、LightGBM 或 MLP")
        if quick:
            self.model_cfg.xgboost.params.n_estimators = self.project_cfg.demo.trees
            self.model_cfg.lightgbm.params.n_estimators = self.project_cfg.demo.trees
            self.model_cfg.xgboost.tuning.n_trials = self.project_cfg.demo.trials
            self.model_cfg.mlp.training.epochs = self.project_cfg.demo.epochs
            self.model_cfg.mlp.architecture.hidden_layers = [64, 32]
        for name in ["xgboost", "lightgbm"]:
            self.model_cfg[name].params.n_jobs = 2
        self.data_source = "合成演示数据" if use_sample else "用户提供的数据"
        steps = [
            ("加载与校验数据", lambda: self._step_load_data(use_sample, data)),
            ("划分训练、验证、测试集", self._step_split_data),
            ("原始单位特征构造", self._step_feature_engineering),
            ("训练集拟合预处理", self._step_preprocess),
            ("训练集拟合特征选择", self._step_feature_selection),
        ]
        if "xgboost" in names:
            steps.append(("训练 XGBoost", lambda: self._step_train_xgb(not skip_tuning)))
        if "lightgbm" in names:
            steps.append(("训练 LightGBM", self._step_train_lgb))
        if "mlp" in names:
            steps.append(("训练 MLP", self._step_train_mlp))
        steps += [("模型融合", self._step_ensemble), ("独立测试集评估", self._step_evaluate),
                  ("保存训练产物", self._step_save_artifacts)]
        for i, (label, operation) in enumerate(steps):
            if progress_callback:
                progress_callback(label, int(i / len(steps) * 100))
            logger.info(label)
            operation()
            self.state[label] = "完成"
        if progress_callback:
            progress_callback("完成", 100)
        return {"run_id": self.run_id, "duration": time.time()-start,
                "eval_results": self.eval_results, "feature_count": self.X_train.shape[1],
                "train_samples": len(self.X_train)}

    def _step_load_data(self, use_sample=False, data=None):
        self.df = validate_training_data(data.copy() if data is not None else
                                         DataLoader(self.data_cfg).load_train(use_sample=use_sample))
        expected = (self.data_cfg.columns.numerical + self.data_cfg.columns.categorical +
                    self.data_cfg.columns.datetime + self.data_cfg.columns.anonymous)
        columns = [c for c in expected if c in self.df]
        self.df = self.df[columns + [self.data_cfg.target_column]].copy()
        for col in self.data_cfg.columns.numerical + self.data_cfg.columns.anonymous:
            if col in self.df:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        self.df = self.df.replace([np.inf, -np.inf], np.nan)

    def _step_split_data(self):
        split = DataSplitter(self.data_cfg).split(self.df)
        self.raw_train, self.raw_val, self.raw_test, self.y_train, self.y_val, self.y_test = split
        self.raw_columns = self.raw_train.columns.tolist()
        self.input_defaults = {}
        for col in self.raw_columns:
            values = self.raw_train[col]
            if pd.api.types.is_numeric_dtype(values):
                value = values.median() if values.notna().any() else 0.0
            else:
                modes = values.mode()
                value = modes.iloc[0] if len(modes) else "unknown"
            self.input_defaults[col] = value.item() if isinstance(value, np.generic) else value

    def _step_feature_engineering(self):
        self.feature_builder = ProjectFeatureBuilder(self.project_cfg.features)
        self.X_train = self.feature_builder.build(self.raw_train)
        self.X_val = self.feature_builder.build(self.raw_val)
        self.X_test = self.feature_builder.build(self.raw_test)
        self.data_cfg.columns.numerical += self.feature_builder.get_created_features()

    def _step_preprocess(self):
        self.preprocessor = ProjectPreprocessor(self.data_cfg, self.feature_cfg)
        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.X_val = self.preprocessor.transform(self.X_val)
        self.X_test = self.preprocessor.transform(self.X_test)
        for frame in [self.X_train, self.X_val, self.X_test]:
            if frame.isna().any().any() or not np.isfinite(frame.to_numpy(dtype=float)).all():
                raise ValueError("预处理后仍存在缺失值或非有限值，请检查输入数据")

    def _step_feature_selection(self):
        if self.feature_cfg.feature_selection.enabled:
            self.feature_selector = FeatureSelector(self.feature_cfg)
            self.X_train = self.feature_selector.fit_transform(self.X_train, self.y_train)
            self.X_val = self.feature_selector.transform(self.X_val)
            self.X_test = self.feature_selector.transform(self.X_test)

    def _step_train_xgb(self, with_tuning=False):
        if with_tuning:
            from xgboost import XGBClassifier
            optimizer = CompatibleOptimizer(self.model_cfg.xgboost.tuning)
            best = optimizer.optimize(XGBClassifier, self.X_train, self.y_train, self.X_val,
                                      self.y_val, base_params=dict(self.model_cfg.xgboost.params))
            self.model_cfg.xgboost.params.update(best)
            self.tuning_history = optimizer.get_history_df().to_dict("records")
        self.xgb_model = XGBModel(self.model_cfg.xgboost)
        self.xgb_model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
        self.models["xgboost"] = self.xgb_model
        self.state["xgb_model"] = self.xgb_model

    def _step_train_lgb(self):
        model = LGBModel(self.model_cfg.lightgbm)
        model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
        self.models["lightgbm"] = model

    def _step_train_mlp(self):
        self.mlp_model = MLPModel(self.model_cfg.mlp)
        self.mlp_model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
        self.models["mlp"] = self.mlp_model

    def _step_ensemble(self):
        if not self.model_cfg.ensemble.enabled or not {"xgboost", "mlp"}.issubset(self.models):
            return
        self.ensemble = EnsembleModel(self.model_cfg.ensemble)
        for name in ["xgboost", "mlp"]:
            self.ensemble.add_model(name, self.models[name])
        self.ensemble.fit(self.X_val, self.y_val)
        self.models["ensemble"] = self.ensemble
        self.state["ensemble_model"] = self.ensemble

    def _step_evaluate(self):
        evaluator = Evaluator()
        self.test_predictions = {}
        self.thresholds = {}
        cfg = self.model_cfg.evaluation.threshold_optimization
        for name, model in self.models.items():
            threshold = cfg.fixed_value if cfg.method == "fixed" or not cfg.enabled else \
                evaluator.find_best_threshold(self.y_val, model.predict_proba(self.X_val), cfg.method)
            p = positive_probability(model.predict_proba(self.X_test))
            self.thresholds[name] = threshold
            self.test_predictions[name] = p
            self.eval_results[name] = evaluator.evaluate(self.y_test, p, threshold).to_dict()
        primary = "ensemble" if "ensemble" in self.models else next(iter(self.models))
        self.state["eval_results"] = self.eval_results
        self.state["evaluate_result"] = evaluator.evaluate(
            self.y_test, self.test_predictions[primary], self.thresholds[primary])

    def _step_save_artifacts(self):
        ArtifactManager(self.model_cfg.persistence.save_dir, self.run_id).save_pipeline(self)


class ArtifactManager:
    def __init__(self, base_dir, run_id=None):
        self.base_dir = project_path(base_dir)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def save_pipeline(self, pipeline):
        meta = {"schema_version": 2, "run_id": pipeline.run_id,
                "timestamp": datetime.now().isoformat(), "source": pipeline.data_source,
                "feature_names": pipeline.X_train.columns.tolist(),
                "raw_columns": pipeline.raw_columns, "train_samples": len(pipeline.X_train),
                "val_samples": len(pipeline.X_val), "test_samples": len(pipeline.X_test),
                "model_names": list(pipeline.models), "seed": 42,
                "fit_train_indices": pipeline.raw_train.index.tolist()}
        bundle = {"meta": meta, "models": pipeline.models, "preprocessor": pipeline.preprocessor,
                  "feature_builder": pipeline.feature_builder, "feature_selector": pipeline.feature_selector,
                  "input_defaults": pipeline.input_defaults, "eval_results": pipeline.eval_results,
                  "thresholds": pipeline.thresholds, "project_config": pipeline.project_cfg.to_dict(),
                  "background": pipeline.X_train.head(100), "raw_test": pipeline.raw_test.head(200),
                  "raw_baseline": pipeline.raw_train.head(200), "y_test": pipeline.y_test.head(200),
                  "test_predictions": {n: p[:200] for n, p in pipeline.test_predictions.items()},
                  "tuning_history": pipeline.tuning_history, "tuning_metric": pipeline.tuning_metric}
        for folder in [self.base_dir / self.run_id, self.base_dir / "latest"]:
            folder.mkdir(parents=True, exist_ok=True)
            pending = folder / f"bundle.{pipeline.run_id}.pending"
            joblib.dump(bundle, pending, compress=3)
            os.replace(pending, folder / "bundle.joblib")
            (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            (folder / "eval_results.json").write_text(json.dumps(pipeline.eval_results, indent=2), encoding="utf-8")
        run_folder = self.base_dir / self.run_id
        if pipeline.xgb_model:
            pipeline.xgb_model.save(str(run_folder / "xgb_model.json"))
        if "lightgbm" in pipeline.models:
            pipeline.models["lightgbm"].save(str(run_folder / "lightgbm_model.txt"))
        if pipeline.mlp_model:
            pipeline.mlp_model.save(str(run_folder / "mlp_model.pth"))
        if pipeline.ensemble:
            pipeline.ensemble.save(str(run_folder / "ensemble.json"))
        import shutil
        latest = self.base_dir / "latest"
        for path in run_folder.iterdir():
            if path.suffix in (".json", ".txt", ".pth", ".pkl"):
                shutil.copy2(path, latest / path.name)
        for name in ["preprocessor", "feature_builder", "feature_selector"]:
            value = getattr(pipeline, name)
            if value is not None:
                joblib.dump(value, latest / (name + ".pkl"))
        logger.info(f"训练产物已保存：{run_folder}")

    @classmethod
    def load_latest(cls, base_dir):
        folder = project_path(base_dir) / "latest"
        path = folder / "bundle.joblib"
        if not path.exists():
            raise FileNotFoundError("没有完整训练产物，请重新训练或使用内置演示模型")
        result = joblib.load(path)
        result["dir"] = str(folder)
        result["ensemble_model"] = result["models"].get("ensemble")
        return result
