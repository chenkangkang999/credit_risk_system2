import numpy as np
import pandas as pd
import pytest

from src.models.ensemble import EnsembleModel
from src.pipeline.orchestrator import TrainingPipeline
from src.utils.config_loader import AttrDict


class ConstantModel:
    is_fitted = True
    feature_names = ["loanAmnt"]

    def __init__(self, probability):
        self.probability = probability
        self.prediction_calls = 0

    def predict_proba(self, data):
        self.prediction_calls += 1
        return np.full(len(data), self.probability)


def pipeline_for(names, enabled=True):
    pipeline = object.__new__(TrainingPipeline)
    pipeline.model_cfg = AttrDict({"ensemble": {"enabled": enabled, "method": "weighted_avg",
                                               "weights": {"xgboost": 0.4, "lightgbm": 0.4, "mlp": 0.2}}})
    probabilities = {"xgboost": 0.1, "lightgbm": 0.8, "mlp": 0.4}
    pipeline.models = {name: ConstantModel(probabilities[name]) for name in names}
    pipeline.state = {}
    pipeline.ensemble = None
    pipeline.X_val = pd.DataFrame({"loanAmnt": [10000, 30000]})
    pipeline.y_val = pd.Series([0, 1])
    return pipeline


def test_pipeline_uses_all_three_models_in_weighted_prediction():
    pipeline = pipeline_for(["xgboost", "lightgbm", "mlp"])
    base_models = dict(pipeline.models)

    pipeline._step_ensemble()

    assert isinstance(pipeline.ensemble, EnsembleModel)
    assert pipeline.models["ensemble"] is pipeline.ensemble
    assert pipeline.state["ensemble_model"] is pipeline.ensemble
    assert pipeline.ensemble.model_weights == pytest.approx({"xgboost": 0.4, "lightgbm": 0.4, "mlp": 0.2})
    np.testing.assert_allclose(pipeline.ensemble.predict_proba(pipeline.X_val), [0.44, 0.44])
    assert all(model.prediction_calls == 1 for model in base_models.values())

    base_models["lightgbm"].probability = 0.3
    np.testing.assert_allclose(pipeline.ensemble.predict_proba(pipeline.X_val), [0.24, 0.24])


@pytest.mark.parametrize("names", [
    ["xgboost", "lightgbm"],
    ["xgboost", "mlp"],
    ["xgboost"],
    ["lightgbm"],
    ["mlp"],
])
def test_incomplete_model_set_skips_ensemble_safely(names):
    pipeline = pipeline_for(names)
    base_models = dict(pipeline.models)

    pipeline._step_ensemble()

    assert pipeline.ensemble is None
    assert pipeline.models == base_models
    assert "ensemble_model" not in pipeline.state
    assert all(model.prediction_calls == 0 for model in base_models.values())


def test_disabled_ensemble_keeps_three_base_models():
    pipeline = pipeline_for(["xgboost", "lightgbm", "mlp"], enabled=False)
    base_models = dict(pipeline.models)

    pipeline._step_ensemble()

    assert pipeline.ensemble is None
    assert pipeline.models == base_models
    assert "ensemble_model" not in pipeline.state
    assert all(model.prediction_calls == 0 for model in base_models.values())
