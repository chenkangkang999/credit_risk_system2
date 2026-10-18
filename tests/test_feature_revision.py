import copy

import numpy as np
import pandas as pd
import pytest

from src.features.project_builder import ProjectFeatureBuilder
from src.pipeline.adapters import ProjectPreprocessor
from src.pipeline.orchestrator import TrainingPipeline
from src.utils.config_loader import ConfigLoader


def test_new_features_use_only_input_and_keep_original_values():
    configs = ConfigLoader("configs")
    from scripts.generate_sample_data import generate_sample_data
    raw = generate_sample_data(80).drop(columns=["isDefault", "id"])
    original = raw.copy(deep=True)
    result = ProjectFeatureBuilder(configs.get("project").features).build(raw)
    pd.testing.assert_frame_equal(raw, original)
    pd.testing.assert_frame_equal(result[original.columns], original)
    assert len(result.columns) > len(raw.columns) + 5
    assert "isDefault" not in result
    assert result["annual_interest_to_income"].iloc[0] == pytest.approx(
        raw.loanAmnt.iloc[0] * raw.interestRate.iloc[0] / 100 / raw.annualIncome.iloc[0])


def test_credit_history_and_flags():
    raw = pd.DataFrame({"issueDate": ["2020-01-01", "2010-01-01"],
                        "earliesCreditLine": ["Jan-2010", "2020-01-01"], "grade": ["F", None]})
    rules = [{"name": "years", "operation": "credit_history"},
             {"name": "flag", "operation": "category_flag", "column": "grade", "values": ["E", "F", "G"]}]
    result = ProjectFeatureBuilder(rules).build(raw)
    assert 9.99 < result.years.iloc[0] < 10.01
    assert result.years.iloc[1] == 0
    assert result.flag.iloc[0] == 1 and np.isnan(result.flag.iloc[1])


def test_interest_ratio_does_not_create_infinity():
    rule = {"name": "load", "operation": "annual_interest_ratio"}
    result = ProjectFeatureBuilder([rule]).build(pd.DataFrame({
        "loanAmnt": [1000], "interestRate": [10], "annualIncome": [0]}))
    assert np.isnan(result.load.iloc[0])


def test_knn_reference_is_capped_and_keeps_training_statistics():
    configs = ConfigLoader("configs")
    from scripts.generate_sample_data import generate_sample_data
    raw = generate_sample_data(120).drop(columns=["isDefault", "id"])
    prep = ProjectPreprocessor(copy.deepcopy(configs.get("data")),
                               copy.deepcopy(configs.get("feature")), knn_reference_limit=30)
    result = prep.fit_transform(raw)
    assert len(prep._knn_imputer._fit_X) == 30
    assert prep._numerical_imputers["annualIncome"] == raw.annualIncome.median()
    assert len(result) == len(raw) and np.isfinite(result.to_numpy()).all()


def test_training_profile_changes_only_private_config_copy():
    configs = ConfigLoader("configs")
    original = configs.get("model").to_dict()
    pipeline = TrainingPipeline(configs)
    pipeline._apply_training_profile({"xgboost": {"max_depth": 2}, "lightgbm": {"num_leaves": 7}})
    assert pipeline.model_cfg.xgboost.params.max_depth == 2
    assert configs.get("model").to_dict() == original
