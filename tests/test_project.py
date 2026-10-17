import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
import joblib
import numpy as np
import pandas as pd
import pytest
from openai import APIConnectionError

from src.ab_test.router import ABTestRouter
from src.features.project_builder import ProjectFeatureBuilder
from src.llm.client import DeepSeekClient
from src.llm.json_utils import parse_json_object
from src.pipeline.inference import InferenceService, monthly_payment
from src.pipeline.orchestrator import TrainingPipeline
from src.utils.config_loader import AttrDict, ConfigLoader
from src.utils.metrics import Evaluator

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def configs():
    return ConfigLoader(str(ROOT / "configs"))


@pytest.fixture
def bundle():
    return joblib.load(ROOT / "demo_assets/bundle.joblib")


def test_nested_plain_dictionary():
    config = AttrDict({"a": {"b": [{"c": 1}]}})
    output = config.to_dict()
    assert type(output["a"]) is dict
    output["a"]["b"][0]["c"] = 2
    assert config.a.b[0].c == 1


def test_interpolation_does_not_reparse_yaml(configs, tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_VALUE", "value: with # punctuation")
    path = tmp_path / "sample.yaml"
    path.write_text("value: " + "$" + "{TEST_VALUE}", encoding="utf-8")
    assert configs._load_yaml(path).value == "value: with # punctuation"


@pytest.mark.parametrize("denominator", [0, np.nan])
def test_derived_invalid_denominator(denominator):
    builder = ProjectFeatureBuilder([{"name": "ratio", "operation": "ratio", "numerator": "a", "denominator": "b"}])
    result = builder.build(pd.DataFrame({"a": [1], "b": [denominator]}))
    assert np.isnan(result.ratio.iloc[0])


def test_no_arbitrary_feature_code(configs):
    from src.llm.agents.feature_agent import FeatureAgent
    agent = FeatureAgent(None, None, configs.get("llm").agents.feature_agent)
    with pytest.raises(ValueError):
        agent.apply_suggestion(pd.DataFrame({"a": [1]}), {"name": "payload", "operation": "exec"})


@pytest.mark.parametrize("text", ['{"a": {"b": 1}}', "text\n" + chr(96)*3 + 'json\n{"a":{"b":1}}\n' + chr(96)*3])
def test_json_response(text):
    assert parse_json_object(text)["a"]["b"] == 1


def test_json_failure():
    with pytest.raises(ValueError):
        parse_json_object("not json")


def test_single_class_metrics():
    result = Evaluator().evaluate([0, 0], np.array([0.1, 0.2]))
    assert result.auc is None and result.ks == 0 and np.isfinite(result.log_loss)


def test_report_names_actual_prediction_and_explanation_models(configs):
    from src.llm.agents.report_agent import ReportAgent
    from src.llm.prompts import PromptManager
    captured = []
    client = SimpleNamespace(chat=lambda prompt, **kw: captured.append(prompt) or "ok")
    agent = ReportAgent(client, PromptManager(configs.get("llm")), configs.get("llm").agents.report_agent)
    agent.generate({"loanAmnt": 10000}, 0.4, None, model_name="mlp", explanation_model="xgboost")
    assert "mlp" in captured[0] and "归因对象：xgboost" in captured[0]


@pytest.mark.parametrize("p", [np.array([np.nan]), np.array([1.1])])
def test_invalid_probabilities(p):
    with pytest.raises(ValueError):
        Evaluator().evaluate([0], p)


def test_inference_roundtrip_and_column_order(bundle, tmp_path):
    service = InferenceService(bundle)
    raw = bundle["raw_test"].head(10)
    for name in service.models:
        expected = service.models[name].predict_proba(service.transform(raw))
        assert np.allclose(service.predict_proba(raw, name), expected)
        assert np.allclose(service.predict_proba(raw[raw.columns[::-1]], name), expected)
    path = tmp_path / "bundle.joblib"
    joblib.dump(bundle, path)
    assert np.allclose(InferenceService(joblib.load(path)).predict_proba(raw), service.predict_proba(raw))


def test_missing_input_defaults_and_errors(bundle):
    service = InferenceService(bundle)
    assert service.predict_proba(pd.DataFrame([{"loanAmnt": 30000}])).shape == (1,)
    with pytest.raises(ValueError):
        service.predict_proba(pd.DataFrame([{"loanAmnt": "bad"}]))
    with pytest.raises(ValueError):
        service.predict_proba(pd.DataFrame())


def test_train_only_fit(configs):
    from scripts.generate_sample_data import generate_sample_data
    pipeline = TrainingPipeline(configs)
    pipeline._step_load_data(True, generate_sample_data(300, 42))
    pipeline._step_split_data()
    expected = pipeline.raw_train["annualIncome"].median()
    pipeline.raw_test["annualIncome"] = 1e9
    pipeline._step_feature_engineering()
    pipeline._step_preprocess()
    assert pipeline.preprocessor._numerical_imputers["annualIncome"] == expected
    assert pipeline.preprocessor._knn_imputer._fit_X.shape[0] == len(pipeline.raw_train)
    assert pipeline.preprocessor._scaler.n_samples_seen_ == len(pipeline.raw_train)
    assert set(pipeline.raw_train.index).isdisjoint(pipeline.raw_test.index)


@pytest.mark.parametrize("ratio, expected", [(0, "B"), (1, "A")])
def test_ab_endpoints(ratio, expected):
    assert all(ABTestRouter(None, None, ratio).group(i) == expected for i in range(20))


def test_ab_stable():
    router = ABTestRouter(None, None, .5)
    groups = [router.group(str(i)) for i in range(1000)]
    assert groups == [ABTestRouter(None, None, .5).group(str(i)) for i in range(1000)]
    assert 400 < groups.count("A") < 600
    with pytest.raises(ValueError):
        router.group("")


def test_monthly_payment():
    assert monthly_payment(12000, 0, 1) == 1000
    assert monthly_payment(12000, 12, 1) > 1000


def test_rules_agent(configs):
    from src.llm.agents.compliance_agent import ComplianceAgent
    agent = ComplianceAgent(None, None, configs.get("project").compliance)
    result = agent.check({"loanAmnt": 50000, "annualIncome": 0, "term": 3, "interestRate": 10},
                         "建议通过", probability=.9)
    assert not result["is_compliant"] and len(result["violations"]) >= 2


@pytest.fixture
def llm(configs, tmp_path, monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-" + "x"*32)
    cfg = copy.deepcopy(configs.get("llm"))
    cfg.deepseek.retry.initial_wait = 0
    client = DeepSeekClient(cfg)
    client.audit_log_path = tmp_path / "audit.log"
    return client


def test_pii_filtered_before_transport(llm):
    secret = os.environ["DEEPSEEK_API_KEY"]
    text = json.dumps(llm._build_messages("13812345678，" + secret, "110101199001011234",
                                          [{"role": "user", "content": secret}]))
    assert secret not in text and "13812345678" not in text and "110101199001011234" not in text


def test_chat_retry_and_audit(llm):
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise APIConnectionError(request=httpx.Request("POST", "https://test.invalid"))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                               usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 3}))
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert llm.chat("private applicant record") == "ok" and len(calls) == 2
    text = llm.audit_log_path.read_text()
    assert "private applicant record" not in text and os.environ["DEEPSEEK_API_KEY"] not in text


def test_stream_empty_usage_chunk(llm):
    class Stream:
        def __iter__(self):
            yield SimpleNamespace(choices=[])
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"))])
        def close(self):
            self.closed = True
    stream = Stream()
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: stream)))
    assert "".join(llm.stream_chat("test")) == "hello" and stream.closed


def test_public_mode_blocks_api(llm, monkeypatch):
    monkeypatch.setenv("APP_ENV", "public_demo")
    with pytest.raises(ValueError):
        llm.chat("test")


def test_multiobjective_adapter(configs):
    from src.pipeline.adapters import CompatibleOptimizer
    from xgboost import XGBClassifier
    cfg = copy.deepcopy(configs.get("model").xgboost.tuning)
    cfg.n_trials, cfg.multi_objective = 2, True
    optimizer = CompatibleOptimizer(cfg)
    X = pd.DataFrame({"x": np.arange(40), "z": np.arange(40) % 3})
    params = optimizer.optimize(XGBClassifier, X, np.arange(40) % 2, X, np.arange(40) % 2,
                                base_params={"n_estimators": 5, "n_jobs": 1})
    assert len(optimizer.study.trials) == 2 and params


def test_protected_files_and_secret_ignores():
    from scripts.check_changes import build_report
    report = build_report()
    assert len(report["protected_unchanged"]) >= 20 and ".env" in report["ignored_secret_paths"]
