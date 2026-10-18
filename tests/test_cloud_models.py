import ast
import copy
import os
import pickle
import shutil
from pathlib import Path
from types import ModuleType, SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


class ModelLoadingStopped(Exception):
    pass


def small_bundle(marker="fixture"):
    return {"meta": {"schema_version": 2, "run_id": marker}, "models": {},
            "preprocessor": None, "feature_builder": None, "feature_selector": None,
            "input_defaults": {}, "eval_results": {}, "thresholds": {}, "project_config": {},
            "background": None, "raw_test": None, "raw_baseline": None,
            "y_test": None, "test_predictions": {}}


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.delenv("CREDIT_RISK_MODEL_PATH", raising=False)
    monkeypatch.delenv("CREDIT_RISK_PUBLIC_DEMO", raising=False)
    monkeypatch.setenv("APP_ENV", "development")
    module = ModuleType("isolated_cloud_model_runtime")
    errors = []

    def stop():
        raise ModelLoadingStopped

    module.__dict__.update(ROOT=tmp_path, Path=Path, os=os, pd=pd, pickle=pickle, joblib=joblib,
                           st=SimpleNamespace(session_state={}, error=errors.append, stop=stop))
    names = {"public_mode", "configured_model_path", "bundle_path", "_load_bundle", "artifacts",
             "current_data", "source_label"}
    path = ROOT / "app/components/runtime.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    nodes = [copy.deepcopy(node) for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), module.__dict__)
    module.REAL_DATA_FILES = ("train_merged.csv", "train.csv")
    module.generated_data = lambda n=5000: pd.DataFrame({"fixture_kind": ["synthetic"]})
    module.errors = errors
    return module


@pytest.mark.parametrize("value", [None, "", "   "])
def test_no_configured_path_preserves_default(runtime, monkeypatch, value):
    if value is not None:
        monkeypatch.setenv("CREDIT_RISK_MODEL_PATH", value)
    assert runtime.configured_model_path() is None
    assert runtime.bundle_path() == runtime.ROOT / "demo_assets/bundle.joblib"


def test_local_latest_selection_is_preserved(runtime):
    runtime.st.session_state["artifact_source"] = "本地最新训练"
    assert runtime.bundle_path() == runtime.ROOT / "data/models/latest/bundle.joblib"


def test_public_mode_ignores_local_latest_without_override(runtime, monkeypatch):
    runtime.st.session_state["artifact_source"] = "本地最新训练"
    monkeypatch.setenv("CREDIT_RISK_PUBLIC_DEMO", "1")
    assert runtime.bundle_path() == runtime.ROOT / "demo_assets/bundle.joblib"


@pytest.mark.parametrize("absolute", [False, True])
def test_configured_path_resolves_and_takes_priority(runtime, monkeypatch, absolute):
    relative = Path("cloud_models") / "nested" / ".." / "model.joblib"
    selected = runtime.ROOT / relative if absolute else relative
    monkeypatch.setenv("CREDIT_RISK_MODEL_PATH", "  " + str(selected) + "  ")
    runtime.st.session_state["artifact_source"] = "本地最新训练"
    monkeypatch.setenv("CREDIT_RISK_PUBLIC_DEMO", "1")
    assert runtime.bundle_path() == (runtime.ROOT / "cloud_models/model.joblib").resolve()


def test_public_override_does_not_read_csv_or_call_api(runtime, monkeypatch):
    import openai

    path = runtime.ROOT / "cloud.joblib"
    joblib.dump(small_bundle("cloud-fixture"), path)
    monkeypatch.setenv("CREDIT_RISK_MODEL_PATH", str(path))
    monkeypatch.setenv("CREDIT_RISK_PUBLIC_DEMO", "1")
    runtime.st.session_state.update(data_source="train_merged.csv", active_data="private-fixture",
                                    data_label="private-label", artifact_source="本地最新训练")

    def forbidden(*args, **kwargs):
        pytest.fail("public model loading must not read customer CSV or call API")

    monkeypatch.setattr(pd, "read_csv", forbidden)
    monkeypatch.setattr(openai, "OpenAI", forbidden)
    runtime.load_real_data = forbidden
    loaded = runtime.artifacts()
    assert loaded["meta"]["run_id"] == "cloud-fixture"
    assert runtime.current_data()["fixture_kind"].tolist() == ["synthetic"]
    assert runtime.source_label() == "合成演示数据"
    assert not runtime.errors


def test_missing_override_stops_without_falling_back(runtime, monkeypatch):
    fallback = runtime.ROOT / "demo_assets/bundle.joblib"
    fallback.parent.mkdir()
    joblib.dump(small_bundle("fallback-fixture"), fallback)
    monkeypatch.setenv("CREDIT_RISK_MODEL_PATH", "missing.joblib")

    def forbidden(*args):
        pytest.fail("a missing explicit model must not silently load the demo bundle")

    runtime._load_bundle = forbidden
    with pytest.raises(ModelLoadingStopped):
        runtime.artifacts()
    assert runtime.errors and "不存在" in runtime.errors[-1]


def test_missing_default_can_still_return_none(runtime):
    assert runtime.artifacts() is None
    assert not runtime.errors


@pytest.mark.parametrize("content", [b"", b"not a model", b"broken", b"\x80\x04badpickle"])
def test_broken_model_file_stops_with_actionable_message(runtime, monkeypatch, content):
    path = runtime.ROOT / "bad.joblib"
    path.write_bytes(content)
    monkeypatch.setenv("CREDIT_RISK_MODEL_PATH", str(path))
    with pytest.raises(ModelLoadingStopped):
        runtime.artifacts()
    assert "预训练模型加载失败" in runtime.errors[-1]
    assert "bundle.joblib" in runtime.errors[-1]


@pytest.mark.parametrize("change", ["schema", "meta_none", "missing_models", "not_mapping"])
def test_incompatible_bundle_stops_without_falling_back(runtime, monkeypatch, change):
    bundle = small_bundle()
    if change == "schema":
        bundle["meta"]["schema_version"] = 1
    elif change == "meta_none":
        bundle["meta"] = None
    elif change == "missing_models":
        bundle.pop("models")
    else:
        bundle = ["not-a-model-mapping"]
    path = runtime.ROOT / "incompatible.joblib"
    joblib.dump(bundle, path)
    monkeypatch.setenv("CREDIT_RISK_MODEL_PATH", str(path))
    with pytest.raises(ModelLoadingStopped):
        runtime.artifacts()
    assert "预训练模型加载失败" in runtime.errors[-1]


def test_copied_demo_bundle_preserves_each_model_prediction(runtime, monkeypatch):
    from src.pipeline.inference import InferenceService

    source = ROOT / "demo_assets/bundle.joblib"
    original = joblib.load(source)
    assert original["meta"]["source"] == "合成演示数据"
    deployed = runtime.ROOT / "cloud-demo.joblib"
    shutil.copy2(source, deployed)
    monkeypatch.setenv("CREDIT_RISK_MODEL_PATH", str(deployed))
    monkeypatch.setenv("CREDIT_RISK_PUBLIC_DEMO", "1")
    loaded = runtime.artifacts()
    local_service = InferenceService(original)
    cloud_service = InferenceService(loaded)
    sample = pd.DataFrame([original["input_defaults"]])
    assert set(cloud_service.models) == set(local_service.models)
    for model in local_service.models:
        np.testing.assert_allclose(cloud_service.predict_proba(sample, model),
                                   local_service.predict_proba(sample, model), rtol=0, atol=1e-12)
