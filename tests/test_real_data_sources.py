import ast
import io
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "app/pages/1_📊_数据探索.py"


class DataSourceStopped(Exception):
    pass


def sample_frame(marker, amount=10000):
    rows = 60
    return pd.DataFrame({
        "loanAmnt": amount + np.arange(rows), "term": 3, "interestRate": 10.0,
        "installment": 350.0, "annualIncome": 80000.0, "dti": 15.0,
        "revolUtil": 25.0, "ficoRangeLow": 660, "ficoRangeHigh": 664,
        "isDefault": np.arange(rows) % 2, "fixture_marker": marker,
    })


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("CREDIT_RISK_PUBLIC_DEMO", raising=False)
    module = ModuleType("isolated_real_data_runtime")
    errors = []

    def stop():
        raise DataSourceStopped

    module.__dict__.update(ROOT=tmp_path, os=os, pd=pd,
                           st=SimpleNamespace(cache_resource=st.cache_resource, session_state={},
                                              error=errors.append, stop=stop))
    names = {"public_mode", "available_real_data", "_read_real_data", "load_real_data",
             "current_data", "source_label"}
    path = ROOT / "app/components/runtime.py"
    source = ast.parse(path.read_text(encoding="utf-8-sig"))
    nodes = [node for node in source.body
             if isinstance(node, ast.FunctionDef) and node.name in names
             or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                                                     and target.id == "REAL_DATA_FILES" for target in node.targets)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), module.__dict__)
    module.errors = errors
    module.generated_data = lambda n=5000: sample_frame("demo_fixture")
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    sample_frame("merged_fixture", 20000).to_csv(raw / "train_merged.csv", index=False)
    sample_frame("local_fixture", 30000).to_csv(raw / "train.csv", index=False)
    demo = tmp_path / "demo_assets"
    demo.mkdir()
    sample_frame("demo_fixture").to_json(demo / "sample.json", orient="table")
    yield module
    module._read_real_data.clear()


@pytest.fixture
def page_runtime(runtime, monkeypatch):
    runtime.st = st
    runtime.init_page = lambda *args, **kwargs: (None, SimpleNamespace(demo=SimpleNamespace(max_upload_mb=20)))
    monkeypatch.setitem(sys.modules, "app.components.runtime", runtime)
    return runtime


def click(app, label):
    next(button for button in app.button if button.label == label).click().run()
    assert not app.exception


def source(app, label):
    next(radio for radio in app.radio if radio.label == "数据来源").set_value(label).run()
    assert not app.exception


def uploaded(frame):
    data = frame.to_csv(index=False).encode("utf-8")
    file = io.BytesIO(data)
    file.name = "test_fixture.csv"
    file.size = len(data)
    return file


@pytest.mark.parametrize("name", ["../outside.csv", "other.csv", "data/raw/train.csv", "", None])
def test_only_named_source_files_are_allowed(runtime, name):
    with pytest.raises(ValueError, match="train_merged.csv.*train.csv"):
        runtime.load_real_data(name)


def test_same_filename_replacement_refreshes_cached_frame(runtime):
    path = runtime.ROOT / "data/raw/train.csv"
    first = runtime.load_real_data("train.csv")
    stamp = path.stat().st_mtime_ns
    sample_frame("replacement_fixture", 90000).to_csv(path, index=False)
    os.utime(path, ns=(stamp + 2000000000, stamp + 2000000000))

    refreshed = runtime.load_real_data("train.csv")

    assert first["fixture_marker"].eq("local_fixture").all()
    assert refreshed["fixture_marker"].eq("replacement_fixture").all()
    assert refreshed["loanAmnt"].iloc[0] == 90000


def test_cached_source_is_read_once_and_callers_receive_independent_copies(runtime):
    calls = []
    read_csv = pd.read_csv

    def counted_read(path):
        calls.append(path)
        return read_csv(path)

    runtime.pd = SimpleNamespace(read_csv=counted_read)
    first = runtime.load_real_data("train.csv")
    first.loc[0, "loanAmnt"] = -1
    first["caller_only"] = True
    second = runtime.load_real_data("train.csv")

    assert len(calls) == 1
    assert second["loanAmnt"].iloc[0] == 30000 and "caller_only" not in second


def test_invalid_source_is_rejected_before_becoming_current(runtime):
    path = runtime.ROOT / "data/raw/train.csv"
    pd.DataFrame({"isDefault": [0, 1] * 30}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="数据缺少字段"):
        runtime.load_real_data("train.csv")


def test_missing_selected_source_stops_instead_of_falling_back(runtime):
    runtime.st.session_state.update(data_source="train.csv", active_data=sample_frame("prior_fixture"))
    (runtime.ROOT / "data/raw/train.csv").unlink()
    with pytest.raises(DataSourceStopped):
        runtime.current_data()
    assert len(runtime.errors) == 1 and "当前真实数据无法读取" in runtime.errors[0]
    assert runtime.st.session_state["data_source"] == "train.csv"


@pytest.mark.parametrize("flag,value", [("APP_ENV", "public_demo"), ("CREDIT_RISK_PUBLIC_DEMO", "1")])
def test_public_mode_ignores_local_source_and_uploaded_session(runtime, monkeypatch, flag, value):
    runtime.st.session_state.update(data_source="train.csv", active_data=sample_frame("private_fixture"),
                                    data_label="会话测试夹具")
    monkeypatch.setenv(flag, value)

    def forbidden(*args, **kwargs):
        raise AssertionError("公开模式不应读取本地 CSV")

    runtime.load_real_data = forbidden
    assert runtime.current_data()["fixture_marker"].eq("demo_fixture").all()
    assert runtime.source_label() == "合成演示数据"


def test_page_source_switches_and_demo_reset(page_runtime, monkeypatch):
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: uploaded(sample_frame("upload_fixture", 40000)))
    app = AppTest.from_file(str(PAGE), default_timeout=30).run()
    assert not app.exception

    source(app, "本地真实数据")
    click(app, "加载本地真实数据")
    assert app.session_state["data_source"] == "train_merged.csv"
    assert "active_data" not in app.session_state
    assert app.dataframe[0].value["fixture_marker"].eq("merged_fixture").all()

    source(app, "上传 CSV")
    click(app, "使用这份数据")
    assert "data_source" not in app.session_state
    assert app.session_state["active_data"]["fixture_marker"].eq("upload_fixture").all()
    assert app.dataframe[0].value["fixture_marker"].eq("upload_fixture").all()

    source(app, "本地真实数据")
    click(app, "加载本地真实数据")
    source(app, "本地天池 CSV")
    click(app, "加载前 20,000 行用于页面探索")
    assert "data_source" not in app.session_state
    assert app.session_state["active_data"]["fixture_marker"].eq("local_fixture").all()
    assert app.dataframe[0].value["fixture_marker"].eq("local_fixture").all()

    source(app, "本地真实数据")
    click(app, "加载本地真实数据")
    source(app, "内置合成数据")
    click(app, "恢复合成演示数据")
    assert all(key not in app.session_state for key in ("data_source", "active_data", "data_label"))
    assert app.dataframe[0].value["fixture_marker"].eq("demo_fixture").all()


@pytest.mark.parametrize("choice", ["本地真实数据", "上传 CSV", "本地天池 CSV"])
def test_failed_load_preserves_previous_source_without_false_success(page_runtime, monkeypatch, choice):
    invalid = pd.DataFrame({"isDefault": [0, 1] * 30})
    invalid.to_csv(page_runtime.ROOT / "data/raw/train.csv", index=False)
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: uploaded(invalid))
    app = AppTest.from_file(str(PAGE), default_timeout=30)
    app.session_state["data_source"] = "train_merged.csv"
    app.session_state["data_label"] = "上一次成功选择的测试夹具"
    app.run()
    source(app, choice)
    if choice == "本地真实数据":
        next(item for item in app.selectbox if item.label == "选择数据文件").set_value("train.csv").run()
        click(app, "加载本地真实数据")
    elif choice == "本地天池 CSV":
        click(app, "加载前 20,000 行用于页面探索")

    assert not app.exception and app.error
    assert app.session_state["data_source"] == "train_merged.csv"
    assert app.session_state["data_label"] == "上一次成功选择的测试夹具"
    assert "active_data" not in app.session_state
    assert not any("已加载" in item.value or "当前数据源已更新" in item.value for item in app.success)
    assert app.dataframe[0].value["fixture_marker"].eq("merged_fixture").all()
