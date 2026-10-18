import sys
from pathlib import Path
from unittest.mock import Mock

import dotenv
import pandas as pd
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_cloud_home_survives_reruns_and_page_navigation(monkeypatch):
    monkeypatch.setenv("APP_ENV", "public_demo")
    monkeypatch.setenv("CREDIT_RISK_PUBLIC_DEMO", "1")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    blocked_dotenv = Mock(return_value=False)
    monkeypatch.setattr(dotenv, "load_dotenv", blocked_dotenv)
    from src.utils import config_loader
    monkeypatch.setattr(config_loader, "load_dotenv", blocked_dotenv)

    def forbidden_csv(*args, **kwargs):
        raise AssertionError("公开入口测试不应读取本地 CSV")

    monkeypatch.setattr(pd, "read_csv", forbidden_csv)
    cached_home = sys.modules.pop("app.streamlit_app", None)
    labels = {"训练样本", "入模特征", "XGBoost 测试 AUC", "可用模型"}

    def assert_home(app, expected=None):
        assert not app.exception
        assert any(title.value == "信贷风险工作台" for title in app.title)
        metrics = {metric.label: metric.value for metric in app.metric}
        assert labels <= metrics.keys()
        values = {label: metrics[label] for label in labels}
        assert all(value not in (None, "") for value in values.values())
        if expected is not None:
            assert values == expected
        return values

    try:
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
        first = assert_home(app)
        for _ in range(2):
            assert_home(app.run(), first)

        app.switch_page("pages/1_📊_数据探索.py").run()
        assert not app.exception
        assert any(title.value == "数据探索" for title in app.title)

        app.switch_page("app.py").run()
        assert_home(app, first)
        assert_home(app.run(), first)
    finally:
        sys.modules.pop("app.streamlit_app", None)
        if cached_home is not None:
            sys.modules["app.streamlit_app"] = cached_home
