import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["APP_ENV"] = "development"
os.environ.pop("CREDIT_RISK_PUBLIC_DEMO", None)

from streamlit.testing.v1 import AppTest
from unittest.mock import PropertyMock, patch


def button(app, label):
    return next(item for item in app.button if item.label == label)


def widget(items, label):
    return next(item for item in items if item.label == label)


def check(app, label, results):
    errors = [e.message for e in app.exception]
    results.append({"step": label, "errors": errors})
    if errors:
        raise AssertionError(json.dumps(results[-1], ensure_ascii=True))


def main():
    results = []
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=90).run()
    app.switch_page("pages/4_🎯_单笔预测.py").run()
    button(app, "开始风险评估").click().run()
    check(app, "prediction_submit", results)
    assert app.session_state["prediction"]["probability"] >= 0
    assert "未调用 DeepSeek" in app.session_state["risk_report"]
    first_report = app.session_state["risk_report"]
    app.run()
    assert app.session_state["risk_report"] == first_report
    check(app, "prediction_persists_on_rerun", results)
    widget(app.selectbox, "待检查的审批建议").set_value("建议通过").run()
    check(app, "compliance_decision_change", results)
    app.switch_page("pages/6_💬_LLM智能助手.py").run()
    app.chat_input[0].set_value("请解释 AUC").run()
    check(app, "chat_demo", results)
    assert len(app.session_state["chat_history"]) == 2
    widget(app.radio, "助手模式").set_value("风险报告").run()
    check(app, "assistant_report_context", results)
    widget(app.radio, "助手模式").set_value("异常分析").run()
    check(app, "assistant_case_analysis", results)
    app.switch_page("pages/10_风险情景对比.py").run()
    widget(app.number_input, "调整后贷款金额（元）").set_value(30000.0).run()
    button(app, "计算前后变化").click().run()
    check(app, "scenario_comparison", results)
    assert len(app.session_state["scenario_result"]["probabilities"]) == 2
    app.switch_page("pages/2_🔧_特征工程.py").run()
    button(app, "预览前 500 行预处理效果").click().run()
    check(app, "knn_preview", results)
    assert app.session_state["clean_preview"][1]["dti"].isna().sum() == 0
    button(app, "检查并应用到预览").click().run()
    check(app, "feature_suggestion_preview", results)
    app.switch_page("pages/8_模型监控.py").run()
    widget(app.radio, "新样本场景").set_value("模拟收入分布变化").run()
    check(app, "drift_simulation", results)
    app.switch_page("pages/9_AB测试.py").run()
    widget(app.slider, "A 组流量比例").set_value(0.0).run()
    check(app, "ab_zero_ratio", results)
    app.switch_page("pages/3_🤖_模型训练.py").run()
    widget(app.multiselect, "选择模型").set_value(["xgboost", "lightgbm"]).run()
    button(app, "开始训练").click().run()
    check(app, "train_and_reload_latest", results)
    assert app.session_state["artifact_source"] == "本地最新训练"
    (ROOT / ".local/interactions.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=True))


if __name__ == "__main__":
    with patch("src.llm.client.DeepSeekClient.configured", new_callable=PropertyMock, return_value=False):
        main()
