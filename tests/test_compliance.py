import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import pytest

from src.llm.agents.compliance_agent import ComplianceAgent
from src.llm.prompts import PromptManager
from src.utils.config_loader import ConfigLoader


ROOT = Path(__file__).resolve().parents[1]
VALID = {"is_compliant": True, "violations": [], "warnings": [], "recommendations": ["人工复核"]}


@pytest.fixture
def applicant():
    return {"loanAmnt": 30000, "annualIncome": 100000, "term": 3, "interestRate": 10,
            "dti": 20, "age": 35, "repayment_proof": False, "loan_purpose": "日常消费", "lpr": 3.0}


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("CREDIT_RISK_PUBLIC_DEMO", raising=False)
    configs = ConfigLoader(str(ROOT / "configs"))
    llm_cfg = copy.deepcopy(configs.get("llm"))
    calls = []

    def chat(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return json.dumps(VALID)

    client = SimpleNamespace(config=llm_cfg, configured=True, chat=chat)
    result = ComplianceAgent(client, PromptManager(llm_cfg), configs.get("project").compliance)
    result.calls = calls
    return result


def test_compliance_registered_and_passes_parameters(agent, applicant):
    assert agent.enabled
    agent.agent_config.temperature = 0.2
    agent.agent_config.max_tokens = 1000
    result = agent.check(applicant, "建议通过", rules=["贷款金额超过 10 万元时需要收入证明"], use_llm=True)
    prompt, kwargs = agent.calls[0]
    assert result["is_compliant"] is True and result["llm_analysis"] == VALID
    assert result["source"] == "DeepSeek 逐条规则检查" and "local_result" in result
    assert "贷款金额超过 10 万元" in prompt and "4 倍" in prompt and "35" in prompt
    assert kwargs["temperature"] == 0.2 and kwargs["max_tokens"] == 1000
    assert kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("use_llm", [False, True])
def test_disabled_agent_does_not_execute(agent, applicant, use_llm):
    agent.agent_config.enabled = False
    with pytest.raises(ValueError, match="已在.*关闭"):
        agent.check(applicant, "建议通过", use_llm=use_llm)
    assert not agent.calls


def test_local_checks_all_three_course_rules(agent, applicant):
    applicant.update(age=65, interestRate=20, loan_purpose="房地产投资")
    result = agent.check(applicant, "建议通过")
    assert not result["is_compliant"] and len(result["violations"]) == 3
    assert any("4 倍" in item["rule"] for item in result["violations"])
    assert any("60 岁" in item["rule"] for item in result["violations"])
    assert any("房地产投资" in item["rule"] for item in result["violations"])
    assert not agent.calls


def test_proof_and_rate_boundary(agent, applicant):
    applicant.update(age=65, repayment_proof=True, interestRate=12)
    assert agent.check(applicant, "建议通过")["is_compliant"]


def test_missing_policy_information_warns(agent, applicant):
    for field in ("age", "lpr", "loan_purpose"):
        applicant.pop(field)
    result = agent.check(applicant, "建议通过")
    assert len(result["warnings"]) == 3


def test_local_custom_rule_is_not_claimed_executed(agent, applicant):
    result = agent.check(applicant, "建议通过", rules=["金额超过 2 万元需额外材料"])
    assert any("尚未由本地程序执行" in item for item in result["warnings"])


@pytest.mark.parametrize("env", ["APP_ENV", "CREDIT_RISK_PUBLIC_DEMO"])
def test_public_mode_never_calls_api(agent, applicant, monkeypatch, env):
    monkeypatch.setenv(env, "public_demo" if env == "APP_ENV" else "1")
    with pytest.raises(ValueError, match="公开演示模式"):
        agent.check(applicant, "建议通过", use_llm=True)
    assert not agent.calls


@pytest.mark.parametrize("text", [
    "not json", '{"is_compliant": true}', '[]',
    json.dumps({**VALID, "is_compliant": "true"}),
    json.dumps({**VALID, "violations": [{"rule": "课程规则"}]}),
    json.dumps({**VALID, "violations": [{"rule": "课程规则", "reason": 1}]}),
    json.dumps({**VALID, "warnings": "警告"}),
    json.dumps({**VALID, "recommendations": [None]}),
    json.dumps({**VALID, "is_compliant": False}),
    json.dumps({**VALID, "summary": "多余字段"}),
    json.dumps(VALID) + " trailing text",
])
def test_bad_json_is_clear_failure(text):
    with pytest.raises(ValueError, match="政策合规检查返回格式错误"):
        ComplianceAgent._validate_response(text)


def test_json_fence_and_failure_result():
    result = {"is_compliant": False, "violations": [{"rule": "课程规则", "reason": "超过阈值"}],
              "warnings": [], "recommendations": ["补充材料"]}
    text = "```json\n" + json.dumps(result) + "\n```"
    assert ComplianceAgent._validate_response(text) == result


def test_llm_does_not_hide_invalid_result(agent, applicant):
    agent.client.chat = lambda *args, **kwargs: '{"is_compliant": "true"}'
    with pytest.raises(ValueError, match="返回格式错误"):
        agent.check(applicant, "建议通过", use_llm=True)


@pytest.mark.parametrize("value", [None, "bad", float("nan"), True])
def test_invalid_money_does_not_crash(agent, applicant, value):
    applicant["annualIncome"] = value
    assert not agent.check(applicant, "建议通过")["is_compliant"]


def test_prediction_page_has_real_compliance_entry(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from src.llm.client import DeepSeekClient
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("CREDIT_RISK_PUBLIC_DEMO", raising=False)
    calls = []

    def chat(self, prompt, **kwargs):
        calls.append(prompt)
        return json.dumps(VALID)

    monkeypatch.setattr(DeepSeekClient, "chat", chat)
    with patch.object(DeepSeekClient, "configured", new_callable=PropertyMock, return_value=True):
        app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=90).run()
        app.switch_page("pages/4_🎯_单笔预测.py").run()
        next(item for item in app.button if item.label == "开始风险评估").click().run()
        assert not app.exception
        assert any(tab.label == "政策合规检查" for tab in app.tabs)
        labels = {item.label for item in app.number_input}
        assert {"检查使用的 LPR（%）", "申请人年龄", "待检查的贷款年利率（%）"} <= labels
        assert any(item.label == "贷款用途" for item in app.selectbox)
        next(item for item in app.text_area if item.label.startswith("附加自定义")).set_value("贷款金额超过 10 万元时需要收入证明").run()
        next(item for item in app.button if item.label == "使用 DeepSeek 执行政策合规检查").click().run()
        assert not app.exception and len(calls) == 1
        assert "10 万元" in calls[0]
        assert app.session_state["compliance_llm"]["llm_analysis"] == VALID
        next(item for item in app.number_input if item.label == "申请人年龄").set_value(65).run()
        assert any("原结果已隐藏" in item.value for item in app.caption)


def test_prediction_page_public_disables_deepseek(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from src.llm.client import DeepSeekClient
    monkeypatch.setenv("CREDIT_RISK_PUBLIC_DEMO", "1")

    def forbidden(*args, **kwargs):
        raise AssertionError("公开模式不应调用 API")

    monkeypatch.setattr(DeepSeekClient, "chat", forbidden)
    with patch.object(DeepSeekClient, "configured", new_callable=PropertyMock, return_value=True):
        app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=90).run()
        app.switch_page("pages/4_🎯_单笔预测.py").run()
        next(item for item in app.button if item.label == "开始风险评估").click().run()
        assert not app.exception
        assert next(item for item in app.button if item.label == "使用 DeepSeek 执行政策合规检查").disabled
        assert any("公开演示只展示本地规则结果" in item.value for item in app.info)
