import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.chart_utils import gauge_chart
from app.components.explanations import render_single, single_explanation
from app.components.forms import loan_form
from app.components.runtime import (example_report, init_page, llm_tools, require_service,
                                    public_mode, risk_limits, saved_prediction, stream_to_state)
from src.llm.agents.compliance_agent import ComplianceAgent
from src.pipeline.inference import decision_for, risk_level


def render_compliance_result(result):
    message = "已执行的课程规则检查通过" if result["is_compliant"] else "存在需要复核的课程规则事项"
    (st.success if result["is_compliant"] else st.warning)(message)
    for item in result["violations"]:
        st.error(item["rule"] + "：" + item["reason"])
    for warning in result["warnings"]:
        st.warning(warning)
    for recommendation in result["recommendations"]:
        st.write("• " + recommendation)


_, cfg = init_page("单笔风险评估", "输入一笔贷款申请，查看模型判断、关键因素和辅助审批建议。", "PREDICTION")
st.info("完成风险评估后，可在“政策合规检查”中核对申请资料和审批结果。")
service, bundle = require_service()
low, high = risk_limits(bundle)
model_name = st.selectbox("评估模型", list(service.models))
st.caption("未填写的其他字段采用该模型训练集的中位数或众数，月供按金额、期限和利率计算。")
submitted, raw = loan_form(bundle["input_defaults"])
if submitted:
    try:
        probability = float(service.predict_proba(raw, model_name)[0])
        explanation, tree_name = single_explanation(service, bundle, raw, model_name)
        prediction = {"features": raw.iloc[0].to_dict(), "probability": probability,
                      "explanation": explanation, "explanation_model": tree_name,
                      "model_name": model_name, "run_id": bundle["meta"]["run_id"]}
        st.session_state["prediction"] = prediction
        st.session_state["risk_report"] = example_report(prediction["features"], probability, model_name,
                                                         explanation, (low, high), tree_name)
    except (ValueError, KeyError, RuntimeError) as error:
        st.error(str(error))
prediction = saved_prediction()
if prediction:
    p = prediction["probability"]
    st.caption("本次结果模型：" + prediction["model_name"] + " · 版本：" + prediction["run_id"])
    left, middle, right = st.columns([1.3, 1, 1])
    with left:
        st.plotly_chart(gauge_chart(p*100), use_container_width=True)
    with middle:
        st.metric("风险等级", risk_level(p, low, high))
        st.metric("审批建议", decision_for(p, low, high))
    with right:
        st.metric("预计月供", f'{prediction["features"]["installment"]:,.0f} 元')
        st.metric("年收入", f'{prediction["features"]["annualIncome"]:,.0f} 元')
    tab1, tab2, tab3 = st.tabs(["风险解释", "风险报告", "政策合规检查"])
    with tab1:
        render_single(prediction["explanation"], f'SHAP 解释对象：{prediction["explanation_model"]}')
        if prediction["model_name"] != prediction["explanation_model"]:
            st.caption("这里解释树子模型，不能作为融合模型或 MLP 的精确归因。")
    with tab2:
        client, prompts, llm_cfg = llm_tools()
        if client.configured:
            if st.button("使用 DeepSeek 生成真实报告", type="primary"):
                from src.llm.agents.report_agent import ReportAgent
                agent = ReportAgent(client, prompts, llm_cfg.agents.report_agent)
                stream_to_state(agent.generate(prediction["features"], p, prediction["explanation"],
                                                stream=True, model_name=prediction["model_name"],
                                                low_risk=low, high_risk=high,
                                                explanation_model=prediction["explanation_model"]), "risk_report")
        else:
            st.info("当前展示本地模板报告，未调用 DeepSeek。")
        report = st.session_state.get("risk_report", "")
        st.markdown(report)
        st.download_button("下载当前报告", report, file_name="risk_report.md", mime="text/markdown")
    with tab3:
        st.subheader("政策合规检查 Agent · 课程演示")
        st.caption("规则来源为课程指导书和项目配置。4 倍 LPR、年龄证明及用途限制是本项目的课程规则，不代表全国通用法律；结果供人工复核参考。")
        client, prompts, llm_cfg = llm_tools()
        agent = ComplianceAgent(client, prompts, cfg.compliance, agent_config=llm_cfg.agents.compliance)
        a, b, c = st.columns(3)
        choices = ["建议通过", "人工复核", "建议拒绝"]
        with a:
            decision = st.selectbox("待检查的审批建议", choices, index=choices.index(decision_for(p, low, high)))
            lpr = st.number_input("检查使用的 LPR（%）", min_value=0.01, max_value=30.0, value=3.0, step=0.05)
            st.caption("3.0% 只是可修改的演示值，请按课程或实际核对需要手工输入。")
        with b:
            age = st.number_input("申请人年龄", min_value=18, max_value=100, value=35)
            repayment_proof = st.checkbox("已提供额外还款能力证明")
        with c:
            rate = st.number_input("待检查的贷款年利率（%）", min_value=0.0, max_value=100.0,
                                   value=float(prediction["features"]["interestRate"]), step=0.1)
            purpose = st.selectbox("贷款用途", ["日常消费", "教育培训", "装修", "经营周转", "房地产投资", "其他"])
            if purpose == "其他":
                purpose = st.text_input("其他贷款用途", max_chars=100)
        custom_text = st.text_area("附加自定义检查规则（每行一条）", placeholder="例如：金额超过 10 万元时需要补充收入证明", max_chars=4000)
        extra_rules = [line.strip() for line in custom_text.splitlines() if line.strip()]
        check_features = dict(prediction["features"])
        check_features.update(age=age, repayment_proof=repayment_proof, loan_purpose=purpose, lpr=lpr,
                              interestRate=rate)
        with st.expander("查看检查规则与实现位置"):
            for rule in agent.rule_list(extra_rules):
                st.write("• " + rule)
            st.code("步骤 1：configs/prompts/compliance_check.j2\n步骤 2：src/llm/agents/compliance_agent.py\n步骤 3：configs/llm_config.yaml → agents.compliance\n步骤 4：单笔预测 → 政策合规检查", language="text")
        st.subheader("1. 本地规则检查 · 无 API 费用")
        if not agent.enabled:
            st.warning("政策合规检查 Agent 已在 agents.compliance.enabled 中关闭，本地与 DeepSeek 检查均停止。")
        else:
            result = agent.check(check_features, decision, interest_rate=rate, rules=extra_rules,
                                 probability=p, low=low, high=high)
            render_compliance_result(result)
        st.caption("修改这里只影响规则检查输入，不会重新训练模型或改动上方预测。附加自定义规则由 DeepSeek 或人工核对，本地程序不执行自由文本。")
        st.subheader("2. DeepSeek 逐条检查 · 真实 API 调用")
        if public_mode():
            st.info("公开演示只展示本地规则结果，不调用 DeepSeek。")
        elif not client.configured:
            st.info("本地配置 DeepSeek 后可进行真实检查；密钥不显示在页面中。")
        signature = json.dumps({"features": check_features, "decision": decision, "rules": extra_rules,
                                "probability": p, "run_id": prediction["run_id"]}, sort_keys=True, default=str)
        if st.button("使用 DeepSeek 执行政策合规检查", type="primary",
                     disabled=not agent.enabled or public_mode() or not client.configured):
            try:
                with st.spinner("正在逐条核对课程规则和附加规则…"):
                    result = agent.check(check_features, decision, interest_rate=rate, rules=extra_rules,
                                         probability=p, low=low, high=high, use_llm=True)
                st.session_state["compliance_llm"] = result
                st.session_state["compliance_signature"] = signature
            except (ValueError, RuntimeError) as error:
                st.session_state.pop("compliance_llm", None)
                st.session_state.pop("compliance_signature", None)
                st.error(str(error))
        if st.session_state.get("compliance_signature") == signature and agent.enabled and not public_mode():
            llm_result = st.session_state["compliance_llm"]
            st.caption("以下为 DeepSeek 返回并通过四字段 JSON 格式检查的结果。")
            render_compliance_result(llm_result)
            with st.expander("查看 DeepSeek 原始结构化结果"):
                st.json(llm_result["llm_analysis"])
        elif "compliance_llm" in st.session_state:
            st.caption("检查输入已变化，原结果已隐藏，请重新执行 DeepSeek 检查。")
else:
    st.info("完成表单后，结果会保留在当前会话，并同步给智能助手与情景对比页面。")
