import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.chart_utils import gauge_chart
from app.components.explanations import render_single, single_explanation
from app.components.forms import loan_form
from app.components.runtime import (example_report, init_page, llm_tools, require_service,
                                    risk_limits, rules_agent, saved_prediction, stream_to_state)
from src.pipeline.inference import decision_for, risk_level

_, cfg = init_page("单笔风险评估", "输入一笔贷款申请，查看模型判断、关键因素和辅助审批建议。", "PREDICTION")
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
    tab1, tab2, tab3 = st.tabs(["风险解释", "风险报告", "教学规则检查"])
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
        choices = ["建议通过", "人工复核", "建议拒绝"]
        decision = st.selectbox("待检查的审批建议", choices, index=choices.index(decision_for(p, low, high)))
        result = rules_agent().check(prediction["features"], decision, probability=p, low=low, high=high)
        (st.success if result["is_compliant"] else st.warning)("当前教学规则检查通过" if result["is_compliant"] else "存在需要核对的事项")
        for item in result["violations"]:
            st.error(item["reason"])
        for warning in result["warnings"]:
            st.warning(warning)
        for recommendation in result["recommendations"]:
            st.write("• " + recommendation)
        client, _, _ = llm_tools()
        if client.configured and st.button("让规则检查 Agent 解释结果"):
            try:
                st.session_state["compliance_llm"] = rules_agent().check(
                    prediction["features"], decision, interest_rate=prediction["features"]["interestRate"],
                    probability=p, low=low, high=high, use_llm=True)
            except (ValueError, RuntimeError) as error:
                st.error(str(error))
        if "compliance_llm" in st.session_state:
            st.json(st.session_state["compliance_llm"])
        st.caption("教学规则和阈值由项目配置决定，结果供课程展示和人工复核参考。")
else:
    st.info("完成表单后，结果会保留在当前会话，并同步给智能助手与情景对比页面。")
