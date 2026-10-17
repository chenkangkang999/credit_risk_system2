import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.components.runtime import init_page, require_service, risk_limits, saved_prediction
from src.pipeline.inference import decision_for, monthly_payment, risk_level

init_page("风险情景对比", "保持其他信息不变，观察申请条件调整后模型如何响应。", "SCENARIOS")
service, bundle = require_service()
prediction = saved_prediction()
base = dict(prediction["features"] if prediction else bundle["input_defaults"])
name = st.selectbox("对比模型", list(service.models))
st.caption("使用最近一笔申请作为基准；尚未评估时使用训练集默认申请。")
with st.form("scenario"):
    a, b, c, d = st.columns(4)
    amount = a.number_input("调整后贷款金额（元）", 1000.0, 500000.0, min(500000.0, max(1000.0, float(base["loanAmnt"]))), step=1000.0)
    income = b.number_input("调整后年收入（元）", 0.0, 10000000.0, min(10000000.0, max(0.0, float(base["annualIncome"]))), step=1000.0)
    interest = c.number_input("调整后年利率（%）", 0.0, 40.0, min(40.0, max(0.0, float(base["interestRate"]))), step=0.5)
    term = d.selectbox("调整后期限（年）", [3, 5], index=1 if base.get("term") == 5 else 0)
    submitted = st.form_submit_button("计算前后变化", type="primary")
if submitted:
    try:
        changed = dict(base)
        base["installment"] = monthly_payment(base["loanAmnt"], base["interestRate"], base["term"])
        changed.update(loanAmnt=amount, annualIncome=income, interestRate=interest, term=term,
                       installment=monthly_payment(amount, interest, term))
        probabilities = service.predict_proba(pd.DataFrame([base, changed]), name)
        st.session_state["scenario_result"] = {"rows": [base, changed], "probabilities": probabilities,
                                              "name": name, "run_id": bundle["meta"]["run_id"]}
        from app.components.explanations import single_explanation
        first, tree = single_explanation(service, bundle, pd.DataFrame([base]), name)
        second, _ = single_explanation(service, bundle, pd.DataFrame([changed]), name)
        if first and second:
            st.session_state["scenario_result"]["shap_change"] = pd.Series(
                second.shap_values-first.shap_values, index=first.feature_names)
            st.session_state["scenario_result"]["explanation_model"] = tree
    except ValueError as error:
        st.error(str(error))
result = st.session_state.get("scenario_result")
if result and result["run_id"] == bundle["meta"]["run_id"]:
    p0, p1 = result["probabilities"]
    low, high = risk_limits(bundle)
    a, b, c = st.columns(3)
    a.metric("原申请违约概率", f"{p0:.2%}")
    b.metric("调整后违约概率", f"{p1:.2%}", f"{(p1-p0)*100:+.2f} 个百分点", delta_color="inverse")
    c.metric("调整后风险等级", risk_level(p1, low, high))
    frame = pd.DataFrame(result["rows"], index=["原申请", "调整后"])
    fields = ["loanAmnt", "annualIncome", "interestRate", "term", "installment"]
    st.dataframe(frame[fields].T.rename(columns={"原申请": "原申请", "调整后": "调整后"}), use_container_width=True)
    st.plotly_chart(px.bar(x=["原申请", "调整后"], y=result["probabilities"],
                           labels={"x": "场景", "y": "违约概率"}, color=["原申请", "调整后"],
                           color_discrete_sequence=["#3468ea", "#57b9a7"]), use_container_width=True)
    st.write(f'模型：{result["name"]}；调整后建议：{decision_for(p1, low, high)}')
    if "shap_change" in result:
        change = result["shap_change"]
        top = change.reindex(change.abs().sort_values(ascending=False).head(8).index).sort_values()
        st.subheader("关键因素贡献变化")
        st.plotly_chart(px.bar(x=top.values, y=top.index, orientation="h",
                               labels={"x": "调整后 SHAP − 调整前 SHAP", "y": "特征"}), use_container_width=True)
        st.caption("解释对象：" + result["explanation_model"] + "；SHAP 差值不等同于概率差值。")
    st.caption("这是同一个模型的条件对比，不代表因果关系。调整收入或额度后，风险变化不保证单调。")
