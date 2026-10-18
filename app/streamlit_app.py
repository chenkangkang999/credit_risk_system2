import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.components.runtime import ROOT, artifacts, current_data, init_page, public_mode

configs, cfg = init_page("信贷风险工作台", "把数据、模型解释与业务判断连接起来。", "OVERVIEW")
bundle = artifacts()
st.markdown('<div class="hero"><small>EXPLAINABLE CREDIT DECISIONS</small>'
            '<h2>让每一次风险判断，都有据可循。</h2>'
            '<p>从贷款申请出发，查看违约概率、理解关键风险因素，'
            '再通过智能报告和情景对比，形成可解释的教学决策流程。</p></div>',
            unsafe_allow_html=True)
data = current_data()
cols = st.columns(4)
cols[0].metric("训练样本", f'{bundle["meta"]["train_samples"]:,}' if bundle else "尚未训练")
cols[1].metric("入模特征", len(bundle["meta"]["feature_names"]) if bundle else "—")
auc = bundle["eval_results"].get("xgboost", {}).get("auc") if bundle else None
cols[2].metric("XGBoost 测试 AUC", f"{auc:.3f}" if auc is not None else "—")
cols[3].metric("可用模型", len(bundle["models"]) if bundle else 0)
st.caption((bundle["meta"]["source"] if bundle else "演示数据") + " · 指标来自独立测试集 · 当前效果不代表真实信贷业务表现")
st.markdown("")
left, right = st.columns([1.6, 1])
with left:
    with st.container(border=True):
        st.subheader("从申请到决策")
        st.caption("按顺序走一遍，也可以直接进入单笔预测体验。")
        parts = st.columns(4)
        for col, no, title, desc in zip(parts, ["01", "02", "03", "04"],
                                        ["了解数据", "训练评估", "解释风险", "辅助决策"],
                                        ["数据质量与衍生特征", "双树模型与神经网络", "SHAP 全局与局部归因", "报告、规则和情景对比"]):
            col.markdown(f'<div class="flow-step"><span>STEP {no}</span><b>{title}</b><p>{desc}</p></div>',
                         unsafe_allow_html=True)
        st.markdown("")
        st.page_link("pages/4_🎯_单笔预测.py", label="进入单笔预测 →", icon="🎯")
        st.page_link("pages/10_风险情景对比.py", label="探索风险情景对比 →", icon="🔄")
    with st.container(border=True):
        st.subheader("模型表现")
        if bundle:
            scores = pd.DataFrame(bundle["eval_results"]).T.reset_index(names="模型")
            fig = px.bar(scores, x="模型", y="auc", color="模型",
                         color_discrete_sequence=["#3468ea", "#57b9a7", "#af95e5", "#f2b968"])
            fig.update_layout(height=255, showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                              yaxis_title="独立测试集 AUC", xaxis_title=None, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
with right:
    with st.container(border=True):
        st.subheader("当前工作区")
        from app.components.runtime import source_label
        st.write("🟢 数据：" + source_label())
        st.write("🟢 模型：已就绪" if bundle else "🟠 模型：请先准备演示产物")
        from app.components.runtime import llm_tools
        st.write("🟢 DeepSeek：已配置" if llm_tools()[0].configured else "⚪ DeepSeek：示例报告模式")
        if bundle:
            st.caption("模型版本：" + bundle["meta"]["run_id"])
    with st.container(border=True):
        st.subheader("本次扩展")
        st.write("✓ 教学审批规则检查 Agent")
        st.write("✓ XGBoost / LightGBM 稳定分流")
        st.write("✓ 调整申请条件，比较风险变化")
        st.write("✓ 接入 KNN、调参和漂移检测")
        st.page_link("pages/9_AB测试.py", label="查看 A/B 测试", icon="🧪")
        st.page_link("pages/8_模型监控.py", label="查看模型监控", icon="📡")
