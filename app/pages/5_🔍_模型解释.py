import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

from app.components.explanations import analyzer_for, available_tree_models, render_single
from app.components.runtime import init_page, require_service

init_page("模型解释", "从整体规律到单笔归因，观察模型如何形成判断。", "EXPLAINABILITY")
service, bundle = require_service()
names = available_tree_models(bundle)
if not names:
    st.info("请训练 XGBoost 或 LightGBM 后查看树模型解释。")
    st.stop()
name = st.selectbox("解释模型", names)
analyzer = analyzer_for(bundle, name)
raw = bundle["raw_test"].head(100)
X = service.transform(raw)
values = analyzer.explain_batch(X)
importance = pd.Series(np.abs(values).mean(axis=0), index=X.columns).sort_values(ascending=False)
global_tab, summary, dependence, local = st.tabs(["全局重要性", "SHAP 分布", "特征依赖", "单笔解释与力图"])
with global_tab:
    top = importance.head(20).sort_values()
    st.plotly_chart(px.bar(x=top.values, y=top.index, orientation="h",
                           labels={"x": "平均绝对 SHAP", "y": "特征"}, color_discrete_sequence=["#3468ea"]),
                    use_container_width=True)
with summary:
    shap.summary_plot(values, X, max_display=15, show=False)
    fig = plt.gcf()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
with dependence:
    column = st.selectbox("观察特征", X.columns)
    index = list(X.columns).index(column)
    fig = px.scatter(x=X[column], y=values[:, index], color=raw["isDefault"] if "isDefault" in raw else None,
                     labels={"x": column + "（预处理后）", "y": "SHAP 值"})
    st.plotly_chart(fig, use_container_width=True)
with local:
    row = st.number_input("测试样本序号", 0, len(X)-1, 0)
    render_single(analyzer.explain_single(X.iloc[row]), f"SHAP 解释对象：{name}")
st.caption(f"本页使用 {len(X)} 条独立测试样本。SHAP 原始输出与模型概率使用不同刻度。")
