import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.components.runtime import init_page, require_service
from src.monitoring.drift_detector import DriftDetector

init_page("模型监控", "衔接同学的漂移检测器，比较基线与新样本的特征分布。", "MONITORING")
_, bundle = require_service()
baseline = bundle["raw_baseline"].copy()
new_data = bundle["raw_test"].copy()
mode = st.radio("新样本场景", ["测试集样本", "模拟收入分布变化"], horizontal=True)
if mode == "模拟收入分布变化":
    multiple = st.slider("年收入放大倍数", 1.0, 5.0, 2.0)
    new_data["annualIncome"] = new_data["annualIncome"] * multiple
    st.caption("本场景人为修改演示样本，用于观察漂移检测效果。")
threshold = st.slider("漂移效应阈值", 0.05, 0.5, 0.15, step=0.01)
detector = DriftDetector(baseline)
drifts = detector.detect(new_data, threshold=threshold)
a, b, c = st.columns(3)
a.metric("基线样本数", len(baseline))
b.metric("新样本数", len(new_data))
c.metric("检测到漂移的字段", len(drifts))
if drifts:
    st.dataframe(pd.DataFrame(drifts).T.astype(str), use_container_width=True)
else:
    st.success("当前阈值下未检测到明显漂移。")
column = st.selectbox("查看分布", [c for c in ["annualIncome", "dti", "revolUtil", "interestRate"] if c in baseline])
plot = pd.concat([baseline[[column]].assign(来源="训练基线"), new_data[[column]].assign(来源="新样本")])
st.plotly_chart(px.histogram(plot, x=column, color="来源", barmode="overlay",
                            color_discrete_sequence=["#3468ea", "#ed9b70"]), use_container_width=True)
report = detector.get_drift_report(new_data, threshold=threshold)
st.download_button("下载监控报告", report, file_name="drift_report.txt")
st.caption("检测逻辑来自同学的原始模块，页面沿用其阈值判定方式。分布变化提示排查，不直接证明模型效果下降。")
