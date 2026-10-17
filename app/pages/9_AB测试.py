import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app.components.runtime import init_page, require_service
from src.ab_test.router import ABTestRouter
from src.utils.metrics import Evaluator

_, cfg = init_page("A/B 模型测试", "按申请编号稳定分流，并在同一批测试样本上比较模型。", "EXPERIMENTS")
service, bundle = require_service()
if len(service.models) < 2:
    st.info("请先训练至少两种模型。")
    st.stop()
names = list(service.models)
a, b, c = st.columns(3)
model_a = a.selectbox("A 组模型", names, index=names.index("xgboost") if "xgboost" in names else 0)
model_b = b.selectbox("B 组模型", names, index=names.index("lightgbm") if "lightgbm" in names else 1)
ratio = c.slider("A 组流量比例", 0.0, 1.0, float(cfg.ab_test.ratio), step=0.05)
if model_a == model_b:
    st.warning("请选择两个不同模型。")
    st.stop()
router = ABTestRouter(service.models[model_a], service.models[model_b], ratio)
user_id = st.text_input("演示申请编号", "DEMO-001")
if user_id.strip():
    sample = bundle["raw_test"].iloc[[0]]
    group, probability = router.route(user_id, service.transform(sample))
    st.info(f"申请 {user_id} 固定进入 {group} 组，本次模型违约概率 {float(probability[0]):.2%}。")
raw = bundle["raw_test"]
ids = [f"demo-{i}" for i in range(len(raw))]
groups = [router.group(value) for value in ids]
pa = service.predict_proba(raw, model_a)
pb = service.predict_proba(raw, model_b)
table = pd.DataFrame({"申请编号": ids, "分组": groups, "A模型概率": pa, "B模型概率": pb})
table["实际路由概率"] = np.where(table["分组"] == "A", pa, pb)
a, b, c = st.columns(3)
a.metric("A 组申请", int((table["分组"] == "A").sum()))
b.metric("B 组申请", int((table["分组"] == "B").sum()))
c.metric("平均绝对概率差", f"{np.mean(np.abs(pa-pb)):.2%}")
st.subheader("同一测试样本上的离线对比")
metrics = {}
for label, name, p in [("A · "+model_a, model_a, pa), ("B · "+model_b, model_b, pb)]:
    metrics[label] = Evaluator().evaluate(bundle["y_test"], p, bundle["thresholds"].get(name, .5)).to_dict()
st.dataframe(pd.DataFrame(metrics).T, use_container_width=True)
st.plotly_chart(px.scatter(table, x="A模型概率", y="B模型概率", color="分组",
                          color_discrete_sequence=["#3468ea", "#57b9a7"]), use_container_width=True)
st.dataframe(table.head(30), hide_index=True, use_container_width=True)
st.download_button("下载本次模拟分流记录", table.to_csv(index=False).encode("utf-8-sig"), "ab_demo.csv")
st.caption("这是合成数据上的离线模拟。没有线上真实还款结果时，不能将分流统计视为线上收益结论。")
