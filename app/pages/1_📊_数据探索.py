import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.components.runtime import ROOT, available_real_data, current_data, init_page, load_real_data, public_mode, source_label
from src.pipeline.orchestrator import validate_training_data

_, cfg = init_page("数据探索", "从样本规模、缺失情况与变量分布，了解当前数据。", "DATA")
if not public_mode():
    choice = st.radio("数据来源", ["内置合成数据", "上传 CSV", "本地真实数据", "本地天池 CSV"], horizontal=True)
    if choice == "上传 CSV":
        uploaded = st.file_uploader("上传与项目字段一致的 CSV", type=["csv"])
        if uploaded is not None:
            try:
                if uploaded.size > cfg.demo.max_upload_mb*1024*1024:
                    raise ValueError("文件超过 20 MB；大数据请放入本地 data/raw 目录")
                frame = pd.read_csv(uploaded)
                validate_training_data(frame)
                if st.button("使用这份数据", type="primary"):
                    st.session_state.pop("data_source", None)
                    st.session_state["active_data"] = frame
                    st.session_state["data_label"] = "用户上传：" + uploaded.name
                    st.success("当前数据源已更新，训练页面将使用此数据。")
            except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
                st.error(str(error))
    elif choice == "本地真实数据":
        files = available_real_data()
        if not files:
            st.info(f"未找到真实数据文件，请将 train_merged.csv 或 train.csv 放入 {ROOT / 'data' / 'raw'}")
        else:
            labels = {"train_merged.csv": "天池 + Lending Club 合并数据",
                      "train.csv": "天池数据"}
            selected = st.selectbox("选择数据文件", files,
                                    format_func=lambda name: f"{name} · {labels.get(name, '')}")
            if st.button("加载本地真实数据", type="primary"):
                try:
                    count = len(load_real_data(selected))
                except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
                    st.error(f"加载失败：{error}")
                else:
                    st.session_state["data_source"] = selected
                    st.session_state["data_label"] = f"本地真实数据：{selected}"
                    st.session_state.pop("active_data", None)
                    st.success(f"已加载 {selected}，共 {count:,} 行，数据源已切换。")
    elif choice == "本地天池 CSV":
        path = ROOT / "data/raw/train.csv"
        if path.exists():
            if st.button("加载前 20,000 行用于页面探索"):
                try:
                    frame = validate_training_data(pd.read_csv(path, nrows=20000))
                except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
                    st.error(f"加载失败：{error}")
                else:
                    st.session_state.pop("data_source", None)
                    st.session_state["active_data"] = frame
                    st.session_state["data_label"] = f"天池数据（前 {len(frame):,} 行）"
        else:
            st.info(f"请将 train.csv 放入 {path.parent}")
    elif st.button("恢复合成演示数据"):
        st.session_state.pop("active_data", None)
        st.session_state.pop("data_label", None)
        st.session_state.pop("data_source", None)
data = current_data()
st.caption("当前数据源：" + source_label())
cols = st.columns(4)
cols[0].metric("当前样本", f"{len(data):,}")
cols[1].metric("字段数量", data.shape[1])
cols[2].metric("缺失比例", f"{data.isna().to_numpy().mean():.1%}")
cols[3].metric("违约样本比例", f'{data["isDefault"].mean():.1%}' if "isDefault" in data else "无标签")
overview, quality, distribution = st.tabs(["样本概览", "数据质量", "变量分布"])
with overview:
    st.dataframe(data.head(100), use_container_width=True, hide_index=True)
    st.caption("仅展示前 100 行；上传数据只用于当前本地会话。")
with quality:
    missing = data.isna().mean().sort_values(ascending=False).head(20)
    st.plotly_chart(px.bar(x=missing.values, y=missing.index, orientation="h",
                           labels={"x": "缺失比例", "y": "字段"}, color_discrete_sequence=["#3468ea"]),
                    use_container_width=True)
    st.write("重复记录：", int(data.duplicated().sum()))
with distribution:
    column = st.selectbox("选择字段", data.columns, index=list(data.columns).index("interestRate"))
    st.plotly_chart(px.histogram(data, x=column, color="isDefault" if "isDefault" in data else None,
                                barmode="overlay", color_discrete_sequence=["#3468ea", "#ed9b70"]),
                    use_container_width=True)
