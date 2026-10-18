from pathlib import Path

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


def _home_page():
    ctx = get_script_run_ctx()
    return Path(ctx.main_script_path).name if ctx and ctx.main_script_path else "streamlit_app.py"


def render_sidebar():
    from app.components.runtime import ROOT, public_mode
    with st.sidebar:
        st.markdown('<div class="sidebar-brand"><div class="brand-icon">◈</div>'
                    '<div class="brand-name">智能风控<span>CREDIT RISK LAB</span></div></div>',
                    unsafe_allow_html=True)
        links = [(_home_page(), "工作台", "🏠"),
                 ("pages/1_📊_数据探索.py", "数据探索", "📊"),
                 ("pages/2_🔧_特征工程.py", "特征工程", "🔧"),
                 ("pages/3_🤖_模型训练.py", "模型训练", "🤖"),
                 ("pages/4_🎯_单笔预测.py", "单笔预测", "🎯"),
                 ("pages/5_🔍_模型解释.py", "模型解释", "🔍"),
                 ("pages/6_💬_LLM智能助手.py", "LLM 智能助手", "💬"),
                 ("pages/7_⚙️_系统配置.py", "系统配置", "⚙️"),
                 ("pages/8_模型监控.py", "模型监控", "📡"),
                 ("pages/9_AB测试.py", "A/B 模型测试", "🧪"),
                 ("pages/10_风险情景对比.py", "风险情景对比", "🔄")]
        for page, label, icon in links:
            st.page_link(page, label=label, icon=icon)
        st.divider()
        options = ["内置演示模型"]
        if not public_mode() and (ROOT / "data/models/latest/bundle.joblib").exists():
            options.append("本地最新训练")
        if st.session_state.get("artifact_source") not in options:
            st.session_state["artifact_source"] = options[0]
        pending = st.session_state.pop("next_artifact_source", None)
        if pending in options:
            st.session_state["artifact_source"] = pending
        st.selectbox("当前模型来源", options, key="artifact_source")
        from app.components.runtime import artifacts
        bundle = artifacts()
        source = bundle["meta"]["source"] if bundle else "暂无训练产物"
        st.success("合成数据演示 · 可放心体验") if source == "合成演示数据" else st.info(source)
        st.caption("公开展示模式" if public_mode() else "本地开发模式")
