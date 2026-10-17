import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import copy
import pandas as pd
import streamlit as st

from app.components.runtime import current_data, init_page, llm_tools, public_mode
from src.features.project_builder import ProjectFeatureBuilder
from src.pipeline.adapters import ProjectPreprocessor

configs, cfg = init_page("特征工程", "查看清洗效果，构造业务特征，再检查智能建议。", "FEATURES")
data = current_data().drop(columns=["id", "isDefault"], errors="ignore")
clean, derived, assistant = st.tabs(["预处理与 KNN", "衍生特征", "LLM 特征建议"])
with clean:
    st.info("沿用同学的 KNN 填充：dti（负债收入比）、revolUtil（信用利用率）。")
    if st.button("预览前 500 行预处理效果", type="primary"):
        feature_cfg = copy.deepcopy(configs.get("feature"))
        feature_cfg.scaling.method = "none"
        preprocessor = ProjectPreprocessor(copy.deepcopy(configs.get("data")), feature_cfg)
        before = data.head(500).copy()
        after = preprocessor.fit_transform(before)
        st.session_state["clean_preview"] = (before, after)
    if "clean_preview" in st.session_state:
        before, after = st.session_state["clean_preview"]
        st.dataframe(pd.DataFrame({"字段": ["dti", "revolUtil"],
                                  "处理前缺失数": [before[c].isna().sum() for c in ["dti", "revolUtil"]],
                                  "处理后缺失数": [after[c].isna().sum() for c in ["dti", "revolUtil"]]}),
                     hide_index=True, use_container_width=True)
        st.dataframe(after.head(20), use_container_width=True)
        st.caption("此处仅演示处理过程；正式训练时只在训练集拟合预处理参数。")
with derived:
    st.dataframe(pd.DataFrame([{"特征": r.name, "业务含义": r.description, "操作": r.operation}
                               for r in cfg.features]), hide_index=True, use_container_width=True)
    result = ProjectFeatureBuilder(cfg.features).build(data)
    st.dataframe(result[[r.name for r in cfg.features]].head(20), hide_index=True, use_container_width=True)
    st.caption("比值遇到零分母时保留为缺失值，由训练集拟合的预处理器处理。")
with assistant:
    client, pm, llm_cfg = llm_tools()
    from src.llm.agents.feature_agent import FeatureAgent
    agent = FeatureAgent(client, pm, llm_cfg.agents.feature_agent)
    if not client.configured:
        st.info("尚未配置 API。下面可以预览并应用一条明确标记的示例建议。")
        suggestion = {"name": "demo_income_load", "operation": "ratio", "numerator": "loanAmnt",
                      "denominator": "annualIncome", "business_meaning": "贷款收入比（示例）"}
        st.session_state.setdefault("feature_suggestions", {"suggestions": [suggestion]})
    if client.configured and st.button("请求 DeepSeek 特征建议", type="primary"):
        try:
            st.session_state["feature_suggestions"] = agent.suggest(data)
        except (ValueError, RuntimeError) as error:
            st.error(str(error))
    result = st.session_state.get("feature_suggestions", {})
    for i, suggestion in enumerate(result.get("suggestions", [])):
        with st.expander(suggestion.get("name", f"建议 {i+1}"), expanded=True):
            st.json(suggestion)
            if st.button("检查并应用到预览", key=f"feature_{i}"):
                try:
                    preview = agent.apply_suggestion(data, suggestion)
                    st.session_state["applied_feature"] = (suggestion, preview)
                except (ValueError, KeyError, TypeError) as error:
                    st.error(f"无法应用建议：{error}")
    if "applied_feature" in st.session_state:
        suggestion, preview = st.session_state["applied_feature"]
        st.dataframe(preview[[suggestion["name"]]].head(20), use_container_width=True)
        if not public_mode() and st.button("保存这条规则到本地扩展配置"):
            from app.components.settings_editor import save_settings
            rules = [r for r in cfg.features if r.name != suggestion["name"]]
            cfg.features = rules + [suggestion]
            try:
                save_settings(cfg.to_dict())
                st.success("已保存。后续训练将使用此规则，已有模型继续使用原训练规则。")
            except ValueError as error:
                st.error(str(error))
