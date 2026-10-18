import copy
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import yaml
from dotenv import load_dotenv

from src.utils.config_loader import AttrDict, ConfigLoader

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=True)
os.environ.setdefault("APP_ENV", "development" if (ROOT / ".env").exists() else "public_demo")


def public_mode():
    return os.getenv("APP_ENV") == "public_demo" or os.getenv("CREDIT_RISK_PUBLIC_DEMO") == "1"


def settings():
    cfg = copy.deepcopy(ConfigLoader(str(ROOT / "configs")).get("project"))
    path = ROOT / ".local/settings.yaml"
    if not public_mode() and path.exists():
        cfg = AttrDict(yaml.safe_load(path.read_text(encoding="utf-8")))
    return cfg


def init_page(title, subtitle, section="WORKSPACE"):
    st.set_page_config(page_title=f"{title} · 智能风控", page_icon="◈", layout="wide")
    st.markdown("<style>" + (ROOT / "app/assets/style.css").read_text(encoding="utf-8") + "</style>",
                unsafe_allow_html=True)
    from app.components.sidebar import render_sidebar
    render_sidebar()
    st.caption(f"CREDIT RISK LAB  /  {section}")
    st.title(title)
    st.markdown(f'<p class="page-subtitle">{subtitle}</p>', unsafe_allow_html=True)
    return ConfigLoader(str(ROOT / "configs")), settings()


@st.cache_data
def generated_data(n=5000):
    from scripts.generate_sample_data import generate_sample_data
    return generate_sample_data(n, random_seed=42)


REAL_DATA_FILES = ("train_merged.csv", "train.csv")


def available_real_data():
    """返回 data/raw 下存在的真实数据文件（合并 105 万数据优先）。"""
    return [name for name in REAL_DATA_FILES if (ROOT / "data" / "raw" / name).exists()]


@st.cache_resource(show_spinner="正在加载本地真实数据…")
def load_real_data(name):
    return pd.read_csv(ROOT / "data" / "raw" / name)


def current_data():
    if not public_mode():
        source = st.session_state.get("data_source")
        if source in REAL_DATA_FILES and (ROOT / "data" / "raw" / source).exists():
            return load_real_data(source)
        if "active_data" in st.session_state:
            return st.session_state["active_data"].copy()
    path = ROOT / "demo_assets/sample.json"
    if path.exists():
        return pd.read_json(path, orient="table")
    return generated_data(5000)


def source_label():
    return st.session_state.get("data_label", "合成演示数据") if not public_mode() else "合成演示数据"


def bundle_path():
    if not public_mode() and st.session_state.get("artifact_source") == "本地最新训练":
        return ROOT / "data/models/latest/bundle.joblib"
    return ROOT / "demo_assets/bundle.joblib"


@st.cache_resource
def _load_bundle(path, stamp):
    return joblib.load(path)


def artifacts():
    path = bundle_path()
    if not path.exists():
        return None
    return _load_bundle(str(path), path.stat().st_mtime_ns)


def require_service():
    from src.pipeline.inference import InferenceService
    bundle = artifacts()
    if bundle is None:
        st.info("请先运行 prepare_demo.py，或在模型训练页完成一次训练。")
        st.stop()
    return InferenceService(bundle), bundle


def llm_tools():
    from src.llm.client import DeepSeekClient
    from src.llm.prompts import PromptManager
    config = ConfigLoader(str(ROOT / "configs")).get("llm")
    if "llm_client" not in st.session_state:
        st.session_state["llm_client"] = DeepSeekClient(config)
    return st.session_state["llm_client"], PromptManager(config), config


def rules_agent():
    from src.llm.agents.compliance_agent import ComplianceAgent
    client, prompts, _ = llm_tools()
    return ComplianceAgent(client, prompts, settings().compliance)


def saved_prediction():
    prediction = st.session_state.get("prediction")
    bundle = artifacts()
    if prediction and bundle and prediction["run_id"] == bundle["meta"]["run_id"]:
        return prediction
    return None


def risk_limits(bundle):
    app = bundle["project_config"]["app"]
    return app["low_risk"], app["high_risk"]


def example_report(features, probability, model_name, explanation=None, limits=(0.3, 0.7), explanation_model=None):
    from src.pipeline.inference import decision_for, risk_level
    low, high = limits
    lines = ["### 示例报告 · 按本次预测生成", "",
             f"当前使用 **{model_name}**，违约概率为 **{probability:.2%}**，"
             f"属于 **{risk_level(probability, low, high)}**。",
             f"贷款金额 {features['loanAmnt']:,.0f} 元，年收入 {features['annualIncome']:,.0f} 元。",
             f"页面建议：**{decision_for(probability, low, high)}**。"]
    if explanation:
        lines += ["", f"以下归因对象为 {explanation_model or model_name}，与预测模型不同时仅解释该子模型。", "本次解释中影响较大的因素："]
        for item in explanation.get_top_features(5):
            lines.append(f"- {item['feature']}：{item['direction']}")
    lines += ["", "此内容由本地模板生成，未调用 DeepSeek；仅用于课程展示，最终判断需要人工复核。"]
    return "\n".join(lines)


def stream_to_state(generator, state_key):
    placeholder = st.empty()
    content = ""
    try:
        for chunk in generator:
            content += chunk
            st.session_state[state_key] = content
            placeholder.markdown(content + " ▌")
        placeholder.markdown(content)
    except (ValueError, RuntimeError) as error:
        placeholder.markdown(content)
        st.error(str(error))
    return content
