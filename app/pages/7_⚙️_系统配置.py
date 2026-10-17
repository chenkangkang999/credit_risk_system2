import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import streamlit as st
import yaml

from app.components.runtime import ROOT, init_page, llm_tools, public_mode
from app.components.settings_editor import save_settings, validate_settings

configs, cfg = init_page("系统配置", "查看基础配置与扩展规则，检查 DeepSeek 连接状态。", "SETTINGS")
settings_tab, original, templates, api = st.tabs(["本地扩展配置", "基础配置（只读）", "提示词模板", "API 状态"])
with settings_tab:
    st.caption("修改保存到 .local/settings.yaml，仅影响之后的训练；同学的基础配置不写回，已有模型继续使用原训练规则。")
    content = yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=False)
    edited = st.text_area("扩展规则 YAML", content, height=420, disabled=public_mode())
    if not public_mode():
        a, b = st.columns(2)
        if a.button("校验格式和关键参数"):
            try:
                validate_settings(yaml.safe_load(edited))
                st.success("校验通过")
            except (ValueError, yaml.YAMLError) as error:
                st.error(str(error))
        if b.button("保存本地配置", type="primary"):
            try:
                save_settings(yaml.safe_load(edited))
                st.success("本地扩展配置已保存，重新打开页面后生效。")
            except (ValueError, yaml.YAMLError) as error:
                st.error(str(error))
    else:
        st.info("公开演示站点只提供配置查看。")
with original:
    choice = st.selectbox("基础配置文件", ["data_config.yaml", "feature_config.yaml", "model_config.yaml",
                                        "pipeline_config.yaml", "llm_config.yaml"])
    st.code((ROOT / "configs" / choice).read_text(encoding="utf-8"), language="yaml")
with templates:
    files = sorted((ROOT / "configs/prompts").glob("*.j2"))
    selected = st.selectbox("模板", [p.name for p in files])
    st.code((ROOT / "configs/prompts" / selected).read_text(encoding="utf-8"))
with api:
    client, _, _ = llm_tools()
    st.write("DeepSeek 密钥状态：" + ("已配置（不显示内容）" if client.configured else "未配置或处于公开展示模式"))
    st.write("模型：" + str(configs.get("llm").deepseek.model))
    if not public_mode():
        st.caption(f"密钥填写位置：{ROOT / '.env'}。修改后重启应用。")
        if client.configured and st.button("测试连接（会发起一次少量 token 请求）"):
            ok = client.health_check()
            (st.success if ok else st.error)("连接成功" if ok else "连接失败，请检查密钥、余额和网络")
