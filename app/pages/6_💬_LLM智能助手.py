import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import streamlit as st

from app.components.explanations import single_explanation
from app.components.runtime import (current_data, example_report, init_page, llm_tools,
                                    require_service, risk_limits, saved_prediction, stream_to_state)

init_page("LLM 智能助手", "围绕当前数据、模型和申请记录，理解预测背后的业务含义。", "ASSISTANT")
service, bundle = require_service()
client, prompts, cfg = llm_tools()
prediction = saved_prediction()
if not client.configured:
    st.info("当前为示例模式，未调用 DeepSeek。配置本地 .env 后可使用真实对话与分析。")
mode = st.radio("助手模式", ["自由对话", "风险报告", "特征建议", "异常分析"], horizontal=True)
if mode == "自由对话":
    key = "chat_history"
    st.session_state.setdefault(key, [])
    for message in st.session_state[key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("例如：AUC 和 KS 有什么区别？")
    if question:
        history = st.session_state[key]
        with st.chat_message("user"):
            st.markdown(question)
        answer = ""
        with st.chat_message("assistant"):
            if client.configured:
                data = current_data()
                context = prompts.render(
                    "chat_assistant", user_question=question,
                    dataset_info={"n_samples": len(data), "n_features": data.shape[1],
                                  "default_rate": f'{data["isDefault"].mean():.1%}'},
                    model_info={"xgb_auc": bundle["eval_results"].get("xgboost", {}).get("auc"),
                                "mlp_auc": bundle["eval_results"].get("mlp", {}).get("auc"),
                                "ensemble_auc": bundle["eval_results"].get("ensemble", {}).get("auc")})
                answer = stream_to_state(client.stream_chat(context,
                                          system_prompt=prompts.get_system_role("chat_assistant"),
                                          history=history), "last_chat")
            else:
                if "SHAP" in question.upper():
                    answer = "【示例答复】SHAP 用来分解特征对模型输出的贡献。正值推动风险输出上升，负值推动下降；贡献值不能直接当作概率。"
                elif "KNN" in question.upper():
                    answer = "【示例答复】项目沿用 KNN 填充 dti 和 revolUtil，利用相似样本补足缺失值。正式训练只使用训练集拟合。"
                else:
                    answer = "【示例答复】AUC 描述整体排序能力，KS 描述两类样本累积分布的最大差异。本项目在独立测试集计算这些指标。"
                st.markdown(answer)
        if answer:
            history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
            st.session_state[key] = history[-cfg.agents.chat_agent.max_history_turns*2:]
elif mode == "风险报告":
    if not prediction:
        st.info("请先到单笔预测页面评估一笔申请。")
    else:
        st.markdown(st.session_state.get("risk_report") or example_report(
            prediction["features"], prediction["probability"], prediction["model_name"],
            prediction["explanation"], risk_limits(bundle), prediction["explanation_model"]))
        st.page_link("pages/4_🎯_单笔预测.py", label="进入单笔预测生成或下载报告")
elif mode == "特征建议":
    st.write("特征建议会使用当前数据的字段和统计信息，不发送原始客户记录。")
    st.page_link("pages/2_🔧_特征工程.py", label="进入特征工程，预览并应用建议")
else:
    from src.llm.agents.case_agent import CaseAgent
    agent = CaseAgent(client, prompts, cfg.agents.case_agent)
    name = st.selectbox("诊断模型", list(service.models))
    raw = bundle["raw_test"]
    y = np.asarray(bundle["y_test"])
    probability = np.asarray(bundle["test_predictions"][name])
    anomalies = agent.find_anomalies(y, probability, raw, threshold=bundle["thresholds"].get(name, 0.5))
    st.caption(f"高置信度错误案例：{len(anomalies)} 条")
    if len(anomalies):
        st.dataframe(anomalies, hide_index=True, use_container_width=True)
    else:
        st.info("当前抽样未发现达到阈值的高置信度错误；下面可以查看普通错误案例。")
    wrong = np.flatnonzero((probability >= bundle["thresholds"].get(name, 0.5)) != y)
    if not len(wrong):
        st.info("当前测试样本中没有预测错误案例。")
    else:
        row = st.selectbox("选择错误案例序号", wrong.tolist())
        sample = raw.iloc[[row]]
        explanation, tree = single_explanation(service, bundle, sample, name)
        st.write(f"真实标签：{y[row]}，违约概率：{probability[row]:.2%}")
        if client.configured and explanation and st.button("请求 DeepSeek 分析案例", type="primary"):
            stream_to_state(agent.analyze(sample.iloc[0].to_dict(), int(y[row]),
                                          float(probability[row]), explanation, stream=True,
                                          threshold=bundle["thresholds"].get(name, 0.5)), "case_report")
        if "case_report" in st.session_state:
            st.markdown(st.session_state["case_report"])
        elif not client.configured:
            st.markdown("**示例诊断**：这是一个错误案例，应结合数据质量、特征覆盖和模型泛化能力排查。没有调用 LLM，不能据此确认错误根因。")
