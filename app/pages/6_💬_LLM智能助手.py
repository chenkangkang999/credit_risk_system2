import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import streamlit as st

from app.components.explanations import single_explanation
from app.components.runtime import (current_data, example_report, init_page, llm_tools,
                                    public_mode, require_service, risk_limits, saved_prediction, stream_to_state)

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

st.divider()
st.subheader("LLM 用量与费用")
st.caption("按 API 返回的 Token 用量估算人民币费用，默认按高峰单价估算上限；不是账户余额，也不会代替 DeepSeek 的实际账单。仅统计启用本功能后的调用。")
if public_mode():
    columns = st.columns(3)
    columns[0].metric("示例输入 Token", "0")
    columns[1].metric("示例输出 Token", "0")
    columns[2].metric("示例费用", "¥0.000000")
    st.caption("示例内容由本地模板生成，没有调用 API，也不会写入计费账本。")
else:
    if not client.configured:
        st.caption("当前未配置真实连接，新示例不计费；下面仍可查看本机已有的历史调用记录。")
    tracker = client.billing
    totals = tracker.summary()
    recent = tracker.records(20)
    latest = client.last_billing or (recent[0] if recent else None)

    def show_tokens(value):
        return "未返回" if value is None else f"{value:,}"

    def show_cost(record):
        if record.get("cost_cny") is not None:
            return f'¥{record["cost_cny"]:.6f}'
        if record.get("cost_min_cny") is not None:
            return f'¥{record["cost_min_cny"]:.6f}～¥{record["cost_max_cny"]:.6f}'
        return "待核实"

    if latest:
        columns = st.columns(3)
        columns[0].metric("最近一次输入 Token", show_tokens(latest["prompt_tokens"]))
        columns[1].metric("最近一次输出 Token", show_tokens(latest["completion_tokens"]))
        columns[2].metric("最近一次费用估算", show_cost(latest))
        st.caption(f'模型：{latest["model"]} · 时间（UTC）：{latest["timestamp"]} · 输入/输出含多轮历史及思考等服务端计入的用量。')
    else:
        st.info("还没有真实调用记录。完成一次对话、风险报告、特征建议或规则检查后可在这里查看。")
    columns = st.columns(4)
    columns[0].metric("累计已知输入 Token", f'{totals["prompt_tokens"]:,}')
    columns[1].metric("累计已知输出 Token", f'{totals["completion_tokens"]:,}')
    columns[2].metric("累计已估算费用", f'¥{totals["cost_cny"]:.6f}')
    columns[3].metric("调用记录数", totals["calls"])
    if totals["pending_calls"]:
        st.warning(f'有 {totals["pending_calls"]} 次调用的用量、缓存拆分或模型价格待核实，未计入累计费用；不能据此认为这些调用免费。')
    if tracker.last_error:
        st.warning(tracker.last_error)
    with st.expander("最近调用、计价单价与统计说明"):
        statuses = {"estimated": "已估算", "usage_missing": "用量未完整返回", "price_unknown": "模型单价待配置",
                    "cache_split_missing": "缓存拆分缺失，显示区间", "usage_invalid": "用量不一致",
                    "price_config_error": "计费配置异常，费用待核实"}
        if recent:
            rows = [{"时间（UTC）": row["timestamp"], "模型": row["model"], "流式": row["stream"],
                     "输入 Token": row["prompt_tokens"], "缓存命中": row["cache_hit_tokens"],
                     "缓存未命中": row["cache_miss_tokens"], "输出 Token": row["completion_tokens"],
                     "费用估算（元）": show_cost(row), "状态": statuses.get(row["billing_status"], row["billing_status"]),
                     "请求状态": row["status"], "计价时段": row["price_period"]} for row in recent]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        try:
            billing_cfg = tracker.config()
            price_rows = []
            for name, item in billing_cfg.get("models", {}).items():
                for period, label in (("peak", "高峰"), ("off_peak", "空闲")):
                    rate = item[period]
                    price_rows.append({"模型": name, "时段": label,
                                       "输入缓存命中": rate["input_cache_hit"],
                                       "输入缓存未命中": rate["input_cache_miss"], "输出": rate["output"]})
            st.markdown("**人民币元 / 百万 Token**")
            st.dataframe(pd.DataFrame(price_rows), hide_index=True, use_container_width=True)
            st.caption(f'价格核实日期：{billing_cfg.get("verified_on")}；当前计价配置：{billing_cfg.get("period")}。')
            st.write(billing_cfg.get("period_note", ""))
            st.caption(billing_cfg.get("official_peak_hours", ""))
            st.link_button("查看 DeepSeek 官方价格", billing_cfg["source_url"])
        except (OSError, ValueError, KeyError, TypeError):
            st.warning("计费配置暂时无法读取，请检查 configs/billing_config.yaml。")
        st.caption("本地账本位于 .local/llm_billing.sqlite3，跨会话保存调用时间、模型、Token、费率快照与估算费用，不保存问题、回答或 API 密钥。历史记录按调用当时配置保存，不随新费率重算。")
