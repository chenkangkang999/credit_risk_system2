import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.components.runtime import ROOT, artifacts, current_data, init_page, public_mode, source_label
from src.pipeline.orchestrator import TrainingPipeline

configs, cfg = init_page("模型训练", "训练集学习参数，验证集选择参数，测试集只做最终评估。", "MODELS")
if public_mode():
    st.info("公开站点展示预训练演示结果。训练功能在本地开发模式中使用。")
else:
    with st.container(border=True):
        st.caption("当前训练数据：" + source_label())
        names = st.multiselect("选择模型", ["xgboost", "lightgbm", "mlp"], default=["xgboost", "lightgbm", "mlp"])
        left, middle, right = st.columns(3)
        quick = left.checkbox("快速教学配置", value=True)
        tune = middle.checkbox("使用 Optuna 调参")
        metric = right.selectbox("调参目标", ["auc", "ks"])
        multi = st.checkbox("同时考虑预测指标和模型深度（多目标优化）", disabled=not tune)
        trials = st.number_input("Optuna 试验次数", 1, 50, 3, disabled=not tune)
        st.caption("快速配置：80 棵树、MLP 最多 8 轮；基础模型配置文件保持原样。")
        if st.button("开始训练", type="primary"):
            try:
                pipeline = TrainingPipeline(configs)
                pipeline.project_cfg = cfg
                pipeline.project_cfg.demo.trials = trials
                pipeline.model_cfg.xgboost.tuning.metric = metric
                pipeline.model_cfg.xgboost.tuning.multi_objective = multi
                pipeline.tuning_metric = metric
                progress = st.progress(0, text="准备中")
                def callback(label, percent):
                    progress.progress(percent, text=label)
                result = pipeline.run(data=current_data(), use_sample=source_label() == "合成演示数据",
                                      model_names=names, quick=quick, skip_tuning=not tune,
                                      progress_callback=callback)
                st.session_state["next_artifact_source"] = "本地最新训练"
                st.session_state.pop("prediction", None)
                st.session_state["training_notice"] = f'训练完成，耗时 {result["duration"]:.1f} 秒。'
                st.rerun()
            except Exception as error:
                st.error(f"训练失败：{error}")
bundle = artifacts()
if st.session_state.get("training_notice"):
    st.success(st.session_state["training_notice"])
if bundle:
    st.subheader("独立测试集结果")
    st.dataframe(pd.DataFrame(bundle["eval_results"]).T, use_container_width=True)
    st.caption(f'版本 {bundle["meta"]["run_id"]} · 训练 {bundle["meta"]["train_samples"]} / '
               f'验证 {bundle["meta"]["val_samples"]} / 测试 {bundle["meta"]["test_samples"]} 条')
    if bundle.get("tuning_history"):
        with st.expander("Optuna 试验记录", expanded=True):
            history = pd.DataFrame(bundle["tuning_history"])
            st.dataframe(history, use_container_width=True)
    from src.utils.metrics import Evaluator
    from sklearn.metrics import precision_recall_curve, roc_curve
    import numpy as np
    selected = st.selectbox("查看评估曲线", list(bundle["models"]))
    truth = np.asarray(bundle["y_test"])
    probabilities = bundle["test_predictions"][selected]
    roc_tab, ks_tab, pr_tab, matrix_tab = st.tabs(["ROC 曲线", "KS 曲线", "PR 曲线", "混淆矩阵"])
    fpr, tpr, _ = roc_curve(truth, probabilities)
    with roc_tab:
        st.plotly_chart(px.line(x=fpr, y=tpr, labels={"x": "假正率 FPR", "y": "真正率 TPR"}), use_container_width=True)
    with ks_tab:
        frame = pd.DataFrame({"TPR": tpr, "FPR": fpr, "差值": tpr-fpr})
        st.plotly_chart(px.line(frame, labels={"index": "阈值位置", "value": "累积比例"}), use_container_width=True)
    with pr_tab:
        precision, recall, _ = precision_recall_curve(truth, probabilities)
        st.plotly_chart(px.line(x=recall, y=precision, labels={"x": "召回率", "y": "精确率"}), use_container_width=True)
    with matrix_tab:
        matrix = Evaluator.get_confusion_matrix(truth, probabilities, bundle["thresholds"][selected])
        st.plotly_chart(px.imshow(matrix, text_auto=True, x=["预测正常", "预测违约"], y=["实际正常", "实际违约"], color_continuous_scale="Blues"), use_container_width=True)
    st.caption(f"曲线使用产物中保留的 {len(truth)} 条测试样本；上表总体指标使用完整独立测试集。")
    mlp = bundle["models"].get("mlp")
    if mlp:
        history = pd.DataFrame(mlp.train_history)
        st.plotly_chart(px.line(history, y=[c for c in ["train_loss", "val_loss"] if c in history],
                               labels={"index": "训练轮次", "value": "损失", "variable": "曲线"}),
                        use_container_width=True)
