import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import copy
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
        profile = st.selectbox("训练方案", ["稳健配置（浅树与正则化）", "原基础配置"])
        enhanced = st.checkbox("使用经验证的增强衍生特征", value=True)
        st.caption("稳健方案默认启用原 5 项衍生特征及等级、稳定就业两个标记。更多候选可在特征工程页查看。")
        left, middle, right = st.columns(3)
        quick = left.checkbox("快速教学配置", value=True)
        tune = middle.checkbox("使用 Optuna 调参")
        metric = right.selectbox("调参目标", ["auc", "ks"])
        multi = st.checkbox("同时考虑预测指标和模型深度（多目标优化）", disabled=not tune)
        trials = st.number_input("Optuna 试验次数", 1, 50, 3, disabled=not tune)
        st.caption("稳健方案从个人扩展配置读取；原基础方案保留快速预算。所有方案都复用同学的模型类。")
        if st.button("开始训练", type="primary"):
            try:
                pipeline = TrainingPipeline(configs)
                pipeline.project_cfg = copy.deepcopy(cfg)
                if not enhanced:
                    pipeline.project_cfg.features = pipeline.project_cfg.features[:5]
                pipeline.project_cfg.demo.trials = trials
                pipeline.model_cfg.xgboost.tuning.metric = metric
                pipeline.model_cfg.xgboost.tuning.multi_objective = multi
                pipeline.tuning_metric = metric
                progress = st.progress(0, text="准备中")
                def callback(label, percent):
                    progress.progress(percent, text=label)
                result = pipeline.run(data=current_data(), use_sample=source_label() == "合成演示数据",
                                      model_names=names, quick=quick, skip_tuning=not tune,
                                      training_profile="robust" if profile.startswith("稳健") else "original",
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
    st.caption("训练方案：" + bundle["meta"].get("training_profile", "原始审阅版"))
    st.info("AUC 衡量排序能力，调整分类阈值不会提高 AUC。增加特征也不保证每项指标同时上升，请结合验证集、KS、F1 和 log loss 查看。")
    if bundle.get("previous_eval_results"):
        with st.expander("同一数据与划分的修改前后对照", expanded=True):
            rows = []
            for name, now in bundle["eval_results"].items():
                before = bundle["previous_eval_results"].get(name, {})
                rows.append({"模型": name, "原 AUC": before.get("auc"), "本次 AUC": now["auc"],
                             "原 KS": before.get("ks"), "本次 KS": now["ks"],
                             "原 log loss": before.get("log_loss"), "本次 log loss": now["log_loss"]})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption("使用原来的合成数据、标签、随机种子与训练/验证/测试划分。只在验证集选择方案，表中测试指标不参与选择。")
    with st.expander("验证集结果与学习到的特征重要性"):
        if bundle.get("validation_results"):
            st.dataframe(pd.DataFrame(bundle["validation_results"]).T, use_container_width=True)
        tree = bundle["models"].get("xgboost")
        if tree is not None:
            importance = tree.get_feature_importance().head(20).rename("gain").rename_axis("特征").reset_index()
            st.dataframe(importance, hide_index=True, use_container_width=True)
            st.caption("重要性来自训练后树模型的分裂收益，并非手工提高显示权重。树模型中单纯放大某列的数值通常不会改善排序。")
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
