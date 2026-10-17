import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import shap
import streamlit as st

from src.explain.shap_analyzer import SHAPAnalyzer

_fonts = set(font_manager.get_font_names())
for _name in ["Microsoft YaHei", "Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]:
    if _name in _fonts:
        plt.rcParams["font.family"] = _name
        break
plt.rcParams["axes.unicode_minus"] = False


def available_tree_models(bundle):
    return [name for name in ("xgboost", "lightgbm") if name in bundle["models"]]


def analyzer_for(bundle, name):
    analyzer = SHAPAnalyzer(max_background_size=100)
    return analyzer.fit(bundle["models"][name].model, bundle["background"])


def single_explanation(service, bundle, raw, model_name):
    names = available_tree_models(bundle)
    tree_name = model_name if model_name in names else names[0] if names else None
    if tree_name is None:
        return None, None
    analyzer = analyzer_for(bundle, tree_name)
    explanation = analyzer.explain_single(service.transform(raw).iloc[0])
    return explanation, tree_name


def render_single(explanation, title="SHAP 局部解释"):
    if explanation is None:
        st.info("当前模型没有树模型解释器。")
        return
    st.caption(title + "。正值推动模型输出上升，负值推动其下降；原始 SHAP 输出不是违约概率。")
    chart, force = st.tabs(["贡献瀑布图", "SHAP 力图"])
    exp = shap.Explanation(values=explanation.shap_values, base_values=explanation.base_value,
                           data=explanation.feature_values, feature_names=explanation.feature_names)
    with chart:
        shap.plots.waterfall(exp, max_display=10, show=False)
        figure = plt.gcf()
        figure.set_size_inches(10, 4.8)
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)
    with force:
        figure = shap.force_plot(explanation.base_value, explanation.shap_values,
                                 explanation.feature_values, feature_names=explanation.feature_names,
                                 matplotlib=True, show=False, figsize=(12, 3))
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)
