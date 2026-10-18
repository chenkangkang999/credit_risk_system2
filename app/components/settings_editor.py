from pathlib import Path

import yaml

from src.features.project_builder import ProjectFeatureBuilder

ROOT = Path(__file__).resolve().parents[2]


def validate_settings(data):
    if not isinstance(data, dict):
        raise ValueError("配置必须为 YAML 对象")
    try:
        low, high = data["app"]["low_risk"], data["app"]["high_risk"]
        if not 0 <= low < high <= 1:
            raise ValueError("风险阈值应满足 0 ≤ 低阈值 < 高阈值 ≤ 1")
        if not 0 <= data["ab_test"]["ratio"] <= 1:
            raise ValueError("A/B 比例应在 0 到 1 之间")
        if not isinstance(data["features"], list):
            raise ValueError("features 必须是列表")
        names = [rule["name"] for rule in data["features"]]
        if len(names) != len(set(names)):
            raise ValueError("衍生特征名称不能重复")
        if any(rule.get("operation") not in ("ratio", "monthly_income", "payment_ratio", "fico_mean",
                                             "bucket", "sum", "product", "difference", "log1p", "category_flag",
                                             "annual_interest_ratio", "credit_history", "debt_surplus",
                                             "ratio_plus_one") for rule in data["features"]):
            raise ValueError("包含不支持的特征操作")
        for key in ["demo", "compliance"]:
            if not isinstance(data[key], dict):
                raise ValueError("缺少必要配置：" + key)
    except (KeyError, TypeError):
        raise ValueError("配置缺少字段或字段类型不正确") from None
    return data


def save_settings(data):
    data = validate_settings(data)
    content = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    if "sk-" in content or "api_key" in content.lower():
        raise ValueError("扩展配置中不允许保存密钥，请使用 .env")
    folder = ROOT / ".local"
    folder.mkdir(exist_ok=True)
    pending = folder / "settings.pending"
    pending.write_text(content, encoding="utf-8")
    pending.replace(folder / "settings.yaml")
