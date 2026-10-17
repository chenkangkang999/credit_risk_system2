import re

import pandas as pd

from src.features.project_builder import ProjectFeatureBuilder
from src.llm.json_utils import parse_json_object


class FeatureAgent:
    def __init__(self, client, prompt_manager, config):
        self.client = client
        self.prompts = prompt_manager
        self.config = config

    def suggest(self, df, current_auc=None, current_model="XGBoost", bottleneck=None):
        excluded = set(self.client.config.safety.pii_filter.fields_to_mask) | {"isDefault"}
        clean = df.drop(columns=list(excluded), errors="ignore")
        fields = [{"name": c, "dtype": str(clean[c].dtype),
                   "stats": clean[c].describe().astype(str).to_dict()} for c in clean.columns]
        prompt = self.prompts.render("feature_suggestion", fields=fields,
                                     current_auc=current_auc, current_model=current_model,
                                     max_suggestions=self.config.max_suggestions,
                                     bottleneck=bottleneck or "")
        response = self.client.chat(prompt, system_prompt=self.prompts.get_system_role("feature_suggestion"))
        return self._parse_response(response)

    def _parse_response(self, response):
        result = parse_json_object(response)
        if not isinstance(result.get("suggestions"), list):
            raise ValueError("特征建议缺少 suggestions 列表")
        result["suggestions"] = result["suggestions"][:self.config.max_suggestions]
        return result

    def apply_suggestion(self, df, suggestion):
        name = suggestion.get("name", "")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name) or name in df.columns:
            raise ValueError("特征名不合法或与现有列重名")
        if suggestion.get("operation") not in ("ratio", "sum", "product", "difference", "bucket"):
            raise ValueError("只支持比值、求和、乘积、相减和分箱，请修改建议后重试")
        return ProjectFeatureBuilder([suggestion]).build(df)
