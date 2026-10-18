import json
import math
import os
import re

from src.pipeline.inference import decision_for


def _get(config, key, default=None):
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


class ComplianceAgent:
    DEFAULT_RULES = [
        "课程规则：贷款年利率不得超过输入 LPR 的 4 倍。",
        "课程规则：60 岁以上申请人需要额外的还款能力证明。",
        "课程规则：贷款用途不得用于房地产投资。",
    ]

    def __init__(self, client, prompt_manager, config, agent_config=None):
        self.client = client
        self.prompts = prompt_manager
        self.config = config
        llm_config = _get(client, "config")
        agents = _get(llm_config, "agents", {})
        self.agent_config = agent_config if agent_config is not None else _get(agents, "compliance", config)

    @property
    def enabled(self):
        return _get(self.agent_config, "enabled", True) is True

    def rule_list(self, extra_rules=None):
        if extra_rules is not None and (not isinstance(extra_rules, list)
                                        or any(not isinstance(rule, str) or not rule.strip() for rule in extra_rules)):
            raise ValueError("附加规则必须是非空文本组成的列表")
        rules = list(_get(self.config, "rules", [])) + self.DEFAULT_RULES + (extra_rules or [])
        return list(dict.fromkeys(rules))

    def check(self, features, decision, interest_rate=None, rules=None, probability=None,
              low=0.3, high=0.7, use_llm=False):
        if not self.enabled:
            raise ValueError("政策合规检查 Agent 已在 configs/llm_config.yaml 中关闭")
        if not isinstance(features, dict):
            raise ValueError("申请资料应为字段字典")
        if decision not in ("建议通过", "人工复核", "建议拒绝"):
            raise ValueError("审批决策应为建议通过、人工复核或建议拒绝")
        effective_rules = self.rule_list(rules)
        local_result = self._check_local(features, decision, interest_rate, probability, low, high)
        if rules:
            local_result["warnings"].append("附加自定义规则尚未由本地程序执行，请使用 DeepSeek 逐条检查或人工核对")
            local_result["recommendations"] = ["结合完整资料逐项人工复核，并核对附加自定义规则"]
        if not use_llm:
            return local_result
        if os.getenv("APP_ENV") == "public_demo" or os.getenv("CREDIT_RISK_PUBLIC_DEMO") == "1":
            raise ValueError("公开演示模式只执行本地规则，不调用 DeepSeek")
        if self.client is None or not _get(self.client, "configured", True):
            raise ValueError("DeepSeek 尚未配置，请在本地配置后使用真实检查")
        prompt = self.prompts.render("compliance_check", features=features, decision=decision,
                                     interest_rate=interest_rate if interest_rate is not None else features.get("interestRate"),
                                     rules=effective_rules, rule_result=local_result)
        response = self.client.chat(
            prompt, system_prompt=self.prompts.get_system_role("compliance_check"),
            temperature=_get(self.agent_config, "temperature", 0.1),
            max_tokens=_get(self.agent_config, "max_tokens", 2000),
            response_format={"type": "json_object"})
        analysis = self._validate_response(response)
        return {**analysis, "source": "DeepSeek 逐条规则检查", "notice": local_result["notice"],
                "local_result": local_result, "llm_analysis": analysis}

    def _check_local(self, features, decision, interest_rate, probability, low, high):
        violations, warnings = [], []
        for field in _get(self.config, "required_fields", ["loanAmnt", "annualIncome", "term", "interestRate"]):
            value = features.get(field)
            if value is None or (isinstance(value, (float, int)) and not math.isfinite(value)):
                violations.append({"rule": "申请资料完整", "reason": f"缺少或无效字段 {field}"})
        income = self._number(features.get("annualIncome"))
        amount = self._number(features.get("loanAmnt"))
        if income is None or amount is None or income <= 0 or amount <= 0:
            violations.append({"rule": "输入合理", "reason": "收入与贷款金额应为有效的正数"})
        elif amount / income > _get(self.config, "max_loan_income_ratio", 1.0):
            warnings.append("贷款收入比超过项目阈值，建议人工复核")
        dti = self._number(features.get("dti"))
        if dti is not None and dti > _get(self.config, "max_dti", 40):
            warnings.append("负债收入比超过项目阈值，建议人工复核")
        if probability is not None and decision != decision_for(probability, low, high):
            violations.append({"rule": "审批建议一致", "reason": "所选审批建议与模型风险区间不一致"})
        rate = self._number(interest_rate if interest_rate is not None else features.get("interestRate"))
        lpr = self._number(features.get("lpr"))
        if rate is None or rate < 0:
            violations.append({"rule": "利率输入合理", "reason": "贷款年利率应为非负数"})
        elif lpr is None or lpr <= 0:
            warnings.append("未提供有效 LPR，未检查课程中的 4 倍 LPR 规则")
        elif rate > 4 * lpr:
            violations.append({"rule": self.DEFAULT_RULES[0], "reason": f"输入年利率 {rate:g}% 超过 4 倍输入 LPR（{4*lpr:g}%）"})
        age = self._number(features.get("age"))
        if age is None or not 18 <= age <= 120:
            warnings.append("未提供有效成年人年龄，未检查额外还款能力证明规则")
        elif age > 60 and features.get("repayment_proof") is not True:
            violations.append({"rule": self.DEFAULT_RULES[1], "reason": "申请人超过 60 岁且未确认提供额外还款能力证明"})
        purpose = features.get("loan_purpose")
        if not isinstance(purpose, str) or not purpose.strip():
            warnings.append("未提供贷款用途，未检查房地产投资用途规则")
        elif "房地产投资" in purpose:
            violations.append({"rule": self.DEFAULT_RULES[2], "reason": "填报贷款用途包含房地产投资"})
        return {"is_compliant": not violations, "violations": violations, "warnings": warnings,
                "recommendations": ["结合完整资料逐项人工复核"] if violations or warnings else ["当前资料符合已执行的课程规则"],
                "source": "本地教学规则检查", "notice": "按课程规则演示，不构成对实际法律、监管要求或审批有效性的认定"}

    @staticmethod
    def _number(value):
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
            return number if math.isfinite(number) else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _validate_response(text):
        error = "政策合规检查返回格式错误，请重新检查"
        if not isinstance(text, str):
            raise ValueError(error + "：结果不是 JSON 文本")
        cleaned = text.strip()
        if cleaned.startswith("```"):
            match = re.fullmatch(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", cleaned)
            if not match:
                raise ValueError(error + "：JSON 代码块不完整")
            cleaned = match.group(1)
        try:
            result = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            raise ValueError(error + "：不是完整的 JSON 对象") from None
        fields = {"is_compliant", "violations", "warnings", "recommendations"}
        if not isinstance(result, dict) or set(result) != fields:
            raise ValueError(error + "：应包含规定的四个字段")
        if type(result["is_compliant"]) is not bool:
            raise ValueError(error + "：is_compliant 应为布尔值")
        if not isinstance(result["violations"], list):
            raise ValueError(error + "：violations 应为列表")
        for item in result["violations"]:
            if (not isinstance(item, dict) or set(item) != {"rule", "reason"}
                    or any(not isinstance(item[key], str) or not item[key].strip() for key in ("rule", "reason"))):
                raise ValueError(error + "：每条违规项应包含 rule 和 reason 文本")
        for key in ("warnings", "recommendations"):
            if not isinstance(result[key], list) or any(not isinstance(item, str) or not item.strip() for item in result[key]):
                raise ValueError(error + f"：{key} 应为文本列表")
        if result["is_compliant"] != (not result["violations"]):
            raise ValueError(error + "：通过状态与违规项不一致")
        return result
