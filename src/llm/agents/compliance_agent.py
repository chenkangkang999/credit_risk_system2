import math

from src.llm.json_utils import parse_json_object
from src.pipeline.inference import decision_for


class ComplianceAgent:
    def __init__(self, client, prompt_manager, config):
        self.client = client
        self.prompts = prompt_manager
        self.config = config

    def check(self, features, decision, interest_rate=None, rules=None, probability=None,
              low=0.3, high=0.7, use_llm=False):
        violations, warnings = [], []
        for field in self.config.required_fields:
            value = features.get(field)
            if value is None or (isinstance(value, (float, int)) and not math.isfinite(value)):
                violations.append({"rule": "申请资料完整", "reason": f"缺少或无效字段 {field}"})
        income, amount = features.get("annualIncome", 0), features.get("loanAmnt", 0)
        if income <= 0 or amount <= 0:
            violations.append({"rule": "输入合理", "reason": "收入与贷款金额应大于零"})
        elif amount/income > self.config.max_loan_income_ratio:
            warnings.append("贷款收入比超过项目阈值，建议人工复核")
        if features.get("dti", 0) > self.config.max_dti:
            warnings.append("负债收入比超过项目阈值，建议人工复核")
        if probability is not None and decision != decision_for(probability, low, high):
            violations.append({"rule": "审批建议一致", "reason": "所选审批建议与模型风险区间不一致"})
        result = {"is_compliant": not violations, "violations": violations, "warnings": warnings,
                  "recommendations": ["结合实际资料进行人工复核"] if violations or warnings else ["资料符合当前教学规则"],
                  "source": "本地教学规则检查", "notice": "规则为课程演示设置"}
        if use_llm:
            prompt = self.prompts.render("compliance_check", features=features, decision=decision,
                                         interest_rate=interest_rate, rules=rules or self.config.rules,
                                         rule_result=result)
            response = self.client.chat(prompt, system_prompt=self.prompts.get_system_role("compliance_check"),
                                        temperature=0.1)
            result["llm_analysis"] = parse_json_object(response)
        return result
