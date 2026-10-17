import copy

import numpy as np
import pandas as pd


class ProjectFeatureBuilder:
    def __init__(self, rules):
        self.rules = copy.deepcopy(list(rules))

    def build(self, df):
        result = df.copy()
        for rule in self.rules:
            name = rule["name"]
            if name in result.columns:
                raise ValueError(f"衍生特征与输入列重名：{name}")
            result[name] = self.calculate(result, rule)
        return result.replace([np.inf, -np.inf], np.nan)

    transform = build

    @staticmethod
    def calculate(df, rule):
        operation = rule["operation"]
        if operation == "ratio":
            return pd.to_numeric(df[rule["numerator"]], errors="coerce") / pd.to_numeric(
                df[rule["denominator"]], errors="coerce").replace(0, np.nan)
        if operation == "monthly_income":
            return df["annualIncome"] / 12
        if operation == "payment_ratio":
            return df["installment"] / (df["annualIncome"] / 12).replace(0, np.nan)
        if operation == "fico_mean":
            return (df["ficoRangeLow"] + df["ficoRangeHigh"]) / 2
        if operation == "bucket":
            return pd.cut(df[rule["column"]], bins=rule["bins"], labels=False, include_lowest=True).astype(float)
        if operation in ("sum", "product", "difference"):
            a = pd.to_numeric(df[rule["left"]], errors="coerce")
            b = pd.to_numeric(df[rule["right"]], errors="coerce")
            return a+b if operation == "sum" else a*b if operation == "product" else a-b
        raise ValueError(f"暂不支持该特征操作：{operation}")

    def get_created_features(self):
        return [rule["name"] for rule in self.rules]
