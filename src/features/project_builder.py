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
        if operation == "log1p":
            return np.log1p(pd.to_numeric(df[rule["column"]], errors="coerce").clip(lower=0))
        if operation == "category_flag":
            values = df[rule["column"]]
            flag = values.astype(str).isin([str(v) for v in rule["values"]]).astype(float)
            return flag.where(values.notna())
        if operation == "annual_interest_ratio":
            return df["loanAmnt"] * df["interestRate"] / 100 / df["annualIncome"].replace(0, np.nan)
        if operation == "credit_history":
            start = pd.to_datetime(df["earliesCreditLine"], format="mixed", errors="coerce")
            end = pd.to_datetime(df["issueDate"], format="mixed", errors="coerce")
            return ((end-start).dt.days / 365.25).clip(lower=0)
        if operation == "debt_surplus":
            return df["annualIncome"] / 12 * (1-df["dti"] / 100) - df["installment"]
        if operation == "ratio_plus_one":
            return pd.to_numeric(df[rule["numerator"]], errors="coerce") / (
                pd.to_numeric(df[rule["denominator"]], errors="coerce") + 1).replace(0, np.nan)
        if operation == "bucket":
            return pd.cut(df[rule["column"]], bins=rule["bins"], labels=False, include_lowest=True).astype(float)
        if operation in ("sum", "product", "difference"):
            a = pd.to_numeric(df[rule["left"]], errors="coerce")
            b = pd.to_numeric(df[rule["right"]], errors="coerce")
            return a+b if operation == "sum" else a*b if operation == "product" else a-b
        raise ValueError(f"暂不支持该特征操作：{operation}")

    def get_created_features(self):
        return [rule["name"] for rule in self.rules]
