"""
Lending Club 数据源适配器
==========================
把 Lending Club (LC) 公开信贷数据集的字段映射到天池标准 schema，
使外部数据可以直接进入现有的预处理 / 模型流程。

适配要点：
    1. 字段重命名：下划线风格 -> 天池驼峰风格
    2. 特殊字段清洗：去掉 % 号、months 后缀、日期解析、类别编码
    3. 目标变量：loan_status 文本 -> isDefault 0/1
    4. 丢弃数据泄漏字段（贷款发放后才产生的数据）
    5. 天池特有字段处理：
       - n0~n14 匿名特征：LC 没有，整列丢弃
       - pubRecBankruptcies：LC 没有，填 0
       - ficoRangeLow/High：LC 没有原始分，用 grade 业务映射反推

使用示例:
    >>> import pandas as pd
    >>> from src.data.adapters import LendingClubAdapter
    >>> df = pd.read_csv("lending_club.csv")
    >>> adapter = LendingClubAdapter()
    >>> df_tianchi = adapter.transform(df)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("data.adapters.lending_club")


class LendingClubAdapter:
    """把 Lending Club 数据 schema 映射到天池 schema"""

    # ===== 字段重命名表：LC 列名 -> 天池列名 =====
    FIELD_MAPPING = {
        "id": "id",
        "loan_amnt": "loanAmnt",
        "installment": "installment",
        "annual_inc": "annualIncome",
        "dti": "dti",
        "delinq_2yrs": "delinquency_2years",
        "open_acc": "openAcc",
        "pub_rec": "pubRec",
        "revol_bal": "revolBal",
        "total_acc": "totalAcc",
        "grade": "grade",
        "sub_grade": "subGrade",
        "emp_title": "employmentTitle",
        "emp_length": "employmentLength",
        "home_ownership": "homeOwnership",
        "verification_status": "verificationStatus",
        "purpose": "purpose",
        "title": "title",
        "policy_code": "policyCode",
        "revol_util": "revolUtil",
        "term": "term",
        "int_rate": "interestRate",
        "zip_code": "postCode",
        "addr_state": "regionCode",
        "issue_d": "issueDate",
        "earliest_cr_line": "earliesCreditLine",
        "initial_list_status": "initialListStatus",
        "application_type": "applicationType",
    }

    # ===== 贷款发放后才产生的字段（数据泄漏，必须丢弃）=====
    LEAKAGE_COLUMNS = [
        "member_id", "funded_amnt", "funded_amnt_inv", "pymnt_plan", "url",
        "desc", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
        "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
        "recoveries", "collection_recovery_fee", "last_pymnt_d",
        "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
        "collections_12_mths_ex_med", "mths_since_last_major_derog",
        "acc_now_delinq", "annual_inc_joint", "dti_joint",
        "verification_status_joint", "inq_last_6mths",
        "mths_since_last_delinq", "mths_since_last_record",
    ]

    # ===== LC 特有、天池没有对应语义的字段（丢弃）=====
    DROP_COLUMNS = LEAKAGE_COLUMNS + [
        "loan_status",  # 目标变量，单独转换后改名为 isDefault
    ]

    # ===== grade 评级反推 FICO 分数区间（LC 官方历史口径的近似中值）=====
    GRADE_TO_FICO_LOW = {
        "A": 720, "B": 680, "C": 640, "D": 600, "E": 560, "F": 520, "G": 500,
    }
    GRADE_TO_FICO_HIGH = {
        "A": 750, "B": 719, "C": 679, "D": 639, "E": 599, "F": 559, "G": 519,
    }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行完整转换。

        参数:
            df: 原始 Lending Club DataFrame

        返回:
            符合天池 schema 的 DataFrame
        """
        df = df.copy()
        n_before = len(df)
        logger.info(f"Lending Club 数据开始转换 - 原始 shape: {df.shape}")

        # 1. 转换目标变量 loan_status -> isDefault（未到期样本先剔除）
        df = self._convert_target(df)

        # 2. 清洗带符号的数值字段（% / months）
        df = self._clean_numeric_fields(df)

        # 3. 清洗类别字段（邮编 / 州 / 上市状态 / 申请类型）
        df = self._clean_categorical_fields(df)

        # 4. 解析日期字段
        df = self._clean_datetime_fields(df)

        # 5. 用 grade 反推 FICO 分数
        df = self._infer_fico_from_grade(df)

        # 6. 补齐天池有、LC 没有的字段
        df["pubRecBankruptcies"] = 0

        # 7. 丢弃无关字段（泄漏字段 + 天池无对应字段）
        drop_cols = [c for c in self.DROP_COLUMNS if c in df.columns]
        df = df.drop(columns=drop_cols)

        # 8. 统一重命名为天池字段
        df = df.rename(columns={k: v for k, v in self.FIELD_MAPPING.items()
                                if k in df.columns})

        # 9. 天池 n0~n14 匿名特征 LC 没有，整列丢弃（不进入后续特征矩阵）
        #    这里不补 0，因为整列单一值无区分度，训练时用天池字段交集即可。

        logger.info(
            f"Lending Club 数据转换完成 - 输出 shape: {df.shape}，"
            f"剔除未到期/无效样本 {n_before - len(df)} 条"
        )
        return df

    # ----------------------------------------------------------
    # 各字段清洗步骤
    # ----------------------------------------------------------

    def _convert_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """loan_status 文本标签 -> isDefault 0/1"""
        if "loan_status" not in df.columns:
            logger.warning("未找到 loan_status 列，跳过目标变量转换")
            return df

        status = df["loan_status"].astype(str).str.strip()

        # 只保留已到期、有明确结论的样本
        paid_mask = status.isin(["Fully Paid"])
        bad_mask = status.isin(["Charged Off", "Default"])
        valid_mask = paid_mask | bad_mask

        n_drop = (~valid_mask).sum()
        if n_drop > 0:
            logger.info(f"剔除未到期/状态不明确样本 {n_drop} 条（Current 等）")
        df = df[valid_mask].copy()

        # 重新计算（因为上面剔除了样本）
        status = status[valid_mask]
        df["isDefault"] = np.where(status.isin(["Charged Off", "Default"]), 1, 0)
        return df

    def _clean_numeric_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗带 % 和 months 的数值字段"""
        # int_rate: "10.65%" -> 10.65
        if "int_rate" in df.columns:
            df["int_rate"] = (
                pd.to_numeric(df["int_rate"].astype(str).str.strip().str.rstrip("%"),
                              errors="coerce")
            )

        # revol_util: "45.2%" -> 45.2
        if "revol_util" in df.columns:
            df["revol_util"] = (
                pd.to_numeric(df["revol_util"].astype(str).str.strip().str.rstrip("%"),
                              errors="coerce")
            )

        # term: " 36 months" -> 3（年），" 60 months" -> 5（年），与天池口径一致
        if "term" in df.columns:
            months = pd.to_numeric(
                df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
            )
            df["term"] = months / 12

        return df

    def _clean_categorical_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗类别字段：邮编、州、上市状态、申请类型"""
        # zip_code: "10xxx" -> 10（天池 postCode 为脱敏整数）
        if "zip_code" in df.columns:
            df["zip_code"] = (
                pd.to_numeric(df["zip_code"].astype(str).str.extract(r"(\d+)")[0],
                              errors="coerce")
            )

        # addr_state: 州缩写 -> 数字编码（按出现顺序映射，保证可复现）
        if "addr_state" in df.columns:
            states = sorted(df["addr_state"].dropna().astype(str).unique())
            state_map = {s: i for i, s in enumerate(states)}
            df["addr_state"] = df["addr_state"].astype(str).map(state_map)

        # initial_list_status: f/w -> 0/1
        if "initial_list_status" in df.columns:
            df["initial_list_status"] = (
                df["initial_list_status"].astype(str).str.strip().map({"w": 1, "f": 0})
            )
            df["initial_list_status"] = df["initial_list_status"].fillna(0).astype(int)

        # application_type: Individual -> 0, Joint App -> 1
        if "application_type" in df.columns:
            df["application_type"] = (
                df["application_type"].astype(str).str.strip()
                .map({"Individual": 0, "Joint App": 1})
            )
            df["application_type"] = df["application_type"].fillna(0).astype(int)

        # policy_code: 转成整数（天池为类别列）
        if "policy_code" in df.columns:
            df["policy_code"] = pd.to_numeric(df["policy_code"], errors="coerce")

        return df

    def _clean_datetime_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """解析 LC 日期格式（如 'Dec-2018'）"""
        for col in ["issue_d", "earliest_cr_line"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format="%b-%Y", errors="coerce")
        return df

    def _infer_fico_from_grade(self, df: pd.DataFrame) -> pd.DataFrame:
        """LC 没有原始 FICO 分，用 grade 评级反推一个估算区间中值"""
        if "grade" not in df.columns:
            return df

        grade = df["grade"].astype(str).str.strip()
        low = grade.map(self.GRADE_TO_FICO_LOW)
        high = grade.map(self.GRADE_TO_FICO_HIGH)

        # 未知 grade 用总体常见区间中值 660 兜底
        df["ficoRangeLow"] = low.fillna(660).astype(float)
        df["ficoRangeHigh"] = high.fillna(660).astype(float)
        return df
