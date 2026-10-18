"""
基础冒烟测试
============
确保各模块基本可用。

运行: pytest tests/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def configs():
    """全局配置"""
    from src.utils.config_loader import ConfigLoader
    return ConfigLoader(str(ROOT / "configs"))


@pytest.fixture(scope="module")
def sample_df():
    """合成测试数据"""
    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame({
        "id": np.arange(n),
        "loanAmnt": rng.lognormal(9.5, 0.5, n),
        "term": rng.choice([3, 5], n),
        "interestRate": rng.uniform(5, 25, n),
        "installment": rng.uniform(100, 2000, n),
        "grade": rng.choice(["A", "B", "C", "D"], n),
        "subGrade": rng.choice(["A1", "B2"], n),
        "employmentTitle": rng.uniform(0, 100, n),
        "employmentLength": rng.choice(["< 1 year", "1 year", "10+ years"], n),
        "homeOwnership": rng.choice([0, 1, 2], n),
        "annualIncome": rng.lognormal(11, 0.5, n),
        "verificationStatus": rng.choice([0, 1, 2], n),
        "issueDate": ["2019-01-01"] * n,
        "purpose": rng.integers(0, 14, n),
        "postCode": rng.uniform(0, 1000, n),
        "regionCode": rng.integers(0, 50, n),
        "dti": rng.uniform(0, 50, n),
        "delinquency_2years": rng.poisson(0.3, n).astype(float),
        "ficoRangeLow": rng.integers(580, 850, n),
        "ficoRangeHigh": rng.integers(580, 850, n),
        "openAcc": rng.poisson(11, n).astype(float),
        "pubRec": rng.poisson(0.2, n).astype(float),
        "pubRecBankruptcies": rng.poisson(0.1, n).astype(float),
        "revolBal": rng.lognormal(8, 1, n),
        "revolUtil": rng.uniform(0, 100, n),
        "totalAcc": rng.poisson(25, n).astype(float),
        "initialListStatus": rng.choice([0, 1], n),
        "applicationType": rng.choice([0, 1], n),
        "earliesCreditLine": ["Jan-2010"] * n,
        "title": rng.uniform(0, 100, n),
        "policyCode": np.ones(n),
        "isDefault": rng.choice([0, 1], n, p=[0.8, 0.2]),
    })
    # 加几个匿名字段
    for i in range(15):
        df[f"n{i}"] = rng.normal(0, 1, n)
    return df


# ===== 测试 ConfigLoader =====
class TestConfig:
    def test_load_all_configs(self, configs):
        assert configs.get("data") is not None
        assert configs.get("model") is not None
        assert configs.get("feature") is not None
        assert configs.get("llm") is not None

    def test_attr_access(self, configs):
        # 应支持点访问
        lr = configs.get("model").xgboost.params.learning_rate
        assert isinstance(lr, (int, float))


# ===== 测试评估指标 =====
class TestMetrics:
    def test_evaluator(self):
        from src.utils.metrics import Evaluator
        rng = np.random.default_rng(0)
        y_true = rng.choice([0, 1], 1000)
        y_pred = rng.uniform(0, 1, 1000)

        evaluator = Evaluator()
        result = evaluator.evaluate(y_true, y_pred)

        assert 0 <= result.auc <= 1
        assert 0 <= result.ks <= 1
        assert 0 <= result.f1 <= 1


# ===== 测试预处理 =====
class TestPreprocessor:
    def test_fit_transform(self, configs, sample_df):
        from src.data.preprocessor import Preprocessor

        prep = Preprocessor(configs.get("data"), configs.get("feature"))
        X = sample_df.drop(columns=["isDefault", "id"])
        X_processed = prep.fit_transform(X)

        # 处理后不应有缺失值
        assert X_processed.isnull().sum().sum() == 0

    def test_knn_imputation(self, configs, sample_df):
        """测试 dti 和 revolUtil 的 KNN 填充"""
        from src.data.preprocessor import Preprocessor

        # 故意给 dti 和 revolUtil 加缺失值
        df = sample_df.drop(columns=["isDefault", "id"]).copy()
        df.loc[0:10, "dti"] = np.nan
        df.loc[20:30, "revolUtil"] = np.nan

        prep = Preprocessor(configs.get("data"), configs.get("feature"))
        X_processed = prep.fit_transform(df)

        # 不应有缺失值
        assert X_processed.isnull().sum().sum() == 0

        # dti 和 revolUtil 应该被填上了合理的值
        assert "dti" in X_processed.columns
        assert "revolUtil" in X_processed.columns
        # KNN填充的值应该在合理范围内（不会是极端值）
        assert X_processed["dti"].between(-100, 200).all()
        assert X_processed["revolUtil"].between(-100, 200).all()


# ===== 测试 XGBoost =====
class TestXGB:
    def test_fit_predict(self, configs, sample_df):
        from src.models.xgb_model import XGBModel
        from src.data.preprocessor import Preprocessor

        prep = Preprocessor(configs.get("data"), configs.get("feature"))
        X = sample_df.drop(columns=["isDefault", "id"])
        y = sample_df["isDefault"]
        X_processed = prep.fit_transform(X)

        # 减少 estimators 加快测试
        configs.get("model").xgboost.params.n_estimators = 50
        configs.get("model").xgboost.early_stopping.enabled = False

        model = XGBModel(configs.get("model").xgboost)
        model.fit(X_processed, y)

        proba = model.predict_proba(X_processed)
        assert proba.shape == (len(y),)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_save_load(self, configs, sample_df, tmp_path):
        from src.models.xgb_model import XGBModel
        from src.data.preprocessor import Preprocessor

        prep = Preprocessor(configs.get("data"), configs.get("feature"))
        X = sample_df.drop(columns=["isDefault", "id"])
        y = sample_df["isDefault"]
        X_processed = prep.fit_transform(X)

        configs.get("model").xgboost.params.n_estimators = 50
        configs.get("model").xgboost.early_stopping.enabled = False

        model = XGBModel(configs.get("model").xgboost)
        model.fit(X_processed, y)

        save_path = tmp_path / "test_model.json"
        model.save(str(save_path))
        assert save_path.exists()

        model2 = XGBModel(configs.get("model").xgboost)
        model2.load(str(save_path))
        proba1 = model.predict_proba(X_processed.head(10))
        proba2 = model2.predict_proba(X_processed.head(10))
        assert np.allclose(proba1, proba2, rtol=1e-5)


# ===== 测试 LightGBM =====
class TestLGB:
    def test_fit_predict(self, configs, sample_df):
        from src.data.preprocessor import Preprocessor
        from src.models.lgb_model import LGBModel

        prep = Preprocessor(configs.get("data"), configs.get("feature"))
        X = sample_df.drop(columns=["isDefault", "id"])
        y = sample_df["isDefault"]
        X_processed = prep.fit_transform(X)

        configs.get("model").lightgbm.params.n_estimators = 50

        model = LGBModel(configs.get("model").lightgbm)
        model.fit(X_processed, y)

        proba = model.predict_proba(X_processed)
        assert proba.shape == (len(y),)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_save_load(self, configs, sample_df, tmp_path):
        from src.data.preprocessor import Preprocessor
        from src.models.lgb_model import LGBModel

        prep = Preprocessor(configs.get("data"), configs.get("feature"))
        X = sample_df.drop(columns=["isDefault", "id"])
        y = sample_df["isDefault"]
        X_processed = prep.fit_transform(X)

        configs.get("model").lightgbm.params.n_estimators = 50

        model = LGBModel(configs.get("model").lightgbm)
        model.fit(X_processed, y)

        save_path = tmp_path / "test_lgb_model.txt"
        model.save(str(save_path))
        assert save_path.exists()


# ===== 测试超参优化器 =====
class TestOptimizer:
    def test_ks_metric(self, configs, sample_df):
        """测试 metric 切换为 KS 时能正常工作"""
        from src.data.preprocessor import Preprocessor
        from src.tuning.optimizer import OptunaOptimizer

        prep = Preprocessor(configs.get("data"), configs.get("feature"))
        X = sample_df.drop(columns=["isDefault", "id"])
        y = sample_df["isDefault"]
        X_processed = prep.fit_transform(X)

        # 只跑2个trial，快速测试
        configs.get("model").xgboost.tuning.n_trials = 2
        configs.get("model").xgboost.tuning.metric = "ks"

        optimizer = OptunaOptimizer(configs.get("model").xgboost.tuning)
        from xgboost import XGBClassifier

        best_params = optimizer.optimize(
            model_class=XGBClassifier,
            X_train=X_processed,
            y_train=y,
            X_val=X_processed.head(100),
            y_val=y.head(100),
            base_params={"objective": "binary:logistic"},
        )

        # 应该返回了最佳参数
        assert isinstance(best_params, dict)
        assert len(best_params) > 0

    def test_auc_metric_default(self, configs, sample_df):
        """测试默认 AUC 指标仍然正常"""
        from src.data.preprocessor import Preprocessor
        from src.tuning.optimizer import OptunaOptimizer

        prep = Preprocessor(configs.get("data"), configs.get("feature"))
        X = sample_df.drop(columns=["isDefault", "id"])
        y = sample_df["isDefault"]
        X_processed = prep.fit_transform(X)

        configs.get("model").xgboost.tuning.n_trials = 2

        optimizer = OptunaOptimizer(configs.get("model").xgboost.tuning)
        from xgboost import XGBClassifier

        best_params = optimizer.optimize(
            model_class=XGBClassifier,
            X_train=X_processed,
            y_train=y,
            X_val=X_processed.head(100),
            y_val=y.head(100),
            base_params={"objective": "binary:logistic"},
        )

        assert isinstance(best_params, dict)


# ===== 测试漂移检测器 =====
class TestDriftDetector:
    def test_no_drift(self, sample_df):
        """同分布数据不应检测到漂移"""
        from src.monitoring.drift_detector import DriftDetector

        baseline = sample_df[["loanAmnt", "annualIncome", "dti", "revolUtil"]].head(200)
        new_data = sample_df[["loanAmnt", "annualIncome", "dti", "revolUtil"]].tail(200)

        detector = DriftDetector(baseline)
        drifts = detector.detect(new_data, threshold=0.1)

        # 同分布数据，不应该有high级别的漂移
        high_drifts = [k for k, v in drifts.items() if v["severity"] == "high"]
        assert len(high_drifts) == 0

    def test_detect_drift(self, sample_df):
        """人为制造分布偏移，应该能检测到"""
        from src.monitoring.drift_detector import DriftDetector

        baseline = sample_df[["loanAmnt", "annualIncome"]].copy()
        # 人为制造偏移：新数据收入整体翻倍
        new_data = sample_df[["loanAmnt", "annualIncome"]].copy()
        new_data["annualIncome"] = new_data["annualIncome"] * 5

        detector = DriftDetector(baseline)
        drifts = detector.detect(new_data, threshold=0.1)

        # annualIncome 应该检测到漂移
        assert "annualIncome" in drifts

    def test_report(self, sample_df):
        """测试漂移报告文本生成"""
        from src.monitoring.drift_detector import DriftDetector

        baseline = sample_df[["loanAmnt", "annualIncome"]].head(100)
        new_data = sample_df[["loanAmnt", "annualIncome"]].tail(100)

        detector = DriftDetector(baseline)
        report = detector.get_drift_report(new_data)

        assert isinstance(report, str)
        assert len(report) > 0


# ===== 测试 Lending Club 适配器 =====
class TestLendingClubAdapter:
    def test_field_mapping_and_target(self):
        """测试字段重命名、目标变量转换、Current 样本剔除"""
        from src.data.adapters import LendingClubAdapter

        n = 100
        rng = np.random.default_rng(42)
        lc_df = pd.DataFrame({
            "id": np.arange(n),
            "loan_amnt": rng.uniform(1000, 20000, n),
            "term": rng.choice([" 36 months", " 60 months"], n),
            "int_rate": [f"{rng.uniform(5, 25):.2f}%" for _ in range(n)],
            "revol_util": [f"{rng.uniform(0, 100):.1f}%" for _ in range(n)],
            "grade": rng.choice(["A", "B", "C", "D"], n),
            "sub_grade": rng.choice(["A1", "B2", "C3"], n),
            "annual_inc": rng.uniform(20000, 150000, n),
            "dti": rng.uniform(0, 40, n),
            "home_ownership": rng.choice(["OWN", "RENT", "MORTGAGE"], n),
            "emp_title": rng.integers(0, 1000, n),
            "emp_length": rng.choice(["1 year", "5 years", "10+ years"], n),
            "zip_code": rng.choice(["10xxx", "20xxx", "94xxx"], n),
            "addr_state": rng.choice(["CA", "NY", "TX"], n),
            "issue_d": ["Dec-2018"] * n,
            "earliest_cr_line": ["Jan-2005"] * n,
            "initial_list_status": rng.choice(["f", "w"], n),
            "application_type": rng.choice(["Individual", "Joint App"], n),
            "policy_code": np.ones(n),
            "purpose": rng.choice(["debt_consolidation", "credit_card"], n),
            "title": rng.integers(0, 100, n),
            "loan_status": rng.choice(["Fully Paid", "Charged Off", "Current"], n),
        })

        adapter = LendingClubAdapter()
        out = adapter.transform(lc_df)

        # 目标变量已生成且只有 0/1
        assert "isDefault" in out.columns
        assert set(out["isDefault"].unique()).issubset({0, 1})
        # Current 未到期样本已剔除
        assert len(out) < n

        # 字段已重命名为天池风格
        assert "loanAmnt" in out.columns
        assert "annualIncome" in out.columns
        # % 和 months 已清洗成数值
        assert pd.api.types.is_numeric_dtype(out["interestRate"])
        assert pd.api.types.is_numeric_dtype(out["term"])
        # FICO 已由 grade 反推
        assert "ficoRangeLow" in out.columns
        assert "ficoRangeHigh" in out.columns
        # 天池补齐字段
        assert "pubRecBankruptcies" in out.columns
        # 泄漏字段已剔除
        assert "total_pymnt" not in out.columns

    def test_registered_in_loader(self):
        """测试适配器已在 DataLoader 注册表中"""
        from src.data.loader import _ADAPTERS
        assert "lending_club" in _ADAPTERS


# ===== 测试 Feature Builder =====
class TestFeatureBuilder:
    def test_build(self, configs, sample_df):
        from src.features.builder import FeatureBuilder

        builder = FeatureBuilder(configs.get("feature"))
        df_new = builder.build(sample_df)

        # 至少应该多出一些衍生特征
        assert df_new.shape[1] >= sample_df.shape[1]


# ===== 测试 Prompt Manager =====
class TestPromptManager:
    def test_render(self, configs):
        from src.llm.prompts import PromptManager

        pm = PromptManager(configs.get("llm"))
        # 简单测试 chat_assistant 模板
        prompt = pm.render("chat_assistant", user_question="什么是 AUC？")
        assert "什么是 AUC？" in prompt
        assert len(prompt) > 50

    def test_system_role(self, configs):
        from src.llm.prompts import PromptManager
        pm = PromptManager(configs.get("llm"))
        role = pm.get_system_role("risk_report")
        assert isinstance(role, str)
        assert len(role) > 0


# ===== 测试 SHAP =====
class TestSHAP:
    def test_explain(self, configs, sample_df):
        from src.models.xgb_model import XGBModel
        from src.data.preprocessor import Preprocessor
        from src.explain.shap_analyzer import SHAPAnalyzer

        prep = Preprocessor(configs.get("data"), configs.get("feature"))
        X = sample_df.drop(columns=["isDefault", "id"])
        y = sample_df["isDefault"]
        X_processed = prep.fit_transform(X)

        configs.get("model").xgboost.params.n_estimators = 30
        configs.get("model").xgboost.early_stopping.enabled = False

        model = XGBModel(configs.get("model").xgboost)
        model.fit(X_processed, y)

        analyzer = SHAPAnalyzer(max_background_size=50)
        analyzer.fit(model.model, X_processed.head(50))

        # 解释单样本
        explanation = analyzer.explain_single(X_processed.iloc[0])
        assert explanation.shap_values is not None
        assert len(explanation.feature_names) == X_processed.shape[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
