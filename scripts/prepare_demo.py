import shutil
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.inference import InferenceService
from src.pipeline.orchestrator import ArtifactManager, TrainingPipeline
from src.utils.config_loader import ConfigLoader


def main():
    configs = ConfigLoader(str(ROOT / "configs"))
    cfg = configs.get("project")
    assets = ROOT / "demo_assets"
    sample = assets / "sample.json"
    if not sample.exists():
        raise FileNotFoundError("缺少固定演示数据 demo_assets/sample.json，请先恢复原数据文件")
    data = pd.read_json(sample, orient="table")
    previous = joblib.load(assets / "bundle.joblib") if (assets / "bundle.joblib").exists() else None
    pipeline = TrainingPipeline(configs)
    result = pipeline.run(data=data, use_sample=True, quick=True, skip_tuning=True)
    assets.mkdir(exist_ok=True)
    bundle = ArtifactManager.load_latest(ROOT / "data/models")
    if bundle["meta"]["source"] != "合成演示数据":
        raise ValueError("仅允许将合成数据训练产物放入公开演示目录")
    service = InferenceService(bundle)
    service.predict_proba(bundle["raw_test"].head(5))
    if previous and previous["meta"]["fit_train_indices"] == bundle["meta"]["fit_train_indices"]:
        bundle["previous_eval_results"] = previous.get("previous_eval_results", previous["eval_results"])
        bundle.pop("dir", None)
        bundle.pop("ensemble_model", None)
        joblib.dump(bundle, ROOT / "data/models/latest/bundle.joblib", compress=3)
    shutil.copy2(ROOT / "data/models/latest/bundle.joblib", assets / "bundle.joblib")
    print(result)


if __name__ == "__main__":
    main()
