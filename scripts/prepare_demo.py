import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_sample_data import generate_sample_data
from src.pipeline.inference import InferenceService
from src.pipeline.orchestrator import ArtifactManager, TrainingPipeline
from src.utils.config_loader import ConfigLoader


def main():
    configs = ConfigLoader(str(ROOT / "configs"))
    cfg = configs.get("project")
    data = generate_sample_data(cfg.demo.samples, cfg.demo.seed)
    path = ROOT / "data/sample"
    path.mkdir(parents=True, exist_ok=True)
    data.to_csv(path / "sample.csv", index=False)
    pipeline = TrainingPipeline(configs)
    result = pipeline.run(data=data, use_sample=True, quick=True, skip_tuning=False)
    assets = ROOT / "demo_assets"
    assets.mkdir(exist_ok=True)
    bundle = ArtifactManager.load_latest(ROOT / "data/models")
    if bundle["meta"]["source"] != "合成演示数据":
        raise ValueError("仅允许将合成数据训练产物放入公开演示目录")
    service = InferenceService(bundle)
    service.predict_proba(bundle["raw_test"].head(5))
    shutil.copy2(ROOT / "data/models/latest/bundle.joblib", assets / "bundle.joblib")
    data.to_json(assets / "sample.json", orient="table", force_ascii=False, index=False)
    print(result)


if __name__ == "__main__":
    main()
