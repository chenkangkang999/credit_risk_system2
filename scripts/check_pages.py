import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["APP_ENV"] = "public_demo" if "--public" in sys.argv else "development"
if "--public" in sys.argv:
    os.environ["CREDIT_RISK_PUBLIC_DEMO"] = "1"
from streamlit.testing.v1 import AppTest


def main():
    results = []
    for file in [ROOT / "app/streamlit_app.py"] + sorted((ROOT / "app/pages").glob("*.py")):
        app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=60).run()
        if file.name != "streamlit_app.py":
            app.switch_page(str(file.relative_to(ROOT / "app"))).run()
        errors = [str(error.message) for error in app.exception]
        results.append({"page": file.name, "errors": errors})
        print(json.dumps(results[-1], ensure_ascii=True))
    folder = ROOT / ".local"
    folder.mkdir(exist_ok=True)
    mode = "public" if "--public" in sys.argv else "local"
    (folder / f"pages_{mode}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    if any(row["errors"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
