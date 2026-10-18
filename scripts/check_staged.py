import io
import json
import pickle
import re
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def git_bytes(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL)


def main():
    from check_changes import CHECKOUT_BASE, protected
    key = dotenv_values(ROOT / ".env").get("DEEPSEEK_API_KEY", "")
    if not key:
        raise ValueError("本地密钥未配置")
    matches, sensitive_paths, changed_protected = [], [], []
    records = git_bytes("ls-files", "-s", "-z").split(b"\0")
    key_forms = [key.encode("utf-8"), key.encode("utf-16-le")]
    patterns = [rb"sk-[A-Za-z0-9_-]{20,}", rb"gh[pousr]_[A-Za-z0-9]{20,}",
                rb"github_pat_[A-Za-z0-9_]{30,}"]
    count = 0
    for record in records:
        if not record:
            continue
        metadata, path_bytes = record.split(b"\t", 1)
        path = path_bytes.decode("utf-8")
        oid = metadata.split()[1].decode("ascii")
        data = git_bytes("cat-file", "blob", oid)
        count += 1
        parts = Path(path).parts
        name = Path(path).name
        data_readme = path == "data/README.md" and data == git_bytes("show", f"{CHECKOUT_BASE}:data/README.md")
        if (name.startswith(".env") and name != ".env.example") or name == "secrets.toml" or \
                any(part in {".venv", ".local", "logs", ".git"} for part in parts) or \
                (path.startswith("data/") and not data_readme):
            sensitive_paths.append(path)
        if any(secret in data for secret in key_forms) or any(re.search(pattern, data) for pattern in patterns):
            matches.append(path)
        if protected(path):
            try:
                original = git_bytes("show", f"{CHECKOUT_BASE}:{path}")
            except subprocess.CalledProcessError:
                changed_protected.append(path)
            else:
                if original != data:
                    changed_protected.append(path)
        if path == "demo_assets/bundle.joblib":
            import joblib
            bundle = joblib.load(io.BytesIO(data))
            if bundle["meta"]["source"] != "合成演示数据":
                sensitive_paths.append(path)
            plain = pickle.dumps(bundle)
            if any(secret in plain for secret in key_forms):
                matches.append(path)
    env_ignored = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=ROOT).returncode == 0
    result = {"index_files_checked": count, "secret_match_paths": sorted(set(matches)),
              "sensitive_paths": sensitive_paths, "changed_protected_paths": changed_protected,
              "env_ignored": env_ignored,
              "branch_is_main": git_bytes("branch", "--show-current").strip() == b"main"}
    folder = ROOT / ".local"
    folder.mkdir(exist_ok=True)
    (folder / "staged_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))
    if matches or sensitive_paths or changed_protected or not env_ignored or not result["branch_is_main"]:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"check_failed": True, "error_type": type(error).__name__}))
        raise SystemExit(1)
