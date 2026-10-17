import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER_BASE = "60318f595a823e7fbb381960464eaad2e7016b5c"
CHECKOUT_BASE = "82b8c132702c61cb571019418f6ccf1fa5e59d91"


def git(*args):
    return subprocess.check_output(["git", "-c", "core.quotepath=false", *args], cwd=ROOT)


def protected(path):
    prefixes = ("src/data/", "src/models/", "src/tuning/", "src/explain/", "src/monitoring/")
    return path.startswith(prefixes) or path in [
        "src/features/builder.py", "src/features/selector.py", "src/features/__init__.py",
        "configs/data_config.yaml", "configs/feature_config.yaml", "configs/model_config.yaml",
        "tests/test_basic.py", "scripts/generate_sample_data.py", "scripts/explore_data.py",
        "scripts/download_data.py"]


def build_report():
    tracked = git("ls-tree", "-r", "--name-only", CHECKOUT_BASE).decode("utf-8").splitlines()
    unchanged = []
    for path in tracked:
        if protected(path):
            original = git("show", f"{CHECKOUT_BASE}:{path}")
            current = (ROOT / path).read_bytes()
            if original.replace(b"\r\n", b"\n") != current.replace(b"\r\n", b"\n"):
                raise ValueError("受保护文件发生变化：" + path)
            unchanged.append(path)
    differences = set(git("diff", "--name-only", CHECKOUT_BASE).decode("utf-8").splitlines())
    untracked = set(git("ls-files", "--others", "--exclude-standard").decode("utf-8").splitlines())
    original_paths = set(tracked)
    changed = sorted(differences & original_paths)
    added = sorted((differences | untracked) - original_paths)
    env_ignored = subprocess.run(["git", "check-ignore", ".env", ".local/settings.yaml",
                                 ".streamlit/secrets.toml"], cwd=ROOT, capture_output=True, text=True).stdout
    result = {"user_base": USER_BASE, "checkout_base": CHECKOUT_BASE,
              "teammate_updates_after_user_base": git("diff", "--name-only", f"{USER_BASE}..{CHECKOUT_BASE}").decode().splitlines(),
              "modified_existing_files": changed, "new_files": added,
              "protected_unchanged": unchanged, "ignored_secret_paths": env_ignored.splitlines()}
    folder = ROOT / "docs"
    folder.mkdir(exist_ok=True)
    (folder / "change_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    patch = git("diff", "--no-ext-diff", CHECKOUT_BASE, "--", *changed) if changed else b""
    (folder / "existing_files.diff").write_bytes(patch)
    print(json.dumps({"modified": len(changed), "new": len(added), "protected_unchanged": len(unchanged)}))
    return result


if __name__ == "__main__":
    build_report()
