import shlex
import shutil
from pathlib import Path
from urllib.parse import urlsplit

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[1]


def dependency_lines(path):
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def requirements_set(path, visited=None):
    visited = set() if visited is None else visited
    path = path.resolve()
    assert path not in visited, "requirements include cycle"
    visited.add(path)
    versions, indexes, references = {}, set(), []
    for line in dependency_lines(path):
        tokens = shlex.split(line)
        if tokens[0] in ("-r", "--requirement"):
            assert len(tokens) == 2
            included = (path.parent / tokens[1]).resolve()
            assert included.is_file(), f"build stage cannot access requirement file: {included}"
            child_versions, child_indexes, child_references = requirements_set(included, visited)
            versions.update(child_versions)
            indexes.update(child_indexes)
            references.extend([included, *child_references])
        elif tokens[0] == "--extra-index-url":
            assert len(tokens) == 2 and urlsplit(tokens[1]).scheme == "https"
            indexes.add(tokens[1])
        else:
            requirement = Requirement(line)
            assert requirement.url is None, "deployment dependencies must not reference local or remote source files"
            specifiers = list(requirement.specifier)
            assert len(specifiers) == 1 and specifiers[0].operator == "=="
            assert "*" not in specifiers[0].version
            name = canonicalize_name(requirement.name)
            if name in versions:
                assert versions[name] == specifiers[0].version
            versions[name] = specifiers[0].version
    visited.remove(path)
    return versions, indexes, references


def test_root_dependency_file_has_no_external_file_references():
    for line in dependency_lines(ROOT / "requirements.txt"):
        assert not line.startswith(("-r", "--requirement", "-c", "--constraint")), line


def test_native_build_can_resolve_requirements_before_source_copy(tmp_path):
    dependency_stage = tmp_path / "tmp/deps"
    dependency_stage.mkdir(parents=True)
    copied = dependency_stage / "requirements.txt"
    shutil.copy2(ROOT / "requirements.txt", copied)
    versions, indexes, references = requirements_set(copied)
    assert versions
    assert not references
    assert indexes == {"https://download.pytorch.org/whl/cpu"}
    assert sorted(path.name for path in dependency_stage.iterdir()) == ["requirements.txt"]


def test_root_pins_match_verified_app_and_review_dependency_chain():
    root_versions, root_indexes, root_references = requirements_set(ROOT / "requirements.txt")
    app_versions, app_indexes, app_references = requirements_set(ROOT / "app/requirements.txt")
    assert root_versions == app_versions
    assert root_indexes == app_indexes
    assert not root_references
    assert (ROOT / "requirements-review.txt").resolve() in app_references
    assert root_versions["torch"] == "2.5.1+cpu"
    assert root_versions["numpy"] == "1.26.4"
    assert root_versions["scikit-learn"] == "1.5.2"


def test_root_dependencies_are_fixed_versions():
    lines = dependency_lines(ROOT / "requirements.txt")
    assert all(">=" not in line and "<=" not in line and "~=" not in line for line in lines)
    versions, _, _ = requirements_set(ROOT / "requirements.txt")
    assert len(versions) >= 20
