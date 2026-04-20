from __future__ import annotations

from pathlib import Path

EXPLICIT_TARGETS = {
    "check_core_terms_syntax.py": [
        "nat_add.py",
        "mk_pair_r.py",
        "zeros.py",
        "uint_val.py",
        "float_val.py",
        "str_val.py",
        "todo_val.py",
    ],
    "check_coinductives.py": [
        "repeat_tree.py",
        "tree_root_of_repeat.py",
        "zeros.py",
        "zeros_pair.py",
    ],
    "check_modules.py": ["Phase10Mod.py"],
    "check_point5.py": [
        "get_p5_v.py",
        "get_p5_w.py",
        "get_p5_x.py",
        "get_p5_y.py",
        "get_p5_z.py",
    ],
    "check_proj_pair_r.py": [
        "proj_first.py",
        "proj_second.py",
        "swap_pair_r.py",
    ],
}


def required_generated_files(path: Path) -> list[str]:
    explicit = EXPLICIT_TARGETS.get(path.name)
    if explicit is not None:
        return explicit
    return [f"{path.stem.removeprefix('check_')}.py"]


def pytest_ignore_collect(collection_path: Path, config) -> bool:
    path = Path(str(collection_path))
    if path.suffix != ".py" or not path.name.startswith("check_"):
        return False

    repo_root = Path(__file__).resolve().parents[2]
    build_default = repo_root / "_build" / "default"
    if not build_default.is_dir():
        return True

    return any(
        not (build_default / target).is_file()
        for target in required_generated_files(path)
    )
