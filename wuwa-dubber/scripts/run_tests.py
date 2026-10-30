from __future__ import annotations

import importlib.util
import inspect
import sys
import tempfile
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
sys.path.insert(0, str(SRC))


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    failures: list[tuple[str, str]] = []
    total = 0
    for path in sorted(TESTS.glob("test_*.py")):
        module = load_module(path)
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            total += 1
            parameters = inspect.signature(function).parameters
            try:
                if "tmp_path" in parameters:
                    with tempfile.TemporaryDirectory(prefix=f"{path.stem}-") as directory:
                        function(Path(directory))
                else:
                    function()
            except Exception:
                failures.append((f"{path.name}::{name}", traceback.format_exc()))
    for name, trace in failures:
        print(f"FAIL {name}\n{trace}", file=sys.stderr)
    print(f"{total - len(failures)} passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
