"""Keep the documented scoring:verl PYTHONPATH unambiguous."""
import os
from pathlib import Path
import subprocess
import sys


def test_documented_pythonpath_selects_project_scripts(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    competing_root = tmp_path / "verl"
    competing_scripts = competing_root / "scripts"
    competing_scripts.mkdir(parents=True)
    (competing_scripts / "__init__.py").write_text(
        "ORIGIN = 'competing-verl-package'\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(project_root), str(competing_root)))
    environment["PYTHONNOUSERSITE"] = "1"
    # -S excludes site-packages; discovery must work without importing torch,
    # initializing a device, or relying on the caller's current directory.
    result = subprocess.run(
        [sys.executable, "-S", "-c", """
import importlib.util
from pathlib import Path
import sys
import scripts
project = Path(sys.argv[1]).resolve()
assert Path(scripts.__file__).resolve() == project / "scripts" / "__init__.py"
spec = importlib.util.find_spec("scripts.long_worker_probe")
assert spec is not None, "project probe is hidden by the competing scripts package"
assert Path(spec.origin).resolve() == project / "scripts" / "long_worker_probe.py"
""", str(project_root)],
        cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
