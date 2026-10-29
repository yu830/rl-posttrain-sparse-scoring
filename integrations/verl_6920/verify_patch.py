"""Check the patch on temporary copies of pinned Git blobs; never edit the source checkout."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent


def verify_patch(source_dir: str | Path) -> dict:
    source_dir = Path(source_dir).resolve()
    manifest = json.loads((HERE / "source-manifest.json").read_text())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_dir, text=True).strip()
    if head != manifest["commit"]:
        raise ValueError(f"expected pinned verl commit {manifest['commit']}, found {head}")
    original = {}
    patched = {}
    with tempfile.TemporaryDirectory(prefix="verl-6920-patch-check-") as tmp:
        stage = Path(tmp)
        for path, expected_blob in manifest["source_blobs"].items():
            blob = subprocess.check_output(["git", "rev-parse", f"{head}:{path}"], cwd=source_dir, text=True).strip()
            if blob != expected_blob:
                raise ValueError(f"pinned source blob mismatch for {path}")
            contents = subprocess.check_output(["git", "show", f"{head}:{path}"], cwd=source_dir)
            original[path] = contents.decode()
            target = stage / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
        patch = HERE / "memory-bounded-teacher.patch"
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=stage, check=True, capture_output=True)
        subprocess.run(["git", "apply", str(patch)], cwd=stage, check=True, capture_output=True)
        for path in manifest["source_blobs"]:
            contents = (stage / path).read_text()
            if path.endswith(".py"):
                ast.parse(contents, filename=path)
                compile(contents, path, "exec")
            patched[path] = contents
    return {
        "source_commit": head,
        "patch_application": "PASS_on_temporary_pinned_blob_copies",
        "python_syntax": "PASS",
        "worker_runtime": "NOT_RUN",
        "original_sources": original,
        "patched_sources": patched,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verl-dir", required=True)
    args = parser.parse_args()
    result = verify_patch(args.verl_dir)
    result.pop("original_sources")
    result.pop("patched_sources")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
