"""Full pipeline: run all benchmarks and produce a manifest.

Steps:
  1. Recovery demo  (4 variants, damage → recovery)
  2. Matched-present ablation (shared snapshot)
  3. Solver verification (5 checks)
  4. Onset-map sweeps (alpha0×lambda_s, chi×damage_scale)
  5. Write manifest.json with timestamps, git hash, all output paths.

Usage:
    python benchmarks/run_full_pipeline.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    ("recovery_demo",        ROOT / "benchmarks" / "run_recovery_demo.py"),
    ("matched_present",      ROOT / "benchmarks" / "run_matched_present.py"),
    ("solver_verification",  ROOT / "benchmarks" / "run_solver_verification.py"),
    ("onset_map",            ROOT / "benchmarks" / "run_onset_map.py"),
]


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_artifacts(outputs_dir: Path) -> list[dict]:
    arts = []
    for p in sorted(outputs_dir.rglob("*")):
        if p.is_file():
            arts.append({
                "path": str(p.relative_to(ROOT)),
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })
    return arts


def main():
    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)

    results = {}
    overall_ok = True

    for name, script in SCRIPTS:
        print(f"\n{'='*60}")
        print(f"  [{name}] {script.name}")
        print(f"{'='*60}")
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - t0
        ok = proc.returncode == 0
        overall_ok = overall_ok and ok
        results[name] = {
            "passed": ok,
            "returncode": proc.returncode,
            "elapsed_s": round(elapsed, 2),
            "stdout_tail": proc.stdout[-500:] if proc.stdout else "",
            "stderr_tail": proc.stderr[-500:] if proc.stderr else "",
        }
        status_str = "OK" if ok else "FAIL"
        print(f"  {status_str} ({elapsed:.1f}s)")
        if proc.stdout:
            for line in proc.stdout.strip().split("\n")[-5:]:
                print(f"    {line}")
        if not ok and proc.stderr:
            for line in proc.stderr.strip().split("\n")[-5:]:
                print(f"    [stderr] {line}")

    # Collect all output artifacts
    artifacts = collect_artifacts(outputs)

    manifest = {
        "submission": "Adaptive Chern Self-Healing Conductance Law — Verified Benchmark Release",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash(),
        "python_version": sys.version,
        "overall": "PASS" if overall_ok else "FAIL",
        "stages": results,
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
    }

    manifest_path = outputs / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n{'='*60}")
    print(f"  Pipeline: {manifest['overall']}")
    print(f"  Artifacts: {len(artifacts)}")
    print(f"  Manifest: {manifest_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
