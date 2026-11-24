"""
Single-command Gate-2 reproducibility & integrity verifier for AIRR.SCIENCE.
Checks:
  1. Lean 4 build (`lake build`) and axiom audit (`lake env lean AuditAxioms.lean`) — zero sorry, standard axioms only.
  2. Full pytest verification & exact rational LDL^T replay of all exported certificates.
  3. SHA-256 byte-lock verification against SHA256SUMS.json (or generation when --write-manifest is passed).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(ROOT, "SHA256SUMS.json")

TRACKED_PATTERNS = [
    "README.md",
    "LICENSE",
    "lean/lean-toolchain",
    "lean/lakefile.toml",
    "lean/QcQmcCert.lean",
    "lean/AuditAxioms.lean",
    "src/__init__.py",
    "src/hubbard.py",
    "src/surrogates.py",
    "src/shadows.py",
    "src/hausdorff_sdp.py",
    "scripts/run_all_experiments.py",
    "scripts/make_figures.py",
    "scripts/verify_all.py",
    "tests/test_suite.py",
    "results/experiment1_backflow_obstruction.json",
    "results/experiment2_variational_boost_audit.json",
    "results/experiment3_hubbard_certificates.json",
    "results/experiment4_shadow_coverage_ppi.json",
    "results/rational_certificates/hubbard4_edge_d0_eps005.cert.json",
    "results/rational_certificates/hubbard4_edge_d1_eps005.cert.json",
    "results/rational_certificates/hubbard4_edge_d2_eps005.cert.json",
    "results/rational_certificates/hubbard4_edge_d2_eps002.cert.json",
    "results/rational_certificates/hubbard4_gap_d2_E30_eps002.cert.json",
    "paper/main.tex",
    "paper/references.bib",
    "paper/figures/fig1_backflow_bias_plucker.pdf",
    "paper/figures/fig2_hubbard_energy_gap_bounds.pdf",
    "paper/figures/fig3_finite_sample_coverage_ppi.pdf",
    "paper/main.pdf",
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_lean():
    print("[1/3] Checking Lean 4 formalization & axiom audit...")
    lean_dir = os.path.join(ROOT, "lean")
    r1 = subprocess.run(["lake", "build"], cwd=lean_dir, capture_output=True, text=True)
    if r1.returncode != 0:
        print(r1.stdout, r1.stderr)
        raise RuntimeError("Lean build failed")
    r2 = subprocess.run(["lake", "env", "lean", "AuditAxioms.lean"], cwd=lean_dir, capture_output=True, text=True)
    if r2.returncode != 0:
        print(r2.stdout, r2.stderr)
        raise RuntimeError("Lean axiom audit failed")
    out = r2.stdout + r2.stderr
    for forbidden in ["sorryAx", "Lean.ofReduceBool", "Lean.trustCompiler"]:
        if forbidden in out:
            raise RuntimeError(f"Forbidden axiom detected in Lean output: {forbidden}")
    # Also scan source for sorry
    with open(os.path.join(lean_dir, "QcQmcCert.lean"), encoding="utf-8") as f:
        src = f.read()
    # Check non-comment lines for sorry
    for lineno, line in enumerate(src.splitlines(), 1):
        code = line.split("--")[0]
        if "sorry" in code.split():
            raise RuntimeError(f"Found 'sorry' in QcQmcCert.lean:{lineno}")
    print("  LEAN CHECK: PASS (0 sorry, standard axioms only)")
    return out.strip()


def check_pytest():
    print("[2/3] Running pytest & exact rational LDL^T certificate replay...")
    r = subprocess.run([sys.executable, "-m", "pytest", "-v", "tests/test_suite.py"], cwd=ROOT, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        raise RuntimeError("pytest failed")
    print("  PYTEST + EXACT RATIONAL REPLAY: PASS")


def build_manifest():
    entries = {}
    for rel in TRACKED_PATTERNS:
        full = os.path.join(ROOT, rel.replace("/", os.sep))
        if not os.path.exists(full):
            raise FileNotFoundError(f"Missing tracked file for manifest: {rel}")
        entries[rel] = {
            "sha256": sha256_file(full),
            "bytes": os.path.getsize(full),
        }
    return entries


def check_manifest(write_mode: bool):
    print("[3/3] Checking SHA-256 manifest...")
    current = build_manifest()
    if write_mode:
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump({"files": current}, f, indent=2, sort_keys=True)
        print(f"  Wrote {len(current)} entries to SHA256SUMS.json")
        return
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError("SHA256SUMS.json not found. Run with --write-manifest after building all artifacts.")
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        recorded = json.load(f)["files"]
    if set(recorded.keys()) != set(current.keys()):
        raise RuntimeError("Manifest file list mismatch")
    for rel, meta in recorded.items():
        if meta["sha256"] != current[rel]["sha256"] or meta["bytes"] != current[rel]["bytes"]:
            raise RuntimeError(f"SHA-256 mismatch for {rel}")
    print(f"  MANIFEST CHECK: PASS ({len(recorded)}/{len(recorded)} files match byte-for-byte)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--skip-manifest", action="store_true")
    args = ap.parse_args()

    check_lean()
    check_pytest()
    if not args.skip_manifest:
        check_manifest(write_mode=args.write_manifest)
    print("\nALL GATE-2 VERIFICATION CHECKS: PASS")
