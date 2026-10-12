"""Reproduce the published results with one command, and check them.

    python reproduce.py                 # headline: data, P1, P2, figures (about 4 minutes)
    python reproduce.py --all           # every experiment (about 2 hours on one machine)
    python reproduce.py --all --jobs 3  # independent experiments in parallel (about 1 hour)
    python reproduce.py --only run_p9   # one experiment, plus the data it needs

Each driver's printed output is compared with the output recorded for this release in
experiments/expected_output/, so the answer is "matches" or "differs" per experiment
rather than a wall of numbers to eyeball. Logs and any differences go to build/reproduce/.
The downloaded data is also checked against its published fingerprints (manifest.py).

Install first: pip install -e ".[experiments]" (or the exact versions in requirements-lock.txt).
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED = ROOT / "experiments" / "expected_output"
OUT = ROOT / "build" / "reproduce"

# (name, command) in dependency order. A stage's drivers are independent of each other.
DATA = [("panel", "panel.py")]
HEADLINE = [("run_p1", "run_p1.py"), ("run_p2", "run_p2.py"), ("figures", "figures.py")]
EXTENSIONS = [
    ("run_p6", "run_p6.py"), ("run_p7", "run_p7.py"), ("run_p21", "run_p21.py"),
    ("run_p22", "run_p22.py"), ("run_p23", "run_p23.py"), ("run_p10", "run_p10.py"),
    ("run_p11", "run_p11.py"), ("run_p8_baseline", "run_p8.py --baseline"), ("run_p8", "run_p8.py"),
    ("run_p9_baseline", "run_p9.py --baseline"), ("run_p9", "run_p9.py"), ("run_p13", "run_p13.py"),
]
REGIME_DATA = [(f"synthetic_regime_seed{s}", f"synthetic_regime.py --seed {s}") for s in range(6)]
REGIME = [("run_p12", "run_p12.py")] + [
    (f"run_p12b_seed{s}", f"run_p12b.py --seed {s}" + (" --nopogo" if s < 3 else "")) for s in range(6)]
REGIME_GATES = [("run_p12b", "run_p12b.py")]
ALL = [DATA, HEADLINE, EXTENSIONS, REGIME_DATA, REGIME, REGIME_GATES]

# Lines a driver prints only when it had to build a cache first (the synthetic generators
# run on demand), the absolute path figures.py reports, and elapsed-time progress lines;
# none of them is a result.
_GEN_START = ("=== Layer 1a synthetic allocator", "=== Layer 1b MULTI-REGIME")


def normalize(text: str, keep_generator: bool = False) -> list[str]:
    lines, skipping = [], False
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.startswith(_GEN_START) and not keep_generator:
            skipping = True
        if skipping:
            if line.startswith("  wrote synth_"):
                skipping = False
            continue
        line = re.sub(r"(\[progress\].*@ )\d+s", r"\1<t>s", line)       # elapsed seconds vary
        lines.append(re.sub(r"wrote (\d+) figures to .*", r"wrote \1 figures to <repo>/figures", line))
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def run(name: str, cmd: str, record: bool) -> tuple[str, str, float, str]:
    t0 = time.time()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}   # same bytes on every platform
    proc = subprocess.run([sys.executable, *f"experiments/{cmd}".split()], cwd=ROOT, env=env, check=False,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    secs = time.time() - t0
    (OUT / f"{name}.log").write_text(proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else ""),
                                     encoding="utf-8")
    if proc.returncode != 0:
        return name, f"FAILED (exit {proc.returncode}; see build/reproduce/{name}.log)", secs, proc.stdout
    gen = name.startswith("synthetic_regime")      # for the generator itself, the printout IS the result
    got = normalize(proc.stdout, keep_generator=gen)
    if record:
        (EXPECTED / f"{name}.txt").write_text("\n".join(got) + "\n", encoding="utf-8", newline="\n")
        return name, "recorded", secs, proc.stdout
    exp_file = EXPECTED / f"{name}.txt"
    if not exp_file.exists():
        return name, "ran (no recorded output to compare)", secs, proc.stdout
    want = normalize(exp_file.read_text(encoding="utf-8"), keep_generator=gen)
    if got == want:
        return name, "matches the recorded output", secs, proc.stdout
    diff = list(difflib.unified_diff(want, got, "recorded", "this run", n=0, lineterm=""))
    (OUT / f"{name}.diff").write_text("\n".join(diff) + "\n", encoding="utf-8")
    changed = sum(1 for d in diff if d.startswith("+") and not d.startswith("+++"))
    return name, f"DIFFERS ({changed} line(s); see build/reproduce/{name}.diff)", secs, proc.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--all", action="store_true", help="run every experiment, not just the headline")
    ap.add_argument("--only", nargs="+", metavar="NAME", help="run these drivers (e.g. run_p9 run_p12b)")
    ap.add_argument("--jobs", type=int, default=1, help="drivers to run at once within a stage")
    ap.add_argument("--record", action="store_true", help="maintainers: save outputs as the new reference")
    args = ap.parse_args()

    stages = ALL if args.all else [DATA, HEADLINE]
    if args.only:
        # the Layer 1a drivers build their synthetic data on demand; P12 needs Layer 1b first,
        # and the P12b gates need all six per-seed runs
        wanted = set(args.only)
        known = {n for st in ALL for n, _ in st}
        if wanted - known:
            ap.error(f"unknown driver(s) {sorted(wanted - known)}; choose from {sorted(known)}")
        if "run_p12b" in wanted:
            wanted |= {n for n, _ in REGIME if n.startswith("run_p12b_seed")}
        stages = [DATA, [d for d in HEADLINE + EXTENSIONS if d[0] in wanted]]
        if any(n.startswith("run_p12") for n in wanted):
            stages += [REGIME_DATA, [d for d in REGIME if d[0] in wanted],
                       [d for d in REGIME_GATES if d[0] in wanted]]
        stages = [st for st in stages if st]

    OUT.mkdir(parents=True, exist_ok=True)
    EXPECTED.mkdir(parents=True, exist_ok=True)
    results, t0 = [], time.time()
    for stage in stages:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            for name, verdict, secs, stdout in pool.map(lambda d: run(*d, args.record), stage):
                print(f"{name:24s} {verdict}  [{secs:.0f} s]", flush=True)
                results.append(verdict)
                if name == "panel" and "=== data manifest ===" in stdout:   # show the data check
                    print(stdout[stdout.index("=== data manifest ==="):].rstrip(), flush=True)
    print(f"\n{len(results)} driver run(s) in {(time.time() - t0) / 60:.1f} min.")
    bad = [v for v in results if v.startswith(("DIFFERS", "FAILED"))]
    if bad:
        print(f"{len(bad)} did not reproduce; see build/reproduce/. If the data check above reported "
              "a difference, that explains it.")
    elif not args.record:
        print("Every output matches the recorded output for this release.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
