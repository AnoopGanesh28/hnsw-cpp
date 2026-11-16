"""Helpers for bench/run_all.sh (Python standard library only).

Subcommands:
  env             environment record of the run, embedding env_manual.txt
  get             print one key of a JSON object (for the shell script)
  pick-efs        smallest swept ef whose recall reaches each target
  spread          list (system, ef, threads) points whose timing spread is high
  check-ablation  same-kernel ablation runs must return identical result ids
  project-layout  project the M2 build time to the full base and pick its size
  check-layout    M2 baseline results must equal the all-off scalar variant's
  head-report     keep the header and the first N entries of `perf report`

Each check exits nonzero when its condition fails, after writing its JSON.
"""

import argparse
import glob
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def load(path):
    with open(path) as f:
        return json.load(f)


def dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def command_output(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return {"cmd": " ".join(cmd), "rc": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"cmd": " ".join(cmd), "error": str(e)}


def read_or_none(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError as e:
        return "unavailable: " + str(e)


def cmd_env(a):
    with open(a.manual, "rb") as f:
        manual = f.read()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    doc = {
        "tool": "run_all.sh env",
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "env_manual": {
            "path": os.path.abspath(a.manual),
            "sha256": hashlib.sha256(manual).hexdigest(),
            "text": manual.decode("utf-8", errors="replace"),
        },
        "pinned_cpu_for_single_thread_runs": int(a.pin_cpu),
        "multi_thread_runs": "not pinned",
        "base_vectors": a.n,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "lscpu": command_output(["lscpu"]),
        "free_k": command_output(["free", "-k"]),
        "nproc": command_output(["nproc"]),
        "uname": command_output(["uname", "-a"]),
        "cpu_governor": read_or_none("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "perf_event_paranoid": read_or_none("/proc/sys/kernel/perf_event_paranoid"),
        "git": {
            "commit": command_output(["git", "-C", root, "rev-parse", "HEAD"]),
            "status": command_output(["git", "-C", root, "status", "--porcelain",
                                      "--untracked-files=no"]),
            "m2_worktree": command_output(["git", "-C", a.m2_worktree, "rev-parse", "HEAD"]),
        },
        "compilers": {
            "g++": command_output(["g++", "--version"]),
            "clang++": command_output(["clang++", "--version"]),
            "cmake": command_output(["cmake", "--version"]),
        },
        "taskset": shutil.which("taskset"),
    }
    dump(doc, a.out)


def cmd_get(a):
    v = load(a.file)[a.key]
    print(json.dumps(v) if isinstance(v, bool) else v)


def results_of(doc, system, threads=None):
    return [r for r in doc["results"]
            if r["system"] == system and (threads is None or r["threads"] == threads)]


def cmd_pick_efs(a):
    doc = load(a.sweep)
    rows = sorted(results_of(doc, a.system, 1), key=lambda r: r["ef"])
    targets = [float(t) for t in a.targets.split(",")]
    out = {"source": os.path.abspath(a.sweep), "system": a.system, "threads": 1,
           "rule": "smallest swept ef whose recall_at_k >= target", "points": []}
    efs = []
    for t in targets:
        hit = next((r for r in rows if r["recall_at_k"] >= t), None)
        if hit is None:
            dump(out, a.out + ".failed")
            sys.exit(f"pick-efs: no swept ef reaches recall {t} for {a.system}")
        out["points"].append({"target": t, "ef": hit["ef"], "recall_at_k": hit["recall_at_k"]})
        out[f"ef_{t}"] = hit["ef"]
        efs.append(hit["ef"])
    out["efs_csv"] = ",".join(str(e) for e in sorted(set(efs)))
    dump(out, a.out)


def spread(r):
    s = r["seconds"]
    return (s["max"] - s["min"]) / s["median"]


def cmd_spread(a):
    flagged = []
    for path in a.files:
        for r in load(path)["results"]:
            sp = spread(r)
            if sp > a.threshold:
                flagged.append({"file": os.path.basename(path), "system": r["system"],
                                "ef": r["ef"], "threads": r["threads"], "spread": sp,
                                "seconds": r["seconds"]})
    dump({"threshold": a.threshold, "definition": "(max - min) / median of the timed passes",
          "points": flagged}, a.out)
    for f in flagged:
        print(f"high spread: {f['system']} ef={f['ef']} threads={f['threads']} "
              f"spread={f['spread']:.3f} ({f['file']})", file=sys.stderr)


ABL_NAME = re.compile(r"abl_e(\d)h(\d)p(\d)_(scalar|avx2)_r(\d+)\.json$")
KERNEL_NAME = {"scalar": "l2_squared_scalar", "avx2": "l2_squared_avx2"}


def cmd_check_ablation(a):
    runs = {}
    problems = []
    for path in sorted(glob.glob(os.path.join(a.dir, "abl_*.json"))):
        m = ABL_NAME.search(path)
        if not m:
            continue
        e, h, p, kernel, rep = int(m[1]), int(m[2]), int(m[3]), m[4], int(m[5])
        doc = load(path)
        ours_env = doc["environment"]["ours"]
        switches = (ours_env["HNSW_OPT_EPOCH_VISITED"], ours_env["HNSW_OPT_ALLOC_FREE_HEAPS"],
                    ours_env["HNSW_OPT_PREFETCH"])
        if switches != (bool(e), bool(h), bool(p)):
            problems.append(f"{path}: binary switches {switches} do not match its name")
        if doc["systems"]["ours"]["kernel"] != KERNEL_NAME[kernel]:
            problems.append(f"{path}: kernel {doc['systems']['ours']['kernel']} != {kernel}")
        if not doc["environment"]["build"]["release_build"]:
            problems.append(f"{path}: not a release build")
        runs[(e, h, p, kernel, rep)] = {r["ef"]: r for r in doc["results"]}
    expected = {(e, h, p, k, r) for e in (0, 1) for h in (0, 1) for p in (0, 1)
                for k in ("scalar", "avx2") for r in range(1, a.reps + 1)}
    missing = sorted(expected - set(runs))
    if missing:
        problems.append(f"missing runs: {missing}")
    hashes = {}
    for (e, h, p, k, rep), by_ef in runs.items():
        for ef, r in by_ef.items():
            hashes.setdefault((k, ef), {}).setdefault(r["ids_hash"], []).append(
                f"e{e}h{h}p{p}_r{rep}")
    kernels = {}
    for (k, ef), by_hash in sorted(hashes.items()):
        identical = len(by_hash) == 1
        if not identical:
            problems.append(f"kernel {k} ef {ef}: result ids differ between runs: "
                            + json.dumps(by_hash))
        recalls = sorted({r[ef]["recall_at_k"] for key, r in runs.items() if key[3] == k})
        kernels[f"{k}_ef{ef}"] = {"identical_ids": identical, "hashes": sorted(by_hash),
                                  "recall_at_k": recalls}
    dump({"rule": "for each kernel and ef, every switch combination and repetition returns "
                  "the same result ids (FNV-1a hash of the id array)",
          "runs": len(runs), "groups": kernels, "ok": not problems, "problems": problems}, a.out)
    if problems:
        sys.exit("check-ablation: " + "; ".join(problems))


def fvecs_count(path):
    with open(path, "rb") as f:
        dim = int.from_bytes(f.read(4), "little")
    return os.path.getsize(path) // (4 + 4 * dim)


def cmd_project_layout(a):
    probe = load(a.probe)
    n_probe = probe["config"]["n"]
    t_probe = probe["build_seconds"]
    full = fvecs_count(a.base) if a.n == "all" else int(a.n)
    # HNSW insertion costs O(log n) per vector, so a build is O(n log n).
    projected = t_probe * (full / n_probe) * (math.log(full) / math.log(n_probe))
    subset = projected > a.limit_seconds
    dump({"probe_n": n_probe, "probe_build_seconds": t_probe, "full_n": full,
          "model": "t_full = t_probe * (n_full / n_probe) * (ln n_full / ln n_probe)",
          "projected_full_build_seconds": projected, "limit_seconds": a.limit_seconds,
          "subset": subset, "layout_n": min(a.subset_n, full) if subset else full}, a.out)


def cmd_check_layout(a):
    m2 = load(a.m2)
    m2_by_ef = {r["ef"]: r for r in m2["results"]}
    problems = []
    brackets = []
    for i in range(a.reps + 1):
        path = os.path.join(a.dir, f"alloff_bracket_{i}.json")
        if not os.path.exists(path):
            problems.append(f"missing {path}")
            continue
        doc = load(path)
        env = doc["environment"]["ours"]
        if any(env[k] for k in ("HNSW_OPT_EPOCH_VISITED", "HNSW_OPT_ALLOC_FREE_HEAPS",
                                "HNSW_OPT_PREFETCH")):
            problems.append(f"{path}: not the all-off variant")
        if doc["systems"]["ours"]["kernel"] != "l2_squared_scalar":
            problems.append(f"{path}: not the scalar kernel")
        for r in doc["results"]:
            ref = m2_by_ef.get(r["ef"])
            if ref is None or ref["ids_hash"] != r["ids_hash"]:
                problems.append(f"ef {r['ef']}: bracket {i} ids hash {r['ids_hash']} != M2 "
                                f"{ref['ids_hash'] if ref else None}")
        brackets.append(i)
    dump({"rule": "M2 baseline result ids equal the all-off scalar variant's (FNV-1a hash)",
          "m2_hashes": {str(ef): r["ids_hash"] for ef, r in m2_by_ef.items()},
          "brackets_checked": brackets, "ok": not problems, "problems": problems}, a.out)
    if problems:
        sys.exit("check-layout: " + "; ".join(problems))


def cmd_head_report(a):
    entries = 0
    for line in sys.stdin:
        if line.startswith("#") or not line.strip():
            if entries == 0:
                sys.stdout.write(line.rstrip() + "\n")
            continue
        if entries < a.entries:  # keep reading so perf never sees a closed pipe
            sys.stdout.write(line.rstrip() + "\n")
            entries += 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("env")
    s.add_argument("--out", required=True)
    s.add_argument("--manual", required=True)
    s.add_argument("--pin-cpu", required=True)
    s.add_argument("--n", required=True)
    s.add_argument("--m2-worktree", required=True)
    s = sub.add_parser("get")
    s.add_argument("--file", required=True)
    s.add_argument("--key", required=True)
    s = sub.add_parser("pick-efs")
    s.add_argument("--sweep", required=True)
    s.add_argument("--system", required=True)
    s.add_argument("--targets", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("spread")
    s.add_argument("--threshold", type=float, required=True)
    s.add_argument("--out", required=True)
    s.add_argument("files", nargs="+")
    s = sub.add_parser("check-ablation")
    s.add_argument("--dir", required=True)
    s.add_argument("--reps", type=int, required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("project-layout")
    s.add_argument("--probe", required=True)
    s.add_argument("--base", required=True)
    s.add_argument("--n", required=True)
    s.add_argument("--limit-seconds", type=float, required=True)
    s.add_argument("--subset-n", type=int, required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("check-layout")
    s.add_argument("--m2", required=True)
    s.add_argument("--dir", required=True)
    s.add_argument("--reps", type=int, required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("head-report")
    s.add_argument("--entries", type=int, required=True)
    a = p.parse_args()
    {"env": cmd_env, "get": cmd_get, "pick-efs": cmd_pick_efs, "spread": cmd_spread,
     "check-ablation": cmd_check_ablation, "project-layout": cmd_project_layout,
     "check-layout": cmd_check_layout, "head-report": cmd_head_report}[a.cmd](a)


if __name__ == "__main__":
    main()
