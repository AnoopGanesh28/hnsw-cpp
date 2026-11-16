"""Builds the benchmark report from the raw results in bench/results/.

  python3 bench/make_report.py [--results bench/results] [--update-readme]

Writes RESULTS/summary.md, RESULTS/recall_vs_qps.png and RESULTS/ablation.png.
With --update-readme it also replaces the block between <!-- RESULTS:BEGIN -->
and <!-- RESULTS:END --> in README.md with sections of summary.md (the same
strings, so bench/check_readme.py can trace every number).

Every number shown is computed here from the JSON/text files written by
bench/run_all.sh. Displayed values are truncated toward zero to the digits
shown, never rounded up. A missing input makes its section say "not measured"
with the file name, rather than failing the whole report.
"""

import argparse
import glob
import json
import math
import os
import re
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = (0.95, 0.99)
SPREAD_LIMIT = 0.15
OPTS = ("epoch", "heaps", "prefetch", "simd")
OPT_LABEL = {"epoch": "epoch visited list", "heaps": "allocation-free heaps",
             "prefetch": "software prefetch", "simd": "AVX2+FMA kernel"}
# Reference categorical palette, slots 1 and 2 (dataviz skill, references/palette.md).
COLOR = {"ours": "#2a78d6", "hnswlib": "#eb6834"}
THREAD_STYLE = {1: ("-", "o"), 4: ("--", "s"), 8: (":", "^")}


# ------------------------------------------------------------ formatting
def trunc(x, decimals):
    """x truncated toward zero to `decimals` places, as a string."""
    f = 10 ** decimals
    v = math.trunc(x * f) / f
    return f"{v:.{decimals}f}"


def trunc_sig(x, sig=3):
    """x truncated toward zero to `sig` significant digits, as a string."""
    if x == 0:
        return "0"
    exp = math.floor(math.log10(abs(x)))
    decimals = max(sig - 1 - exp, 0)
    return trunc(x, decimals)


def qps(x):
    return str(math.trunc(x))


def ratio(x):
    return trunc(x, 3)


def rec(x):
    return trunc(x, 4)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def load(path):
    with open(path) as f:
        return json.load(f)


class Missing(Exception):
    pass


def need(path):
    if not os.path.exists(path):
        raise Missing(os.path.relpath(path, ROOT))
    return path


def section(fn):
    """Runs a section builder; a missing input becomes a 'not measured' note."""
    def wrapped(*a):
        try:
            return fn(*a)
        except Missing as e:
            return f"Not measured: `{e}` does not exist.\n"
    return wrapped


# ------------------------------------------------------------ data access
def spread(seconds):
    return (seconds["max"] - seconds["min"]) / seconds["median"]


class Results:
    def __init__(self, d):
        self.d = d

    def path(self, *p):
        return os.path.join(self.d, *p)

    def sweep(self, threads):
        return load(need(self.path(f"sweep_t{threads}.json")))

    def sweep_threads(self):
        return [t for t in (1, 4, 8) if os.path.exists(self.path(f"sweep_t{t}.json"))]


def rows_for(doc, system):
    return sorted((r for r in doc["results"] if r["system"] == system), key=lambda r: r["ef"])


def first_reaching(rows, target):
    return next((r for r in rows if r["recall_at_k"] >= target), None)


def pass_qps(r):
    n = r["queries"]
    s = r["seconds"]
    return n / s["median"], n / s["max"], n / s["min"]


# ------------------------------------------------------------ sections
@section
def sec_environment(R):
    env_doc = load(need(R.path("environment.json")))
    sweep = R.sweep(1)
    e = sweep["environment"]
    ds = sweep["dataset"]
    b = e["build"]
    lines = [
        f"- Dataset: `{ds['name']}`, {ds['index_vectors']} base vectors, "
        f"{sweep['config']['queries']} queries, recall@{sweep['config']['k']} against the "
        f"first {sweep['config']['k']} ground-truth ids.",
        f"- CPU: {e['cpu']['model']}, {e['cpu']['physical_cores']} cores / "
        f"{e['cpu']['logical_cpus']} logical CPUs; MemTotal {e['mem_total_kib']} kB.",
        f"- Kernel: `{e['kernel']['release']}` (WSL detected: {e['wsl']['detected']}).",
        f"- Compiler: {e['compiler']['gxx_version_line']} (`{e['compiler']['path']}`).",
        f"- Flags: `{b['cxx_flags']}`; per file: "
        + ", ".join(f"`{k}`: `{v}`" for k, v in b["per_file_options"].items())
        + f"; release build: {b['release_build']}.",
        f"- Source: commit `{e['source']['commit']}` (`{e['source']['describe']}`, "
        f"dirty: {e['source']['dirty']}).",
        f"- hnswlib: {e['hnswlib']['tag_pinned']} (`{e['hnswlib']['commit_at_build']}`), "
        f"L2 function `{e['hnswlib']['l2_function']}`. Ours: `{e['ours']['l2_function']}`.",
        f"- Single-thread runs pinned to logical CPU "
        f"{env_doc['pinned_cpu_for_single_thread_runs']} (`taskset`; recorded affinity "
        f"{e['cpu_affinity']}); multi-thread runs not pinned.",
    ]
    manual = env_doc["env_manual"]
    return ("\n".join(lines) + "\n\nHand-recorded environment (`env_manual.txt`, sha256 "
            f"`{manual['sha256']}`):\n\n```\n{manual['text'].rstrip()}\n```\n")


def headline_points(R):
    """{(system, threads, target): row or None}."""
    pts = {}
    for t in R.sweep_threads():
        doc = R.sweep(t)
        for s in ("ours", "hnswlib"):
            rows = rows_for(doc, s)
            for target in TARGETS:
                pts[(s, t, target)] = first_reaching(rows, target)
    return pts


def compare_word(ours_qps, hnswlib_qps):
    return "faster" if ours_qps > hnswlib_qps else "slower" if ours_qps < hnswlib_qps else "equal"


@section
def sec_headline(R):
    need(R.path("sweep_t1.json"))
    pts = headline_points(R)
    rows = []
    for t in R.sweep_threads():
        for target in TARGETS:
            for s in ("ours", "hnswlib"):
                r = pts[(s, t, target)]
                if r is None:
                    rows.append([t, f">= {target}", s, "no swept ef reaches it", "", "", ""])
                    continue
                med, lo, hi = pass_qps(r)
                rows.append([t, f">= {target}", s, r["ef"], rec(r["recall_at_k"]), qps(med),
                             f"{qps(lo)} to {qps(hi)}"])
    out = table(["threads", "recall@10 target", "system", "ef", "recall@10", "QPS (median)",
                 "QPS range (slowest to fastest pass)"], rows)
    cmp_rows = []
    for t in R.sweep_threads():
        for target in TARGETS:
            o, h = pts[(("ours", t, target))], pts[("hnswlib", t, target)]
            if o is None or h is None:
                cmp_rows.append([t, f">= {target}", "not comparable (target not reached)", ""])
                continue
            oq, hq = pass_qps(o)[0], pass_qps(h)[0]
            cmp_rows.append([t, f">= {target}", ratio(oq / hq), compare_word(oq, hq)])
    out += "\n\n" + table(["threads", "recall@10 target", "ours QPS / hnswlib QPS",
                           "ours is"], cmp_rows)
    return out + "\n"


@section
def sec_verdict(R):
    need(R.path("sweep_t1.json"))
    pts = headline_points(R)
    parts = []
    for target in TARGETS:
        o, h = pts[("ours", 1, target)], pts[("hnswlib", 1, target)]
        if o is None or h is None:
            parts.append(f"not comparable at recall@10 >= {target} (a system never reaches it)")
            continue
        oq, hq = pass_qps(o)[0], pass_qps(h)[0]
        parts.append(f"{compare_word(oq, hq)} than hnswlib at recall@10 >= {target} "
                     f"({qps(oq)} vs {qps(hq)} QPS, ratio {ratio(oq / hq)})")
    multi = []
    for t in R.sweep_threads():
        if t == 1:
            continue
        words = []
        for target in TARGETS:
            o, h = pts[("ours", t, target)], pts[("hnswlib", t, target)]
            if o is not None and h is not None:
                words.append(f"{compare_word(pass_qps(o)[0], pass_qps(h)[0])} at >= {target}")
        if words:
            multi.append(f"with {t} threads it is " + " and ".join(words))
    text = "**Verdict:** single-threaded on this machine, ours is " + "; ".join(parts) + "."
    if multi:
        text += " " + "; ".join(multi).capitalize() + "."
    return text + "\n"


@section
def sec_thread_consistency(R):
    single = {(r["system"], r["ef"]): r["ids_hash"] for r in R.sweep(1)["results"]}
    rows = []
    for t in R.sweep_threads():
        if t == 1:
            continue
        doc = R.sweep(t)
        same = sum(1 for r in doc["results"] if single.get((r["system"], r["ef"])) == r["ids_hash"])
        rows.append([t, f"{same} of {len(doc['results'])}"])
    return ("Multi-thread result ids identical to single-thread (same system and ef, "
            "compared by hash):\n\n" + table(["threads", "identical points"], rows) + "\n")


@section
def sec_build(R):
    rows, mem = [], []
    for s in ("ours", "hnswlib"):
        d = load(need(R.path(f"build_{s}.json")))
        b = d["build_seconds"]
        rows.append([s, d["config"]["n"], d["config"]["M"], d["config"]["ef_construction"],
                     len(b["raw"]), trunc(b["median"], 1), f"{trunc(b['min'], 1)} to {trunc(b['max'], 1)}",
                     d["index"]["bytes"], trunc(d["index"]["bytes"] / 2**20, 1)])
        m = d["memory"]
        mem.append([s, m["vm_hwm_end_kib"], m["vm_hwm_after_load_kib"],
                    m["vm_hwm_end_kib"] - m["vm_hwm_after_load_kib"]])
    return (table(["system", "vectors", "M", "efConstruction", "runs", "build s (median)",
                   "build s (range)", "index bytes", "index MiB"], rows)
            + "\n\nPeak memory of the build process (VmHWM, kB); the dataset-only value is "
            "VmHWM right after loading the base vectors:\n\n"
            + table(["system", "VmHWM end", "VmHWM after dataset load", "delta"], mem) + "\n")


@section
def sec_load(R):
    d = load(need(R.path("loadtime_warm.json")))
    c = d["config"]
    rows = []
    for r in d["results"]:
        ls, fq = r["load_seconds"], r["first_queries_seconds"]
        rows.append([r["strategy"], trunc_sig(ls["median"] * 1000), trunc_sig(ls["min"] * 1000)
                     + " to " + trunc_sig(ls["max"] * 1000), trunc_sig(fq["median"] * 1000)])
    out = (f"Warm page cache, {c['reps']} repetitions; first {c['first_queries']} queries at "
           f"ef {c['ef']}, single-threaded:\n\n"
           + table(["strategy", "load ms (median)", "load ms (range)",
                    f"first {c['first_queries']} queries ms (median)"], rows) + "\n")
    cold = R.path("loadtime_cold.json")
    if os.path.exists(cold):
        d = load(cold)
        rows = [[r["strategy"], trunc_sig(r["load_seconds"]["median"] * 1000),
                 trunc_sig(r["first_queries_seconds"]["median"] * 1000)] for r in d["results"]]
        out += ("\nCold page cache (Linux page cache dropped before every load):\n\n"
                + table(["strategy", "load ms (median)", "first queries ms (median)"], rows) + "\n")
    else:
        out += "\nCold page cache: not measured (needs sudo; `bench/run_cold_loadtime.sh`).\n"
    return out


# ------------------------------------------------------------ ablation
ABL_NAME = re.compile(r"abl_e(\d)h(\d)p(\d)_(scalar|avx2)_r(\d+)\.json$")


def ablation_data(R):
    files = glob.glob(R.path("ablation", "abl_*.json"))
    if not files:
        raise Missing(os.path.relpath(R.path("ablation"), ROOT) + "/abl_*.json")
    times, recall, nq = {}, {}, None
    for path in files:
        m = ABL_NAME.search(path)
        key = (int(m[1]), int(m[2]), int(m[3]), int(m[4] == "avx2"))
        for r in load(path)["results"]:
            times.setdefault((key, r["ef"]), []).append(r["seconds"]["raw"][0])
            recall[(key[3], r["ef"])] = r["recall_at_k"]
            nq = r["queries"]
    efs = load(need(R.path("ablation_efs.json")))
    return times, recall, nq, efs


def combo_name(c):
    on = [o for o, v in zip(OPTS, c) if v]
    return "all off (scalar)" if not on else "+".join(on)


def abl_qps(times, nq, c, ef):
    t = times[(c, ef)]
    return nq / statistics.median(t), nq / max(t), nq / min(t)


@section
def sec_ablation(R):
    times, recall, nq, efs = ablation_data(R)
    pts = efs["points"]
    off, full = (0, 0, 0, 0), (1, 1, 1, 1)
    reps = len(times[(off, pts[0]["ef"])])
    hdr_pts = [f"ef {p['ef']} (recall >= {p['target']})" for p in pts]
    out = [f"Single-threaded, pinned, the same `ours.hnsw` for every variant; {reps} "
           "repetitions per combination, each its own process (1 warm-up + 1 timed pass), "
           "combinations interleaved round robin. QPS is queries / median timed pass. "
           "Recall@10 per kernel: "
           + "; ".join(f"ef {p['ef']}: scalar {rec(recall[(0, p['ef'])])}, AVX2 "
                       f"{rec(recall[(1, p['ef'])])}" for p in pts) + ".\n"]

    def cell(c, ef, base=None):
        q = abl_qps(times, nq, c, ef)[0]
        return qps(q) + (f" (x{ratio(q / abl_qps(times, nq, base, ef)[0])})" if base else "")

    rows = []
    for i, o in enumerate(OPTS):
        c = tuple(int(j == i) for j in range(4))
        rows.append([OPT_LABEL[o]] + [cell(c, p["ef"], off) for p in pts])
    out.append("**Each optimization alone** (QPS, speedup over all off with the scalar "
               "kernel):\n\n" + table(["optimization"] + hdr_pts, rows)
               + "\n\nBaseline, all off (scalar): "
               + ", ".join(f"ef {p['ef']}: {cell(off, p['ef'])} QPS" for p in pts) + ".\n")

    chain = [(0, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 1, 1, 0), (1, 1, 1, 1)]
    names = ["all off (scalar)", "+ epoch visited list", "+ allocation-free heaps",
             "+ software prefetch", "+ AVX2+FMA kernel"]
    rows = []
    for k, c in enumerate(chain):
        row = [names[k]]
        for p in pts:
            q = abl_qps(times, nq, c, p["ef"])[0]
            step = "" if k == 0 else f", step x{ratio(q / abl_qps(times, nq, chain[k - 1], p['ef'])[0])}"
            row.append(f"{qps(q)} (total x{ratio(q / abl_qps(times, nq, off, p['ef'])[0])}{step})")
        rows.append(row)
    out.append("**Cumulative** (QPS; total speedup over all off, and the step over the row "
               "above):\n\n" + table(["configuration"] + hdr_pts, rows) + "\n")

    rows = []
    for i, o in enumerate(OPTS):
        c = tuple(0 if j == i else 1 for j in range(4))
        row = [f"without {OPT_LABEL[o]}"]
        for p in pts:
            qf, qw = abl_qps(times, nq, full, p["ef"])[0], abl_qps(times, nq, c, p["ef"])[0]
            row.append(f"{qps(qw)} (full / this: x{ratio(qf / qw)})")
        rows.append(row)
    out.append("**Leave one out** from the full configuration ("
               + ", ".join(f"ef {p['ef']}: {cell(full, p['ef'])} QPS" for p in pts)
               + "):\n\n" + table(["configuration"] + hdr_pts, rows) + "\n")

    rows = []
    for c in sorted({k for k, _ in times}):
        rows.append([combo_name(c)] + [
            f"{qps(abl_qps(times, nq, c, p['ef'])[0])} ({qps(abl_qps(times, nq, c, p['ef'])[1])} "
            f"to {qps(abl_qps(times, nq, c, p['ef'])[2])})" for p in pts])
    out.append("All 16 combinations (QPS median, slowest to fastest repetition):\n\n"
               + table(["enabled"] + hdr_pts, rows) + "\n")
    chk = load(need(R.path("ablation_check.json")))
    out.append(f"Result-id check (`ablation_check.json`): identical ids for every same-kernel "
               f"combination: {chk['ok']}.\n")
    return "\n".join(out)


@section
def sec_layout(R):
    proj = load(need(R.path("layout", "projection.json")))
    m2 = load(need(R.path("layout", "m2.json")))
    chk = load(need(R.path("layout_check.json")))
    reps = m2["reps"]
    nq = m2["queries"]
    before, after = {}, {}
    for i in range(reps + 1):
        for r in load(need(R.path("layout", f"alloff_bracket_{i}.json")))["results"]:
            t = r["seconds"]["raw"][0]
            if i < reps:
                before.setdefault(r["ef"], []).append(t)
            if i > 0:
                after.setdefault(r["ef"], []).append(t)
    size = (f"{proj['layout_n']}-vector subset (ground truth recomputed for it)"
            if proj["subset"] else f"full base ({proj['layout_n']} vectors)")
    rows = []
    for r in m2["results"]:
        ef = r["ef"]
        qm = nq / r["seconds"]["median"]
        qb = nq / statistics.median(before[ef])
        qa = nq / statistics.median(after[ef])
        rows.append([ef, rec(r["recall_at_k"]), qps(qm), f"{qps(qb)} (x{ratio(qb / qm)})",
                     f"{qps(qa)} (x{ratio(qa / qm)})"])
    return (f"Memory layout, measured on the {size}. The M2 build time was projected from a "
            f"{proj['probe_n']}-vector build ({trunc(proj['probe_build_seconds'], 1)} s) to "
            f"{trunc(proj['projected_full_build_seconds'], 1)} s for {proj['full_n']} vectors "
            f"(n log n; limit {trunc(proj['limit_seconds'], 1)} s). M2 baseline (pre-layout, "
            f"scalar) versus the all-off scalar variant on an index with the same graph, both "
            f"single-threaded and pinned, {reps} repetitions; the all-off variant was measured "
            f"just before and just after every M2 repetition:\n\n"
            + table(["ef", "recall@10", "M2 QPS", "all-off QPS before (vs M2)",
                     "all-off QPS after (vs M2)"], rows)
            + f"\n\nResult-id check (`layout_check.json`): M2 ids equal the all-off variant's: "
            f"{chk['ok']}.\n")


# ------------------------------------------------------------ profile
REPORT_LINE = re.compile(r"^\s*([\d.]+)%\s+(\S+)\s+\[\.\]\s+(.*)$")


def short_symbol(sym):
    sym = sym.strip()
    depth, out = 0, ""
    for ch in sym:  # drop template and argument lists
        if ch in "(<":
            depth += 1
        elif ch in ")>":
            depth -= 1
        elif depth == 0:
            out += ch
    return out.strip() or sym


STAT_LINE = re.compile(r"^\s*([\d,.]+|<not supported>|<not counted>)\s+(?:msec\s+)?([\w-]+)(:u)?")


def parse_stat(path):
    vals = {}
    with open(path) as f:
        for line in f:
            m = STAT_LINE.match(line)
            if m:
                v = m[1]
                vals[m[2]] = None if v.startswith("<") else float(v.replace(",", ""))
    return vals


@section
def sec_profile(R):
    if os.path.exists(R.path("perf", "perf_unavailable.txt")):
        with open(R.path("perf", "perf_unavailable.txt")) as f:
            return ("Profile: not available. perf could not run here; the recorded errors are "
                    "in `bench/results/perf/perf_unavailable.txt`:\n\n```\n"
                    + f.read().strip() + "\n```\n")
    with open(need(R.path("perf", "perf_info.txt"))) as f:
        info = f.read().strip()
    out = ["Sampled with `perf record -e cpu-clock` (user-space samples, whole process: "
           "includes loading and the untimed warm-up and recall code), single-threaded and "
           "pinned. Top symbols (`perf report --stdio --no-children`, full lists in "
           "`bench/results/perf/report_*.txt`):\n"]
    for s in ("ours", "hnswlib"):
        rows = []
        with open(need(R.path("perf", f"report_{s}.txt"))) as f:
            for line in f:
                m = REPORT_LINE.match(line)
                if m and len(rows) < 8:
                    sym = re.split(r"\s{2,}", m[3].strip())[0]  # drop perf's IPC columns
                    rows.append([m[1] + "%", m[2], "`" + short_symbol(sym) + "`"])
        out.append(f"{s}:\n\n" + table(["samples", "object", "symbol"], rows) + "\n")
    rows = []
    unsupported = []
    for s in ("ours", "hnswlib"):
        v = parse_stat(need(R.path("perf", f"stat_{s}.txt")))
        for k in ("cycles", "instructions", "cache-references", "cache-misses"):
            if v.get(k) is None:
                unsupported.append(f"{s} {k}")
        ipc = (trunc(v["instructions"] / v["cycles"], 2)
               if v.get("instructions") and v.get("cycles") else "n/a")
        miss = (trunc(100 * v["cache-misses"] / v["cache-references"], 1) + "%"
                if v.get("cache-misses") and v.get("cache-references") else "n/a")
        rows.append([s, trunc(v["task-clock"], 0) if v.get("task-clock") else "n/a",
                     int(v["cycles"]) if v.get("cycles") else "n/a",
                     int(v["instructions"]) if v.get("instructions") else "n/a", ipc, miss])
    out.append("`perf stat` over the same runs (user space):\n\n"
               + table(["system", "task-clock ms", "cycles", "instructions", "IPC",
                        "cache misses / references"], rows) + "\n")
    out.append("Hardware counters: " + ("not exposed for " + ", ".join(unsupported)
                                        if unsupported else "exposed (user space)") + ".\n")
    out.append("```\n" + info + "\n```\n")
    return "\n".join(out)


@section
def sec_repeat(R):
    rep = load(need(R.path("repeat_t1.json")))
    base = {(r["system"], r["ef"]): r for r in R.sweep(1)["results"]}
    rows = []
    for r in sorted(rep["results"], key=lambda r: (r["system"], r["ef"])):
        b = base[(r["system"], r["ef"])]
        q1, q2 = pass_qps(b)[0], pass_qps(r)[0]
        rows.append([r["system"], r["ef"], qps(q1), qps(q2), ratio(q2 / q1),
                     r["ids_hash"] == b["ids_hash"]])
    return ("Headline single-thread points measured again at the end of the run "
            "(`repeat_t1.json`) against the main sweep (`sweep_t1.json`):\n\n"
            + table(["system", "ef", "sweep QPS", "repeat QPS", "repeat / sweep",
                     "same result ids"], rows) + "\n")


@section
def sec_spread(R):
    flagged = []

    def add(where, label, secs):
        s = {"median": statistics.median(secs), "min": min(secs), "max": max(secs)}
        if spread(s) > SPREAD_LIMIT:
            flagged.append([where, label, len(secs), trunc(100 * spread(s), 1) + "%"])

    for f in [f"sweep_t{t}.json" for t in R.sweep_threads()] + ["repeat_t1.json"]:
        if not os.path.exists(R.path(f)):
            continue
        for r in load(R.path(f))["results"]:
            add(f, f"{r['system']} ef {r['ef']} threads {r['threads']}", r["seconds"]["raw"])
    try:
        times, _, _, _ = ablation_data(R)
        for (c, ef), t in sorted(times.items()):
            add("ablation", f"{combo_name(c)} ef {ef}", t)
    except Missing:
        pass
    if os.path.exists(R.path("layout", "m2.json")):
        for r in load(R.path("layout", "m2.json"))["results"]:
            add("layout", f"M2 ef {r['ef']}", r["seconds"]["raw"])
    head = (f"Points whose timed passes spread by more than {trunc(100 * SPREAD_LIMIT, 0)}% "
            "((max - min) / median):")
    if not flagged:
        return head + " none.\n"
    return head + "\n\n" + table(["file", "point", "passes", "spread"], flagged) + "\n"


def stage_times(R):
    path = R.path("stage_times.tsv")
    if not os.path.exists(path):
        return "Not measured: `stage_times.tsv` does not exist.\n"
    rows = []
    with open(path) as f:
        for line in f:
            name, start, end, secs = line.rstrip("\n").split("\t")
            rows.append([name, start, end, secs])
    return ("Wall time per stage (`stage_times.tsv`; a resumed stage appears once per "
            "invocation):\n\n" + table(["stage", "start (UTC)", "end (UTC)", "seconds"], rows)
            + "\n")


# ------------------------------------------------------------ plots
def plot_recall_qps(R, out):
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    lo = 1.0
    for t in R.sweep_threads():
        doc = R.sweep(t)
        for s in ("ours", "hnswlib"):
            rows = rows_for(doc, s)
            x = [r["recall_at_k"] for r in rows]
            y = [pass_qps(r)[0] for r in rows]
            ls, mk = THREAD_STYLE[t]
            ax.plot(x, y, linestyle=ls, marker=mk, markersize=4, linewidth=2, color=COLOR[s],
                    label=f"{s}, {t} thread{'s' if t > 1 else ''}")
            lo = min(lo, min(x))
    lo = max(math.floor(lo * 100) / 100, 0.80)
    ax.set_xlim(lo, 1.0)
    ax.set_yscale("log")
    for target in TARGETS:
        ax.axvline(target, color="#b0b0a8", linewidth=1, zorder=0)
        ax.annotate(f"recall {target}", (target, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(-3, -3), textcoords="offset points", ha="right", va="top",
                    fontsize=7, color="#6b6b66")
    ax.set_xlabel("recall@10")
    ax.set_ylabel("queries per second (log scale)")
    ds = R.sweep(1)["dataset"]["name"]
    ax.set_title(f"Recall vs throughput on {ds} (M=16, efConstruction=200)", loc="left")
    ax.grid(True, which="major", color="#e4e4df", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_ablation(R, out):
    times, _, nq, efs = ablation_data(R)
    chain = [(0, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 1, 1, 0), (1, 1, 1, 1)]
    names = ["all off (scalar)", "+ epoch visited", "+ alloc-free heaps", "+ prefetch",
             "+ AVX2+FMA"]
    pts = efs["points"]
    fig, axes = plt.subplots(1, len(pts), figsize=(9, 3.6), dpi=150, sharey=True)
    for ax, p in zip(axes, pts):
        q = [abl_qps(times, nq, c, p["ef"])[0] for c in chain]
        y = list(range(len(chain)))[::-1]
        ax.barh(y, q, color=COLOR["ours"], height=0.6)
        for yi, qi in zip(y, q):
            ax.text(qi, yi, f" x{ratio(qi / q[0])}", va="center", fontsize=8, color="#3d3d3a")
        ax.set_yticks(y, names)
        ax.set_xlabel("QPS, single thread")
        ax.set_title(f"ef {p['ef']} (recall@10 >= {p['target']})", loc="left", fontsize=10)
        ax.set_xlim(0, max(q) * 1.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Cumulative ablation (speedup over all off, scalar kernel)", x=0.01,
                 ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------ main
README_BEGIN, README_END = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=os.path.join(ROOT, "bench", "results"))
    p.add_argument("--update-readme", action="store_true")
    p.add_argument("--readme", default=os.path.join(ROOT, "README.md"))
    a = p.parse_args()
    R = Results(os.path.abspath(a.results))

    secs = {
        "headline": sec_headline(R), "verdict": sec_verdict(R),
        "threads": sec_thread_consistency(R), "build": sec_build(R), "load": sec_load(R),
        "ablation": sec_ablation(R), "layout": sec_layout(R), "profile": sec_profile(R),
        "repeat": sec_repeat(R), "spread": sec_spread(R), "env": sec_environment(R),
        "stages": stage_times(R),
    }
    plots = []
    for name, fn in (("recall_vs_qps.png", plot_recall_qps), ("ablation.png", plot_ablation)):
        try:
            fn(R, R.path(name))
            plots.append(name)
        except Missing as e:
            print(f"make_report: {name} not drawn: {e} missing", file=sys.stderr)

    def body(level, img_prefix):
        h = "#" * level
        parts = [secs["verdict"], f"{h} Headline: QPS at the first ef reaching each recall "
                 "target\n", "Each system at its own first swept ef whose recall@10 reaches the "
                 "target (no interpolation).\n", secs["headline"], secs["threads"]]
        if "recall_vs_qps.png" in plots:
            parts.append(f"![recall vs QPS]({img_prefix}recall_vs_qps.png)\n")
        parts += [f"{h} Build, index size, memory\n", secs["build"],
                  f"{h} Load time\n", secs["load"],
                  f"{h} Ablation\n", secs["ablation"]]
        if "ablation.png" in plots:
            parts.append(f"![ablation]({img_prefix}ablation.png)\n")
        parts += [f"{h} Layout ablation\n", secs["layout"],
                  f"{h} Profile\n", secs["profile"],
                  f"{h} Reproducibility\n", secs["repeat"],
                  f"{h} High-spread points\n", secs["spread"]]
        return "\n".join(parts)

    summary = ("# Benchmark summary\n\nGenerated by `bench/make_report.py` from the files in "
               "this directory. Values are truncated, not rounded, to the digits shown.\n\n"
               "## Environment\n\n" + secs["env"] + "\n" + body(2, "") + "\n## Stage wall "
               "times\n\n" + secs["stages"])
    with open(R.path("summary.md"), "w") as f:
        f.write(summary)

    if a.update_readme:
        readme = a.readme
        with open(readme) as f:
            text = f.read()
        if text.count(README_BEGIN) != 1 or text.count(README_END) != 1:
            sys.exit("make_report: README.md needs exactly one RESULTS:BEGIN/END marker pair")
        rel = os.path.relpath(R.d, ROOT) + "/"
        block = ("<!-- Generated by bench/make_report.py --update-readme; do not edit. -->\n\n"
                 "Generated from [" + rel + "summary.md](" + rel + "summary.md), which also "
                 "lists the full environment. Values are truncated, not rounded, to the "
                 "digits shown.\n\n" + body(3, rel))
        pre, rest = text.split(README_BEGIN)
        _, post = rest.split(README_END)
        with open(readme, "w") as f:
            f.write(pre + README_BEGIN + "\n" + block + "\n" + README_END + post)


if __name__ == "__main__":
    main()
