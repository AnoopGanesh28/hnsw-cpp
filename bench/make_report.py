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
NOISE = "not distinguishable from noise (ranges overlap)"


def ranges_overlap(a, b):
    """a, b: (median, lo, hi) QPS triples; True if [lo, hi] intervals overlap."""
    return not (a[2] < b[1] or b[2] < a[1])


def mark(ratio_value, a, b):
    """'x1.234' when the ranges of a and b are disjoint, else 'x1.234, within noise'."""
    return f"x{ratio(ratio_value)}" + (", within noise" if ranges_overlap(a, b) else "")


@section
def sec_environment(R):
    env_doc = load(need(R.path("environment.json")))
    sweep = R.sweep(1)
    e = sweep["environment"]
    ds = sweep["dataset"]
    b = e["build"]
    base = load(need(R.path("build_ours.json")))["dataset"]["base"]
    commits = set()
    for f in glob.glob(R.path("*.json")) + glob.glob(R.path("*", "*.json")):
        env = load(f).get("environment")
        if isinstance(env, dict) and "source" in env:
            commits.add((env["source"]["commit"], env["source"]["dirty"]))
    m2_commit = load(need(R.path("layout", "m2.json")))["environment"]
    pm = b["predefined_macros"]
    h = e["hnswlib"]
    files = [("base", base), ("queries", ds["queries"]), ("ground truth", ds["ground_truth"])]
    lines = [
        f"- Dataset: `{ds['name']}`, {ds['index_vectors']} base vectors, "
        f"{sweep['config']['queries']} queries, recall@{sweep['config']['k']} against the "
        f"first {sweep['config']['k']} ground-truth ids. Files (bytes, sha256, checked "
        "against `bench/data/sift/MANIFEST` by every tool):",
    ] + [f"  - {name}: `{os.path.basename(r['path'])}`, {r['bytes']} bytes, `{r['sha256']}`"
         for name, r in files] + [
        f"- CPU: {e['cpu']['model']}, {e['cpu']['physical_cores']} cores / "
        f"{e['cpu']['logical_cpus']} logical CPUs, {e['cpu']['sockets']} socket; MemTotal "
        f"{e['mem_total_kib']} kB.",
        f"- Kernel: `{e['kernel']['release']}` (WSL detected: {e['wsl']['detected']}).",
        f"- Compiler: {e['compiler']['gxx_version_line']} (`{e['compiler']['path']}`).",
        f"- Flags: `{b['cxx_flags']}`; per file: "
        + ", ".join(f"`{k}`: `{v}`" for k, v in b["per_file_options"].items())
        + f"; sanitizers: {b['sanitize']}; release build: {b['release_build']}.",
        f"- AVX status: compiled with `__AVX2__` {pm['__AVX2__']}, `__FMA__` {pm['__FMA__']}, "
        f"`__AVX512F__` {pm['__AVX512F__']}; CPU supports AVX2+FMA at run time: "
        f"{e['ours']['cpu_supports_avx2_fma']}. Ours: `{e['ours']['l2_function']}`. hnswlib "
        f"header macros: " + ", ".join(f"`{k}` {v}" for k, v in h["header_macros"].items())
        + f"; L2 function `{h['l2_function']}`, AVX path active: {h['avx_path_active']}.",
        "- Source commit of every result file's binaries: "
        + "; ".join(f"`{c}` (dirty: {d})" for c, d in sorted(commits))
        + f" (`{e['source']['describe']}`).",
        f"- hnswlib: {h['tag_pinned']}, commit `{h['commit_at_build']}` (pinned "
        f"`{h['commit_pinned']}`).",
        f"- M2 layout baseline: commit `{m2_commit['m2_commit']}`, compiled with "
        f"`{m2_commit['cxx_flags']}`.",
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


def compare_word(o, h):
    """o, h: (median, lo, hi) QPS. Faster/slower only when the pass ranges are disjoint."""
    if ranges_overlap(o, h):
        return NOISE
    return "faster" if o[0] > h[0] else "slower"


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
            o, h = pts[("ours", t, target)], pts[("hnswlib", t, target)]
            if o is None or h is None:
                cmp_rows.append([t, f">= {target}", "not comparable (target not reached)", ""])
                continue
            oq, hq = pass_qps(o), pass_qps(h)
            cmp_rows.append([t, f">= {target}", ratio(oq[0] / hq[0]), compare_word(oq, hq)])
    out += ("\n\nA system is called faster or slower only when the two systems' pass-QPS "
            "ranges (slowest to fastest pass) do not overlap; otherwise the comparison is "
            "reported as not distinguishable from noise. The median ratio is shown either "
            "way.\n\n" + table(["threads", "recall@10 target", "ours QPS / hnswlib QPS (median)",
                                "ours is"], cmp_rows))
    return out + "\n"


@section
def sec_verdict(R):
    need(R.path("sweep_t1.json"))
    pts = headline_points(R)
    groups = []
    for t in R.sweep_threads():
        parts = []
        for target in TARGETS:
            o, h = pts[("ours", t, target)], pts[("hnswlib", t, target)]
            if o is None or h is None:
                parts.append(f"not comparable at recall@10 >= {target} (a system never "
                             "reaches it)")
                continue
            oq, hq = pass_qps(o), pass_qps(h)
            word = compare_word(oq, hq)
            what = (f"ours is {word} than hnswlib" if word != NOISE else
                    f"the difference between ours and hnswlib is {word}")
            parts.append(f"{what} at recall@10 >= {target} ({qps(oq[0])} vs {qps(hq[0])} "
                         f"median QPS, ratio {ratio(oq[0] / hq[0])})")
        label = "Single-threaded" if t == 1 else f"With {t} threads"
        groups.append(f"{label}, " + "; ".join(parts))
    return ("**Verdict** (faster or slower only where the pass-QPS ranges do not overlap): "
            + ". ".join(groups) + ".\n")


@section
def sec_noise(R):
    rep = load(need(R.path("repeat_t1.json")))
    base = {(r["system"], r["ef"]): r for r in R.sweep(1)["results"]}
    ratios = [pass_qps(r)[0] / pass_qps(base[(r["system"], r["ef"])])[0]
              for r in rep["results"]]
    flagged = high_spread(R)
    by_place = {}
    for where, label, _, _ in flagged:
        by_place[where] = by_place.get(where, 0) + 1
    low_ef = sum(1 for where, label, _, _ in flagged
                 if where.startswith("sweep") and int(label.split(" ef ")[1].split()[0]) <= 64)
    sweep_n = sum(n for w, n in by_place.items() if w.startswith("sweep"))
    return (f"**Noise.** Repeating the {len(ratios)} headline single-thread measurements at the "
            f"end of the run gave repeat / sweep median-QPS ratios from {ratio(min(ratios))} "
            f"to {ratio(max(ratios))}. {len(flagged)} measured points have a pass spread "
            f"above {trunc(100 * SPREAD_LIMIT, 0)}% ((max - min) / median): "
            + ", ".join(f"{n} in `{w}`" for w, n in by_place.items())
            + f"; {low_ef} of the {sweep_n} flagged sweep points are at ef <= 64, where a "
            "pass is shortest. They are listed under High-spread points.\n")


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
                     ", ".join(trunc(x, 1) for x in b["raw"]), trunc(b["median"], 1),
                     f"{trunc(b['min'], 1)} to {trunc(b['max'], 1)}",
                     d["index"]["bytes"], trunc(d["index"]["bytes"] / 2**20, 1)])
        m = d["memory"]
        mem.append([s, m["vm_hwm_end_kib"], m["vm_hwm_after_load_kib"],
                    m["vm_hwm_end_kib"] - m["vm_hwm_after_load_kib"]])
    return (table(["system", "vectors", "M", "efConstruction", "build s (each run, in order)",
                   "build s (median)", "build s (range)", "index bytes on disk", "index MiB"],
                  rows)
            + "\n\nPeak memory of the build process (VmHWM, kB); the dataset-only value is "
            "VmHWM right after loading the base vectors, and the delta is the peak above it:\n\n"
            + table(["system", "VmHWM end", "VmHWM after dataset load (dataset only)",
                     "delta"], mem) + "\n")


def ms(x):
    return trunc_sig(x * 1000)


@section
def sec_load(R):
    warm = load(need(R.path("loadtime_warm.json")))
    c = warm["config"]
    nfirst = c["first_queries"]
    rows = []
    for r in warm["results"]:
        ls, fq = r["load_seconds"], r["first_queries_seconds"]
        rows.append([r["strategy"], ms(ls["median"]), f"{ms(ls['min'])} to {ms(ls['max'])}",
                     ms(fq["median"]), f"{ms(fq['min'])} to {ms(fq['max'])}"])
    hdr = ["strategy", "load ms (median)", "load ms (range)",
           f"first {nfirst} queries ms (median)", f"first {nfirst} queries ms (range)"]
    out = (f"**Warm page cache** (each strategy loaded once untimed first), {c['reps']} "
           f"repetitions; first {nfirst} queries at ef {c['ef']}, single-threaded, fresh "
           "scratch:\n\n" + table(hdr, rows) + "\n")
    cold_path = R.path("loadtime_cold.json")
    if not os.path.exists(cold_path):
        return out + "\nCold page cache: not measured (needs sudo; `bench/run_cold_loadtime.sh`).\n"
    cold = load(cold_path)
    cc = cold["config"]
    rows = []
    for r in cold["results"]:
        ls, fq = r["load_seconds"], r["first_queries_seconds"]
        rows.append([r["strategy"], ms(ls["median"]), f"{ms(ls['min'])} to {ms(ls['max'])}",
                     ms(fq["median"]), f"{ms(fq['min'])} to {ms(fq['max'])}"])
    out += (f"\n**Cold page cache**: `{cc['pre_load_command']}` before every load, "
            f"{cc['reps']} repetitions, same query settings:\n\n" + table(hdr, rows) + "\n")
    w = {r["strategy"]: r for r in warm["results"]}
    rows = []
    for r in cold["results"]:
        s = r["strategy"]
        if s not in w:
            continue
        wt = w[s]["load_seconds"]["median"] + w[s]["first_queries_seconds"]["median"]
        ct = r["load_seconds"]["median"] + r["first_queries_seconds"]["median"]
        rows.append([s, ms(wt), ms(ct)])
    out += ("\n**Cold vs warm**, time until the first " f"{nfirst} queries are answered (median "
            "load + median first-queries time):\n\n"
            + table(["strategy", "warm ms", "cold ms"], rows)
            + "\n\nWSL2 caveat: dropping caches empties the Linux VM's page cache only. The "
            "VM's virtual disk is a file on the Windows host, which may still hold it in the "
            "Windows file cache, so cold here means cold for Linux, not necessarily a read "
            "from the physical drive. With mmap, the cold cost moves from the load into the "
            "first queries, which fault pages in on demand.\n")
    return out


# ------------------------------------------------------------ ablation
ABL_NAME = re.compile(r"abl_e(\d)h(\d)p(\d)_(scalar|avx2)_r(\d+)\.json$")
CHAIN = [(0, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 1, 1, 0), (1, 1, 1, 1)]


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
    """(median, lo, hi) QPS over the repetitions of combination c at ef."""
    t = times[(c, ef)]
    return nq / statistics.median(t), nq / max(t), nq / min(t)


@section
def sec_ablation(R):
    times, recall, nq, efs = ablation_data(R)
    pts = efs["points"]
    off, full = (0, 0, 0, 0), (1, 1, 1, 1)
    reps = len(times[(off, pts[0]["ef"])])
    hdr_pts = [f"ef {p['ef']} (recall >= {p['target']})" for p in pts]

    def q(c, ef):
        return abl_qps(times, nq, c, ef)

    def vs(c, base, ef):
        """Ratio of c over base, marked within noise when their ranges overlap."""
        return mark(q(c, ef)[0] / q(base, ef)[0], q(c, ef), q(base, ef))

    out = [f"Single-threaded, pinned, the same `ours.hnsw` for every variant; {reps} "
           "repetitions per combination, each its own process (1 warm-up + 1 timed pass), "
           "combinations interleaved round robin. QPS is queries / median timed pass. Every "
           "ratio is of medians; it is marked \"within noise\" when the QPS ranges (slowest to "
           "fastest repetition) of the two configurations overlap, and such a ratio is not "
           "a measured speedup or slowdown. Recall@10 per kernel: "
           + "; ".join(f"ef {p['ef']}: scalar {rec(recall[(0, p['ef'])])}, AVX2 "
                       f"{rec(recall[(1, p['ef'])])}" for p in pts) + ".\n"]

    rows = []
    for i, o in enumerate(OPTS):
        c = tuple(int(j == i) for j in range(4))
        rows.append([OPT_LABEL[o]] + [f"{qps(q(c, p['ef'])[0])} ({vs(c, off, p['ef'])})"
                                      for p in pts])
    out.append("**Each optimization alone** (QPS; ratio over all off with the scalar "
               "kernel):\n\n" + table(["optimization"] + hdr_pts, rows)
               + "\n\nBaseline, all off (scalar): "
               + ", ".join(f"ef {p['ef']}: {qps(q(off, p['ef'])[0])} QPS "
                           f"({qps(q(off, p['ef'])[1])} to {qps(q(off, p['ef'])[2])})"
                           for p in pts) + ".\n")

    names = ["all off (scalar)", "+ epoch visited list", "+ allocation-free heaps",
             "+ software prefetch", "+ AVX2+FMA kernel"]
    rows = []
    for k, c in enumerate(CHAIN):
        row = [names[k]]
        for p in pts:
            ef = p["ef"]
            cell = f"{qps(q(c, ef)[0])}"
            if k > 0:
                cell += f" (total {vs(c, off, ef)}; step {vs(c, CHAIN[k - 1], ef)})"
            row.append(cell)
        rows.append(row)
    out.append("**Cumulative** (QPS; total ratio over all off, and the step over the row "
               "above):\n\n" + table(["configuration"] + hdr_pts, rows) + "\n")

    rows = []
    for i, o in enumerate(OPTS):
        c = tuple(0 if j == i else 1 for j in range(4))
        rows.append([f"without {OPT_LABEL[o]}"]
                    + [f"{qps(q(c, p['ef'])[0])} (full / this: {vs(full, c, p['ef'])})"
                       for p in pts])
    out.append("**Leave one out** from the full configuration ("
               + ", ".join(f"ef {p['ef']}: {qps(q(full, p['ef'])[0])} QPS "
                           f"({qps(q(full, p['ef'])[1])} to {qps(q(full, p['ef'])[2])})"
                           for p in pts)
               + "):\n\n" + table(["configuration"] + hdr_pts, rows) + "\n")

    rows = []
    for c in sorted({k for k, _ in times}):
        rows.append([combo_name(c)] + [
            f"{qps(q(c, p['ef'])[0])} ({qps(q(c, p['ef'])[1])} to {qps(q(c, p['ef'])[2])})"
            for p in pts])
    out.append("All 16 combinations (QPS median, slowest to fastest repetition):\n\n"
               + table(["enabled"] + hdr_pts, rows) + "\n")
    chk = load(need(R.path("ablation_check.json")))
    out.append(f"Result-id check (`ablation_check.json`): identical ids for every same-kernel "
               f"combination: {chk['ok']}.\n")
    return "\n".join(out)


def secs4(x):
    return trunc(x, 4)


@section
def sec_layout(R):
    proj = load(need(R.path("layout", "projection.json")))
    m2 = load(need(R.path("layout", "m2.json")))
    chk = load(need(R.path("layout_check.json")))
    reps = m2["reps"]
    nq = m2["queries"]
    efs = [r["ef"] for r in m2["results"]]
    bracket = {}  # ef -> [A_0 .. A_reps] seconds
    for i in range(reps + 1):
        for r in load(need(R.path("layout", f"alloff_bracket_{i}.json")))["results"]:
            bracket.setdefault(r["ef"], []).append(r["seconds"]["raw"][0])
    m2_by_ef = {r["ef"]: r for r in m2["results"]}

    def trip(secs):
        return nq / statistics.median(secs), nq / max(secs), nq / min(secs)

    def median_note(ef):
        """Which samples are the medians of the before and after groups (odd group size)."""
        b, a = bracket[ef][:reps], bracket[ef][1:]
        bi = bracket[ef].index(statistics.median_low(b))
        ai = bracket[ef].index(statistics.median_low(a))
        if bi == ai:
            return (f"At ef {ef} both groups' median is the same sample, A_{bi}, so the two "
                    "values are identical.")
        same = qps(nq / bracket[ef][bi]) == qps(nq / bracket[ef][ai])
        return (f"At ef {ef} the medians are A_{bi} ({secs4(bracket[ef][bi])} s) and A_{ai} "
                f"({secs4(bracket[ef][ai])} s)"
                + (", close enough that both truncate to the same QPS." if same else "."))

    size = (f"{proj['layout_n']}-vector subset (ground truth recomputed for it)"
            if proj["subset"] else f"full base ({proj['layout_n']} vectors)")
    out = [f"Memory layout, measured on the {size}. The M2 build time was projected from a "
           f"{proj['probe_n']}-vector build ({trunc(proj['probe_build_seconds'], 1)} s) to "
           f"{trunc(proj['projected_full_build_seconds'], 1)} s for {proj['full_n']} vectors "
           f"(n log n; limit {trunc(proj['limit_seconds'], 1)} s). M2 baseline (pre-layout, "
           "scalar kernel) versus the all-off scalar variant (current layout) on an index with "
           "the same graph, both single-threaded and pinned.\n",
           f"How the samples were taken: one M2 process built the index once, then ran "
           f"{reps} repetitions M_0 .. M_{reps - 1} (each 1 warm-up + 1 timed pass per ef). "
           f"Before M_0 and after every M_r it ran the all-off variant once as a separate "
           f"process, giving {reps + 1} all-off samples A_0 .. A_{reps} in the order "
           f"A_0 M_0 A_1 M_1 ... M_{reps - 1} A_{reps}. M_r sits between A_r and A_(r+1). The "
           f"\"before\" group A_0 .. A_{reps - 1} and the \"after\" group A_1 .. A_{reps} "
           f"share {reps - 1} of their {reps} samples (A_1 .. A_{reps - 1}), so they are not "
           f"independent: they differ only in A_0 versus A_{reps}. "
           + " ".join(median_note(ef) for ef in efs)
           + f" The comparison below therefore uses all {reps + 1} all-off samples as one "
           "group, and lists both overlapping groups and every raw sample.\n"]
    rows = []
    for ef in efs:
        m = trip(m2_by_ef[ef]["seconds"]["raw"])
        a = trip(bracket[ef])
        word = ("within noise (ranges overlap)" if ranges_overlap(a, m) else
                "faster" if a[0] > m[0] else "slower")
        rows.append([ef, rec(m2_by_ef[ef]["recall_at_k"]), f"{qps(m[0])} ({qps(m[1])} to "
                     f"{qps(m[2])})", f"{qps(a[0])} ({qps(a[1])} to {qps(a[2])})",
                     f"x{ratio(a[0] / m[0])}", word])
    out.append(table(["ef", "recall@10", "M2 QPS median (range)",
                      f"all-off QPS median (range), all {reps + 1} samples",
                      "all-off / M2 (medians)", "current layout is"], rows) + "\n")
    rows = []
    for ef in efs:
        b, a = bracket[ef][:reps], bracket[ef][1:]
        rows.append([ef, qps(nq / statistics.median(b)), qps(nq / statistics.median(a)),
                     secs4(statistics.median(b)), secs4(statistics.median(a))])
    out.append("Overlapping groups (not independent; shown only to explain the identical "
               "values reported before):\n\n"
               + table(["ef", f"before A_0..A_{reps - 1} QPS", f"after A_1..A_{reps} QPS",
                        "before median s", "after median s"], rows) + "\n")
    rows = []
    for i in range(reps + 1):
        row = [f"A_{i}"] + [secs4(bracket[ef][i]) for ef in efs]
        rows.append(row)
        if i < reps:
            rows.append([f"M_{i}"] + [secs4(m2_by_ef[ef]["seconds"]["raw"][i]) for ef in efs])
    out.append("Every raw timed pass, in execution order (seconds for "
               f"{nq} queries):\n\n" + table(["run"] + [f"ef {ef} s" for ef in efs], rows)
               + "\n")
    out.append(f"Result-id check (`layout_check.json`): M2 ids equal the all-off variant's: "
               f"{chk['ok']}.\n")
    return "\n".join(out)


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


def high_spread(R):
    """[where, label, passes, spread%] for every measured point above SPREAD_LIMIT."""
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
    return flagged


@section
def sec_spread(R):
    flagged = high_spread(R)
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
    chain = CHAIN
    names = ["all off (scalar)", "+ epoch visited", "+ alloc-free heaps", "+ prefetch",
             "+ AVX2+FMA"]
    pts = efs["points"]
    fig, axes = plt.subplots(1, len(pts), figsize=(9, 3.6), dpi=150, sharey=True)
    for ax, p in zip(axes, pts):
        trips = [abl_qps(times, nq, c, p["ef"]) for c in chain]
        q = [t[0] for t in trips]
        y = list(range(len(chain)))[::-1]
        ax.barh(y, q, color=COLOR["ours"], height=0.6)
        for k, (yi, qi) in enumerate(zip(y, q)):
            label = " baseline" if k == 0 else " total " + mark(qi / q[0], trips[k], trips[0])
            ax.text(qi, yi, label, va="center", fontsize=8, color="#3d3d3a")
        ax.set_yticks(y, names)
        ax.set_xlabel("QPS, single thread")
        ax.set_title(f"ef {p['ef']} (recall@10 >= {p['target']})", loc="left", fontsize=10)
        ax.set_xlim(0, max(q) * 1.45)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Cumulative ablation (median QPS; ratio over all off, scalar kernel)", x=0.01,
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
        "noise": sec_noise(R),
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
        parts = [secs["verdict"], secs["noise"], f"{h} Headline: QPS at the first ef reaching each recall "
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
