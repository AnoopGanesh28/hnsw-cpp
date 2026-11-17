#!/usr/bin/env python3
"""Builds the quality block of README.md from the files in docs/evidence/.

Usage:
  scripts/quality_report.py                 print the block
  scripts/quality_report.py --update-readme replace the block in README.md
  scripts/quality_report.py --check         exit 1 if README.md's block differs
                                            from what the evidence files give

Every figure is parsed from a saved file; nothing is typed by hand. A file that
is missing or has an unexpected format makes the script fail instead of
printing a guess.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "docs" / "evidence"
README = ROOT / "README.md"
BEGIN = "<!-- QUALITY:BEGIN -->"
END = "<!-- QUALITY:END -->"

COMPILERS = ["gcc", "clang"]
MODES = ["address", "undefined", "address+undefined", "thread"]

ENTRY = re.compile(r"Test\s+#(\d+): (\S+) \.+\s*(?:\*\*\*)?(\w+)")
SUMMARY = re.compile(r"(\d+)% tests passed, (\d+) tests failed out of (\d+)")


class EvidenceError(Exception):
    pass


def read(name):
    path = EVIDENCE / name
    if not path.is_file():
        raise EvidenceError(f"missing evidence file docs/evidence/{name}")
    return path.read_text(encoding="utf-8", errors="replace")


def need(match, what, name):
    if match is None:
        raise EvidenceError(f"docs/evidence/{name}: cannot find {what}")
    return match


def parse_ctest(name):
    """One ctest log: options, entries, outcome counts, commit."""
    text = read(name)
    entries = {}
    for m in ENTRY.finditer(text):
        entries[int(m.group(1))] = (m.group(2), m.group(3))
    s = need(SUMMARY.search(text), "the ctest summary line", name)
    options = need(re.search(r"^((?:CMAKE_BUILD_TYPE=\S+ )?HNSW_SANITIZE=.+)$", text, re.M),
                   "the options line", name).group(1)
    flags = need(re.search(r"^compile flags \(test_distance\): (.*)$", text, re.M),
                 "the compile flags line", name).group(1)
    opt = re.search(r"(?<!\S)-O\d", flags)
    debug = re.search(r"(?<!\S)-g(?!\S)", flags) is not None
    status = need(re.search(r"^ctest exit status: (\d+)", text, re.M), "the exit status", name)
    total = int(s.group(3))
    if len(entries) != total:
        raise EvidenceError(f"docs/evidence/{name}: {len(entries)} entries parsed, summary says {total}")
    passed = sum(1 for _, st in entries.values() if st == "Passed")
    return {
        "text": text,
        "entries": entries,
        "names": {n for n, _ in entries.values()},
        "total": total,
        "passed": passed,
        "failed": sum(1 for _, st in entries.values() if st == "Failed"),
        "skipped": sum(1 for _, st in entries.values() if st == "Skipped"),
        "exit_status": int(status.group(1)),
        "options": dict(re.findall(r"(\w+)=(\S+)", options)),
        "opt_level": (opt.group(0) if opt else "no -O") + (" -g" if debug else ""),
        "commit": need(re.search(r"^commit: (\S+)", text, re.M), "the commit", name).group(1),
    }


def check_counts(log):
    """Sum of 'N checks, M failed' lines, attributed to ctest entries (verbose logs only)."""
    by_entry = {}
    for m in re.finditer(r"^(\d+): (\d+) checks, (\d+) failed$", log["text"], re.M):
        by_entry.setdefault(int(m.group(1)), []).append((int(m.group(2)), int(m.group(3))))
    if any(len(v) != 1 for v in by_entry.values()):
        return None  # an entry printed several counts: not reliably attributable
    if any(n not in log["entries"] for n in by_entry):
        return None
    return {
        "checks": sum(v[0][0] for v in by_entry.values()),
        "failed": sum(v[0][1] for v in by_entry.values()),
        "entries": len(by_entry),
    }


def python_unittest(log):
    """The unittest summary printed by the python_tests entry of a verbose log."""
    num = next((n for n, (name, _) in log["entries"].items() if name == "python_tests"), None)
    if num is None:
        return None
    body = [m for m in re.findall(rf"^{num}: (.*)$", log["text"], re.M)]
    ran = next((re.match(r"Ran (\d+) tests", b) for b in body if b.startswith("Ran ")), None)
    ok = any(b.strip() == "OK" for b in body)
    return {"ran": int(ran.group(1)), "ok": ok} if ran else None


def python_asan(compiler):
    name = f"python_asan_{compiler}.txt"
    text = read(name)
    ran = need(re.search(r"^Ran (\d+) tests", text, re.M), "the unittest summary", name)
    status = need(re.search(r"^exit status: (\d+)", text, re.M), "the exit status", name)
    return {"ran": int(ran.group(1)), "ok": bool(re.search(r"^OK$", text, re.M)),
            "status": int(status.group(1))}


def fmt(n):
    return f"{n:,}"


def tests_section():
    out = ["#### Tests (ctest)", ""]
    canonical = {c: parse_ctest(f"ctest_{c}_release.txt") for c in COMPILERS}
    cells = {(c, m): parse_ctest(f"ctest_{c}_{m}.txt") for c in COMPILERS for m in MODES}

    out += ["| Configuration | Build | Options | ctest entries | Passed | Failed | Skipped |",
            "|---|---|---|---:|---:|---:|---:|"]
    for c in COMPILERS:
        l = canonical[c]
        o = l["options"]
        out.append(f"| {c}, canonical full | {o['CMAKE_BUILD_TYPE']} {l['opt_level']} | "
                   f"BENCH={o['HNSW_BUILD_BENCH']}, PYTHON={o['HNSW_BUILD_PYTHON']}, "
                   f"FUZZ={o['HNSW_BUILD_FUZZ']}, SANITIZE={o['HNSW_SANITIZE']} | "
                   f"{l['total']} | {l['passed']} | {l['failed']} | {l['skipped']} |")
    for (c, m), l in cells.items():
        o = l["options"]
        out.append(f"| {c}, {m} | {l['opt_level']} | SANITIZE={o['HNSW_SANITIZE']}, "
                   f"BENCH={o['HNSW_BUILD_BENCH']}, PYTHON={o['HNSW_BUILD_PYTHON']} | "
                   f"{l['total']} | {l['passed']} | {l['failed']} | {l['skipped']} |")
    out.append("")

    # Why the counts differ, derived from the entry names in the logs.
    canon_sets = {frozenset(l["names"]) for l in canonical.values()}
    cell_sets = {frozenset(l["names"]) for l in cells.values()}
    if len(canon_sets) == 1 and len(cell_sets) == 1:
        canon, cell = next(iter(canon_sets)), next(iter(cell_sets))
        extra, missing = sorted(canon - cell), sorted(cell - canon)
        if not missing:
            n_canon, n_cell = len(canon), len(cell)
            py_built = {l["options"]["HNSW_BUILD_PYTHON"] for l in cells.values()}
            out.append(
                f"The canonical runs have {n_canon} ctest entries and every sanitizer cell {n_cell}. "
                f"The {n_cell} entries in the sanitizer cells are the same entries as in the canonical "
                f"runs; the {len(extra)} that exist only in the canonical runs are "
                f"{', '.join(f'`{e}`' for e in extra)}. The sanitizer cells were configured with "
                f"HNSW_BUILD_PYTHON={'/'.join(sorted(py_built))}, so the extension was built there, "
                f"but its tests are not registered with ctest in a sanitizer build (the interpreter is "
                f"not instrumented). They are run separately under ASan, below.")
        else:
            out.append("The ctest entry names differ between the canonical runs and the sanitizer "
                       f"cells in both directions (only in the sanitizer cells: {', '.join(missing)}).")
    else:
        out.append("The entry names are not identical across all canonical runs or across all "
                   "sanitizer cells; see the per-log counts above.")
    out.append("")

    out += ["| Canonical run | Checks (C++ test assertions) | Failed | Entries reporting a count | Python unittests |",
            "|---|---:|---:|---:|---|"]
    for c in COMPILERS:
        cc = check_counts(canonical[c])
        pu = python_unittest(canonical[c])
        if cc is None:
            raise EvidenceError(f"docs/evidence/ctest_{c}_release.txt: check counts are not parseable")
        py = f"{pu['ran']} ran, {'OK' if pu['ok'] else 'FAILED'}" if pu else "not found"
        out.append(f"| {c} | {fmt(cc['checks'])} | {cc['failed']} | {cc['entries']} of "
                   f"{canonical[c]['total']} | {py} |")
    out += ["",
            "Checks are the `N checks, M failed` lines the C++ tests print; entries that do not use "
            "that helper (the Python tests, the quickstart, the replay tests, the benchmark tests) "
            "report no count and are not in the sum.", ""]
    return out, canonical, cells


def sanitizer_section(cells):
    out = ["#### Sanitizer matrix", "",
           "Full ctest suite per cell (passed/total); UBSan aborts on any finding, ASan runs with "
           "leak detection.", "",
           "| Compiler | " + " | ".join(MODES) + " |", "|---|" + "---:|" * len(MODES)]
    for c in COMPILERS:
        row = [f"{cells[(c, m)]['passed']}/{cells[(c, m)]['total']}" for m in MODES]
        out.append(f"| {c} | " + " | ".join(row) + " |")
    out += ["", "Python tests on the ASan-built extension (interpreter not instrumented, runtime preloaded):", "",
            "| Compiler | unittests | Result |", "|---|---:|---|"]
    for c in COMPILERS:
        p = python_asan(c)
        out.append(f"| {c} | {p['ran']} | {'OK' if p['ok'] and p['status'] == 0 else 'FAILED'} |")
    out.append("")
    return out


def fuzz_section():
    summary = json.loads(read("fuzz_summary.json"))
    out = ["#### Fuzzing (libFuzzer, clang, ASan+UBSan)", "",
           "| Target | Wall time | Workers | Executions | Crashes | OOMs | Timeouts | Edges | Features | "
           "Committed corpus |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for s in summary:
        out.append(f"| fuzz_{s['target']} | {s['wall_time_s']} s | {s['workers']} | {fmt(s['executions'])} | "
                   f"{s['crashes']} | {s['ooms']} | {s['timeouts']} | {s['coverage_edges']} | "
                   f"{fmt(s['features'])} | {s['committed_corpus_files']} files, "
                   f"{fmt(s['committed_corpus_bytes'])} bytes |")
    out.append("")

    loader = next(s for s in summary if s["target"] == "loader")
    log = read(Path(loader["log"]).name)
    pts = [tuple(map(int, m)) for m in re.findall(
        r"^#(\d+): cov: (\d+) ft: (\d+) corp: \d+ exec/s: \d+ oom/timeout/crash: \d+/\d+/\d+ time: (\d+)s",
        log, re.M)]
    if not pts:
        raise EvidenceError("docs/evidence/fuzz_loader.log: no statistics lines")
    final_cov, final_ft = pts[-1][1], pts[-1][2]
    first_final = next(t for _, c, _, t in pts if c == final_cov)
    threshold = -(-final_cov * 99 // 100)  # ceil(99% of the final edge count)
    first_99 = next(t for _, c, _, t in pts if c >= threshold)
    sample = [p for p in pts if p[3] <= 60][-1]
    out.append(f"Loader edge coverage plateaued early: 99% of the final {final_cov} edges "
               f"(at least {threshold}) was reached by {first_99} s, and the final value was first seen at "
               f"{first_final} s of {loader['wall_time_s']} s. Features kept growing: {fmt(sample[2])} at the "
               f"{sample[3]} s sample, {fmt(final_ft)} at the end. Read from the statistics lines of the log.")
    out.append("")

    eff = read("fuzz_effectiveness.log")
    removed = [l[1:].strip() for l in eff.splitlines() if l.startswith("-") and not l.startswith("---")]
    san = need(re.search(r"^SUMMARY: (\w+): ([^(]+?)\s*(?:\(| in |$)", eff, re.M), "a sanitizer summary",
               "fuzz_effectiveness.log")
    wall = need(re.search(r"^wall seconds until exit: (\d+)", eff, re.M), "the wall time",
                "fuzz_effectiveness.log").group(1)
    restored = "identical" in eff and "restored:" in eff
    out.append(f"Effectiveness check: with this line removed from deep validation (`{removed[0]}`), "
               f"fuzz_loader started from the seeds alone stopped on {san.group(1)} "
               f"`{san.group(2).strip()}` at a recorded wall time of {wall} s (whole seconds; limit 300 s). "
               f"The code was {'restored and verified with cmp' if restored else 'NOT confirmed restored'}. "
               f"This was an easy bug to find, so it shows the setup works, not that it finds subtle ones.")
    out.append("")
    return out


def ci_section():
    ci = json.loads(read("ci_latest.json"))
    out = ["#### Continuous integration", "",
           f"Workflow `{ci['workflowName']}`, run [{ci['databaseId']}]({ci['url']}) on commit "
           f"`{ci['headSha'][:7]}` (\"{ci['displayTitle']}\"), event `{ci['event']}`: "
           f"**{ci['conclusion']}**. Later commits are not covered by this record.", "",
           "| Job | Conclusion |", "|---|---|"]
    for j in sorted(ci["jobs"], key=lambda j: j["name"]):
        out.append(f"| {j['name']} | {j['conclusion']} |")
    out.append("")
    return out


def build():
    lines, _, cells = tests_section()
    lines += sanitizer_section(cells) + fuzz_section() + ci_section()
    head = ["_Generated by `scripts/quality_report.py` from `docs/evidence/`; do not edit by hand. "
            "`scripts/quality_report.py --check` fails if this block disagrees with those files._", ""]
    return "\n".join(head + lines).rstrip() + "\n"


def block_bounds(readme):
    b, e = readme.find(BEGIN), readme.find(END)
    if b < 0 or e < 0 or e < b:
        raise EvidenceError(f"README.md has no {BEGIN} ... {END} block")
    return b + len(BEGIN), e


def main():
    args = sys.argv[1:]
    if len(args) > 1 or (args and args[0] not in ("--update-readme", "--check")):
        sys.exit(__doc__)
    try:
        block = "\n" + build()
        if not args:
            sys.stdout.write(block)
            return 0
        readme = README.read_text(encoding="utf-8")
        start, end = block_bounds(readme)
        if args[0] == "--update-readme":
            README.write_text(readme[:start] + block + readme[end:], encoding="utf-8")
            print("README.md quality block updated")
            return 0
        if readme[start:end] != block:
            print("quality_report: the README quality block disagrees with docs/evidence; "
                  "run scripts/quality_report.py --update-readme", file=sys.stderr)
            return 1
        print("quality_report: ok (README block matches docs/evidence)")
        return 0
    except EvidenceError as e:
        print(f"quality_report: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
