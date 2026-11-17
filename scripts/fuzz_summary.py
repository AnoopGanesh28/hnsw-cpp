#!/usr/bin/env python3
"""Summarizes fuzz logs written by scripts/run_fuzz.sh as JSON.

Usage: scripts/fuzz_summary.py <log>... > docs/evidence/fuzz_summary.json

Every value is parsed from the logs. A log that lacks a field gives null, so a
missing number is visible rather than invented.
"""
import json
import re
import sys

STATS = re.compile(
    r"^#(\d+): cov: (\d+) ft: (\d+) corp: (\d+) exec/s: (\d+) "
    r"oom/timeout/crash: (\d+)/(\d+)/(\d+) time: (\d+)s job: (\d+)")


def first(pattern, text, cast=str):
    m = re.search(pattern, text, re.MULTILINE)
    return cast(m.group(1)) if m else None


def summarize(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    stats = [m.groups() for m in map(STATS.match, text.splitlines()) if m]
    last = [int(x) for x in stats[-1]] if stats else None
    merge = re.search(r"^MERGE-OUTER: (\d+) new files with (\d+) new features added; "
                      r"(\d+) new coverage edges", text, re.MULTILINE)
    corpus = re.search(r"^corpus after merge: (\d+) files, (\d+) bytes", text, re.MULTILINE)
    sanitizer_reports = len(re.findall(r"^==\d+==ERROR: |^SUMMARY: \w+Sanitizer", text,
                                       re.MULTILINE))
    return {
        "log": path,
        "target": first(r"^# scripts/run_fuzz\.sh (\S+)", text),
        "start": first(r"^start: (\S+)", text),
        "end": first(r"^end: (\S+)", text),
        "commit": first(r"^commit: (.+)$", text),
        "command": first(r"^command: (.+)$", text),
        "workers": first(r"-fork=(\d+)", text, int),
        "max_total_time_s": first(r"-max_total_time=(\d+)", text, int),
        "wall_time_s": first(r"^INFO: exiting: \d+ time: (\d+)s", text, int),
        "executions": last[0] if last else None,
        "coverage_edges": last[1] if last else None,
        "features": last[2] if last else None,
        "corpus_units_at_end": last[3] if last else None,
        "fork_jobs": last[9] if last else None,
        "ooms": last[5] if last else None,
        "timeouts": last[6] if last else None,
        "crashes": last[7] if last else None,
        "crash_artifacts": first(r"^crash artifacts: (\d+)", text, int),
        "sanitizer_report_lines": sanitizer_reports,
        "fuzzer_exit_status": first(r"^fuzzer exit status: (\d+)", text, int),
        "merge_new_files_beyond_seeds": int(merge.group(1)) if merge else None,
        "merge_new_features_beyond_seeds": int(merge.group(2)) if merge else None,
        "merge_new_edges_beyond_seeds": int(merge.group(3)) if merge else None,
        "committed_corpus_files": int(corpus.group(1)) if corpus else None,
        "committed_corpus_bytes": int(corpus.group(2)) if corpus else None,
    }


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    json.dump([summarize(p) for p in sys.argv[1:]], sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
