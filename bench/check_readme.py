"""Checks that the README's benchmark numbers are traceable to the report.

  python3 bench/check_readme.py [--readme README.md] [--summary bench/results/summary.md]

Fails (exit 1) if
  - any number in the generated block between <!-- RESULTS:BEGIN --> and
    <!-- RESULTS:END --> does not appear as a number in summary.md, or
  - the hand-written rest of the "## Results" section contains a number
    (digits inside names such as SIFT1M, AVX2 or WSL2 are not numbers):
    hand-written text there must carry no numbers of its own.
"""

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEGIN, END = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"
# A number not embedded in an identifier or hash (an "x" speedup prefix is allowed).
NUMBER = re.compile(r"(?<![A-Za-z0-9_.])x?(\d+(?:\.\d+)?)(?![A-Za-z0-9_])")


def numbers(text):
    return set(NUMBER.findall(text))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--readme", default=os.path.join(ROOT, "README.md"))
    p.add_argument("--summary", default=os.path.join(ROOT, "bench", "results", "summary.md"))
    a = p.parse_args()
    with open(a.readme) as f:
        readme = f.read()
    with open(a.summary) as f:
        summary = f.read()
    if readme.count(BEGIN) != 1 or readme.count(END) != 1:
        sys.exit("check_readme: README needs exactly one RESULTS:BEGIN/END marker pair")
    pre, rest = readme.split(BEGIN)
    block, post = rest.split(END)
    problems = []

    missing = sorted(numbers(block) - numbers(summary), key=lambda s: float(s))
    if missing:
        problems.append("numbers in the generated block not found in summary.md: "
                        + ", ".join(missing))

    # Hand-written part of the Results section: from its heading to the next
    # level-2 heading, minus the generated block.
    m = re.search(r"^## Results\b.*$", pre, re.M)
    if m is None:
        problems.append("README has no '## Results' section before the generated block")
    else:
        nxt = re.search(r"^## ", post, re.M)
        hand = pre[m.end():] + (post[:nxt.start()] if nxt else post)
        for line in hand.splitlines():
            if numbers(line):
                problems.append(f"hand-written Results text has a number: {line.strip()!r}")

    if problems:
        for pr in problems:
            print("check_readme: " + pr, file=sys.stderr)
        sys.exit(1)
    print(f"check_readme: ok ({len(numbers(block))} distinct numbers in the generated block, "
          "all found in summary.md; no numbers in the hand-written Results text)")


if __name__ == "__main__":
    main()
