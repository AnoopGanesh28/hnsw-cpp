# Project: from-scratch HNSW (C++17)

Full spec is in docs/TASK.md. Follow it exactly.

## Rules (non-negotiable)
- Never invent, estimate, or round up a number. Every README figure comes from a script in bench/ whose raw output is saved in bench/results/.
- Each milestone ends with passing tests and its own commit ("feat: ...", "test: ...", "bench: ..."). Never move on with failing tests.
- No datasets or large files in git. No dead code, no debug prints.
- If something is blocked (e.g. no network), note it in README under "Status", continue with what can proceed, and tell me.
- Work on ONE milestone at a time. Stop after finishing it and report: what was done, test output, commit hash, anything I should know.
- Don't start the next milestone until I say so.

## Conventions
- C++17, CMake, -O3 -march=native for release builds, warnings on (-Wall -Wextra).
- Library in src/ + include/, tests in tests/, benchmarks in bench/.
- Use ctest for tests. Use conventional commit prefixes.
