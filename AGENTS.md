# Project Continuity

Before making changes in a new session:

1. Read `AGENTS.md` completely.
2. Read `docs/PROJECT_STATE.md` completely.
3. Inspect Git status. If `.git` is absent, report that limitation instead of inventing a clean or dirty status.
4. Treat actual code, tests, and outputs as the final source of truth.
5. Treat `docs/PROJECT_STATE.md` as the persistent project handoff.
6. Do not repeat already verified expensive experiments without a concrete reason.
7. Do not modify `FROZEN` components without explicit new evidence or user request.
8. Do not introduce testcase-specific hacks or task-ID branches.
9. Prefer general fixes with regression tests.
10. Clearly distinguish `PASS`, `FAIL`, `LIMITATION`, `NOT TESTED`, `PLANNED`, and `FROZEN`.
11. The presence of legacy World Model, collector, MPPI, V-JEPA, Dreamer, or TD-MPC2 code does not mean it is integrated with or validated on the current Helsinki navigation stack.

Before ending a substantial session:

1. Update `docs/PROJECT_STATE.md`.
2. Record only work actually completed.
3. Record exact validation results.
4. Record important output paths.
5. Record new limitations.
6. Record architecture decisions.
7. Record the next milestone.
8. Make the file usable by a fresh Codex session with no chat history.

