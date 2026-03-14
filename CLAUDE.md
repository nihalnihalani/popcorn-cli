# CLAUDE.md

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, architecture, and admin commands.

## Quick Reference

```bash
cargo build                                                  # Build
cargo test                                                   # Run tests
cargo fmt --all                                              # Format (required before commits)
cargo clippy --all-targets --all-features -- -D warnings     # Lint (must pass with no warnings)
```

## Auto-Commit and Push Rule

**MANDATORY**: After every change you make to any file in this repository, you MUST:

1. Stage the changed files: `git add <specific files you changed>`
2. Commit with a clear message describing what changed: `git commit -m "description of change"`
3. Push to `nihal`: `git push origin nihal`

This applies to EVERY change — no exceptions. Do not batch changes. Commit and push immediately after each logical change.

- Always push to `nihal`
- Never force push
- Use descriptive commit messages that explain the "why"
- If a pre-commit hook fails, fix the issue and create a NEW commit (never amend)

## Agent Team Strategy

Use agent teams for any task that benefits from parallel work across independent modules. Teams are enabled via `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` in settings.

### When to Use Teams
- Multi-file features spanning popcorn-cli (Rust) and helion-main (Python)
- Research + implementation in parallel (one teammate explores, another builds)
- Code review with competing perspectives (security, performance, correctness)
- Debugging with competing hypotheses — teammates test different theories simultaneously
- Any task with 3+ independent subtasks that don't touch the same files

### When NOT to Use Teams
- Sequential tasks with heavy dependencies between steps
- Changes to a single file or tightly coupled files
- Simple bug fixes or small tweaks
- Tasks where coordination overhead exceeds the benefit

### Team Configuration
- Start with **3-5 teammates** for most workflows
- Aim for **5-6 tasks per teammate** to keep everyone productive
- Use **Opus for the lead** (reasoning/coordination), **Opus for teammates** (focused implementation)
- Use **delegate mode** (`Shift+Tab`) when the lead should only coordinate, not write code

### Independent Modules

Each teammate should own a separate module to avoid file conflicts:

| Module | Directory | Notes |
|--------|-----------|-------|
| CLI Core | `src/cmd/` | Rust CLI commands (auth, submit, admin, setup) |
| Models | `src/models/` | Data models and types |
| Services | `src/service/` | Business logic layer |
| Views | `src/views/` | Terminal UI (loading, results) |
| Utils | `src/utils/` | Shared utilities |
| Helion Core | `helion-main/helion/` | Python DSL, compiler, runtime |
| Helion Tests | `helion-main/test/` | PyTest test suite |
| Helion Examples | `helion-main/examples/` | Runnable example scripts |
| Docs | `docs/` | Documentation |

### Team Communication Rules
- Use `SendMessage` (type: "message") for direct teammate communication — always refer to teammates by **name**
- Use `SendMessage` (type: "broadcast") **only** for critical blockers affecting everyone
- Use `TaskCreate`/`TaskUpdate`/`TaskList` for work coordination — teammates self-claim unblocked tasks
- When a teammate finishes, they check `TaskList` for the next available task (prefer lowest ID first)
- Mark tasks `completed` only after verification passes

### Task Dependencies
- Use `addBlockedBy` to express task ordering (e.g., "models depend on types being done")
- Teammates skip blocked tasks and pick up unblocked work
- When a blocking task completes, dependent tasks auto-unblock

### Plan Approval for Risky Work
- For architectural changes or risky refactors, require **plan approval** before implementation
- The teammate works in read-only mode, submits a plan, lead approves/rejects
- Only after approval does the teammate implement

### Team Quality Hooks
- `TaskCompleted` hook: prevents marking tasks done unless tests pass
- `TeammateIdle` hook: auto-assigns follow-up work to idle teammates
- Every teammate must run verification before reporting completion

### Shutdown Protocol
- When all tasks are complete, the lead sends `shutdown_request` to each teammate
- Teammates approve shutdown after confirming their work is committed
- Lead calls `TeamDelete` to clean up team resources

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Run `cargo test` and `cargo clippy` for Rust changes
- Run `pytest` for Helion Python changes
- Run `./lint.sh` in helion-main before claiming Python work is done

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- Skip this for simple, obvious fixes — don't over-engineer

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them

## Project Context

- **Popcorn CLI**: Rust (Cargo), CLI tool for hackathon submissions
  - Entry: `src/main.rs`
  - Commands: `src/cmd/` (auth, submit, submissions, admin, setup)
  - Build: `cargo build` | Test: `cargo test` | Lint: `cargo clippy`
- **Helion**: Python 3.10+, GPU kernel DSL/compiler
  - Entry: `helion-main/helion/`
  - Test: `pytest` | Lint: `./lint.sh`
  - Env vars: `HELION_LOGS`, `HELION_PRINT_OUTPUT_CODE`, `HELION_USE_DEFAULT_CONFIG`

## Verification Standards

Before marking any task complete, run:
- `cargo fmt --all --check` — Rust formatting
- `cargo clippy --all-targets --all-features -- -D warnings` — Rust lint
- `cargo test` — Rust tests pass
- For Helion: `./lint.sh` and `pytest <relevant test file>`

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Minimal code impact.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
