# Sub-Agent Orchestration Architecture

Specialized sub-agent architecture for parallelizing investigation, analysis, and implementation tasks.

## When to Use Sub-Agents

**Use sub-agents when**:
- Multi-file analysis required (3+ files)
- Multiple independent investigations can run in parallel
- Complex root cause analysis needed
- Expected gains: 40-80% speedup, 43% context reduction per agent

**Use sequential when**:
- Single file, single bug, known location
- Quick iteration on obvious issues

## Available Agent Types

**Investigation (Blue)** - Read-only, parallel:
- `test-analyzer` - Parse test files, identify failures
- `source-inspector` - Read source code, trace implementations
- `matlab-reference` - Search reference implementations
- `fixture-comparator` - Parse test data fixtures

**Analysis (Green)** - Synthesize reports:
- `differential-analyzer` - Compare implementations, identify differences
- `root-cause-investigator` - Form hypotheses from investigation reports

**Implementation (Red)** - Apply changes:
- `code-patcher` - Apply targeted code fixes

**Existing**:
- `rust-octave-tracer` (Yellow) - Static verification
- `octave-rust-debugger` (Orange) - Runtime debugging

## 4-Phase Workflow

**Phase 1: Parallel Investigation**
```
Launch 4 investigation agents in parallel (single message, 4 Task calls):
├─ test-analyzer → What's failing
├─ source-inspector → How code works
├─ reference-checker → Expected behavior
└─ data-comparator → Expected vs actual
```

**Phase 2: Analysis & Decision**
```
Synthesize reports:
├─ High confidence → Phase 3 (implement)
├─ Conflicting → Launch differential-analyzer
└─ Unclear → Launch root-cause-investigator
```

**Phase 3: Parallel Implementation**
```
Launch code-patcher agents in parallel (one per fix):
├─ Fix #1
├─ Fix #2
└─ Fix #N
```

**Phase 4: Verification**
```
Run tests:
├─ PASS → Done
└─ FAIL → Phase 1 (refined focus)
```

## Invocation

**Syntax**: `Task` tool with `subagent_type` = agent name (without .md)

**Parallel launch**: Single message with multiple `Task` calls

**Prompt guidelines**:
- Specify exact file paths
- Request specific information
- Define expected report format

## Orchestrator Decision Rules

**After Phase 1**:
- High confidence → Phase 3
- Conflicting reports → differential-analyzer
- Unclear → root-cause-investigator

**After Phase 3**:
- Always verify (Phase 4)

**After Phase 4**:
- Pass → Done
- Fail → Phase 1 (refined)
- Regression → Rollback

## Resource Limits

- Max 10 agents in parallel
- Agent timeout: 60s
- Max 3 iterations per bug
- Priority: Verification > Implementation > Analysis > Investigation

## Best Practices

**Agent usage**:
- Investigation: Parallel (independent)
- Analysis: Sequential (synthesis needed)
- Implementation: Parallel (independent changes)

**Context management**:
- Each agent: <500 lines
- Orchestrator: Summaries only
- Reports: Structured format

**Iteration**:
- First: Wide net (all investigation agents)
- Subsequent: Targeted (relevant agents only)
