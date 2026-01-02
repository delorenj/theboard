# Meeting CLI Improvements

## Summary

Three critical UX improvements to the meeting workflow:

1. **Fixed premature meeting termination** - Meetings now require minimum rounds before convergence
2. **Added interactive TUI** - No more copy-pasting meeting UUIDs
3. **Document artifacts** - Full meeting details in markdown instead of truncated tables

---

## Issue 1: Premature Meeting Termination

**Problem**: Meetings ended after just one round due to aggressive convergence detection.

**Root Cause**: Convergence check (`avg_novelty < 0.3`) ran immediately after round 1 without minimum round requirement.

**Solution**: Added `min_rounds` parameter (default: 2) to `MultiAgentMeetingWorkflow`:

```python
# src/theboard/workflows/multi_agent_meeting.py:49-74
def __init__(
    self,
    meeting_id: UUID,
    model_override: str | None = None,
    novelty_threshold: float = 0.3,
    min_rounds: int = 2,  # NEW: minimum rounds before convergence
    enable_compression: bool = True,
    compression_threshold: int = 10000,
) -> None:
    # ...
    self.min_rounds = min_rounds
```

Convergence logic now checks minimum rounds first:

```python
# src/theboard/workflows/multi_agent_meeting.py:227-261
if round_num >= self.min_rounds and avg_novelty < self.novelty_threshold:
    # Convergence detected
elif round_num < self.min_rounds:
    # Skip convergence check (log debug message)
```

**Impact**: Meetings will always run at least 2 rounds (configurable), preventing false convergence.

---

## Issue 2: Meeting ID Friction

**Problem**: Users had to manually copy/paste UUIDs to run or view meetings.

**Old Workflow**:
```bash
board create -t "My topic"
# Meeting created: a3f2e1d4-...
board run a3f2e1d4-...        # Manual UUID copy
board status a3f2e1d4-...     # Manual UUID copy again
```

**Solution**: Added interactive TUI selector to `status` command (matching existing `run` command pattern):

```python
# src/theboard/cli.py:363-449
@app.command()
def status(
    meeting_id: str | None = typer.Argument(None, ...),  # Now optional
    last: bool = typer.Option(False, "--last", ...),      # NEW: --last flag
) -> None:
    if meeting_id is None:
        if last:
            # Use most recent meeting
        else:
            # Show interactive selector (numbered list)
```

**New Workflow**:
```bash
board create -t "My topic"
board run --last           # Auto-runs most recent
board status --last        # Auto-shows most recent

# OR with interactive selection:
board run                  # Shows numbered list, select with 1-20
board status               # Shows numbered list, select with 1-20
```

**Impact**: Zero UUID management required. Both `run` and `status` commands now support:
- No arguments → interactive selector
- `--last` → use most recent meeting
- UUID argument → direct access (still supported)

---

## Issue 3: Truncated Table Display

**Problem**: `board status` showed truncated comments (100 chars max) in a table format, making it useless for review.

**Old Output**:
```
┌─────┬───────────┬──────────┬──────────────────────────────┐
│ ... │ Agent     │ Category │ Comment                      │
├─────┼───────────┼──────────┼──────────────────────────────┤
│ 1   │ Architect │ concern  │ This approach might introdu... │
└─────┴───────────┴──────────┴──────────────────────────────┘
```

**Solution**: Auto-generate full markdown artifact when viewing status:

```python
# src/theboard/cli.py:479-536
if show_comments:
    # Generate markdown artifact
    artifacts_dir = Path("artifacts/meetings")
    artifact_path = artifacts_dir / f"meeting_{timestamp}_{safe_topic}_{uuid_id}.md"

    export_service.export_markdown(uuid_id, artifact_path)

    # Display artifact location instead of truncated table
    console.print(Panel.fit(
        f"Meeting artifact generated\n"
        f"Path: {artifact_path}\n"
        f"Comments: {total_comments}\n"
        f"Rounds: {current_round}",
        title="Meeting Document"
    ))

    console.print(f"View full details: cat {artifact_path}")
```

**New Output**:
```
┌─────────────────────────────────────────────┐
│ Meeting Document                            │
├─────────────────────────────────────────────┤
│ ✓ Meeting artifact generated                │
│                                             │
│ Path: artifacts/meetings/meeting_....md    │
│ Comments: 47                                │
│ Rounds: 3                                   │
└─────────────────────────────────────────────┘

View full details: cat artifacts/meetings/meeting_....md
Or open in editor: $EDITOR artifacts/meetings/meeting_....md
```

The markdown artifact includes:
- Full meeting metadata (topic, strategy, status, cost)
- Complete agent responses (no truncation)
- All comments grouped by round and agent
- Convergence metrics

**Impact**:
- Full meeting details accessible in structured markdown
- Can be viewed in terminal (`cat`), editor (`$EDITOR`), or version controlled
- Already auto-generated on meeting completion (stored in `logs/meetings/`)
- Artifacts stored in `artifacts/meetings/` (gitignored)

---

## Files Modified

### /home/delorenj/code/theboard/src/theboard/workflows/multi_agent_meeting.py
- Added `min_rounds` parameter (line 54, default: 2)
- Updated convergence check to respect minimum rounds (lines 228-261)
- Added debug logging for skipped convergence checks

### /home/delorenj/code/theboard/src/theboard/cli.py
- Made `status` meeting_id argument optional (line 364)
- Added `--last` flag to `status` command (line 371-373)
- Added interactive TUI selector to `status` (lines 389-439)
- Replaced truncated table with artifact generation (lines 479-536)
- Updated help text in `create` and `run` commands to reference new TUI pattern

---

## Testing Recommendations

1. **Premature Termination Fix**:
   ```bash
   board create -t "Test topic with low novelty" -r 5
   board run --last
   # Verify: Meeting runs at least 2 rounds even if novelty < 0.3 after round 1
   ```

2. **TUI Workflow**:
   ```bash
   board create -t "Meeting 1"
   board create -t "Meeting 2"
   board create -t "Meeting 3"
   board run          # Should show selector with 3 meetings
   board run --last   # Should auto-run Meeting 3
   board status       # Should show selector
   board status --last # Should show Meeting 3 status
   ```

3. **Document Artifacts**:
   ```bash
   board create -t "Document test"
   board run --last
   board status --last
   # Verify:
   # - Artifact path displayed
   # - File exists at artifacts/meetings/meeting_*.md
   # - Contains full comment text (not truncated)
   # - Can be opened with cat or $EDITOR
   ```

---

## Migration Notes

**No breaking changes** - all existing workflows still supported:
- `board run <uuid>` still works (no UUID required but supported)
- `board status <uuid>` still works (no UUID required but supported)
- Truncated table fallback if artifact generation fails

**New patterns available**:
- `board run` → interactive selector
- `board run --last` → use most recent
- `board status` → interactive selector
- `board status --last` → use most recent

**Configuration**:
- Minimum rounds configurable via `MultiAgentMeetingWorkflow.min_rounds` (default: 2)
- Artifact directory: `artifacts/meetings/` (add to .gitignore)
- Meeting logs still auto-generated in `logs/meetings/` on completion
