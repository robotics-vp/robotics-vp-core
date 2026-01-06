# Codex Postmortem Template

Use this template to diagnose and document Codex issues, especially "silent" failures where no output is produced for extended periods.

## Incident Summary

| Field | Value |
|-------|-------|
| Date | YYYY-MM-DD |
| Duration | How long the issue lasted |
| Severity | High/Medium/Low |
| Resolved | Yes/No |

## Observed Symptoms

Describe what you observed:

- [ ] No output for extended period (specify duration: ___)
- [ ] Partial output then silence
- [ ] Error messages (copy exact text below)
- [ ] Process appeared running but produced nothing
- [ ] Other: ___

### Error Messages

```
Paste any error messages here
```

### Timeline

| Time | Event |
|------|-------|
| HH:MM | Task started |
| HH:MM | Last observed output |
| HH:MM | Issue detected |
| HH:MM | Resolution attempted |
| HH:MM | Resolved |

## Evidence Collection

### Commands Run

```bash
# What command was used to invoke Codex?
./scripts/codex/run.sh "..."
```

### Environment

```bash
# Check and document:
echo $OPENAI_API_KEY | head -c 8  # First 8 chars only
codex --version
which codex
```

### Logs

```bash
# Collect logs
cat .agent/runs/<job_id>/meta.json
cat .agent/runs/<job_id>/events.jsonl | head -50
cat .agent/runs/<job_id>/events.jsonl | tail -50
```

### Diagnostic Output

```bash
./scripts/codex/diagnose.sh
# Paste output here
```

## Root Cause Analysis

### Most Likely Causes (check all that apply)

- [ ] **Auth failure/retry loop** - API key invalid or expired, causing silent retries
- [ ] **Buffering/no flush** - Output buffered, no `--json` flag, events not visible
- [ ] **Missing timeouts** - No timeout set, process hung indefinitely
- [ ] **Tool output limit** - Output truncated silently due to low limit
- [ ] **Network/proxy issues** - Firewall, proxy, or network blocking requests
- [ ] **Rate limiting** - API rate limits hit with no surfaced error
- [ ] **Wrong model/tool** - Requested model unavailable or misconfigured
- [ ] **Task too large** - Task scope caused excessive processing time
- [ ] **Other**: ___

### Evidence for Root Cause

Explain why you believe this is the root cause:

```
Evidence here...
```

## Resolution

### Immediate Fix

What was done to resolve the immediate issue:

```bash
# Commands run to fix
```

### Permanent Fix

What changes were made to prevent recurrence:

- [ ] Added `--json` flag for event streaming
- [ ] Increased tool output limits in config
- [ ] Added timeout wrapper
- [ ] Set up watchdog monitoring
- [ ] Split task into smaller pieces
- [ ] Updated API key
- [ ] Other: ___

### Configuration Changes

```toml
# Config changes made
[tools]
max_output_chars = 50000
```

## Verification

How was the fix verified?

```bash
# Commands run to verify
./scripts/codex/diagnose.sh
./scripts/codex/run.sh "quick test task"
```

## Lessons Learned

### What went well

-

### What could be improved

-

### Action items

| Item | Owner | Due |
|------|-------|-----|
| | | |

## Appendix

### Full Log Output

<details>
<summary>Click to expand full logs</summary>

```
Paste full logs here
```

</details>

### Config at Time of Incident

<details>
<summary>Click to expand config</summary>

```toml
# Contents of ~/.codex/config.toml at incident time
```

</details>

---

## Common Issues Quick Reference

### "No output for ~4 hours"

| Cause | Detection | Fix |
|-------|-----------|-----|
| No `--json` flag | No events.jsonl | Add `--json` to command |
| Output limit | Truncated logs | Increase max_output_chars |
| Auth failure | Check API key | Refresh/verify API key |
| Network | Check connectivity | Verify network access |
| Timeout | No completion | Add shell timeout |

### "Partial output then silence"

| Cause | Detection | Fix |
|-------|-----------|-----|
| Task too large | Last event shows large scope | Split task |
| Rate limit | 429 errors in logs | Add delays, reduce rate |
| Model overload | Slow responses | Try different time |

### "Error but no details"

| Cause | Detection | Fix |
|-------|-----------|-----|
| Silent truncation | Output ends mid-sentence | Increase limits |
| Buffering | No intermediate output | Enable streaming |
| Process killed | SIGKILL in logs | Check system resources |
