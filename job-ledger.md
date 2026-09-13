# Job ledger — campaign-2026-09-11

`job-ledger.jsonl` is one line per **submission** (not per cell), appended by hand after
running an `sbatch` command from `jobs/` — never by an agent, since agents in this campaign
never execute `sbatch` (contract.md, decisions.md #2). Empty until the first submission.

JSONL schema, one object per line:

```
{
  "ts": "2026-09-11T18:00:00Z",       // submission wall-clock time, ISO 8601 UTC
  "job_id": "12345678",                // Slurm job ID from the sbatch output
  "sbatch_script": "jobs/campaign-genoa.sbatch",
  "env_overrides": {"PS_REV": "ac74682", "THREADS": "1 96"},  // non-default vars passed at submit time
  "revision": "ac74682ef0f80c7d4bd54a02cd4caf354aed4af0",     // resolved PS_REV, once known
  "status": "queued"                   // queued | running | completed | failed | cancelled
}
```

Update `status` (and add `revision` once the job log prints the resolved `PS_REV`) by appending
a corrected line — this file is append-only history, not a database to edit in place; the
latest line for a given `job_id` is authoritative.
