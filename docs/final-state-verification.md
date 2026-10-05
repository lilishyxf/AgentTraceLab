# Independent final-state verification

Trace checks prove what an Agent recorded. Final-state verification separately reads what actually
exists after the run and checks that state against a declarative contract.

## Workflow

1. Reset a filesystem tree or SQLite database from a versioned fixture.
2. Save the initial `EnvironmentSnapshot`.
3. Run the target Agent outside the evaluator with its normal tools.
4. Ask the same adapter to snapshot the final workspace or database independently.
5. Run `verify-final-state` against the contract and the two snapshot artifacts.

```powershell
uv run agenttracelab verify-final-state contract.json `
  --initial initial-snapshot.json `
  --final final-snapshot.json `
  --output final-state-report.json `
  --summary final-state-report.md
```

The command accepts previously captured snapshots, so the target may run in another process or
browser. AgentTraceLab does not trust the target trace as proof of the external result.

## Contract boundary

Contracts allow only fixed predicates for file existence/hash, CSV cells/rows, SQLite rows/counts/
values, schema equality, and protected-state equality. They cannot contain Python, shell commands,
SQL, or arbitrary expressions. Multiple alternatives may describe different legal paths to an
accepted state. Protected-state failures and side-effect-bound violations fail even when a required
alternative matches.

Filesystem fixtures reject entries that resolve outside their root. SQLite snapshots open the copied
database read-only and order schema objects and rows deterministically; manifest-provided SQL is never
executed.

The checked-in offline fixture index contains three filesystem and three SQLite contracts. Each
mutating task has two accepted outcomes, and each adapter includes a no-write clarification case.
