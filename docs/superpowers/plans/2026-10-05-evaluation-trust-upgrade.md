# Evaluation Trust Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add held-out Judge calibration, independent file/SQLite final-state verification, and runtime fault campaigns that feed fail-closed release readiness without paid APIs.

**Architecture:** Three new focused modules produce immutable, separately scored reports; existing readiness logic binds those reports and decides whether the evidence remains diagnostic-only. Existing calibration v1 and snapshot mutation behavior stay unchanged, while new v2 entry points make stronger evidence explicit.

**Tech Stack:** Python 3.11, Pydantic 2, SQLite stdlib, csv/openpyxl-free file hashing, argparse, pytest, Ruff

**Spec:** `docs/superpowers/specs/2026-10-05-evaluation-trust-upgrade-design.md`

## Global Constraints

- Preserve calibration v1 and snapshot-mutation command behavior.
- Keep process, final-state, Judge-reliability, and runtime-resilience reports separate.
- Final state must be read independently after the target run; trace claims cannot satisfy it.
- Accept any declared valid final state rather than enforce a single tool trajectory.
- Fail closed on malformed contracts, missing protected baselines, identity mismatch, or incomplete observations.
- New unit and integration tests must run offline without model credentials.
- Bind manifests, snapshots, fixtures, and reports by SHA-256.
- Do not add a dashboard, authentication system, model fine-tuning, live SaaS injection, or container orchestrator.
- Do not persist secrets, unrestricted tool arguments, or database credentials.
- Refuse to overwrite existing evidence artifacts.

## Review Focus

- Symlink or path traversal in a filesystem fixture must be rejected, covered by `test_filesystem_adapter_rejects_escape` in Task 2.
- A SQLite database with a view, trigger, or non-table object must snapshot deterministically without executing evaluator-supplied SQL, covered by `test_sqlite_snapshot_is_read_only_and_stable` in Task 2.
- A Judge key reused with different provider/model/prompt identity must fail closed, covered by `test_calibration_v2_rejects_identity_drift` in Task 1.
- A fault runner that reports safe stop without evidence linkage must fail the campaign, covered by `test_campaign_rejects_unlinked_safe_stop` in Task 3.
- A readiness bundle whose bound artifact changes after hashing must be rejected, covered by `test_trust_readiness_rejects_changed_artifact` in Task 4.

---

### Task 1: Held-out Judge calibration v2

**Files:**
- Create: `src/agenttracelab/calibration_v2.py`
- Create: `tests/test_calibration_v2.py`
- Modify: `src/agenttracelab/cli.py`
- Create: `evaluation/judge-calibration/offline-v2/manifest.json`
- Create: `docs/judge-calibration-v2.md`

**Interfaces:**
- Consumes: existing `JudgeDimensionName`, `JudgeVerdict`, `FrozenModel`, `_cohen_kappa`, and existing CLI output conventions.
- Produces: `CalibrationManifestV2`, `CalibrationReportV2`, `load_calibration_manifest_v2(path: str | Path) -> tuple[CalibrationManifestV2, bytes]`, `analyze_judge_calibration_v2(manifest: CalibrationManifestV2) -> CalibrationReportV2`, and `render_calibration_v2_markdown(report: CalibrationReportV2) -> str`.

- [ ] **Step 1: Write failing schema and validation tests**

Add tests named `test_calibration_v2_requires_two_holdout_annotators`, `test_calibration_v2_requires_adjudication_for_disagreement`, and `test_calibration_v2_rejects_identity_drift`. Assert exact validation failures for missing second annotation, unresolved disagreement, and one `judge_key` mapped to two identities.

- [ ] **Step 2: Run validation tests and verify RED**

Run: `uv run pytest tests/test_calibration_v2.py -k "requires or identity" -q`

Expected: collection/import failure because `agenttracelab.calibration_v2` does not exist.

- [ ] **Step 3: Implement the v2 immutable schemas and loader**

Create `CalibrationSplit`, `JudgeOutcomeStatus`, `JudgeIdentity`, `HumanAnnotation`, `AdjudicatedLabel`, `RecordedJudgeOutcomeV2`, `CalibrationCaseV2`, `CalibrationPolicyV2`, and `CalibrationManifestV2`. The loader enforces a 1 MiB limit, JSON object root, schema version, unique case IDs, exact identity consistency, minimum annotations, and adjudication rules.

- [ ] **Step 4: Run validation tests and verify GREEN**

Run: `uv run pytest tests/test_calibration_v2.py -k "requires or identity" -q`

Expected: all selected tests pass.

- [ ] **Step 5: Write failing split-metric and gate tests**

Add `test_calibration_v2_keeps_development_and_holdout_separate`, `test_calibration_v2_counts_false_allow_false_block_abstain_and_error`, `test_development_success_cannot_mask_holdout_failure`, and `test_safety_critical_false_allow_blocks_holdout` with a small in-memory manifest and exact expected counts/rates.

- [ ] **Step 6: Run metric tests and verify RED**

Run: `uv run pytest tests/test_calibration_v2.py -k "separate or counts or mask or safety" -q`

Expected: failures because the analyzer and report models are missing.

- [ ] **Step 7: Implement metrics, gates, rendering, and CLI command**

Implement `CalibrationSplitMetrics`, `JudgeCalibrationV2Metrics`, and `CalibrationReportV2`. Exclude abstentions/errors from agreement denominators but retain them in total, coverage, abstention, and scoring-error metrics. Add `analyze-judge-calibration-v2 MANIFEST --output JSON --summary MD`, with no overwrite and atomic parent creation behavior matching current CLI conventions.

- [ ] **Step 8: Add the offline fixture and documentation**

Create a deliberately small development/holdout fixture that exercises all outcome statuses and states that it validates implementation only, not any commercial Judge. Document identity binding, denominator rules, and the 24/12 production gate.

- [ ] **Step 9: Run Task 1 verification**

Run: `uv run pytest tests/test_calibration.py tests/test_calibration_v2.py -q && uv run ruff check src/agenttracelab/calibration_v2.py tests/test_calibration_v2.py src/agenttracelab/cli.py`

Expected: all selected tests pass and Ruff reports `All checks passed!`.

- [ ] **Step 10: Commit Task 1**

```powershell
git add src/agenttracelab/calibration_v2.py src/agenttracelab/cli.py tests/test_calibration_v2.py evaluation/judge-calibration/offline-v2/manifest.json docs/judge-calibration-v2.md
git commit -m "feat: add held-out judge calibration"
```

### Task 2: Independent filesystem and SQLite final-state verification

**Files:**
- Create: `src/agenttracelab/final_state.py`
- Create: `tests/test_final_state.py`
- Modify: `src/agenttracelab/cli.py`
- Create: `evaluation/final-state/offline-v1/manifest.json`
- Create: `evaluation/final-state/offline-v1/fixtures/files/`
- Create: `evaluation/final-state/offline-v1/fixtures/sqlite/fixture.sqlite3`
- Create: `docs/final-state-verification.md`

**Interfaces:**
- Consumes: Task 1 CLI conventions only; no calibration types.
- Produces: `EnvironmentSnapshot`, `FilesystemEnvironmentAdapter`, `SQLiteEnvironmentAdapter`, `FinalStateContract`, `FinalStateReport`, `load_final_state_contract(path: str | Path) -> tuple[FinalStateContract, bytes]`, `verify_final_state(contract: FinalStateContract, initial: EnvironmentSnapshot, final: EnvironmentSnapshot) -> FinalStateReport`, and Markdown rendering.

- [ ] **Step 1: Write failing adapter reset and security tests**

Add `test_filesystem_reset_reproduces_contents_in_distinct_workspaces`, `test_filesystem_adapter_rejects_escape`, and `test_sqlite_snapshot_is_read_only_and_stable`. Include a filesystem symlink escape where supported and skip only when the platform cannot create symlinks.

- [ ] **Step 2: Run adapter tests and verify RED**

Run: `uv run pytest tests/test_final_state.py -k "reset or escape or sqlite_snapshot" -q`

Expected: collection/import failure because `agenttracelab.final_state` does not exist.

- [ ] **Step 3: Implement environment snapshots and adapters**

Implement `FilesystemEnvironmentAdapter(fixture_root: Path, workspace_root: Path)` and `SQLiteEnvironmentAdapter(fixture_database: Path, workspace_root: Path)`. Use resolved-root containment, reject escaping symlinks, copy fixtures to unique workspaces, open SQLite snapshots with `mode=ro`, order objects and rows deterministically, and hash canonical JSON.

- [ ] **Step 4: Run adapter tests and verify GREEN**

Run: `uv run pytest tests/test_final_state.py -k "reset or escape or sqlite_snapshot" -q`

Expected: all selected tests pass.

- [ ] **Step 5: Write failing contract and evaluation tests**

Add `test_two_legal_trajectories_satisfy_one_contract`, `test_wrong_record_fails_required_state`, `test_extra_write_fails_side_effect_bound`, `test_protected_state_change_always_fails`, `test_no_write_contract_rejects_false_completion`, `test_contract_rejects_unsafe_predicate`, and `test_report_hashes_are_stable`.

- [ ] **Step 6: Run contract tests and verify RED**

Run: `uv run pytest tests/test_final_state.py -k "trajectories or wrong_record or extra_write or protected or no_write or unsafe or hashes" -q`

Expected: failures because contract predicates and verification are missing.

- [ ] **Step 7: Implement declarative contracts and verifier**

Support only `file_exists`, `file_sha256`, `csv_cell_equals`, `csv_row_exists`, `sqlite_row_exists`, `sqlite_row_absent`, `sqlite_count_equals`, `sqlite_value_equals`, `schema_equals`, and `protected_equals`. Evaluate all protected predicates and side-effect bounds after selecting a matching valid alternative. Never execute manifest-provided code or SQL.

- [ ] **Step 8: Add final-state CLI, six offline fixtures, and docs**

Add `verify-final-state CONTRACT --initial PATH --final PATH --output JSON --summary MD`. Provide three filesystem and three SQLite contracts, including one no-write task and two alternative accepted end states for each mutating task. Document importing a captured final workspace versus running a target externally.

- [ ] **Step 9: Run Task 2 verification**

Run: `uv run pytest tests/test_final_state.py -q && uv run ruff check src/agenttracelab/final_state.py tests/test_final_state.py src/agenttracelab/cli.py`

Expected: all tests pass and Ruff reports `All checks passed!`.

- [ ] **Step 10: Commit Task 2**

```powershell
git add src/agenttracelab/final_state.py src/agenttracelab/cli.py tests/test_final_state.py evaluation/final-state/offline-v1 docs/final-state-verification.md
git commit -m "feat: verify independent final state"
```

### Task 3: Runtime fault campaigns

**Files:**
- Create: `src/agenttracelab/runtime_faults.py`
- Create: `tests/test_runtime_faults.py`
- Modify: `src/agenttracelab/cli.py`
- Create: `evaluation/runtime-faults/offline-v1/manifest.json`
- Create: `docs/runtime-fault-campaigns.md`

**Interfaces:**
- Consumes: Task 2 `EnvironmentSnapshot` hashes and existing `Disposition` semantics.
- Produces: `RuntimeFaultType`, `RuntimeOperation`, `RuntimeFaultObservation`, `RuntimeFaultManifest`, `RuntimeFaultCampaignReport`, `RuntimeToolProxy`, `run_runtime_fault_campaign(manifest: RuntimeFaultManifest, runner: RuntimeFaultRunner) -> RuntimeFaultCampaignReport`, and Markdown rendering.

- [ ] **Step 1: Write failing proxy behavior tests**

Add `test_applied_response_lost_commits_once_without_blind_retry`, `test_stale_approval_prevents_commit`, and `test_readback_mismatch_prevents_completion`, using a deterministic in-memory key/value target runner.

- [ ] **Step 2: Run proxy tests and verify RED**

Run: `uv run pytest tests/test_runtime_faults.py -k "lost or stale or mismatch" -q`

Expected: collection/import failure because `agenttracelab.runtime_faults` does not exist.

- [ ] **Step 3: Implement the proxy, faults, and observation model**

Implement one-shot fault hooks at `before_commit`, `after_commit_before_response`, and `after_commit_before_readback`. Bind approval to source fingerprint and proposal hash. Bind each attempted action, actual effect, returned response/error, readback, and terminal disposition by operation ID.

- [ ] **Step 4: Run proxy tests and verify GREEN**

Run: `uv run pytest tests/test_runtime_faults.py -k "lost or stale or mismatch" -q`

Expected: all selected tests pass.

- [ ] **Step 5: Write failing campaign and unsafe-runner tests**

Add `test_campaign_runs_three_faults_five_times`, `test_campaign_detects_duplicate_side_effect`, `test_campaign_detects_false_completion`, `test_campaign_rejects_unlinked_safe_stop`, and `test_campaign_artifact_hashes_are_stable`.

- [ ] **Step 6: Run campaign tests and verify RED**

Run: `uv run pytest tests/test_runtime_faults.py -k "campaign" -q`

Expected: failures because campaign aggregation and gates are missing.

- [ ] **Step 7: Implement campaign aggregation, CLI, fixture, and docs**

Implement repetitions, duplicate-effect counting, false-completion counting, recovered/safe-stop/unresolved counts, linkage completeness, artifact hashes, and pass policy. Add `run-runtime-faults MANIFEST --output-directory DIR`; its built-in offline runner is explicitly a harness validation target, not proof of a third-party Agent's recovery.

- [ ] **Step 8: Run Task 3 verification**

Run: `uv run pytest tests/test_fault_injection.py tests/test_runtime_faults.py -q && uv run ruff check src/agenttracelab/runtime_faults.py tests/test_runtime_faults.py src/agenttracelab/cli.py`

Expected: legacy mutation tests and new runtime tests all pass; Ruff reports `All checks passed!`.

- [ ] **Step 9: Commit Task 3**

```powershell
git add src/agenttracelab/runtime_faults.py src/agenttracelab/cli.py tests/test_runtime_faults.py evaluation/runtime-faults/offline-v1/manifest.json docs/runtime-fault-campaigns.md
git commit -m "feat: add runtime fault campaigns"
```

### Task 4: Trust evidence readiness and end-to-end offline release gate

**Files:**
- Create: `src/agenttracelab/trust_readiness.py`
- Create: `tests/test_trust_readiness.py`
- Modify: `src/agenttracelab/cli.py`
- Modify: `README.md`
- Create: `docs/evaluation-trust-release.md`

**Interfaces:**
- Consumes: Task 1 `CalibrationReportV2`, Task 2 `FinalStateReport`, Task 3 `RuntimeFaultCampaignReport` JSON artifacts and their SHA-256 values.
- Produces: `TrustEvidenceManifest`, `TrustReadinessReport`, `audit_trust_evidence(manifest_path: str | Path) -> TrustReadinessReport`, and Markdown rendering.

- [ ] **Step 1: Write failing artifact-binding and status tests**

Add `test_trust_readiness_passes_complete_offline_evidence`, `test_trust_readiness_rejects_missing_artifact`, `test_trust_readiness_rejects_changed_artifact`, `test_trust_readiness_rejects_identity_mismatch`, and `test_trust_readiness_keeps_unreviewed_evidence_diagnostic_only`.

- [ ] **Step 2: Run readiness tests and verify RED**

Run: `uv run pytest tests/test_trust_readiness.py -q`

Expected: collection/import failure because `agenttracelab.trust_readiness` does not exist.

- [ ] **Step 3: Implement the evidence manifest and fail-closed audit**

Load bound artifacts from paths contained under the manifest directory, verify every declared hash before parsing, verify schema versions and cross-artifact Judge/environment identities, and emit required checks with `pass`, `fail`, or `unknown`. Recommendation remains `diagnostic_only` unless all new gates pass; the offline fixture still carries an explicit non-commercial-evidence limitation.

- [ ] **Step 4: Add readiness CLI and release documentation**

Add `audit-trust-evidence MANIFEST --output JSON --summary MD`. Update README with the four offline workflows, boundaries between mutation and runtime recovery, and a reproducible command sequence that uses only checked-in fixtures and temporary output directories.

- [ ] **Step 5: Run Task 4 targeted verification**

Run: `uv run pytest tests/test_trust_readiness.py tests/test_readiness.py -q && uv run ruff check src/agenttracelab/trust_readiness.py tests/test_trust_readiness.py src/agenttracelab/cli.py`

Expected: all targeted tests pass and Ruff reports `All checks passed!`.

- [ ] **Step 6: Run full regression and CLI smoke verification**

Run: `uv run pytest -q`

Expected: all existing 211 tests plus new tests pass with zero failures.

Run: `uv run ruff check .`

Expected: `All checks passed!`.

Run each new CLI once against checked-in offline fixtures with outputs under a fresh temporary directory.

Expected: calibration, final-state, runtime-fault, and trust-readiness JSON/Markdown artifacts are created without network access; a second invocation refuses overwrite.

- [ ] **Step 7: Commit Task 4**

```powershell
git add src/agenttracelab/trust_readiness.py src/agenttracelab/cli.py tests/test_trust_readiness.py README.md docs/evaluation-trust-release.md
git commit -m "feat: gate releases on trusted evaluation evidence"
```

- [ ] **Step 8: Inspect final repository state**

Run: `git status --short --branch && git log --oneline -5`

Expected: no unintended generated files, four implementation commits after the approved design, and the branch ahead only by deliberate local commits.
