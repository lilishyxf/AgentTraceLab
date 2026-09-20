# ruff: noqa: E501
from __future__ import annotations

from html import escape
from xml.etree.ElementTree import Element, SubElement, tostring

from agenttracelab.batch_comparison import BatchComparisonReport, ScenarioComparison


def _status_label(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _scenario_details(item: ScenarioComparison) -> str:
    details: list[str] = []
    if item.check_regressions:
        details.append("Regressions: " + ", ".join(item.check_regressions))
    if item.check_improvements:
        details.append("Improvements: " + ", ".join(item.check_improvements))
    return " · ".join(details) or "No deterministic check changes."


def render_comparison_html(report: BatchComparisonReport) -> str:
    decision = escape(report.recommendation.upper())
    decision_class = "promote" if report.recommendation == "promote" else "hold"
    blockers = "".join(f"<li>{escape(reason)}</li>" for reason in report.reasons)
    blocker_section = (
        f'<section><h2>Promotion blockers</h2><ul class="blockers">{blockers}</ul></section>'
        if blockers
        else '<section><h2>Promotion blockers</h2><p class="muted">None.</p></section>'
    )
    rows = "".join(
        (
            "<tr>"
            f"<td><code>{escape(item.scenario_id)}</code></td>"
            f'<td><span class="status {_status_label(item.baseline_passed).lower()}">'
            f"{_status_label(item.baseline_passed)}</span> {item.baseline_score:.2f}</td>"
            f'<td><span class="status {_status_label(item.candidate_passed).lower()}">'
            f"{_status_label(item.candidate_passed)}</span> {item.candidate_score:.2f}</td>"
            f'<td class="delta">{item.score_delta:+.2f}</td>'
            f"<td>{escape(_scenario_details(item))}</td>"
            "</tr>"
        )
        for item in report.scenarios
    )
    if not rows:
        rows = '<tr><td colspan="5" class="muted">No paired scenarios.</td></tr>'
    baseline_only = (
        "".join(f"<li><code>{escape(item)}</code></li>" for item in report.baseline_only_scenarios)
        or "<li>None</li>"
    )
    candidate_only = (
        "".join(f"<li><code>{escape(item)}</code></li>" for item in report.candidate_only_scenarios)
        or "<li>None</li>"
    )
    notice = escape(report.comparison_notice)
    baseline_label = f"{escape(report.baseline_batch_id)} · {escape(report.baseline_version)}"
    candidate_label = f"{escape(report.candidate_batch_id)} · {escape(report.candidate_version)}"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
  <title>AgentTraceLab comparison · {decision}</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0b1020; --panel:#121a2f; --line:#273250;
      --text:#edf3ff; --muted:#9eabc7; --good:#51d88a; --bad:#ff7085; --accent:#78a6ff; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:linear-gradient(145deg,#09101e,#11182c); color:var(--text);
      font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif; }}
    main {{ width:min(1180px,calc(100% - 32px)); margin:40px auto 72px; }}
    header {{ display:flex; justify-content:space-between; gap:24px; align-items:flex-end; margin-bottom:24px; }}
    h1,h2 {{ margin:0; letter-spacing:-.02em; }} h1 {{ font-size:36px; }} h2 {{ font-size:18px; margin-bottom:14px; }}
    .eyebrow,.muted {{ color:var(--muted); }} .eyebrow {{ text-transform:uppercase; letter-spacing:.14em; font-size:12px; }}
    .decision {{ padding:9px 14px; border-radius:999px; font-weight:800; letter-spacing:.08em; }}
    .decision.promote {{ color:#06190e; background:var(--good); }} .decision.hold {{ color:#26050b; background:var(--bad); }}
    .grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }}
    .card,section {{ background:rgba(18,26,47,.88); border:1px solid var(--line); border-radius:16px; }}
    .card {{ padding:18px; }} .card strong {{ display:block; font-size:25px; }} .card span {{ color:var(--muted); }}
    section {{ padding:22px; margin-top:16px; overflow:auto; }}
    table {{ width:100%; border-collapse:collapse; min-width:760px; }} th,td {{ padding:12px; text-align:left; border-bottom:1px solid var(--line); }}
    th {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }}
    code {{ color:#bcd2ff; }} .status {{ font-size:11px; font-weight:800; }} .status.pass {{ color:var(--good); }} .status.fail {{ color:var(--bad); }}
    .delta {{ color:var(--accent); font-variant-numeric:tabular-nums; }}
    .coverage {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }} ul {{ margin:0; padding-left:20px; }}
    .blockers {{ color:#ffd0d7; }} footer {{ margin-top:20px; color:var(--muted); font-size:13px; }}
    @media (max-width:800px) {{ .grid {{ grid-template-columns:1fr 1fr; }} header {{ align-items:flex-start; flex-direction:column; }} }}
  </style>
</head>
<body><main>
  <header><div><div class="eyebrow">AgentTraceLab paired experiment</div><h1>{baseline_label}<br>→ {candidate_label}</h1></div>
    <div class="decision {decision_class}">{decision}</div></header>
  <div class="grid">
    <div class="card"><strong>{report.paired_scenario_count}</strong><span>paired scenarios</span></div>
    <div class="card"><strong>{report.mean_paired_score_delta:+.2f}</strong><span>mean score delta</span></div>
    <div class="card"><strong>{len(report.run_regressions)}</strong><span>run regressions</span></div>
    <div class="card"><strong>{report.check_regression_count}</strong><span>check regressions</span></div>
  </div>
  {blocker_section}
  <section><h2>Paired scenarios</h2><table><thead><tr><th>Scenario</th><th>Baseline</th><th>Candidate</th><th>Delta</th><th>Evidence</th></tr></thead>
    <tbody>{rows}</tbody></table></section>
  <section class="coverage"><div><h2>Baseline-only</h2><ul>{baseline_only}</ul></div><div><h2>Candidate-only</h2><ul>{candidate_only}</ul></div></section>
  <footer>{notice}</footer>
</main></body></html>"""


def render_comparison_junit(report: BatchComparisonReport) -> str:
    paired_failures = sum(
        (not item.candidate_passed) or bool(item.check_regressions) for item in report.scenarios
    )
    total_tests = (
        len(report.scenarios) + len(report.baseline_only_scenarios) + len(report.candidate_only_scenarios) + 1
    )
    failure_count = paired_failures + len(report.baseline_only_scenarios)
    if report.recommendation == "hold":
        failure_count += 1
    suite = Element(
        "testsuite",
        {
            "name": "AgentTraceLab paired comparison",
            "tests": str(total_tests),
            "failures": str(failure_count),
            "errors": "0",
            "skipped": str(len(report.candidate_only_scenarios)),
        },
    )
    properties = SubElement(suite, "properties")
    SubElement(properties, "property", {"name": "recommendation", "value": report.recommendation})
    SubElement(
        properties,
        "property",
        {"name": "paired_scenario_count", "value": str(report.paired_scenario_count)},
    )
    for item in report.scenarios:
        case = SubElement(
            suite,
            "testcase",
            {"classname": "agenttracelab.scenario", "name": item.scenario_id},
        )
        if not item.candidate_passed or item.check_regressions:
            failure = SubElement(case, "failure", {"type": "deterministic-regression"})
            failure.text = _scenario_details(item)
        output = SubElement(case, "system-out")
        output.text = (
            f"baseline_score={item.baseline_score:.2f} "
            f"candidate_score={item.candidate_score:.2f} delta={item.score_delta:+.2f}"
        )
    for scenario_id in report.baseline_only_scenarios:
        case = SubElement(
            suite,
            "testcase",
            {"classname": "agenttracelab.coverage", "name": scenario_id},
        )
        failure = SubElement(case, "failure", {"type": "missing-baseline-scenario"})
        failure.text = "Candidate batch omitted a scenario present in the baseline."
    for scenario_id in report.candidate_only_scenarios:
        case = SubElement(
            suite,
            "testcase",
            {"classname": "agenttracelab.coverage", "name": scenario_id},
        )
        SubElement(case, "skipped", {"message": "Candidate-only scenario has no paired baseline."})
    gate = SubElement(
        suite,
        "testcase",
        {"classname": "agenttracelab.gate", "name": "promotion-recommendation"},
    )
    if report.recommendation == "hold":
        failure = SubElement(gate, "failure", {"type": "promotion-hold"})
        failure.text = "\n".join(report.reasons)
    output = SubElement(suite, "system-out")
    output.text = report.comparison_notice
    return tostring(suite, encoding="unicode", xml_declaration=True)
