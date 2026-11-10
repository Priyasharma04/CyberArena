from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _load(summary_path):
    """Accept both the new {"config", "summary"} file and the legacy flat one."""
    data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "summary" in data and "config" in data:
        return data["config"], data["summary"]
    return None, data


def build_report(summary_path="results/baselines.summary.json", output="results/cyberarena_report.pdf"):
    config, summary = _load(summary_path)
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(output, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("CyberArena – Simulation Assessment", styles["Title"]),
        Paragraph(
            "Scope: abstract, procedurally generated network simulation only. "
            "No external systems were scanned or modified.",
            styles["BodyText"],
        ),
        Spacer(1, 14),
    ]

    # Experimental setup: results are meaningless without the conditions.
    if config:
        story.append(Paragraph("Experimental setup", styles["Heading2"]))
        setup = (
            f"Topology: <b>{config.get('topology_family')}</b> &nbsp;|&nbsp; "
            f"Defender: <b>{config.get('defender_mode')}</b> &nbsp;|&nbsp; "
            f"Seeds: <b>{config.get('seed_start')}–{config.get('seed_end')}</b> "
            f"({config.get('episodes_per_agent')} episodes per agent) &nbsp;|&nbsp; "
            f"Max steps: <b>{config.get('max_steps')}</b> &nbsp;|&nbsp; "
            f"Reward mode: <b>{config.get('reward_mode')}</b> &nbsp;|&nbsp; "
            f"Stochastic policy: <b>{config.get('stochastic')}</b>"
        )
        story.extend([Paragraph(setup, styles["BodyText"]), Spacer(1, 10)])
    else:
        story.extend([
            Paragraph(
                "<i>Configuration was not recorded in this summary file "
                "(legacy format); re-run the evaluation to capture it.</i>",
                styles["BodyText"],
            ),
            Spacer(1, 10),
        ])

    rows = [["Agent", "Episodes", "Success (95% CI)", "Detection", "Mean steps", "Mean reward"]]
    for name, m in summary.items():
        if "success_ci95_low" in m:
            success = (
                f"{m['success_rate']:.1%} "
                f"[{m['success_ci95_low']:.0%}–{m['success_ci95_high']:.0%}]"
            )
        else:
            success = f"{m['success_rate']:.1%}"
        detection = f"{m['detection_rate']:.0%}" if "detection_rate" in m else "n/a"
        steps = f"{m['mean_steps']:.1f}"
        if "std_steps" in m:
            steps += f" ± {m['std_steps']:.1f}"
        rows.append([
            name,
            str(m.get("episodes", "n/a")),
            success,
            detection,
            steps,
            f"{m['mean_reward']:.3f}",
        ])

    table = Table(rows, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF3F8")]),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))

    story.extend([
        Paragraph("Baseline results", styles["Heading2"]),
        table,
        Spacer(1, 12),
        Paragraph("Interpretation", styles["Heading2"]),
        Paragraph(
            "These values are simulator measurements, not evidence of real-world "
            "exploit capability. The Oracle is a full-knowledge reference planner, not a fair "
            "partial-observation attacker; against an adaptive defender its narrow "
            "shortest-path strategy is not guaranteed to outperform broader "
            "attackers. Intervals "
            "are 95% Wilson score intervals; small episode counts give wide "
            "intervals, so compare agents across many seeds and topology families "
            "before drawing research conclusions.",
            styles["BodyText"],
        ),
    ])

    doc.build(story)
    return output
