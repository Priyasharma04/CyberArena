"""
Matched-seed defender comparison using the SHARED runtime.

Every (attacker, seed) pair is run under each defender with identical topology,
dice and (for PPO) policy sampling, using the same live action mask, defender
cadence and stall watchdog as the live API and cyberarena.evaluate.

Causal attribution: the no-defender run for each (attacker, seed) is executed
first and its result is passed to every defended run, so an episode is labelled
`contained_by_defender` only when the attacker would have won without the
defender. Stalls that happen anyway are `attacker_stalled`.

Reports, per attacker and pooled over all attackers:
  * attacker success rate with Wilson 95% CI
  * paired effect vs no defender: prevented / harmed episodes, prevention rate
    (Wilson CI), exact two-sided sign test on discordant pairs
  * mean compromised hosts, steps, interventions, budget spent (t-based 95% CI)
  * stall rate, containment rate, mean containment step / time to detection

Pooled significance caveat: the same seeds are reused across attackers, so
pairs are not independent. The pooled summary therefore also reports a
seed-clustered sign test (one signed value per seed). Prefer per-attacker
results for inference.

Before running it validates each checkpoint (structure, state version,
visibility) and warns when the evaluation setting differs from the setting the
defender was trained for (budget, interval, topology family, seed overlap).

Usage:
    python evaluate_defenders.py --episodes 100 --seed-start 1000
    python evaluate_defenders.py --visibility alerts --v5 models/learned_defender_v5_alerts.json
    python evaluate_defenders.py --strict      # abort on any compatibility warning
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from cyberarena.config import ArenaConfig
from cyberarena.defender import LearnedDefender
from cyberarena.evaluate import CSV_FIELDS, run_episode, wilson_interval
from cyberarena.runtime import BASELINE_AGENTS, MODEL_PATHS, RL_AGENTS


def available_attackers(requested):
    names = sorted(BASELINE_AGENTS)
    for name in sorted(RL_AGENTS):
        if Path(MODEL_PATHS[name]).is_file():
            names.append(name)
        else:
            print(f"Skipping missing attacker checkpoint: {MODEL_PATHS[name]}")
    if requested:
        unknown = set(requested) - set(names)
        if unknown:
            raise SystemExit(f"Attacker(s) unavailable: {sorted(unknown)}")
        names = [n for n in names if n in requested]
    return names


# --------------------------------------------------- checkpoint validation

def training_range(info, path):
    """(seed_start, seed_end, source) of a checkpoint's training seeds, or None."""
    meta = info["metadata"] or {}
    if "seed_start" in meta and "seed_end" in meta:
        return meta["seed_start"], meta["seed_end"], "checkpoint metadata"
    # Legacy v3: the old trainer wrote no seed range. It used seeds
    # 7 .. 7+episodes-1 (default --seed 7) and a summary file with `episodes`.
    legacy = Path("results/learned_defender_v3_training_summary.json")
    if info["state_version"] == 3 and legacy.is_file():
        episodes = json.loads(legacy.read_text(encoding="utf-8")).get("episodes")
        if episodes:
            return 7, 7 + int(episodes) - 1, "legacy summary (ASSUMES trainer --seed 7)"
    return None


def compatibility_warnings(label, info, path, args, seeds):
    meta = info["metadata"] or {}
    warnings = []

    budget = meta.get("budget", info["initial_budget"])
    if budget is not None and budget != args.budget:
        warnings.append(f"{label}: trained with budget {budget} but evaluating with {args.budget}")
    if "interval" in meta and meta["interval"] != args.interval:
        warnings.append(f"{label}: trained with interval {meta['interval']} but evaluating with {args.interval}")
    if "topology_family" in meta and meta["topology_family"] != args.family:
        warnings.append(f"{label}: trained on '{meta['topology_family']}' but evaluating on '{args.family}' "
                        f"(fine for a generalisation test, otherwise a mismatch)")
    if not meta:
        warnings.append(f"{label}: no training metadata in checkpoint; interval/topology "
                        f"match cannot be verified")
    if meta.get("patch_semantics") != "strong":
        warnings.append(
            f"{label}: checkpoint does not declare strong-patching training semantics; "
            "evaluation uses strong patching, so this is a simulator-policy mismatch"
        )

    if meta.get("defender_semantics") != "confirmed_compromise_v1":
        warnings.append(f"{label}: trained before confirmed-compromise isolation semantics")
    if meta.get("initial_access_alert") != "potential_alert_v1":
        warnings.append(f"{label}: trained without the initial-access alert")
    if meta.get("pre_action_response") is not True:
        warnings.append(f"{label}: trained without the step-0 pre-action defender response")
    if meta.get("attack_path_aware_patching") is not True:
        warnings.append(f"{label}: trained without attack-path-aware patch targeting")

    rng = training_range(info, path)
    if rng is None:
        warnings.append(f"{label}: training seed range unknown; seed overlap cannot be checked")
    else:
        lo, hi, source = rng
        if seeds[0] <= hi and seeds[-1] >= lo:
            warnings.append(f"{label}: evaluation seeds {seeds[0]}-{seeds[-1]} overlap training "
                            f"seeds {lo}-{hi} ({source}); results would be optimistic")
    return warnings


def build_modes(args, seeds):
    modes = {"off": "off", "rule": "rule"}
    seen = {}
    warnings = []
    for label, path in (
        ("learned_v3", args.v3),
        ("learned_v4", args.v4),
        ("learned_v5_legacy", args.v5),
        ("learned_v6", args.v6),
    ):
        p = Path(path)
        if not p.is_file():
            print(f"Skipping {label}: checkpoint not found ({p})")
            continue
        try:
            info = LearnedDefender.inspect_checkpoint(str(p))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"Skipping {label}: invalid checkpoint ({exc})")
            continue
        if info["visibility"] != args.visibility:
            print(f"Skipping {label}: trained with visibility '{info['visibility']}' "
                  f"but evaluating '{args.visibility}'")
            continue
        modes[label] = f"learned:{p}"
        seen[label] = (info["metadata"] or {}).get("attackers")
        warnings += compatibility_warnings(label, info, p, args, seeds)
    return modes, seen, warnings


# ---------------------------------------------------------------- statistics

_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
         15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
         21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
         27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def t_crit(df: int) -> float:
    """Two-sided 95% Student-t critical value."""
    if df <= 30:
        return _T975[max(1, df)]
    if df <= 40:
        return 2.021
    if df <= 60:
        return 2.000
    if df <= 120:
        return 1.980
    return 1.960


def mean_ci(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"n": 1, "mean": round(mean, 4), "ci95_low": None, "ci95_high": None}
    sd = statistics.stdev(values)
    half = t_crit(len(values) - 1) * sd / math.sqrt(len(values))
    return {"n": len(values), "mean": round(mean, 4),
            "ci95_low": round(mean - half, 4), "ci95_high": round(mean + half, 4)}


def sign_test_p(a: int, b: int) -> float:
    """Exact two-sided sign test on discordant pairs."""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def summarise(rows, labels):
    """rows: dicts with 'label', 'attacker', 'seed' and metrics."""
    off = {(r["attacker"], r["seed"]): r for r in rows if r["label"] == "off"}
    out = {}
    for label in labels:
        group = [r for r in rows if r["label"] == label]
        n = len(group)
        wins = sum(r["success"] for r in group)
        lo, hi = wilson_interval(wins, n)

        prevented = harmed = 0
        compromised_diff = []
        per_seed = {}
        for r in group:
            base = off.get((r["attacker"], r["seed"]))
            if base is None or label == "off":
                continue
            p = int(base["success"] == 1 and r["success"] == 0)
            h = int(base["success"] == 0 and r["success"] == 1)
            prevented += p
            harmed += h
            per_seed[r["seed"]] = per_seed.get(r["seed"], 0) + p - h
            compromised_diff.append(r["compromised"] - base["compromised"])
        p_lo, p_hi = wilson_interval(prevented, n)

        paired = None
        if label != "off":
            pos = sum(1 for d in per_seed.values() if d > 0)
            neg = sum(1 for d in per_seed.values() if d < 0)
            paired = {
                "prevented": prevented,
                "harmed": harmed,
                "prevention_rate": round(prevented / n, 4) if n else None,
                "prevention_ci95": [round(p_lo, 4), round(p_hi, 4)],
                "sign_test_p": round(sign_test_p(prevented, harmed), 6),
                # One signed value per seed, so attackers sharing a seed are
                # not treated as independent evidence.
                "sign_test_p_seed_clustered": round(sign_test_p(pos, neg), 6),
                "mean_compromised_change": mean_ci(compromised_diff),
            }

        out[label] = {
            "episodes": n,
            "attacker_success_rate": round(wins / n, 4) if n else None,
            "attacker_success_ci95": [round(lo, 4), round(hi, 4)],
            "paired_vs_off": paired,
            "compromised_hosts": mean_ci([r["compromised"] for r in group]),
            "steps": mean_ci([r["steps"] for r in group]),
            "interventions": mean_ci([r["defender_interventions"] for r in group]),
            "budget_spent": mean_ci([r["budget_spent"] for r in group]),
            "time_to_detection": mean_ci([r["time_to_detection"] for r in group]),
            "confirmed_compromises": mean_ci([r["confirmed_compromises"] for r in group]),
            "first_defender_decision_step": mean_ci(
                [r["first_defender_decision_step"] for r in group]
            ),
            # last step of attacker progress in VERIFIED contained episodes
            "containment_step": mean_ci([r["containment_step"] for r in group]),
            "stall_rate": round(statistics.fmean(r["stalled"] for r in group), 4) if n else None,
            "containment_rate": round(
                statistics.fmean(r["outcome"] == "contained_by_defender" for r in group), 4
            ) if n else None,
            "outcomes": {o: sum(1 for r in group if r["outcome"] == o)
                         for o in sorted({r["outcome"] for r in group})},
        }
    return out


# ---------------------------------------------------------------------- main

def main(args):
    attackers = available_attackers(args.attackers.split(",") if args.attackers else None)
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    modes, trained_on, warnings = build_modes(args, seeds)

    for label, names in trained_on.items():
        if names:
            unseen = [a for a in attackers if a not in names]
            if unseen:
                print(f"Note: {label} did not train against {unseen} (held-out attackers).")

    for w in warnings:
        print(f"WARNING: {w}")
    if warnings and args.strict:
        raise SystemExit("Aborting: compatibility warnings with --strict.")

    cfg = ArenaConfig(
        topology_family=args.family,
        defender_budget=args.budget,
        defender_interval=args.interval,
        defender_visibility=args.visibility,
    )

    rows = []
    total = len(attackers) * len(seeds) * len(modes)
    done = 0
    for attacker in attackers:
        for seed in seeds:
            baseline_success = None
            for label, mode in modes.items():          # "off" runs first
                r = run_episode(
                    attacker, seed, config=cfg, defender_mode=mode,
                    stochastic=not args.deterministic,
                    baseline_success=baseline_success,
                )
                if label == "off":
                    baseline_success = bool(r["success"])
                row = {k: r[k] for k in CSV_FIELDS}
                row["outcome"] = r["outcome"]
                row["label"] = label
                row["attacker"] = attacker
                rows.append(row)
                done += 1
        print(f"Finished {attacker} ({done}/{total})")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "defender_comparison.csv"
    fields = ["attacker", "label", "outcome"] + [f for f in CSV_FIELDS if f != "agent"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    labels = list(modes)
    summary = {
        "config": {
            "attackers": attackers, "modes": modes,
            "seed_start": seeds[0], "seed_end": seeds[-1], "episodes_per_cell": len(seeds),
            "topology_family": args.family, "defender_budget": args.budget,
            "defender_interval": args.interval, "visibility": args.visibility,
            "patch_semantics": "strong",
            "initial_access_alert": "potential_alert_v1",
            "defender_semantics": "confirmed_compromise_v1",
            "defender_first_response": "pre_action_step_0",
            "attack_path_aware_patching": True,
            "stochastic_attackers": not args.deterministic,
            "max_steps": cfg.max_steps, "stall_window": cfg.stall_window,
            "compatibility_warnings": warnings,
            "notes": [
                "contained_by_defender requires the matched no-defender run to reach the objective.",
                "containment_step = last step of attacker progress in verified contained episodes.",
                "Pooled p-values reuse seeds across attackers; see sign_test_p_seed_clustered "
                "and prefer per-attacker results.",
            ],
        },
        "by_attacker": {a: summarise([r for r in rows if r["attacker"] == a], labels)
                        for a in attackers},
        "pooled": summarise(rows, labels),
    }
    summary_path = out_dir / "defender_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nPooled over all attackers (attacker success, paired prevention vs off):")
    for label in labels:
        s = summary["pooled"][label]
        extra = ""
        if s["paired_vs_off"]:
            p = s["paired_vs_off"]
            extra = (f" | prevented {p['prevented']} harmed {p['harmed']} "
                     f"p={p['sign_test_p']} (seed-clustered p={p['sign_test_p_seed_clustered']})")
        print(f"  {label:11s} success {s['attacker_success_rate']:.1%} "
              f"{s['attacker_success_ci95']} compromised "
              f"{s['compromised_hosts']['mean']}{extra}")
    print(f"\nCSV: {csv_path}\nSummary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--family", default="mixed")
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--interval", type=int, default=4)
    parser.add_argument("--visibility", choices=["full", "alerts"], default="alerts")
    parser.add_argument("--attackers", default=None)
    parser.add_argument("--deterministic", action="store_true",
                        help="Deterministic PPO actions (default: stochastic)")
    parser.add_argument("--strict", action="store_true",
                        help="Abort if a checkpoint's training setup differs from this evaluation")
    parser.add_argument("--v3", default="models/learned_defender_v3.json")
    parser.add_argument("--v4", default="models/learned_defender_v4.json")
    parser.add_argument("--v5", default="models/learned_defender_v5_alerts.json")
    parser.add_argument("--v6", default="models/learned_defender_v6_alerts.json")
    args = parser.parse_args()
    if args.episodes < 1 or args.interval < 1 or args.budget < 0:
        parser.error("episodes/interval must be positive and budget non-negative")
    main(args)
