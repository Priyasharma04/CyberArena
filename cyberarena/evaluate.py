from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import replace
from pathlib import Path

from .config import ArenaConfig
from .env import CyberArenaEnv, ACTION_TYPES  # noqa: F401  (re-exported for scripts)
from .runtime import (
    AGENTS,  # noqa: F401  (re-exported: api.py / scripts import it from here)
    ALL_AGENTS,
    RL_AGENTS,
    DEFENDER_MODES,  # noqa: F401
    TOPOLOGY_CHOICES,
    valid_defender_mode,
    classify_outcome,
    create_attacker,
    create_defender,
    select_action,
    StallMonitor,
    stall_event,
)

# Columns written to CSV. Full logs/topology stay in the in-memory result
# (used by /api/episode) but are not dumped into the CSV.
CSV_FIELDS = (
    "agent", "seed", "defender_mode", "topology_family",
    "success", "termination_reason", "stalled", "steps", "reward",
    "compromised", "discovered", "privileged", "detected",
    "detection_events", "initial_access_alerts", "confirmed_compromises",
    "alerted_hosts", "defender_interventions", "time_to_detection",
    "first_defender_decision_step", "first_defender_action_step",
    "last_attacker_progress_step", "containment_step", "episode_end_step",
    "budget_spent", "defender_visibility",
    "crown_jewel", "attack_path", "defender_actions",
)


_PROGRESS_PREFIXES = (
    "discovered_host_",
    "abstract_compromise_success",
    "privilege_level_raised",
    "lateral_movement_success_",
)


def last_attacker_progress_step(env) -> int:
    """Step of the attacker's last successful discovery / compromise /
    escalation / lateral move (0 if it never progressed)."""
    last = 0
    for entry in env.logs:
        result = str(entry.get("result", ""))
        if any(result.startswith(p) for p in _PROGRESS_PREFIXES):
            last = max(last, int(entry.get("step", 0)))
    return last


def run_episode(
    agent_name: str,
    seed: int,
    defender_enabled: bool = False,
    config: ArenaConfig | None = None,
    defender_mode: str | None = None,
    stochastic: bool = True,
    baseline_success: bool | None = None,
):
    """
    Run one episode under exactly the same rules as the live API:
    shared live action mask, defender cadence from config, stall watchdog.

    `defender_enabled=True` is kept for backward compatibility and means
    the rule-based defender.

    `baseline_success` is the result of the SAME attacker on the SAME seed with
    no defender. When supplied, a non-success episode in which the defender
    acted is labelled `contained_by_defender` only if the baseline attacker
    reached the objective. Without it, stalls after an intervention are
    labelled `stalled_after_intervention` (causation unproven).
    """
    if defender_mode is None:
        defender_mode = "rule" if defender_enabled else "off"
    if not valid_defender_mode(defender_mode):
        raise ValueError(f"Unknown defender mode: {defender_mode}")
    if agent_name not in ALL_AGENTS:
        raise ValueError(f"Unknown attacker: {agent_name}")

    cfg = replace(config or ArenaConfig(), seed=int(seed))
    env = CyberArenaEnv(cfg, render_mode="ansi")
    obs, reset_info = env.reset(seed=seed)

    attacker = create_attacker(agent_name, env)
    if agent_name in RL_AGENTS:
        # Make stochastic policy sampling reproducible per seed.
        attacker.set_random_seed(int(seed))
    defense = create_defender(
        defender_mode, seed, cfg.defender_budget, cfg.defender_visibility
    )
    monitor = StallMonitor(env, defense)

    # Give the defender one initial decision at step 0, before the attacker
    # receives its first action. Training and the live API use the same timing.
    if defense is not None:
        defense.respond(env)

    total_reward = 0.0
    success = False
    stalled = False
    truncated = False
    stalled_reason = None

    while True:
        mask = env.live_action_mask()
        if not mask.any():
            stalled, truncated = True, True
            stalled_reason = "no_unfailed_legal_actions"
            stall_event(env, stalled_reason)
            break

        action = select_action(agent_name, attacker, obs, env, mask, stochastic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            success = bool(info["success"])
            break

        if defense is not None and env.steps % cfg.defender_interval == 0:
            defense.respond(env)

        if monitor.update():
            stalled, truncated = True, True
            stalled_reason = f"no_network_progress_for_{monitor.window}_steps"
            stall_event(env, stalled_reason)
            break

    defender_actions = defense.actions if defense else []
    defender_decisions = getattr(defense, "decisions", []) if defense else []
    behavioural_events = [
        event for event in env.detection_events
        if event.get("signal") != "initial_access" or event.get("confirmed_compromise")
    ]
    detection_times = [
        int(event.get("confirmation_step", event.get("step", 0)))
        if event.get("signal") == "initial_access"
        else int(event.get("step", 0))
        for event in behavioural_events
    ]
    outcome = classify_outcome(
        success, stalled, truncated, defense,
        objective_without_defender=(
            None if baseline_success is None else bool(baseline_success)
        ),
    )
    last_progress = last_attacker_progress_step(env)

    return {
        "agent": agent_name,
        "seed": seed,
        "defender_mode": defender_mode,
        "topology_family": reset_info.get("topology_family", cfg.topology_family),
        "success": int(success),
        "termination_reason": stalled_reason
        if stalled else classify_outcome(success, False, truncated, defense),
        "outcome": outcome,
        "stalled": int(stalled),
        "steps": env.steps,
        "reward": round(total_reward, 5),
        "compromised": len(env.compromised),
        "discovered": len(env.discovered),
        "privileged": len(env.privileged),
        # Exclude the potential initial-access alert from behavioural detection
        # metrics until corroborating activity confirms the foothold.
        "detected": int(bool(behavioural_events)),
        "detection_events": len(behavioural_events),
        "initial_access_alerts": sum(
            event.get("signal") == "initial_access" for event in env.detection_events
        ),
        "confirmed_compromises": len(env.confirmed_compromised_hosts),
        "alerted_hosts": len(env.detected_hosts),
        "defender_interventions": len(defender_actions),
        "time_to_detection": min(detection_times) if detection_times else None,
        "first_defender_decision_step": (
            defender_decisions[0].get("step") if defender_decisions else None
        ),
        "first_defender_action_step": (
            defender_actions[0].get("step") if defender_actions else None
        ),
        # Last step at which the attacker made any progress. For a VERIFIED
        # contained_by_defender episode, nothing progressed after it, so it is
        # when containment took effect. It is NOT the episode-end step.
        "last_attacker_progress_step": last_progress,
        "containment_step": (
            last_progress if outcome == "contained_by_defender" else None
        ),
        "episode_end_step": env.steps,
        "budget_spent": getattr(defense, "budget_spent", 0) if defense else 0,
        "defender_visibility": cfg.defender_visibility,
        "attack_path": json.dumps(env.last_path),
        "crown_jewel": env.crown_jewel,
        "topology": {
            "nodes": [
                {
                    "id": n,
                    "discovered": n in env.discovered,
                    "compromised": n in env.compromised,
                    "privileged": n in env.privileged,
                    "crown_jewel": n == env.crown_jewel,
                }
                for n in sorted(env.discovered)
            ],
            "edges": [
                {"source": u, "target": v}
                for u, v in env.graph.edges
                if u in env.discovered and v in env.discovered
            ],
        },
        "logs": env.logs.copy(),
        "defender_actions": json.dumps(defender_actions),
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _mean(values):
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def _std(values):
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def wilson_interval(successes: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a success proportion."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _mean_defined(values):
    """Mean over values that exist (e.g. time-to-detection is None if never
    detected). Returns None when no episode has a value."""
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def summarize(rows, agent_names):
    summary = {}
    for name in agent_names:
        group = [r for r in rows if r["agent"] == name]
        n = len(group)
        wins = sum(r["success"] for r in group)
        lo, hi = wilson_interval(wins, n)
        outcomes = {}
        for r in group:
            outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
        summary[name] = {
            "episodes": n,
            "success_rate": _mean(r["success"] for r in group),
            "success_ci95_low": lo,
            "success_ci95_high": hi,
            "mean_steps": _mean(r["steps"] for r in group),
            "std_steps": _std(r["steps"] for r in group),
            "mean_reward": _mean(r["reward"] for r in group),
            "std_reward": _std(r["reward"] for r in group),
            "detection_rate": _mean(r["detected"] for r in group),
            "mean_detection_events": _mean(r["detection_events"] for r in group),
            "mean_confirmed_compromises": _mean(r["confirmed_compromises"] for r in group),
            "mean_first_defender_decision_step": _mean_defined(
                r["first_defender_decision_step"] for r in group
            ),
            "mean_compromised": _mean(r["compromised"] for r in group),
            "mean_discovered": _mean(r["discovered"] for r in group),
            "mean_defender_interventions": _mean(
                r["defender_interventions"] for r in group
            ),
            "stall_rate": _mean(r["stalled"] for r in group),
            "containment_rate": _mean(
                r["outcome"] == "contained_by_defender" for r in group
            ),
            "mean_time_to_detection": _mean_defined(
                r["time_to_detection"] for r in group
            ),
            "mean_containment_step": _mean_defined(
                r["containment_step"] for r in group
            ),
            "mean_budget_spent": _mean(r["budget_spent"] for r in group),
            "outcomes": outcomes,
        }
    return summary


def _compact(row):
    return {key: row[key] for key in CSV_FIELDS}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_evaluation(
    agent_names=("random", "greedy", "oracle"),
    seeds=range(10),
    defender_mode: str = "off",
    topology_family: str = "enterprise",
    stochastic: bool = True,
    config: ArenaConfig | None = None,
):
    """
    Evaluate agents and return {"config", "summary", "episodes"}.

    The returned config identifies the experiment, so results can never be
    confused with a run made under different conditions.
    """
    agent_names = tuple(agent_names)
    seeds = list(seeds)

    unknown = set(agent_names) - ALL_AGENTS
    if unknown:
        raise ValueError(f"Unknown attacker(s): {sorted(unknown)}")
    if not valid_defender_mode(defender_mode):
        raise ValueError(f"Unknown defender mode: {defender_mode}")
    if topology_family not in TOPOLOGY_CHOICES:
        raise ValueError(f"Unknown topology family: {topology_family}")
    if not seeds:
        raise ValueError("At least one seed is required")

    base = replace(config or ArenaConfig(), topology_family=topology_family)

    baseline_cache: dict = {}

    def _baseline(name, seed):
        if defender_mode == "off":
            return None
        key = (name, seed)
        if key not in baseline_cache:
            baseline_cache[key] = run_episode(
                name, seed, config=base,
                defender_mode="off", stochastic=stochastic,
            )["success"]
        return baseline_cache[key]

    rows = [
        run_episode(
            name, seed, config=base,
            defender_mode=defender_mode, stochastic=stochastic,
            baseline_success=_baseline(name, seed),
        )
        for name in agent_names
        for seed in seeds
    ]

    return {
        "config": {
            "agents": list(agent_names),
            "seed_start": min(seeds),
            "seed_end": max(seeds),
            "episodes_per_agent": len(seeds),
            "defender_mode": defender_mode,
            "topology_family": topology_family,
            "stochastic": stochastic,
            "min_hosts": base.min_hosts,
            "max_hosts": base.max_hosts,
            "max_steps": base.max_steps,
            "reward_mode": base.reward_mode,
            "defender_budget": base.defender_budget,
            "defender_interval": base.defender_interval,
            "defender_visibility": base.defender_visibility,
            "defender_first_response": "pre_action_step_0",
            "initial_access_alert": "potential_alert_v1",
            "confirmed_compromise_telemetry": True,
            "stall_window": base.stall_window,
        },
        "summary": summarize(rows, agent_names),
        "episodes": [_compact(r) for r in rows],
    }


def paired_defender_effect(
    agent_name: str,
    seeds,
    defender_mode: str,
    config: ArenaConfig | None = None,
    stochastic: bool = True,
):
    """
    Causal, matched-seed measure of what a defender adds.

    Each seed is run twice with the same attacker and topology: once with no
    defender and once with `defender_mode`. 'prevented' means the attacker
    wins without the defender and loses with it, so containment is attributed
    to the defender rather than to an attacker that merely got stuck.
    Test significance on the discordant pairs (prevented vs harmed) with an
    exact McNemar / sign test.
    """
    seeds = list(seeds)
    prevented = harmed = both_win = both_lose = 0
    for s in seeds:
        base = run_episode(agent_name, s, config=config,
                           defender_mode="off", stochastic=stochastic)
        test = run_episode(agent_name, s, config=config,
                           defender_mode=defender_mode, stochastic=stochastic)
        b, t = base["success"], test["success"]
        prevented += int(b == 1 and t == 0)
        harmed += int(b == 0 and t == 1)
        both_win += int(b == 1 and t == 1)
        both_lose += int(b == 0 and t == 0)

    n = len(seeds)
    lo, hi = wilson_interval(prevented, n)
    return {
        "agent": agent_name,
        "defender_mode": defender_mode,
        "episodes": n,
        "prevented": prevented,
        "harmed": harmed,
        "both_attacker_win": both_win,
        "both_attacker_lose": both_lose,
        "prevention_rate": prevented / n if n else 0.0,
        "prevention_ci95_low": lo,
        "prevention_ci95_high": hi,
    }


def write_evaluation(result: dict, out: str):
    """Write <out> (episode CSV) and <out>.summary.json (config + summary)."""
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(result["episodes"])
    path.with_suffix(".summary.json").write_text(
        json.dumps(
            {"config": result["config"], "summary": result["summary"]},
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def evaluate(
    agent_names=("random", "greedy", "oracle"),
    seeds=range(10),
    defender_enabled=False,
    out="results/baselines.csv",
    defender_mode: str | None = None,
    topology_family: str = "enterprise",
    stochastic: bool = True,
):
    """
    Backward-compatible wrapper: returns {agent: metrics}.
    `defender_enabled=True` means the rule-based defender.
    """
    if defender_mode is None:
        defender_mode = "rule" if defender_enabled else "off"

    result = run_evaluation(
        agent_names, seeds,
        defender_mode=defender_mode,
        topology_family=topology_family,
        stochastic=stochastic,
    )
    if out:
        write_evaluation(result, out)
    return result["summary"]
