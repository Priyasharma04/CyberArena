"""
Train the tabular Q-learning defender (state v5) against the real simulator.

Differences from the previous trainer
-------------------------------------
* Uses the SAME runtime as the live API / evaluator: live action mask for every
  attacker (incl. PPO), per-seed PPO sampling, stall watchdog, defender cadence.
* The crown-jewel outcome is actually credited. (In the old trainer the +/-6
  terminal reward sat in unreachable code, so the defender never learned from
  winning or losing.)
* Credit is CAUSAL. The "attacker did not win" bonus is paid only if the same
  attacker on the same seed WITHOUT a defender does win (a matched baseline
  run). An attacker that stalls, times out or fails on its own earns the
  defender nothing.
* The defender is asked for a decision every interval even at budget 0, so
  late-episode / budget-exhausted states get value estimates too.
* No direct bonus for isolating or for the attacker triggering alerts (the old
  +1.5 per isolated host and +0.8 per alert taught "isolate ASAP"). Reward is
  based on attacker damage, intervention cost and the final outcome.
* Training seeds start at --seed-start (default 10000) so they never overlap
  the evaluation seeds.
* Optional --visibility alerts trains a defender that only sees alerts.

Usage:
    python train_learned_defender.py --episodes 3000
    python train_learned_defender.py --episodes 3000 --visibility alerts \
        --out models/learned_defender_v5_alerts.json
    python train_learned_defender.py --attackers random,greedy,ppo   # hold out oracle
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import replace
from pathlib import Path

from cyberarena.config import ArenaConfig
from cyberarena.defender import LearnedDefender
from cyberarena.env import CyberArenaEnv
from cyberarena.evaluate import run_episode
from cyberarena.runtime import (
    BASELINE_AGENTS,
    MODEL_PATHS,
    RL_AGENTS,
    StallMonitor,
    create_attacker,
    select_action,
)

# Reward weights (training signal only; never part of the defender's state).
REWARD = {
    "per_new_compromise": -0.30,      # each host the attacker newly compromises
    "per_intervention": -0.10,        # budget is a scarce resource
    "crown_jewel_lost": -10.0,        # attacker reaches the objective
    "prevented_objective": 6.0,       # attacker would have won with no defender, did not
    "final_compromise_fraction": -4.0,  # scaled by compromised / hosts at the end
}


def available_attackers(requested: list[str] | None) -> list[str]:
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


def run_training_episode(defender, attacker_name, seed, cfg, interval,
                         baseline_success=None):
    env = CyberArenaEnv(replace(cfg, seed=int(seed)))
    obs, _ = env.reset(seed=seed)

    attacker = create_attacker(attacker_name, env)
    if attacker_name in RL_AGENTS:
        attacker.set_random_seed(int(seed))

    defender.reset()
    monitor = StallMonitor(env, defender)

    pending = None
    last_compromised = len(env.compromised)
    success = False
    stalled = False
    decisions = 0

    # Match evaluation/live timing: first defender decision at step 0, before
    # the attacker gets its first action. The decision receives subsequent
    # compromise/outcome credit through the normal pending-transition logic.
    initial_decision = defender.respond(env, explore=True)
    decisions += 1
    pending = {
        "state": initial_decision["state"],
        "action": initial_decision["action_index"],
        "reward": (
            REWARD["per_intervention"]
            if initial_decision["action"] != "wait" else 0.0
        ),
    }

    while True:
        terminal = False
        mask = env.live_action_mask()

        if not mask.any():
            stalled, terminal = True, True
        else:
            action = select_action(attacker_name, attacker, obs, env, mask, True)
            obs, _, terminated, truncated, info = env.step(action)

            gained = len(env.compromised) - last_compromised
            last_compromised = len(env.compromised)
            if pending is not None and gained > 0:
                pending["reward"] += REWARD["per_new_compromise"] * gained

            if terminated or truncated:
                success, terminal = bool(info["success"]), True
            else:
                if env.steps % interval == 0:
                    state = defender.state(env)
                    valid = defender.valid_actions(env)
                    if pending is not None:
                        defender.learn(
                            pending["state"], pending["action"], pending["reward"],
                            state, next_valid_actions=valid, terminal=False,
                        )
                    decision = defender.respond(env, explore=True)
                    decisions += 1
                    pending = {
                        "state": decision["state"],
                        "action": decision["action_index"],
                        "reward": (
                            REWARD["per_intervention"]
                            if decision["action"] != "wait" else 0.0
                        ),
                    }
                if monitor.update():
                    stalled, terminal = True, True

        if terminal:
            if pending is not None:
                frac = len(env.compromised) / max(1, env.n_hosts)
                if success:
                    final = REWARD["crown_jewel_lost"]
                else:
                    final = REWARD["final_compromise_fraction"] * frac
                    # Causal credit only: the objective was prevented AND the
                    # no-defender baseline shows the attacker would have won.
                    if baseline_success:
                        final += REWARD["prevented_objective"]
                defender.learn(
                    pending["state"], pending["action"],
                    pending["reward"] + final, None, terminal=True,
                )
            break

    row = {
        "attacker": attacker_name,
        "seed": seed,
        "baseline_success": None if baseline_success is None else int(baseline_success),
        "success": int(success),
        "stalled": int(stalled),
        "steps": env.steps,
        "compromised": len(env.compromised),
        "hosts": env.n_hosts,
        "interventions": len(defender.actions),
        "decisions": decisions,
        "budget_remaining": defender.budget,
        "epsilon": round(defender.epsilon, 5),
        "q_states": len(defender.q_table),
    }
    env.close()
    return row


def train(args):
    random.seed(args.seed_start)
    attackers = available_attackers(args.attackers.split(",") if args.attackers else None)
    if not attackers:
        raise SystemExit("No attackers available.")
    print(f"Training against: {attackers}")

    cfg = ArenaConfig(
        topology_family=args.family,
        defender_budget=args.budget,
        defender_interval=args.interval,
        defender_visibility=args.visibility,
    )

    eps_min = 0.05
    decay = (eps_min / args.epsilon) ** (1.0 / max(1, int(0.6 * args.episodes)))
    defender = LearnedDefender(
        budget=args.budget,
        seed=args.seed_start,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=eps_min,
        epsilon_decay=decay,
        visibility=args.visibility,
        state_version=6,
    )

    rows = []
    for episode in range(args.episodes):
        attacker_name = attackers[episode % len(attackers)]
        seed = args.seed_start + episode
        baseline = None
        if not args.no_counterfactual:
            baseline = bool(run_episode(
                attacker_name, seed, config=cfg, defender_mode="off",
            )["success"])
        row = run_training_episode(
            defender, attacker_name, seed, cfg, args.interval, baseline
        )
        row["episode"] = episode + 1
        rows.append(row)
        defender.decay_exploration()

        if (episode + 1) % 100 == 0:
            recent = rows[-100:]
            print(
                f"Episode {episode + 1}/{args.episodes} | "
                f"attacker success {sum(r['success'] for r in recent) / len(recent):.1%} | "
                f"mean compromised {sum(r['compromised'] for r in recent) / len(recent):.1f} | "
                f"mean interventions {sum(r['interventions'] for r in recent) / len(recent):.2f} | "
                f"Q states {len(defender.q_table)} | eps {defender.epsilon:.3f}"
            )

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "learned_defender_v6_training.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "topology_family": args.family,
        "budget": args.budget,
        "interval": args.interval,
        "visibility": args.visibility,
        "attackers": attackers,
        "seed_start": args.seed_start,
        "seed_end": args.seed_start + args.episodes - 1,
        "episodes": args.episodes,
        "counterfactual_credit": not args.no_counterfactual,
        "patch_semantics": "strong",
        "defender_semantics": "confirmed_compromise_v1",
        "initial_access_alert": "potential_alert_v1",
        "pre_action_response": True,
        "attack_path_aware_patching": True,
        "reward": REWARD,
    }
    defender.save(args.out, metadata=metadata)

    last = rows[-max(1, len(rows) // 5):]
    summary = {
        "episodes": args.episodes,
        "seed_start": args.seed_start,
        "seed_end": args.seed_start + args.episodes - 1,
        "attackers": attackers,
        "topology_family": args.family,
        "budget": args.budget,
        "interval": args.interval,
        "visibility": args.visibility,
        "state_version": 6,
        "patch_semantics": "strong",
        "defender_semantics": "confirmed_compromise_v1",
        "initial_access_alert": "potential_alert_v1",
        "pre_action_response": True,
        "attack_path_aware_patching": True,
        "reward": REWARD,
        "q_table_states": len(defender.q_table),
        "final_epsilon": defender.epsilon,
        "last_20pct_attacker_success": sum(r["success"] for r in last) / len(last),
        "last_20pct_mean_compromised": sum(r["compromised"] for r in last) / len(last),
        "last_20pct_mean_interventions": sum(r["interventions"] for r in last) / len(last),
        "checkpoint": args.out,
        "training_csv": str(csv_path),
    }
    (out_dir / "learned_defender_v6_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nTraining complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed-start", type=int, default=10_000,
                        help="First training seed (keep evaluation seeds outside this range)")
    parser.add_argument("--family", default="mixed")
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--interval", type=int, default=4)
    parser.add_argument("--visibility", choices=["full", "alerts"], default="alerts")
    parser.add_argument("--attackers", default=None,
                        help="Comma list, e.g. random,greedy,ppo (omit to use all available)")
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.92)
    parser.add_argument("--epsilon", type=float, default=0.30)
    parser.add_argument("--no-counterfactual", action="store_true",
                        help="Skip the matched no-defender baseline (faster, but the "
                             "'prevented objective' bonus is then never paid)")
    parser.add_argument("--out", default="models/learned_defender_v6_alerts.json")
    args = parser.parse_args()

    if args.episodes < 1 or args.interval < 1 or args.budget < 0:
        parser.error("episodes/interval must be positive and budget non-negative")
    train(args)
