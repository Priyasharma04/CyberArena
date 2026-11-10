from __future__ import annotations

import csv
import json
import random
import statistics
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from cyberarena.env import CyberArenaEnv, ACTION_TYPES
from cyberarena.config import ArenaConfig
from cyberarena.baselines import RandomAgent, GreedyAgent, FullKnowledgePlanner


SEEDS = range(1000, 1020)
MODEL_PATH = "models/ppo_v3_30k.zip"
OUTPUT_CSV = "results/matched_agent_comparison.csv"
OUTPUT_JSON = "results/matched_agent_comparison.summary.json"


def run_baseline(name, seed):
    env = CyberArenaEnv(ArenaConfig(), render_mode="ansi")
    obs, _ = env.reset(seed=seed)

    agent_class = {
        "random": RandomAgent,
        "greedy": GreedyAgent,
        "oracle": FullKnowledgePlanner,
    }[name]
    agent = agent_class(env)

    return run_episode(env, obs, lambda current_obs: agent.act(current_obs), name, seed)


def run_ppo(seed, deterministic):
    env = CyberArenaEnv(ArenaConfig(), render_mode="ansi")
    obs, _ = env.reset(seed=seed)

    # Make stochastic action sampling reproducible for this seed.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = run_ppo.model
    mode = "ppo_deterministic" if deterministic else "ppo_stochastic"

    def choose_action(current_obs):
        action, _ = model.predict(
            current_obs,
            deterministic=deterministic,
            action_masks=env.action_masks(),
        )
        return int(action)

    return run_episode(env, obs, choose_action, mode, seed)


def run_episode(env, obs, choose_action, name, seed):
    total_reward = 0.0
    action_counts = {kind: 0 for kind in ACTION_TYPES}

    while True:
        action = choose_action(obs)
        action_type = action // env.config.max_nodes
        if 0 <= action_type < len(ACTION_TYPES):
            action_counts[ACTION_TYPES[action_type]] += 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            return {
                "agent": name,
                "seed": seed,
                "success": int(bool(info.get("success", False))),
                "steps": env.steps,
                "reward": round(total_reward, 5),
                "compromised": len(env.compromised),
                "discovered": len(env.discovered),
                "detected": int(bool(env.detected)),
                "invalid_actions": sum(
                    1 for entry in env.logs
                    if entry.get("result") in ("invalid_action", "invalid")
                ),
                "action_counts": json.dumps(action_counts),
            }


def summarize(rows):
    summary = {}
    for name in sorted(set(row["agent"] for row in rows)):
        group = [row for row in rows if row["agent"] == name]
        summary[name] = {}

        for metric in (
            "success", "steps", "reward",
            "compromised", "discovered", "detected", "invalid_actions"
        ):
            values = [float(row[metric]) for row in group]
            summary[name][metric] = {
                "mean": round(statistics.mean(values), 4),
                "std": round(statistics.pstdev(values), 4),
            }

    return summary


def main():
    print("Loading PPO checkpoint:", MODEL_PATH)
    run_ppo.model = MaskablePPO.load(MODEL_PATH)

    rows = []

    for seed in SEEDS:
        print(f"Evaluating seed {seed}...")
        for name in ("random", "greedy", "oracle"):
            rows.append(run_baseline(name, seed))

        rows.append(run_ppo(seed, deterministic=False))
        rows.append(run_ppo(seed, deterministic=True))

    Path("results").mkdir(exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    Path(OUTPUT_JSON).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n========== MATCHED 20-SEED RESULTS ==========")
    print(json.dumps(summary, indent=2))
    print("\nSaved CSV:", OUTPUT_CSV)
    print("Saved summary:", OUTPUT_JSON)


if __name__ == "__main__":
    main()
