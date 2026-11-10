import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from cyberarena.config import ArenaConfig
from cyberarena.env import CyberArenaEnv, ACTION_TYPES

MODEL_PATH = "models/ppo_v3_30k.zip"
SEEDS = range(1000, 1020)


def evaluate_mode(model, deterministic):
    rows = []
    action_counts = Counter()
    action_type_counts = Counter()

    for seed in SEEDS:
        env = CyberArenaEnv(ArenaConfig())
        obs, _ = env.reset(seed=seed)
        total_reward = 0.0
        invalid = 0

        while True:
            mask = env.action_masks()
            action, _ = model.predict(
                obs,
                deterministic=deterministic,
                action_masks=mask,
            )
            action = int(np.asarray(action).item())

            if action < 0 or action >= len(mask) or not mask[action]:
                invalid += 1

            kind_idx, node = divmod(action, env.config.max_nodes)
            kind = (
                ACTION_TYPES[kind_idx]
                if kind_idx < len(ACTION_TYPES)
                else "invalid"
            )
            action_counts[kind] += 1
            action_type_counts[kind] += 1

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

            if terminated or truncated:
                break

        rows.append({
            "seed": seed,
            "success": int(info["success"]),
            "steps": env.steps,
            "reward": total_reward,
            "compromised": len(env.compromised),
            "discovered": len(env.discovered),
            "invalid_actions": invalid,
        })
        env.close()

    return {
        "mode": "deterministic" if deterministic else "stochastic",
        "success_rate": statistics.mean(x["success"] for x in rows),
        "mean_steps": statistics.mean(x["steps"] for x in rows),
        "mean_reward": statistics.mean(x["reward"] for x in rows),
        "mean_compromised": statistics.mean(x["compromised"] for x in rows),
        "mean_discovered": statistics.mean(x["discovered"] for x in rows),
        "invalid_actions": sum(x["invalid_actions"] for x in rows),
        "action_counts": dict(action_counts),
        "episodes": rows,
    }


def main():
    model = MaskablePPO.load(MODEL_PATH, device="cpu")

    results = [
        evaluate_mode(model, True),
        evaluate_mode(model, False),
    ]

    for result in results:
        print("\n==========", result["mode"].upper(), "==========")
        print(json.dumps(
            {k: v for k, v in result.items() if k != "episodes"},
            indent=2,
        ))

    Path("results/ppo_diagnostic_v2.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print("\nSaved: results/ppo_diagnostic_v2.json")


if __name__ == "__main__":
    main()

