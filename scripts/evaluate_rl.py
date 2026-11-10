import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from sb3_contrib import MaskablePPO

from cyberarena.config import ArenaConfig
from cyberarena.env import CyberArenaEnv


def load_agent(agent_type, model_path, env):
    if agent_type == "dqn":
        return DQN.load(model_path, env=env, device="cpu")
    return MaskablePPO.load(model_path, env=env, device="cpu")


def run_episode(model, agent_type, env, seed, deterministic=True):
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    invalid_actions = 0
    action_count = 0

    while True:
        mask = env.action_masks()

        if agent_type in ("ppo", "gnn_ppo"):
            action, _ = model.predict(
                obs,
                deterministic=deterministic,
                action_masks=mask,
            )
        else:
            action, _ = model.predict(obs, deterministic=True)

        action = int(np.asarray(action).item())

        if action < 0 or action >= len(mask) or not mask[action]:
            invalid_actions += 1

        action_count += 1
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        if terminated or truncated:
            break

    return {
        "agent": agent_type,
        "seed": seed,
        "success": int(info["success"]),
        "steps": env.steps,
        "reward": round(total_reward, 6),
        "compromised": len(env.compromised),
        "discovered": len(env.discovered),
        "detected": int(env.detected),
        "invalid_actions": invalid_actions,
        "action_count": action_count,
        "invalid_action_rate": invalid_actions / max(1, action_count),
    }


def summarize(rows):
    metrics = (
        "success", "steps", "reward", "compromised",
        "discovered", "detected", "invalid_action_rate",
    )
    output = {}

    for agent in sorted({row["agent"] for row in rows}):
        group = [row for row in rows if row["agent"] == agent]
        output[agent] = {"episodes": len(group)}

        for metric in metrics:
            values = [float(row[metric]) for row in group]
            output[agent][metric + "_mean"] = round(
                statistics.mean(values), 6
            )
            output[agent][metric + "_std"] = round(
                statistics.stdev(values) if len(values) > 1 else 0.0, 6
            )

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained CyberArena RL agents on held-out seeds."
    )
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(range(1000, 1020)))
    parser.add_argument("--topology-family",
                        choices=["enterprise", "erdos_renyi", "watts_strogatz", "barabasi_albert", "mixed"],
                        default="enterprise")
    parser.add_argument("--stochastic", action="store_true",
                        help="Evaluate PPO with stochastic actions")
    parser.add_argument("--ppo", default="models/ppo_v2.zip")
    parser.add_argument("--gnn-ppo", default="models/gnn_ppo_v2.zip")
    parser.add_argument("--dqn", default="models/dqn_v2.zip")
    parser.add_argument("--output", default="results/rl_evaluation_v2")
    args = parser.parse_args()

    specifications = [
        ("ppo", args.ppo),
        ("gnn_ppo", args.gnn_ppo),
        ("dqn", args.dqn),
    ]

    available = [
        (name, path) for name, path in specifications
        if Path(path).exists()
    ]

    if not available:
        raise FileNotFoundError(
            "No trained model files found. Train an agent first."
        )

    rows = []

    for agent_type, model_path in available:
        print(f"\nEvaluating {agent_type}: {model_path}")
        env = CyberArenaEnv(ArenaConfig(topology_family=args.topology_family))
        model = load_agent(agent_type, model_path, env)

        for seed in args.seeds:
            result = run_episode(model, agent_type, env, seed,
                             deterministic=not args.stochastic)
            rows.append(result)

        env.close()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = output_path.with_suffix(".csv")
    json_path = output_path.with_suffix(".summary.json")

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n========== HELD-OUT EVALUATION ==========")
    print(json.dumps(summary, indent=2))
    print(f"\nDetailed results: {csv_path}")
    print(f"Summary: {json_path}")


if __name__ == "__main__":
    main()
