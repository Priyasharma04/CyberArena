import argparse

from cyberarena.rl import train_dqn, train_ppo

parser = argparse.ArgumentParser(
    description="Train CyberArena reinforcement-learning agents."
)
parser.add_argument(
    "--agent",
    choices=["dqn", "ppo", "gnn-ppo"],
    default="ppo",
)
parser.add_argument("--steps", type=int, default=50000)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--out", default=None)

args = parser.parse_args()

if args.steps <= 0:
    parser.error("--steps must be positive")

default_paths = {
    "dqn": "models/dqn_v2",
    "ppo": "models/ppo_v2",
    "gnn-ppo": "models/gnn_ppo_v2",
}

output = args.out or default_paths[args.agent]

print(f"Agent: {args.agent}")
print(f"Timesteps: {args.steps}")
print(f"Seed: {args.seed}")
print(f"Checkpoint: {output}")

if args.agent == "dqn":
    train_dqn(
        total_timesteps=args.steps,
        seed=args.seed,
        out=output,
    )
else:
    train_ppo(
        total_timesteps=args.steps,
        seed=args.seed,
        graph_encoder=(args.agent == "gnn-ppo"),
        out=output,
    )

print(f"Training finished. Saved checkpoint: {output}.zip")
