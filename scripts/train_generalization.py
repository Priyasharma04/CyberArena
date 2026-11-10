import argparse
from cyberarena.rl import train_ppo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.steps < 1024:
        parser.error("--steps must be at least 1024")

    print("Experiment: Standard PPO")
    print("Training distribution: Enterprise + Erdős–Rényi")
    print("Graph encoder: Disabled")
    print("Timesteps:", args.steps)
    print("Seed:", args.seed)
    print("Checkpoint: models/ppo_mixed_30k")

    train_ppo(
        total_timesteps=args.steps,
        seed=args.seed,
        graph_encoder=False,
        out="models/ppo_mixed_30k",
        topology_family="mixed",
    )

    print("Standard mixed-topology PPO training completed.")


if __name__ == "__main__":
    main()
