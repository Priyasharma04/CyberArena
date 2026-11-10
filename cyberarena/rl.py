"""
Optional RL training. Install with: pip install -e '.[rl]'.

CLI (file names match what the API loads from models/):
    python -m cyberarena.rl --algo ppo     --family mixed --steps 30000
    python -m cyberarena.rl --algo gnn_ppo --family mixed --steps 30000
"""
from __future__ import annotations
import numpy as np
import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from .env import CyberArenaEnv, NODE_FEATURES
from .config import ArenaConfig

class GraphFeatureExtractor(BaseFeaturesExtractor):
    """Small message-passing encoder over fixed-size node features and adjacency."""
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, max_nodes: int = 100):
        super().__init__(observation_space, features_dim)
        self.max_nodes = max_nodes
        self.node_dim = NODE_FEATURES
        self.node_encoder = nn.Sequential(nn.Linear(NODE_FEATURES, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.message = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.pool = nn.Sequential(nn.Linear(128, features_dim), nn.ReLU())
    def forward(self, observations: th.Tensor) -> th.Tensor:
        n, f = self.max_nodes, self.node_dim
        nodes = observations[:, :n*f].reshape(-1, n, f)
        adj = observations[:, n*f:].reshape(-1, n, n)
        present = nodes[:, :, 0:1]
        h = self.node_encoder(nodes) * present
        denom = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        neigh = th.bmm(adj, h) / denom
        h = self.message(th.cat([h, neigh], dim=-1)) * present
        # Pool only discovered nodes; zero-padding must not bias graph embeddings.
        count = present.sum(dim=1).clamp(min=1.0)
        mean = h.sum(dim=1) / count
        maximum = h.masked_fill(present == 0, -1e9).max(dim=1).values
        return self.pool(th.cat([mean, maximum], dim=-1))

def _default_out(algo: str, family: str, total_timesteps: int) -> str:
    return f"models/{algo}_{family}_{total_timesteps // 1000}k"


def _mask_fn(env: CyberArenaEnv) -> np.ndarray:
    """
    Train with the SAME mask the live API / evaluator use (legal actions minus
    proven-dead ones), so a trained policy sees identical rules at test time.
    Never return an all-False mask: that makes the policy output NaNs.
    """
    base = env.unwrapped  # wrappers do not forward custom methods on gymnasium 1.x
    mask = base.live_action_mask()
    return mask if mask.any() else base.action_masks()


def make_env(seed: int = 7, topology_family: str = "enterprise") -> CyberArenaEnv:
    return CyberArenaEnv(ArenaConfig(seed=seed, topology_family=topology_family))


def train_dqn(total_timesteps=50_000, seed=7, out=None, topology_family="enterprise"):
    """DQN baseline. NOTE: DQN has no action masking."""
    env = Monitor(make_env(seed, topology_family))
    model = DQN("MlpPolicy", env, learning_rate=1e-4, buffer_size=5_000, learning_starts=2_000,
                batch_size=64, gamma=0.99, train_freq=4, target_update_interval=1_000,
                exploration_fraction=0.25, verbose=1, seed=seed)
    model.learn(total_timesteps=total_timesteps)
    from pathlib import Path
    out = out or _default_out("dqn", topology_family, total_timesteps)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    return model


def train_ppo(total_timesteps=50_000, seed=7, graph_encoder=False, out=None,
              topology_family="enterprise"):
    # ActionMasker exposes action_masks() reliably through SB3's vec-env
    # wrappers (plain attribute forwarding is not guaranteed on gymnasium 1.x).
    # The topology family is now actually passed to the environment.
    env = ActionMasker(Monitor(make_env(seed, topology_family)), _mask_fn)
    kwargs = {"features_extractor_class": GraphFeatureExtractor,
              "features_extractor_kwargs": {"features_dim": 256, "max_nodes": env.unwrapped.config.max_nodes}} if graph_encoder else {}
    model = MaskablePPO("MlpPolicy", env, learning_rate=3e-4, n_steps=1024, batch_size=128,
                        gamma=0.99, gae_lambda=0.95, ent_coef=0.01, policy_kwargs=kwargs,
                        verbose=1, seed=seed)
    model.learn(total_timesteps=total_timesteps)
    from pathlib import Path
    out = out or _default_out("gnn_ppo" if graph_encoder else "ppo", topology_family, total_timesteps)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a CyberArena attacker.")
    parser.add_argument("--algo", choices=["ppo", "gnn_ppo", "dqn"], default="ppo")
    parser.add_argument("--family", default="mixed",
                        choices=["mixed", "enterprise", "erdos_renyi", "watts_strogatz", "barabasi_albert"])
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.algo == "dqn":
        train_dqn(args.steps, args.seed, args.out, args.family)
    else:
        train_ppo(args.steps, args.seed, args.algo == "gnn_ppo", args.out, args.family)
