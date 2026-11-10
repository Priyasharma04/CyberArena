"""Small, reproducible baseline studies for topology and reward ablations."""
from __future__ import annotations

import csv
from pathlib import Path

from .config import ArenaConfig
from .evaluate import run_episode

FIELDS_GEN = ("split", "topology_size", "topology_family", "defender_mode", "agent",
              "seed", "success", "steps", "reward")


def _write(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def topology_generalization(train_sizes=(10, 20, 30), held_out_sizes=(50, 75, 100), seeds=range(10),
                            out="results/generalization.csv", topology_family="enterprise",
                            defender_mode="off", agents=("greedy", "oracle")):
    rows = []
    for split, sizes in (("train_range", train_sizes), ("held_out", held_out_sizes)):
        for size in sizes:
            cfg = ArenaConfig(min_hosts=size, max_hosts=size, max_nodes=100, max_steps=300,
                              topology_family=topology_family)
            for agent in agents:
                for seed in seeds:
                    result = run_episode(agent, seed, config=cfg, defender_mode=defender_mode)
                    rows.append({"split": split, "topology_size": size,
                                 "topology_family": topology_family, "defender_mode": defender_mode,
                                 "agent": agent, "seed": seed, "success": result["success"],
                                 "steps": result["steps"], "reward": result["reward"]})
    return _write(rows, out)


def reward_ablation(seeds=range(10), topology_size=20, out="results/reward_ablation.csv",
                    topology_family="enterprise", defender_mode="off", agents=("greedy", "oracle")):
    rows = []
    for mode in ("sparse", "shaped", "potential"):
        cfg = ArenaConfig(min_hosts=topology_size, max_hosts=topology_size, max_nodes=100,
                          max_steps=200, reward_mode=mode, topology_family=topology_family)
        for agent in agents:
            for seed in seeds:
                result = run_episode(agent, seed, config=cfg, defender_mode=defender_mode)
                rows.append({"reward_mode": mode, "topology_family": topology_family,
                             "defender_mode": defender_mode, "agent": agent, "seed": seed,
                             "success": result["success"], "steps": result["steps"],
                             "reward": result["reward"]})
    return _write(rows, out)


def topology_family_comparison(seeds=range(20), defender_mode="off", agents=("random", "greedy", "oracle"),
                               out="results/topology_families.csv"):
    """Success of each attacker on every topology family (research question: do
    graph structures such as hubs or small-world shortcuts change difficulty?)."""
    from .topology import TOPOLOGY_FAMILIES
    rows = []
    for family in TOPOLOGY_FAMILIES:
        cfg = ArenaConfig(topology_family=family)
        for agent in agents:
            for seed in seeds:
                result = run_episode(agent, seed, config=cfg, defender_mode=defender_mode)
                rows.append({"topology_family": family, "defender_mode": defender_mode, "agent": agent,
                             "seed": seed, "success": result["success"], "steps": result["steps"],
                             "reward": result["reward"], "detected": result["detected"]})
    return _write(rows, out)
