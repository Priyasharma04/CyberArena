"""
CyberArena smoke test. Run from the project root:  python scripts/smoke_test.py
Exits non-zero if any check fails. Needs only numpy/networkx/gymnasium (no RL stack).
"""
import itertools
import sys
from collections import Counter

import networkx as nx

from cyberarena.config import ArenaConfig
from cyberarena.env import CyberArenaEnv
from cyberarena.evaluate import run_episode, run_evaluation
from cyberarena.topology import TOPOLOGY_FAMILIES

failures = 0


def check(condition, message):
    global failures
    print(("PASS  " if condition else "FAIL  ") + message)
    failures += 0 if condition else 1


# 1. reset() without a seed must give a NEW episode (RL trains on this)
env = CyberArenaEnv(ArenaConfig())
seen = set()
for _ in range(8):
    env.reset()
    seen.add((env.n_hosts, env.crown_jewel, tuple(sorted(env.graph.edges))[:6]))
check(len(seen) >= 6, f"reset() without seed gives varied episodes ({len(seen)}/8)")

# 2. crown jewel always reachable (firewalls never disconnect the graph)
bad = 0
for family in TOPOLOGY_FAMILIES:
    for seed in range(50):
        e = CyberArenaEnv(ArenaConfig(topology_family=family))
        e.reset(seed=seed)
        g = nx.Graph()
        g.add_nodes_from(e.graph.nodes)
        g.add_edges_from((u, v) for u, v, d in e.graph.edges(data=True) if not d["firewall"])
        bad += not nx.is_connected(g)
check(bad == 0, f"no unwinnable topologies ({bad} found)")

# 3. partial observability: crown jewel (feature 7) and honeypots (feature 5) hidden
e = CyberArenaEnv(ArenaConfig())
e.reset(seed=1)
e.graph.nodes[0]["honeypot"] = True
feats = e._obs()[: 100 * 10].reshape(100, 10)
check(feats[:, 7].sum() == 0 and feats[:, 5].sum() == 0, "observation hides crown jewel + honeypots")

# 4. every agent x defender x topology: no dead-action loops, deterministic per seed
dead = nondeterministic = runs = 0
for family, mode, agent in itertools.product(
    (*TOPOLOGY_FAMILIES, "mixed"), ("off", "rule"), ("random", "greedy", "oracle")
):
    cfg = ArenaConfig(topology_family=family)
    for seed in (0, 1):
        a = run_episode(agent, seed, config=cfg, defender_mode=mode)
        b = run_episode(agent, seed, config=cfg, defender_mode=mode)
        runs += 1
        nondeterministic += (a["steps"], a["reward"], a["success"]) != (b["steps"], b["reward"], b["success"])
        counts = Counter(
            (l["action"], l["host"]) for l in a["logs"]
            if l["result"] in ("exfiltration_attempt_failed", "control_blocked", "privilege_escalation_blocked")
        )
        dead += sum(v - 1 for v in counts.values() if v > 1)
check(dead == 0, f"{runs} runs: no repeated dead actions")
check(nondeterministic == 0, f"{runs} runs: same seed -> same result")

# 5. evaluation reports its own configuration (live/offline comparability)
res = run_evaluation(("greedy", "oracle"), range(5), defender_mode="rule", topology_family="mixed")
check(res["config"]["defender_mode"] == "rule" and res["config"]["topology_family"] == "mixed",
      "evaluation echoes defender_mode + topology_family")
check(all("success_ci95_low" in m for m in res["summary"].values()), "summary includes confidence intervals")

print("\nALL CHECKS PASSED" if not failures else f"\n{failures} CHECK(S) FAILED")
sys.exit(1 if failures else 0)
