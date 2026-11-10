from __future__ import annotations

import numpy as np

from .env import ACTION_TYPES


def _usable_mask(env, action_mask):
    """
    Resolve the action mask an agent should respect.

    Default is env.live_action_mask(): legal actions minus actions that have
    already proven unproductive in this episode. This is the single source of
    truth for failure memory, so every agent (and the live API / evaluator)
    behaves identically. If the attacker is genuinely stuck the live mask is
    empty; callers that want to detect that should check it before acting.
    """
    if action_mask is not None:
        return np.asarray(action_mask, dtype=bool).copy()

    mask = env.live_action_mask()
    if mask.any():
        return mask
    return env.action_masks()


class RandomAgent:
    def __init__(self, env):
        self.env = env

    def act(self, obs=None, action_mask=None):
        mask = _usable_mask(self.env, action_mask)
        valid = np.flatnonzero(mask)
        return int(self.env.np_random.choice(valid))


class GreedyAgent:
    """
    Partial-observation heuristic.
    Uses discovered node features and the shared live action mask only.
    """

    def __init__(self, env):
        self.env = env

    def _action(self, kind, node):
        return ACTION_TYPES.index(kind) * self.env.config.max_nodes + node

    def _features(self, obs):
        n = self.env.config.max_nodes
        return np.asarray(obs[:n * 10]).reshape(n, 10)

    def act(self, obs=None, action_mask=None):
        env = self.env
        if obs is None:
            obs = env._obs()

        mask = _usable_mask(env, action_mask)
        features = self._features(obs)

        # 1. Try exfiltration from privileged hosts. A host whose exfiltration
        #    already failed is retired by the live mask, so no private memory
        #    is needed here.
        for node in sorted(env.privileged):
            action = self._action("exfiltrate", node)
            if mask[action]:
                return action

        # 2. Escalate compromised hosts.
        candidates = []
        for node in env.compromised:
            action = self._action("escalate", node)
            if mask[action]:
                score = features[node, 9] + features[node, 6]
                candidates.append((score, node))

        if candidates:
            _, node = max(candidates)
            return self._action("escalate", node)

        # 3. Exploit discovered hosts, preferring vulnerable/weak-credential hosts.
        candidates = []
        for node in env.discovered:
            action = self._action("exploit", node)
            if mask[action]:
                score = (
                    2.0 * features[node, 9]
                    + features[node, 8]
                    + 0.25 * features[node, 6]
                )
                candidates.append((score, node))

        if candidates:
            _, node = max(candidates)
            return self._action("exploit", node)

        # 4. Scan compromised hosts while unexplored reachable hosts remain.
        n = env.config.max_nodes
        scan_candidates = []
        for node in sorted(env.compromised):
            action = self._action("scan", node)
            if mask[action]:
                known_degree = sum(
                    1 for other in env.discovered
                    if obs[n * 10 + node * n + other] > 0
                )
                scan_candidates.append((known_degree, node))

        if scan_candidates:
            _, node = max(scan_candidates)
            return self._action("scan", node)

        # 5. Move laterally from a compromised host when possible.
        for node in sorted(env.compromised):
            action = self._action("move", node)
            if mask[action]:
                return action

        # Final fallback: first remaining legal action, else the
        # environment's never-empty legal mask.
        valid = np.flatnonzero(mask)
        if valid.size:
            return int(valid[0])

        base_valid = np.flatnonzero(env.action_masks())
        if base_valid.size:
            return int(base_valid[0])

        return 0


class FullKnowledgePlanner:
    """
    Adaptive full-knowledge shortest-path reference planner.

    This is an upper-bound reference, NOT a fair partial-observation
    attacker: it knows the crown-jewel identity and the whole graph.

    Re-plans on every call, so it reacts to patching and isolation (hosts
    behind a control are routed around). If the planned route cannot be acted
    on it tries lateral movement, then falls back to the greedy heuristic
    over the same mask.
    """

    def __init__(self, env):
        self.env = env
        self._fallback = GreedyAgent(env)

    def _action(self, kind, node):
        return ACTION_TYPES.index(kind) * self.env.config.max_nodes + node

    def act(self, obs=None, action_mask=None):
        import networkx as nx

        env = self.env
        if obs is None:
            obs = env._obs()
        mask = _usable_mask(env, action_mask)
        n = env.config.max_nodes

        def traversable(node):
            data = env.graph.nodes[node]
            # Isolated hosts stay in env.compromised but cannot be used.
            if data.get("isolated", False):
                return False
            if node == 0 or node in env.compromised:
                return True
            return not data.get("patched", False)

        target = env.crown_jewel
        try:
            safe_graph = nx.subgraph_view(
                env.graph,
                filter_node=traversable,
                filter_edge=lambda u, v: not env.graph.edges[u, v].get(
                    "firewall", False
                ),
            )
            path = nx.shortest_path(safe_graph, 0, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return self._fallback.act(obs, action_mask=mask)

        for u, v in zip(path, path[1:]):
            if v not in env.discovered:
                action = self._action("scan", u)
                if mask[action]:
                    return action
                break
            if v not in env.compromised:
                action = self._action("exploit", v)
                if mask[action]:
                    return action
                action = self._action("move", u)
                if mask[action]:
                    return action
                break

        if target in env.compromised:
            if target not in env.privileged:
                action = self._action("escalate", target)
                if mask[action]:
                    return action
            else:
                action = self._action("exfiltrate", target)
                if mask[action]:
                    return action

        return self._fallback.act(obs, action_mask=mask)
