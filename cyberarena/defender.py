from __future__ import annotations

import json
import random
from bisect import bisect_right
from collections import deque
from pathlib import Path
from typing import Any


ACTIONS = (
    "isolate",
    "patch",
    "rotate_credentials",
    "deploy_honeypot",
    "wait",
)

VISIBILITIES = ("full", "alerts")

# ---------------------------------------------------------------------------
# Information model
#
# visibility="full"   (legacy) the defender reads ground truth: which hosts are
#                     compromised/privileged and which hosts the attacker has
#                     discovered. Kept so earlier results stay reproducible.
#
# visibility="alerts" the defender may use ONLY:
#     * its own asset inventory and static topology/metadata (env.graph),
#     * its own controls (isolated / patched / honeypot / rotated flags),
#     * detection telemetry: env.detected, env.detected_hosts,
#       env.confirmed_compromised_hosts, env.risk_scores, env.detection_events,
#     * env.steps / env.n_hosts.
#   It must NOT read env.compromised, env.privileged or env.discovered.
#   tests/test_defender_visibility.py enforces this with a guarded env.
#
# Consequences of acting on hidden state in alerts mode (resolved by the
# simulator, never visible to the defender):
#     isolate  on a host that is not compromised  -> budget spent, no effect
#     patch    on a secretly compromised host     -> applies; the attacker
#                                                    remains present, but the
#                                                    host blocks escalation
#                                                    and lateral entry
#     honeypot on a secretly compromised host     -> budget spent, no decoy
#                                                    is created (a live host
#                                                    cannot be turned into one)
# ---------------------------------------------------------------------------


def _host_risk(env, node: int, visibility: str = "full") -> float:
    data = env.graph.nodes[node]
    score = float(getattr(env, "risk_scores", {}).get(node, 0.0))

    if visibility == "full":
        if node in env.compromised:
            score += 3.0
        if node in env.privileged:
            score += 2.0
    elif node in env.detected_hosts:
        score += 3.0  # an alert was raised on this host

    if data.get("credential_posture") == "weak":
        score += 1.0
    if data.get("services") and "database" in data["services"]:
        score += 1.0
    score += 0.5 * len(data.get("vulnerabilities", []))

    return score


def _bucket(value: int, edges: tuple) -> int:
    """Index of the first threshold above `value` (edges ascending)."""
    return bisect_right(edges, int(value))


def _inventory(env, visibility: str) -> list[int]:
    """Hosts the defender can reason about."""
    if visibility == "full":
        return sorted(env.discovered)           # legacy: attacker's knowledge
    return list(range(env.n_hosts))             # alerts: its own asset inventory


def _suspected(env, visibility: str) -> set[int]:
    """Hosts treated as already lost to the attacker."""
    if visibility == "full":
        return set(env.compromised)             # legacy ground truth
    return set(env.detected_hosts)              # alerts only


def _known_degree(env, node: int, visibility: str = "full") -> int:
    if visibility == "full":
        return sum(1 for nb in env.graph.neighbors(node) if nb in env.discovered)
    return sum(1 for _ in env.graph.neighbors(node))   # static topology


def _active_targets(
    env, visibility: str = "full", include_initial: bool = False,
) -> list[int]:
    """Hosts eligible for isolation under the selected observation model.

    Full visibility uses simulator ground truth. Alerts visibility uses only
    the explicit confirmed-compromise telemetry set, not every suspicious alert.
    """
    pool = (
        env.compromised
        if visibility == "full"
        else getattr(env, "confirmed_compromised_hosts", set())
    )
    return [
        n for n in pool
        if (visibility == "alerts" or include_initial or n != 0)
        and not env.graph.nodes[n].get("isolated", False)
    ]


def _exposed_hosts(env, footholds, visibility: str = "full") -> int:
    """
    Attack surface a patch could still close: unpatched, non-isolated hosts
    one open edge away from an active foothold.

    full   : footholds = compromised hosts (+ entry host 0); only discovered,
             uncompromised hosts count (legacy).
    alerts : footholds = alerted hosts; no ground truth is consulted.
    """
    exposed: set[int] = set()
    if visibility == "full":
        sources = set(footholds) | {0}
    else:
        # Confirmed footholds are preferred. Before confirmation, the initial
        # access alert still gives the patch selector a plausible source.
        sources = set(footholds) or set(env.detected_hosts)
    alerted = env.detected_hosts if visibility == "alerts" else set()

    for src in sources:
        if env.graph.nodes[src].get("isolated", False):
            continue
        for nb in env.graph.neighbors(src):
            data = env.graph.nodes[nb]
            if data.get("patched", False) or data.get("isolated", False):
                continue
            if env.graph.edges[src, nb].get("firewall", False):
                continue
            if visibility == "full":
                if nb in env.discovered and nb not in env.compromised:
                    exposed.add(nb)
            elif nb not in alerted:
                exposed.add(nb)
    return len(exposed)


def _eligible(
    env, action: str, visibility: str, include_initial: bool = False,
) -> list[int]:
    """Candidate hosts for an action, using only what `visibility` allows."""
    if action == "isolate":
        return _active_targets(env, visibility, include_initial=include_initial)

    inventory = _inventory(env, visibility)
    suspected = _suspected(env, visibility)
    nodes = env.graph.nodes

    if action == "patch":
        return [
            n for n in inventory
            if n not in suspected
            and not nodes[n].get("patched", False)
            and not nodes[n].get("isolated", False)
        ]
    if action == "rotate_credentials":
        return [
            n for n in inventory
            if nodes[n].get("credential_posture") == "weak"
            and not nodes[n].get("credential_rotated", False)
            and not nodes[n].get("isolated", False)
        ]
    if action == "deploy_honeypot":
        return [
            n for n in inventory
            if n not in suspected
            and not nodes[n].get("honeypot", False)
            and not nodes[n].get("isolated", False)
        ]
    return []


def _apply_action(env, action: str, host: int | None, visibility: str = "full") -> bool:
    """
    Resolve one defensive intervention inside the abstract simulator.

    Returns True if the budget is consumed. In alerts mode this never returns
    False because of hidden ground truth, so the defender's control flow cannot
    leak it. (Ground truth is read here because this is the simulator
    resolving the action, not the defender deciding.)
    """
    if action == "wait" or host is None:
        return True

    data = env.graph.nodes[host]

    if action == "isolate":
        if data.get("isolated"):
            return False
        if visibility == "full" and host not in env.compromised:
            return False
        if visibility == "alerts" and host not in getattr(
            env, "confirmed_compromised_hosts", set()
        ):
            # Suspicious alerts (including failed exploits) are not enough to
            # justify taking a host offline.
            return False
        data["isolated"] = True      # alerts mode: false positive wastes budget
        return True

    if action == "patch":
        if data.get("patched") or data.get("isolated"):
            return False
        data["patched"] = True
        return True

    if action == "rotate_credentials":
        if data.get("credential_posture") != "weak" or data.get("credential_rotated"):
            return False
        data["credential_rotated"] = True
        return True

    if action == "deploy_honeypot":
        if data.get("honeypot") or data.get("isolated"):
            return False
        if host in env.compromised:
            # A live, attacker-held host cannot become a decoy.
            # full: refused (legacy). alerts: budget spent, nothing happens.
            return visibility == "alerts"
        data["honeypot"] = True
        return True

    return False


class BudgetedDefender:
    """Transparent rule-based defender with a limited intervention budget."""

    def __init__(self, budget: int = 5, seed: int = 0, visibility: str = "full"):
        if budget < 0:
            raise ValueError("Defender budget cannot be negative.")
        if visibility not in VISIBILITIES:
            raise ValueError(f"visibility must be one of {VISIBILITIES}")
        self.initial_budget = int(budget)
        self.visibility = visibility
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        self.budget = self.initial_budget
        self.actions: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []
        self.budget_spent = 0

    def _risk(self, env, node: int) -> float:
        return _host_risk(env, node, self.visibility)

    def _execute(self, env, action: str, host: int | None, reason: str):
        before = self.budget
        if action != "wait":
            if self.budget <= 0 or not _apply_action(env, action, host, self.visibility):
                return None
            self.budget -= 1
            self.budget_spent += 1

        record = {
            "step": int(env.steps),
            "action": action,
            "host": host,
            "reason": reason,
            "budget_before": before,
            "budget_after": self.budget,
            "risk_score": round(self._risk(env, host), 2) if host is not None else 0.0,
        }
        self.decisions.append(record)
        if action != "wait":
            self.actions.append(record)
        return record

    def respond(self, env):
        if self.budget <= 0:
            return None
        vis = self.visibility

        active = _eligible(env, "isolate", vis)
        if active:
            host = max(active, key=lambda n: (
                self._risk(env, n),
                n in env.privileged if vis == "full" else False,
            ))
            return self._execute(
                env, "isolate", host,
                "Contain the highest-risk active compromised host."
                if vis == "full" else "Contain the highest-risk alerted host."
            )

        patchable = _eligible(env, "patch", vis)
        if patchable:
            host = max(patchable, key=lambda n: self._risk(env, n))
            return self._execute(
                env, "patch", host,
                "Patch the host with the highest estimated exposure."
            )

        weak = _eligible(env, "rotate_credentials", vis)
        if weak:
            host = max(weak, key=lambda n: self._risk(env, n))
            return self._execute(
                env, "rotate_credentials", host,
                "Rotate weak credentials on a known host."
            )

        decoys = _eligible(env, "deploy_honeypot", vis)
        if decoys:
            host = max(decoys, key=lambda n: (
                _known_degree(env, n, vis),
                self._risk(env, n),
            ))
            return self._execute(
                env, "deploy_honeypot", host,
                "Place a decoy on a known host."
            )

        return self._execute(env, "wait", None, "No eligible intervention.")


class LearnedDefender:
    """
    Tabular Q-learning defender (chooses an action CATEGORY; the target host
    is picked by a heuristic, see _choose_host).

    state_version 3 : legacy, ground-truth state (visibility="full" only).
    state_version 4 : wider buckets, legacy active footholds, attack-surface exposure.
    state_version 5 : legacy compact state with H0 inclusion.
    state_version 6 : compact state for confirmed-compromise telemetry,
                      pre-action response timing and path-aware patching.
    A checkpoint is only valid for the visibility/state version it was
    trained with (both are stored in the file and checked on load).
    """

    STATE_VERSIONS = (3, 4, 5, 6)
    STATE_LENGTH = {3: 8, 4: 7, 5: 7, 6: 7}
    ACTIONS = ACTIONS

    def __init__(
        self,
        budget: int = 5,
        seed: int = 0,
        alpha: float = 0.15,
        gamma: float = 0.92,
        epsilon: float = 0.25,
        epsilon_min: float = 0.03,
        epsilon_decay: float = 0.995,
        visibility: str = "full",
        state_version: int = 4,
    ):
        if budget < 0:
            raise ValueError("Defender budget cannot be negative.")
        if visibility not in VISIBILITIES:
            raise ValueError(f"visibility must be one of {VISIBILITIES}")
        if state_version not in self.STATE_VERSIONS:
            raise ValueError(f"state_version must be one of {self.STATE_VERSIONS}")
        if state_version == 3 and visibility != "full":
            raise ValueError("state_version 3 reads ground truth; use visibility='full'.")
        self.initial_budget = int(budget)
        self.visibility = visibility
        self.state_version = int(state_version)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.rng = random.Random(seed)
        self.q_table: dict[tuple, list[float]] = {}
        self.metadata: dict[str, Any] = {}
        self.reset()

    def reset(self):
        self.budget = self.initial_budget
        self.actions: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []
        self.budget_spent = 0

    @staticmethod
    def _bin(value: int, cap: int = 5) -> int:
        return min(cap, max(0, int(value)))

    def _risk(self, env, node: int) -> float:
        return _host_risk(env, node, self.visibility)

    # ------------------------------------------------------------- state

    def state(self, env) -> tuple:
        if self.state_version == 3:
            return self._state_v3(env)
        if self.state_version == 4:
            return self._state_v4(env)
        if self.state_version == 5:
            return self._state_v5(env)
        return self._state_v6(env)

    def _state_v4(self, env) -> tuple:
        return self._state_with_foothold_policy(env, include_initial=False)

    def _state_v5(self, env) -> tuple:
        return self._state_with_foothold_policy(env, include_initial=True)

    def _state_v6(self, env) -> tuple:
        # v6 uses confirmed-compromise telemetry and pre-action response timing.
        return self._state_with_foothold_policy(env, include_initial=True)

    def _state_with_foothold_policy(self, env, include_initial: bool) -> tuple:
        vis = self.visibility
        footholds = _active_targets(env, vis, include_initial=include_initial)
        risk = env.risk_scores

        if vis == "full":
            known = env.discovered
            privileged = len([n for n in env.privileged
                              if not env.graph.nodes[n].get("isolated", False)])
            size_edges = (3, 6, 12)
            size = len(known)
        else:
            known = range(env.n_hosts)
            privileged = sum(1 for n in footholds if risk.get(n, 0.0) >= 8.0)
            size_edges = (15, 22, 27)
            size = env.n_hosts

        high_risk = sum(1 for n in known if risk.get(n, 0.0) >= 4.0)

        return (
            _bucket(len(footholds), (1, 2, 3, 5, 9)),   # 0,1,2,3-4,5-8,9+
            _bucket(privileged, (1, 2)),                # 0,1,2+
            int(bool(env.detected)),
            _bucket(size, size_edges),
            _bucket(high_risk, (1, 3)),
            _bucket(_exposed_hosts(env, footholds, vis), (1, 3, 6)),
            _bucket(self.budget, (1, 3, 6)),            # 0,1-2,3-5,6+
        )

    def _state_v3(self, env) -> tuple:
        discovered = list(env.discovered)
        high_risk = sum(
            1 for n in discovered
            if getattr(env, "risk_scores", {}).get(n, 0.0) >= 4.0
        )
        unpatched = sum(
            1 for n in discovered
            if not env.graph.nodes[n].get("patched", False)
        )
        weak = sum(
            1 for n in discovered
            if env.graph.nodes[n].get("credential_posture") == "weak"
            and not env.graph.nodes[n].get("credential_rotated", False)
        )
        return (
            self._bin(len(env.compromised)),
            self._bin(len(env.privileged)),
            int(bool(env.detected)),
            self._bin(len(discovered)),
            self._bin(high_risk),
            self._bin(unpatched),
            self._bin(weak),
            self._bin(self.budget),
        )

    # ----------------------------------------------------------- actions

    def _eligible_hosts(self, env, action: str) -> list[int]:
        return _eligible(
            env, action, self.visibility,
            include_initial=(self.state_version >= 5),
        )

    def valid_actions(self, env) -> list[int]:
        valid = [4]  # wait is always valid
        if self.budget <= 0:
            return valid
        for i, action in enumerate(self.ACTIONS[:4]):
            if self._eligible_hosts(env, action):
                valid.append(i)
        return valid

    @staticmethod
    def _distance_from_sources(env, sources: set[int], target: int) -> int | None:
        """Shortest open-path distance from a known/suspected foothold."""
        if not sources:
            return None
        queue = deque((source, 0) for source in sources)
        visited = set(sources)
        while queue:
            node, distance = queue.popleft()
            if node == target:
                return distance
            for neighbour in env.graph.neighbors(node):
                if neighbour in visited:
                    continue
                edge = env.graph.edges[node, neighbour]
                data = env.graph.nodes[neighbour]
                if edge.get("firewall", False):
                    continue
                if data.get("isolated", False) or data.get("patched", False):
                    continue
                visited.add(neighbour)
                queue.append((neighbour, distance + 1))
        return None

    def _patch_priority(self, env, node: int) -> tuple:
        vis = self.visibility
        if vis == "full":
            sources = set(env.compromised)
        else:
            sources = set(getattr(env, "confirmed_compromised_hosts", set()))
            if not sources:
                sources = set(env.detected_hosts)
        sources = {
            source for source in sources
            if not env.graph.nodes[source].get("isolated", False)
            and not env.graph.nodes[source].get("patched", False)
        }
        distance = self._distance_from_sources(env, sources, node)
        # Prefer hosts close to a foothold; among equal distances, prioritise
        # vulnerability exposure and weak credentials. Crown-jewel identity is
        # intentionally never consulted.
        proximity = -(distance if distance is not None else env.n_hosts + 1)
        data = env.graph.nodes[node]
        return (
            proximity,
            len(data.get("vulnerabilities", [])),
            data.get("credential_posture") == "weak",
            self._risk(env, node),
            -int(node),
        )

    def _choose_host(self, env, action: str) -> int | None:
        candidates = self._eligible_hosts(env, action)
        if not candidates:
            return None
        vis = self.visibility

        if action == "isolate":
            return max(candidates, key=lambda n: (
                self._risk(env, n),
                n in env.privileged if vis == "full" else False,
            ))
        if action == "patch":
            return max(candidates, key=lambda n: self._patch_priority(env, n))
        if action == "rotate_credentials":
            return max(candidates, key=lambda n: self._risk(env, n))
        if action == "deploy_honeypot":
            return max(candidates, key=lambda n: (
                _known_degree(env, n, vis),
                self._risk(env, n),
            ))
        return None

    def _q(self, state: tuple) -> list[float]:
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(self.ACTIONS)
        return self.q_table[state]

    def select_action(self, state: tuple, valid_actions: list[int], explore=True) -> int:
        if not valid_actions:
            return 4
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(valid_actions)
        values = self._q(state)
        best_value = max(values[a] for a in valid_actions)
        # Do not let ACTIONS ordering decide between equally valued actions.
        # Seeded randomness keeps runs reproducible while removing the systematic
        # preference for lower-index actions in unseen or weakly learned states.
        best_actions = [a for a in valid_actions if values[a] == best_value]
        return self.rng.choice(best_actions)

    def learn(
        self,
        state: tuple,
        action_index: int,
        reward: float,
        next_state: tuple | None,
        next_valid_actions: list[int] | None = None,
        terminal: bool = False,
    ) -> None:
        current = self._q(state)
        target = float(reward)

        if not terminal and next_state is not None:
            valid = next_valid_actions or [4]
            next_values = self._q(next_state)
            target += self.gamma * max(next_values[a] for a in valid)

        current[action_index] += self.alpha * (target - current[action_index])

    def respond(self, env, explore: bool = True):
        state = self.state(env)
        valid = self.valid_actions(env)

        # Safety override: once telemetry confirms a compromised host, isolate
        # it before allowing the learned policy to spend a decision on a lower-
        # urgency action. Suspicious/failed-attempt alerts do not qualify.
        confirmed = self._eligible_hosts(env, "isolate") if self.budget > 0 else []
        if confirmed:
            action_index = 0
            action = "isolate"
            host = self._choose_host(env, action)
            forced_reason = "Safety priority: isolate a telemetry-confirmed compromised host."
        else:
            action_index = self.select_action(state, valid, explore=explore)
            action = self.ACTIONS[action_index]
            host = self._choose_host(env, action)
            forced_reason = None

        reason_map = {
            "isolate": "Q-policy selected containment.",
            "patch": "Q-policy selected attack-surface reduction.",
            "rotate_credentials": "Q-policy selected credential hardening.",
            "deploy_honeypot": "Q-policy selected decoy placement for earlier detection.",
            "wait": "Q-policy selected wait for a more informative state.",
        }

        before = self.budget
        if action != "wait":
            if (
                host is None
                or self.budget <= 0
                or not _apply_action(env, action, host, self.visibility)
            ):
                action_index = 4
                action = "wait"
                host = None
            else:
                self.budget -= 1
                self.budget_spent += 1

        record = {
            "step": int(env.steps),
            "action": action,
            "host": host,
            "reason": forced_reason or reason_map[action],
            "budget_before": before,
            "budget_after": self.budget,
            "state": state,
            "action_index": action_index,
            "valid_actions": valid,
            "risk_score": round(self._risk(env, host), 2) if host is not None else 0.0,
        }
        self.decisions.append(record)
        if action != "wait":
            self.actions.append(record)
        return record

    def decay_exploration(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # --------------------------------------------------------- persistence

    @classmethod
    def inspect_checkpoint(cls, path: str) -> dict[str, Any]:
        """
        Validate a checkpoint without building a defender. Raises ValueError
        on an unsupported version or a malformed Q-table; otherwise returns
        {state_version, visibility, initial_budget, n_states, metadata}.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = payload.get("state_version")
        if version not in cls.STATE_VERSIONS:
            raise ValueError(
                f"{path}: unsupported state_version {version!r}; "
                f"supported: {cls.STATE_VERSIONS}."
            )
        table = payload.get("q_table")
        if not isinstance(table, list) or not table:
            raise ValueError(f"{path}: missing or empty q_table.")
        length = cls.STATE_LENGTH[version]
        for item in table:
            if len(item.get("state", ())) != length:
                raise ValueError(
                    f"{path}: state length {len(item.get('state', ()))} does not "
                    f"match state_version {version} (expected {length})."
                )
            if len(item.get("values", ())) != len(cls.ACTIONS):
                raise ValueError(f"{path}: Q-values must have {len(cls.ACTIONS)} entries.")
        visibility = payload.get("visibility", "full")
        if version == 3 and visibility != "full":
            raise ValueError(f"{path}: v3 checkpoints are ground-truth ('full') only.")
        return {
            "state_version": int(version),
            "visibility": visibility,
            "initial_budget": payload.get("initial_budget"),
            "n_states": len(table),
            "metadata": payload.get("metadata", {}),
        }

    def save(self, path: str = "models/learned_defender_v6_alerts.json",
             metadata: dict[str, Any] | None = None):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_version": self.state_version,
            "visibility": self.visibility,
            "initial_budget": self.initial_budget,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "metadata": metadata if metadata is not None else self.metadata,
            "q_table": [
                {"state": list(state), "values": values}
                for state, values in self.q_table.items()
            ],
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: str = "models/learned_defender_v3.json"):
        info = self.inspect_checkpoint(path)          # validates structure
        if info["visibility"] != self.visibility:
            raise ValueError(
                f"Checkpoint was trained with visibility='{info['visibility']}' but "
                f"this defender uses '{self.visibility}'. Retrain it."
            )
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.state_version = info["state_version"]
        self.metadata = info["metadata"]
        self.q_table = {
            tuple(item["state"]): [float(x) for x in item["values"]]
            for item in payload["q_table"]
        }
        self.alpha = float(payload.get("alpha", self.alpha))
        self.gamma = float(payload.get("gamma", self.gamma))
        self.epsilon = float(payload.get("epsilon", self.epsilon))
        self.epsilon_min = float(payload.get("epsilon_min", self.epsilon_min))
        self.epsilon_decay = float(payload.get("epsilon_decay", self.epsilon_decay))
        return self
