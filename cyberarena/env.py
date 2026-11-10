
from __future__ import annotations

from typing import Any

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from .config import ArenaConfig
from .topology import generate_topology, TOPOLOGY_FAMILIES


# Five abstract actions per possible host.
# The simulator generates no real network traffic or exploit payloads.
ACTION_TYPES = (
    "scan",
    "exploit",
    "escalate",
    "move",
    "exfiltrate",
)

NODE_FEATURES = 10


class CyberArenaEnv(gym.Env):
    """
    Partially observable enterprise cyber-range simulator.

    The attacker sees only discovered hosts and their observable
    properties. The crown-jewel identity is hidden.

    Detection is generated from observable simulator events:
    repeated reconnaissance, exploit attempts, privilege escalation,
    lateral movement, exfiltration attempts and honeypot triggers. Patched
    hosts reject direct exploitation, privilege escalation and lateral entry.
    """

    metadata = {"render_modes": ["ansi"]}

    # Detection engine thresholds.
    ALERT_THRESHOLD = 4.0
    MAX_RISK_SCORE = 10.0

    SIGNAL_WEIGHTS = {
        "reconnaissance": 0.6,
        # The initial entry event is a medium-confidence alert, not proof that
        # the defender can already isolate the foothold safely.
        "initial_access": 4.0,
        "exploit_failed": 2.0,
        "exploit_success": 2.5,
        "privilege_escalation": 2.5,
        "lateral_movement_success": 1.8,
        "lateral_movement_denied": 0.8,
        "exfiltration_attempt": 3.0,
        "honeypot_trigger": 10.0,
    }

    def __init__(
        self,
        config: ArenaConfig | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.config = config or ArenaConfig()
        self.config.validate()

        self.render_mode = render_mode

        n = self.config.max_nodes

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(
                n * NODE_FEATURES + n * n,
            ),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(
            len(ACTION_TYPES) * n
        )

        self.graph: nx.Graph = nx.Graph()
        self.rng = np.random.default_rng(
            self.config.seed
        )

        self.discovered: set[int] = set()
        self.compromised: set[int] = set()
        self.privileged: set[int] = set()

        self.detected = False
        self.detected_hosts: set[int] = set()
        # Explicit high-confidence compromise telemetry. This is separate from
        # detected_hosts because failed attempts can also generate alerts.
        self.confirmed_compromised_hosts: set[int] = set()

        self.risk_scores: dict[int, float] = {}
        self.detection_events: list[dict[str, Any]] = []
        self._alerted_hosts: set[int] = set()

        self.done = False
        self.steps = 0
        self.budget = self.config.initial_budget

        self.logs: list[dict[str, Any]] = []
        self.last_path: list[int] = []

        self.reset(seed=self.config.seed)

    @property
    def n_hosts(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def crown_jewel(self) -> int:
        """
        Return the actual randomly generated crown-jewel node.

        This property is for simulator administration and final
        reporting. It must not be included in attacker observations
        or used by a partially observable defender policy.
        """
        for node, data in self.graph.nodes(data=True):
            if data.get("is_crown_jewel", False):
                return int(node)

        raise RuntimeError(
            "Topology does not contain a crown-jewel host."
        )

    # -----------------------------------------------------
    # TOPOLOGY INITIALIZATION
    # -----------------------------------------------------

    def _select_topology_family(self) -> str:
        configured = self.config.topology_family

        if configured != "mixed":
            return configured

        return str(
            self.rng.choice(TOPOLOGY_FAMILIES)
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)

        if seed is None:
            # Fresh episode drawn from the env's own (already seeded) stream.
            # Previously this fell back to config.seed, so RL training saw the
            # exact same topology and the exact same dice rolls every episode.
            actual_seed = int(
                self.np_random.integers(0, 2**31 - 1)
            )
        else:
            actual_seed = int(seed)

        self.rng = np.random.default_rng(
            actual_seed
        )

        n = int(
            self.rng.integers(
                self.config.min_hosts,
                self.config.max_hosts + 1,
            )
        )

        topology_seed = int(
            self.rng.integers(
                0,
                2**31 - 1,
            )
        )

        family = self._select_topology_family()

        self.graph = generate_topology(
            n_hosts=n,
            seed=topology_seed,
            family=family,
        )

        self.graph.graph["selected_family"] = family

        # Initial foothold.
        self.discovered = {0}
        self.compromised = {0}
        self.privileged = set()

        # Detection state.
        self.detected = False
        self.detected_hosts = set()
        self.confirmed_compromised_hosts = set()
        self.risk_scores = {
            node: 0.0
            for node in self.graph.nodes
        }
        self.detection_events = []
        self._alerted_hosts = set()

        self.done = False
        self.steps = 0
        self.budget = self.config.initial_budget

        self.logs = [
            {
                "step": 0,
                "action": "initial_access",
                "host": 0,
                "result": "entry foothold",
                "reward": 0.0,
            }
        ]

        self.last_path = [0]

        # Surface the simulator's initial-entry event to the defender at time
        # zero. It is deliberately an alert (not confirmed-compromise
        # telemetry), so it cannot trigger an automatic isolation before the
        # attacker has produced corroborating behaviour.
        initial_alert = self._record_signal(
            0,
            "initial_access",
            "Potential initial-access activity observed on entry host H0.",
        )
        if initial_alert is not None:
            self.logs[0]["detection"] = initial_alert

        return self._obs(), {
            "topology_family": family,
            "host_count": n,
            "initial_alert": initial_alert,
        }

    # -----------------------------------------------------
    # PARTIALLY OBSERVABLE ATTACKER OBSERVATION
    # -----------------------------------------------------

    def _obs(self) -> np.ndarray:
        """
        Fixed-size observation.

        Feature index 7 is deliberately zero. It used to expose
        is_crown_jewel and therefore leaked the hidden objective.
        Feature index 5 is deliberately zero as well: it used to expose
        the honeypot flag, which defeats the purpose of a decoy.
        """
        n = self.config.max_nodes

        feats = np.zeros(
            (n, NODE_FEATURES),
            dtype=np.float32,
        )

        for node in self.discovered:
            data = self.graph.nodes[node]

            feats[node] = [
                1.0,
                float(node in self.compromised),
                float(node in self.privileged),
                float(data.get("patched", False)),
                float(data.get("isolated", False)),
                # Honeypots are deceptive by design: never reveal them.
                0.0,
                len(data.get("services", [])) / 5.0,

                # Hidden crown-jewel flag. Never expose it.
                0.0,

                (
                    1.0
                    if data.get("credential_posture") == "weak"
                    else 0.5
                    if data.get("credential_posture") == "standard"
                    else 0.0
                ),

                len(
                    data.get("vulnerabilities", [])
                ) / 3.0,
            ]

        adj = np.zeros(
            (n, n),
            dtype=np.float32,
        )

        # Only edges between discovered nodes are observable.
        for u, v in self.graph.edges:
            if (
                u in self.discovered
                and v in self.discovered
            ):
                adj[u, v] = 1.0
                adj[v, u] = 1.0

        return np.concatenate(
            [
                feats.ravel(),
                adj.ravel(),
            ]
        ).astype(np.float32)

    # -----------------------------------------------------
    # ACTION MASKING
    # -----------------------------------------------------

    def _reachable_from_foothold(self, node: int) -> bool:
        """
        True if an active attacker foothold (compromised AND not isolated)
        has an open (non-firewalled) edge to `node`.

        Exploitation requires network reachability, so isolating the only
        compromised neighbour of a host really does cut the attacker off.
        """
        return any(
            nb in self.compromised
            and not self.graph.nodes[nb]["isolated"]
            and not self.graph.edges[nb, node].get("firewall", False)
            for nb in self.graph.neighbors(node)
        )

    def legal_action_mask(self) -> np.ndarray:
        """
        Raw legality mask (no fallback, no failure memory).

        Only observable state is used to determine valid actions.
        Exfiltration is allowed from any privileged host; the
        environment determines whether it is the actual objective.
        """
        mask = np.zeros(
            self.action_space.n,
            dtype=bool,
        )

        max_nodes = self.config.max_nodes

        for kind_idx, kind in enumerate(ACTION_TYPES):
            for node in range(self.n_hosts):

                data = self.graph.nodes[node]
                valid = False

                if kind == "scan":
                    valid = (
                        node in self.compromised
                        and not data["isolated"]
                        and any(
                            neighbour not in self.discovered
                            and not self.graph.nodes[neighbour]["isolated"]
                            and not self.graph.edges[
                                node, neighbour
                            ].get("firewall", False)
                            for neighbour in self.graph.neighbors(node)
                        )
                    )

                elif kind == "exploit":
                    valid = (
                        node in self.discovered
                        and node not in self.compromised
                        and not data["patched"]
                        and not data["isolated"]
                        and self._reachable_from_foothold(node)
                    )

                elif kind == "escalate":
                    valid = (
                        node in self.compromised
                        and node not in self.privileged
                        and not data["patched"]
                        and not data["isolated"]
                    )

                elif kind == "move":
                    valid = (
                        node in self.compromised
                        and not data["isolated"]
                        and any(
                            neighbour in self.discovered
                            and neighbour not in self.compromised
                            and not self.graph.nodes[neighbour]["isolated"]
                            and not self.graph.nodes[neighbour]["patched"]
                            and not self.graph.edges[
                                node, neighbour
                            ].get("firewall", False)
                            for neighbour in self.graph.neighbors(node)
                        )
                    )

                elif kind == "exfiltrate":
                    valid = (
                        node in self.privileged
                        and not data["isolated"]
                    )

                mask[
                    kind_idx * max_nodes + node
                ] = valid

        return mask

    def action_masks(self) -> np.ndarray:
        """
        Legal-action mask for MaskablePPO / SB3.

        Never all-False (an all-False mask makes the policy emit NaNs):
        if nothing is legal it falls back to a no-progress scan of H0.
        Use live_action_mask() / has_live_action() to know whether the
        attacker can still make real progress.
        """
        mask = self.legal_action_mask()

        if not mask.any():
            mask[0] = True

        return mask

    def _failure_counts(self):
        """Episode history of actions that did not work."""
        failed_exploits: dict[int, int] = {}
        blocked_exploits: set[int] = set()
        failed_exfiltration: set[int] = set()
        blocked_escalations: set[int] = set()
        move_denials: dict[int, int] = {}
        dead_moves: set[int] = set()

        for entry in self.logs:
            host = entry.get("host")
            if host is None or host < 0 or host >= self.n_hosts:
                continue

            action = entry.get("action")
            result = str(entry.get("result"))

            if action == "exploit":
                if result == "abstract_attempt_failed":
                    failed_exploits[host] = failed_exploits.get(host, 0) + 1
                elif result == "control_blocked":
                    blocked_exploits.add(host)
            elif action == "exfiltrate":
                if result == "exfiltration_attempt_failed":
                    failed_exfiltration.add(host)
            elif action == "escalate":
                if result == "privilege_escalation_blocked":
                    blocked_escalations.add(host)
            elif action == "move":
                if result == "lateral_movement_denied":
                    move_denials[host] = move_denials.get(host, 0) + 1
                elif result == "no_valid_adjacent_target":
                    dead_moves.add(host)

        return (
            failed_exploits, blocked_exploits, failed_exfiltration,
            blocked_escalations, move_denials, dead_moves,
        )

    def live_action_mask(self) -> np.ndarray:
        """
        Legal actions minus actions that have proven unproductive in this
        episode. Shared by every attacker (random, greedy, oracle, PPO,
        Graph-PPO), the evaluator and the live API, so benchmark runs and
        live missions follow identical rules.

        HARD retirement (can never work again, always masked):
          exploit blocked by a patch/isolation; exfiltration from a host that
          already failed; escalation blocked by a patch; lateral movement
          into patched/isolated hosts or from a source with no valid target.

        SOFT retirement (stochastic, may still succeed, so only de-prioritised):
          exploit targets that failed `max_exploit_failures` times and
          lateral-move sources denied `max_move_denials` times. These are
          masked while other live actions exist, and restored when nothing
          else is left. (Retiring them permanently made hard-to-crack but
          mandatory hosts - e.g. the crown jewel - unwinnable.)

        May be all-False: that means the attacker is genuinely stuck.
        """
        hard = self.legal_action_mask()
        n = self.config.max_nodes
        (failed_exploits, blocked_exploits, failed_exfil,
         blocked_escalations, move_denials, dead_moves) = self._failure_counts()

        for host in blocked_exploits:
            hard[1 * n + host] = False
        for host in blocked_escalations:
            hard[2 * n + host] = False
        for host in dead_moves:
            hard[3 * n + host] = False
        for host in failed_exfil:
            hard[4 * n + host] = False

        soft = hard.copy()
        for host, failures in failed_exploits.items():
            if failures >= self.config.max_exploit_failures:
                soft[1 * n + host] = False
        for host, denials in move_denials.items():
            if denials >= self.config.max_move_denials:
                soft[3 * n + host] = False

        return soft if soft.any() else hard

    def has_live_action(self, mask: np.ndarray | None = None) -> bool:
        """True if at least one non-retired legal action remains."""
        live = self.live_action_mask() if mask is None else mask
        return bool(np.asarray(live).any())

    # -----------------------------------------------------
    # BEHAVIOUR-BASED DETECTION ENGINE
    # -----------------------------------------------------

    def _record_signal(
        self,
        host: int,
        signal: str,
        message: str,
        force_alert: bool = False,
    ) -> dict[str, Any] | None:
        """
        Add a simulator event to a host's risk score.

        One alert is generated per host when its cumulative
        risk crosses the threshold. Honeypot triggers alert
        immediately.
        """
        if host not in self.risk_scores:
            self.risk_scores[host] = 0.0

        weight = self.SIGNAL_WEIGHTS.get(
            signal,
            0.0,
        )

        before = self.risk_scores[host]

        after = min(
            self.MAX_RISK_SCORE,
            before + weight,
        )

        self.risk_scores[host] = after

        crossed_threshold = (
            before < self.ALERT_THRESHOLD
            and after >= self.ALERT_THRESHOLD
        )

        if not (
            force_alert
            or crossed_threshold
        ):
            return None

        if host in self._alerted_hosts:
            return None

        self._alerted_hosts.add(host)
        self.detected_hosts.add(host)
        self.detected = True

        if force_alert or after >= 8.0:
            severity = "critical"
        elif after >= 6.0:
            severity = "high"
        else:
            severity = "medium"

        incident = {
            "id": len(self.detection_events) + 1,
            "step": self.steps,
            "host": int(host),
            "signal": signal,
            "severity": severity,
            "risk_score": round(after, 2),
            "message": message,
        }

        self.detection_events.append(incident)

        return incident

    def _record_activity(
        self,
        host: int,
        signal: str,
        message: str,
    ) -> dict[str, Any] | None:
        data = self.graph.nodes[host]

        # A honeypot is an immediate high-confidence signal.
        if data.get("honeypot", False):
            return self._record_signal(
                host=host,
                signal="honeypot_trigger",
                message=(
                    "Honeypot interaction detected. "
                    + message
                ),
                force_alert=True,
            )

        return self._record_signal(
            host=host,
            signal=signal,
            message=message,
        )

    def _confirm_compromise(
        self, host: int, signal: str, message: str,
    ) -> dict[str, Any] | None:
        """Record high-confidence telemetry for a confirmed attacker foothold."""
        host = int(host)
        if host in self.confirmed_compromised_hosts:
            return None
        self.confirmed_compromised_hosts.add(host)
        self.risk_scores[host] = max(self.risk_scores.get(host, 0.0), 8.0)
        incident = self._record_signal(
            host=host,
            signal=signal,
            message=message,
            force_alert=True,
        )
        if incident is not None:
            incident["confirmed_compromise"] = True
            incident["confirmation_signal"] = signal
            incident["confirmation_step"] = int(self.steps)
            return incident

        # A prior low-confidence alert may already exist for this host. Upgrade
        # that telemetry record instead of silently leaving it labelled medium.
        for existing in reversed(self.detection_events):
            if int(existing.get("host", -1)) == host:
                existing["confirmed_compromise"] = True
                existing["severity"] = "critical"
                existing["risk_score"] = round(self.risk_scores[host], 2)
                existing["confirmation_signal"] = signal
                existing["confirmation_step"] = int(self.steps)
                existing["message"] = message
                return dict(existing)
        return None

    # -----------------------------------------------------
    # SIMULATION STEP
    # -----------------------------------------------------

    def step(self, action: int):

        if self.done:
            raise RuntimeError(
                "Episode is over; call reset() before step()."
            )

        self.steps += 1

        action = int(action)

        max_nodes = self.config.max_nodes

        if action < 0 or action >= self.action_space.n:
            kind = "invalid"
            node = -1
        else:
            kind_idx, node = divmod(
                action,
                max_nodes,
            )

            kind = (
                ACTION_TYPES[kind_idx]
                if kind_idx < len(ACTION_TYPES)
                else "invalid"
            )

        potential_before = (
            len(self.compromised)
            / max(1, self.n_hosts)
        )

        reward = float(
            self.config.step_penalty
        )

        result = "invalid_action"
        new_incident = None

        # -------------------------------------------------
        # SCAN
        # -------------------------------------------------

        if (
            kind == "scan"
            and 0 <= node < self.n_hosts
            and node in self.compromised
            and not self.graph.nodes[node]["isolated"]
        ):

            if node == 0 and node not in self.confirmed_compromised_hosts:
                new_incident = self._confirm_compromise(
                    node,
                    "reconnaissance",
                    "Reconnaissance from the initial foothold confirmed active access.",
                )
            else:
                new_incident = self._record_activity(
                    node,
                    "reconnaissance",
                    "Repeated network reconnaissance activity.",
                )

            neighbors = [
                v
                for v in self.graph.neighbors(node)
                if (
                    v not in self.discovered
                    and not self.graph.nodes[v]["isolated"]
                    and not self.graph.edges[node, v].get(
                        "firewall",
                        False,
                    )
                )
            ]

            if neighbors:
                target = int(
                    self.rng.choice(neighbors)
                )

                self.discovered.add(target)

                if self.config.reward_mode == "shaped":
                    reward += self.config.discovery_reward

                result = f"discovered_host_{target}"

            else:
                result = "nothing_new_found"

        # -------------------------------------------------
        # EXPLOIT
        # -------------------------------------------------

        elif (
            kind == "exploit"
            and 0 <= node < self.n_hosts
            and node in self.discovered
            and node not in self.compromised
        ):

            data = self.graph.nodes[node]

            if data["patched"] or data["isolated"]:
                result = "control_blocked"

            elif not self._reachable_from_foothold(node):
                # No active foothold can reach this host any more. Not
                # hard-retired: a new foothold may make it reachable again.
                result = "unreachable"

            else:
                credential_factor = {
                    "weak": 0.12,
                    "standard": 0.0,
                    "strong": -0.10,
                }[data["credential_posture"]]

                if data["credential_rotated"]:
                    credential_factor -= 0.15

                vuln_factor = (
                    0.18
                    + 0.25 * len(data["vulnerabilities"])
                    + credential_factor
                )

                success_prob = min(
                    0.85,
                    max(0.05, vuln_factor),
                )

                if self.rng.random() < success_prob:

                    self.compromised.add(node)

                    self.last_path.append(node)

                    if self.config.reward_mode == "shaped":
                        reward += self.config.compromise_reward

                    result = "abstract_compromise_success"

                    new_incident = self._confirm_compromise(
                        node,
                        "exploit_success",
                        "Successful abstract compromise confirmed by simulator telemetry.",
                    )

                else:

                    result = "abstract_attempt_failed"

                    new_incident = self._record_activity(
                        node,
                        "exploit_failed",
                        "Failed exploit attempt detected.",
                    )

        # -------------------------------------------------
        # PRIVILEGE ESCALATION
        # -------------------------------------------------

        elif (
            kind == "escalate"
            and 0 <= node < self.n_hosts
            and node in self.compromised
            and node not in self.privileged
        ):

            data = self.graph.nodes[node]

            if data["patched"]:
                result = "privilege_escalation_blocked"

            elif data["isolated"]:
                result = "privilege_escalation_blocked"

            else:

                self.privileged.add(node)

                if self.config.reward_mode == "shaped":
                    reward += self.config.privilege_reward

                result = "privilege_level_raised"

                new_incident = self._confirm_compromise(
                    node,
                    "privilege_escalation",
                    "Privilege escalation confirms active compromise on this host.",
                )

        # -------------------------------------------------
        # LATERAL MOVEMENT
        # -------------------------------------------------

        elif (
            kind == "move"
            and 0 <= node < self.n_hosts
            and node in self.compromised
            and not self.graph.nodes[node]["isolated"]
        ):

            targets = [
                v
                for v in self.graph.neighbors(node)
                if (
                    v in self.discovered
                    and v not in self.compromised
                    and not self.graph.nodes[v]["isolated"]
                    and not self.graph.nodes[v]["patched"]
                    and not self.graph.edges[node, v].get(
                        "firewall",
                        False,
                    )
                )
            ]

            if targets:

                target = int(
                    self.rng.choice(targets)
                )

                if self.rng.random() < 0.65:

                    self.compromised.add(target)
                    self.last_path.append(target)

                    if self.config.reward_mode == "shaped":
                        reward += self.config.compromise_reward

                    result = (
                        f"lateral_movement_success_{target}"
                    )

                    self._confirm_compromise(
                        node,
                        "lateral_movement_success",
                        f"Successful lateral movement confirms active access from host H{node}.",
                    )
                    new_incident = self._confirm_compromise(
                        target,
                        "lateral_movement_success",
                        f"Successful lateral movement confirmed compromise of host H{target}.",
                    )

                    # Honeypot on the destination is also an alert.
                    if self.graph.nodes[target].get("honeypot", False):
                        new_incident = self._record_activity(
                            target,
                            "honeypot_trigger",
                            "Attacker accessed a decoy host.",
                        )

                else:

                    result = "lateral_movement_denied"

                    new_incident = self._record_activity(
                        node,
                        "lateral_movement_denied",
                        "Denied lateral movement attempt.",
                    )

            else:
                result = "no_valid_adjacent_target"

        # -------------------------------------------------
        # EXFILTRATION
        # -------------------------------------------------

        elif (
            kind == "exfiltrate"
            and 0 <= node < self.n_hosts
            and node in self.privileged
        ):

            new_incident = self._record_activity(
                node,
                "exfiltration_attempt",
                "Privileged host attempted data exfiltration.",
            )

            if (
                node == self.crown_jewel
                and not self.graph.nodes[node]["isolated"]
            ):

                reward += self.config.crown_jewel_reward

                result = "crown_jewel_reached"

                self.done = True

            else:

                result = "exfiltration_attempt_failed"

        # -------------------------------------------------
        # GLOBAL REWARD AND TERMINATION
        # -------------------------------------------------

        if self.detected:
            reward += (
                self.config.detection_penalty * 0.1
            )

        terminated = (
            result == "crown_jewel_reached"
        )

        truncated = (
            self.steps >= self.config.max_steps
            and not terminated
        )

        self.done = terminated or truncated

        if self.config.reward_mode == "potential":

            potential_after = (
                0.0
                if self.done
                else len(self.compromised)
                / max(1, self.n_hosts)
            )

            reward += (
                0.99 * potential_after
                - potential_before
            )

        log_entry = {
            "step": self.steps,
            "action": kind,
            "host": node,
            "result": result,
            "reward": float(reward),
        }

        if new_incident is not None:
            log_entry["detection"] = new_incident

        self.logs.append(log_entry)

        info = {
            "success": bool(terminated),
            "detected": bool(self.detected),
            "detection_count": len(self.detection_events),
            "detected_hosts": sorted(
                self.detected_hosts
            ),
            "confirmed_compromised_hosts": sorted(
                self.confirmed_compromised_hosts
            ),
            "compromised": len(self.compromised),
            "discovered": len(self.discovered),
            "privileged": len(self.privileged),
            "attack_path": self.last_path.copy(),
            "action_result": result,
            "detection_event": new_incident,
        }

        return (
            self._obs(),
            float(reward),
            terminated,
            truncated,
            info,
        )

    # -----------------------------------------------------
    # RENDERING
    # -----------------------------------------------------

    def render(self):
        return "\n".join(
            (
                f"[{entry['step']}] "
                f"{entry['action']} "
                f"host={entry['host']}: "
                f"{entry['result']}"
            )
            for entry in self.logs
        )
