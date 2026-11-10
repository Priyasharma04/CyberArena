from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArenaConfig:
    min_hosts: int = 10
    max_hosts: int = 30
    max_nodes: int = 100
    max_steps: int = 120
    initial_budget: int = 8
    defender_budget: int = 5
    step_penalty: float = -0.02
    discovery_reward: float = 0.15
    compromise_reward: float = 0.8
    privilege_reward: float = 0.5
    crown_jewel_reward: float = 10.0
    detection_penalty: float = -0.8
    seed: int = 7
    reward_mode: str = "shaped"  # sparse | shaped | potential
    topology_family: str = "enterprise"

    # --- Shared by the simulator, the evaluator and the live API so that
    # --- benchmark runs and live missions use identical rules.
    defender_interval: int = 4        # defender acts every N attacker steps
    defender_visibility: str = "alerts"  # full | alerts; observation model for defenders
    max_exploit_failures: int = 10    # de-prioritise an exploit target after N failures (soft)
    max_move_denials: int = 6         # de-prioritise a move source after N denials (soft)
    stall_window: int = 20            # steps without network progress => stalled

    def validate(self) -> None:
        from .topology import TOPOLOGY_FAMILIES

        if self.topology_family not in (*TOPOLOGY_FAMILIES, "mixed"):
            raise ValueError(f"Unknown topology family: {self.topology_family}")
        if not (2 <= self.min_hosts <= self.max_hosts <= self.max_nodes):
            raise ValueError("Require 2 <= min_hosts <= max_hosts <= max_nodes")
        if self.max_steps < 1 or self.initial_budget < 0 or self.defender_budget < 0:
            raise ValueError(
                "max_steps must be >= 1; initial_budget and defender_budget "
                "must be non-negative"
            )
        if self.reward_mode not in {"sparse", "shaped", "potential"}:
            raise ValueError("reward_mode must be sparse, shaped, or potential")
        if self.defender_interval < 1 or self.stall_window < 1:
            raise ValueError("defender_interval and stall_window must be >= 1")
        if self.defender_visibility not in {"full", "alerts"}:
            raise ValueError("defender_visibility must be 'full' or 'alerts'")
        if self.max_exploit_failures < 1 or self.max_move_denials < 1:
            raise ValueError("failure limits must be >= 1")
