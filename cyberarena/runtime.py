"""
Shared runtime used by BOTH the live API and the offline evaluator.

Keeping agent/defender construction, action selection, stall detection and
outcome classification in one place is what makes benchmark results
comparable to live missions.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import FullKnowledgePlanner, GreedyAgent, RandomAgent
from .defender import BudgetedDefender, LearnedDefender
from .env import CyberArenaEnv

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

AGENTS = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "oracle": FullKnowledgePlanner,
}

BASELINE_AGENTS = frozenset(AGENTS)
RL_AGENTS = frozenset({"ppo", "gnn_ppo"})
ALL_AGENTS = BASELINE_AGENTS | RL_AGENTS

DEFENDER_MODES = ("off", "rule", "learned")


def valid_defender_mode(mode: str) -> bool:
    """'off' | 'rule' | 'learned' | 'learned:<checkpoint path>'."""
    return mode in DEFENDER_MODES or (
        isinstance(mode, str) and mode.startswith("learned:") and len(mode) > 8
    )

TOPOLOGY_CHOICES = (
    "mixed",
    "enterprise",
    "erdos_renyi",
    "watts_strogatz",
    "barabasi_albert",
)

# File names match what `python -m cyberarena.rl` writes by default.
MODEL_PATHS = {
    "ppo": "models/ppo_mixed_30k.zip",
    "gnn_ppo": "models/gnn_ppo_mixed_30k.zip",
}

# V6 is the new alert-aware policy. Legacy v3/v4/v5 checkpoints are preserved
# and can still be evaluated explicitly through learned:<checkpoint>.
LEARNED_DEFENDER_PATH = Path("models/learned_defender_v6_alerts.json")


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2)
def load_attacker_model(agent_name: str):
    """Load a MaskablePPO checkpoint (imports torch/SB3 lazily)."""
    if agent_name not in RL_AGENTS:
        raise ValueError(f"Unsupported RL agent: {agent_name}")

    model_path = Path(MODEL_PATHS[agent_name])
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            f"Train one with: python -m cyberarena.rl --algo {agent_name} "
            f"--family mixed --steps 30000"
        )

    from sb3_contrib import MaskablePPO

    return MaskablePPO.load(str(model_path), device="cpu")


def create_attacker(agent_name: str, env: CyberArenaEnv):
    if agent_name in BASELINE_AGENTS:
        return AGENTS[agent_name](env)

    if agent_name in RL_AGENTS:
        model = load_attacker_model(agent_name)
        expected = tuple(env.observation_space.shape)
        actual = tuple(model.observation_space.shape)
        if expected != actual:
            raise ValueError(
                f"Checkpoint '{MODEL_PATHS[agent_name]}' expects observations "
                f"of shape {actual} but this environment produces {expected}. "
                "Retrain the attacker with the current simulator."
            )
        return model

    raise ValueError(f"Unknown attacker: {agent_name}")


def create_defender(mode: str, seed: int, budget: int, visibility: str = "full"):
    if mode == "off":
        return None

    if mode == "rule":
        return BudgetedDefender(budget=budget, seed=seed, visibility=visibility)

    if mode == "learned" or mode.startswith("learned:"):
        path = (
            Path(mode.split(":", 1)[1]) if ":" in mode else LEARNED_DEFENDER_PATH
        )
        if not path.is_file():
            raise FileNotFoundError(f"Learned defender checkpoint not found: {path}")
        defender = LearnedDefender(budget=budget, seed=seed, visibility=visibility)
        defender.load(str(path))
        defender.epsilon = 0.0  # greedy policy at evaluation time
        return defender

    raise ValueError(f"Unknown defender mode: {mode}")


# ---------------------------------------------------------------------------
# Action selection (identical for live missions and offline evaluation)
# ---------------------------------------------------------------------------

def select_action(
    agent_name: str,
    attacker: Any,
    obs: np.ndarray,
    env: CyberArenaEnv,
    mask: np.ndarray,
    stochastic: bool = True,
) -> int:
    """Pick an action that is guaranteed to be allowed by `mask`."""
    if agent_name in RL_AGENTS:
        action, _ = attacker.predict(
            obs,
            deterministic=not stochastic,
            action_masks=mask,
        )
        selected = int(np.asarray(action).item())
    else:
        selected = int(attacker.act(obs, action_mask=mask))

    if 0 <= selected < len(mask) and mask[selected]:
        return selected

    valid = np.flatnonzero(mask)
    return int(valid[0])


# ---------------------------------------------------------------------------
# Stall detection
# ---------------------------------------------------------------------------

def progress_signature(env: CyberArenaEnv, defense=None) -> tuple:
    """Everything that counts as 'the situation changed'."""
    known = sorted(env.discovered)
    return (
        tuple(known),
        tuple(sorted(env.compromised)),
        tuple(sorted(env.privileged)),
        len(env.detection_events),
        tuple(
            (
                node,
                bool(env.graph.nodes[node].get("isolated", False)),
                bool(env.graph.nodes[node].get("patched", False)),
                bool(env.graph.nodes[node].get("honeypot", False)),
                bool(env.graph.nodes[node].get("credential_rotated", False)),
            )
            for node in known
        ),
        # Defender interventions are real change: don't call it a stall
        # while the defender is still actively reshaping the network.
        len(getattr(defense, "actions", [])) if defense is not None else 0,
    )


class StallMonitor:
    """No-network-progress watchdog (window comes from ArenaConfig)."""

    def __init__(self, env: CyberArenaEnv, defense=None, window: int | None = None):
        self.env = env
        self.defense = defense
        self.window = window or env.config.stall_window
        self.previous = progress_signature(env, defense)
        self.stagnant = 0

    def _was_retry(self) -> bool:
        """A failed stochastic attempt (exploit / lateral move) can still
        succeed next time, so it is a retry, not a dead loop."""
        if not self.env.logs:
            return False
        last = self.env.logs[-1]
        return (
            (last.get("action") == "exploit"
             and last.get("result") == "abstract_attempt_failed")
            or (last.get("action") == "move"
                and last.get("result") == "lateral_movement_denied")
        )

    def update(self) -> bool:
        current = progress_signature(self.env, self.defense)
        if current == self.previous:
            if not self._was_retry():
                self.stagnant += 1
        else:
            self.stagnant = 0
        self.previous = current
        return self.stagnant >= self.window


def stall_event(env: CyberArenaEnv, reason: str) -> dict[str, Any]:
    event = {
        "step": int(env.steps),
        "action": "none",
        "host": None,
        "result": "attacker_stalled",
        "reason": reason,
    }
    env.logs.append(event.copy())
    return event


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------

def classify_outcome(
    success: bool,
    stalled: bool,
    truncated: bool,
    defense,
    objective_without_defender: bool | None = None,
) -> str:
    """
    Say *why* an episode ended without crediting the defender unless that is
    supported by evidence.

    objective_reached        attacker reached the crown jewel
    contained_by_defender    VERIFIED: the defender intervened, the attacker did
                             not reach the objective, and the same attacker on
                             the same seed WITHOUT a defender did
                             (objective_without_defender=True)
    stalled_after_intervention  attacker ran out of moves after the defender
                             acted, but no counterfactual was supplied, so
                             causation is unproven
    attacker_stalled         attacker ran out of moves with no defender action,
                             or would have stalled without the defender too
    time_limit               step budget exhausted while still acting

    `objective_without_defender` comes from a matched no-defender run (see
    evaluate_defenders.py / evaluate.paired_defender_effect). Pass None when no
    counterfactual exists (e.g. a single live mission).
    """
    if success:
        return "objective_reached"

    intervened = defense is not None and len(getattr(defense, "actions", [])) > 0

    if intervened and objective_without_defender is True:
        return "contained_by_defender"

    if stalled:
        if not intervened or objective_without_defender is False:
            return "attacker_stalled"
        return "stalled_after_intervention"

    if truncated:
        return "time_limit"
    return "terminated"
