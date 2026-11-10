from __future__ import annotations

import networkx as nx

from cyberarena.config import ArenaConfig
from cyberarena.defender import LearnedDefender
from cyberarena.env import CyberArenaEnv
from cyberarena.evaluate import run_episode
from scripts.train_learned_defender import run_training_episode


def make_config():
    return ArenaConfig(
        min_hosts=8,
        max_hosts=8,
        max_nodes=20,
        max_steps=30,
        stall_window=8,
        topology_family="enterprise",
        defender_budget=5,
        defender_visibility="alerts",
    )


def test_initial_access_is_potential_alert_not_confirmed_compromise():
    env = CyberArenaEnv(make_config())
    _, info = env.reset(seed=19)

    assert info["initial_alert"]["step"] == 0
    assert info["initial_alert"]["signal"] == "initial_access"
    assert info["initial_alert"]["severity"] == "medium"
    assert env.detected_hosts == {0}
    assert env.confirmed_compromised_hosts == set()


def test_confirmed_foothold_is_isolated_but_suspicious_alert_is_not():
    env = CyberArenaEnv(make_config())
    env.reset(seed=20)
    defender = LearnedDefender(
        budget=5, seed=1, epsilon=0, visibility="alerts", state_version=6
    )

    # A potential initial-access alert alone must not trigger isolation.
    state = defender.state(env)
    defender._q(state)[:] = [-1.0, -1.0, -1.0, -1.0, 10.0]
    first = defender.respond(env, explore=False)
    assert first["action"] == "wait"
    assert not env.graph.nodes[0]["isolated"]

    # Reconnaissance from H0 corroborates the foothold.
    env.step(0)  # scan H0
    assert 0 in env.confirmed_compromised_hosts
    assert env.detection_events[0]["confirmed_compromise"] is True

    second = defender.respond(env, explore=False)
    assert second["action"] == "isolate"
    assert second["host"] == 0
    assert env.graph.nodes[0]["isolated"] is True
    assert not env.live_action_mask().any()

    # A different host with only a suspicious alert is not isolatable.
    env2 = CyberArenaEnv(make_config())
    env2.reset(seed=21)
    false_positive_host = 1
    env2.detected_hosts.add(false_positive_host)
    env2.confirmed_compromised_hosts.clear()
    defender2 = LearnedDefender(
        budget=5, seed=2, epsilon=0, visibility="alerts", state_version=6
    )
    assert 0 not in defender2.valid_actions(env2)


def test_patch_target_is_nearest_then_most_vulnerable():
    env = CyberArenaEnv(make_config())
    env.reset(seed=22)

    graph = nx.Graph()
    for node in range(5):
        graph.add_node(
            node,
            isolated=False,
            patched=False,
            honeypot=False,
            credential_rotated=False,
            credential_posture="standard",
            services=[],
            vulnerabilities=[],
        )
    graph.nodes[1]["vulnerabilities"] = ["CVE-A"]
    graph.nodes[3]["vulnerabilities"] = ["CVE-A", "CVE-B"]
    graph.add_edges_from([(0, 1), (1, 2), (0, 3), (3, 4)])

    env.graph = graph
    env.detected_hosts = {0}
    env.confirmed_compromised_hosts = {0}
    env.compromised = {0}
    env.discovered = {0}
    env.risk_scores = {node: 0.0 for node in graph.nodes}

    defender = LearnedDefender(
        budget=5, seed=3, epsilon=0, visibility="alerts", state_version=6
    )
    assert defender._choose_host(env, "patch") == 3


def test_offline_evaluator_records_step_zero_defender_decision():
    result = run_episode(
        "random",
        31,
        config=make_config(),
        defender_mode="rule",
        stochastic=True,
    )
    assert result["first_defender_decision_step"] == 0
    assert result["first_defender_action_step"] == 0
    assert result["time_to_detection"] == 1
    assert set(result) >= set(__import__("cyberarena.evaluate", fromlist=["CSV_FIELDS"]).CSV_FIELDS)


def test_training_first_decision_is_also_at_step_zero():
    config = make_config()
    defender = LearnedDefender(
        budget=5, seed=4, epsilon=0.0, visibility="alerts", state_version=6
    )
    row = run_training_episode(
        defender, "random", 41, config, interval=4, baseline_success=False
    )
    assert defender.decisions
    assert defender.decisions[0]["step"] == 0
    assert row["decisions"] >= 1
