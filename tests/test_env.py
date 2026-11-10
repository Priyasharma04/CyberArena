import numpy as np

from cyberarena.env import CyberArenaEnv
from cyberarena.config import ArenaConfig


def make_env():
    config = ArenaConfig(seed=7)
    return CyberArenaEnv(config)


def test_reset_observation_shape_and_determinism():
    env = make_env()

    a, info_a = env.reset(seed=123)
    hosts_a = env.n_hosts

    b, info_b = env.reset(seed=123)
    hosts_b = env.n_hosts

    assert a.shape == env.observation_space.shape
    assert np.array_equal(a, b)
    assert hosts_a == hosts_b

    # Hidden topology information must not be exposed in reset metadata.
    assert "n_hosts" not in info_a
    assert "crown_jewel" not in info_a
    assert "n_hosts" not in info_b
    assert "crown_jewel" not in info_b


def test_observation_is_within_declared_space():
    env = make_env()
    obs, _ = env.reset(seed=42)

    assert env.observation_space.contains(obs)


def test_action_mask_has_valid_action():
    env = make_env()
    env.reset(seed=42)

    mask = env.action_masks()

    assert mask.shape == (env.action_space.n,)
    assert mask.any()


def test_step_returns_valid_transition():
    env = make_env()
    obs, _ = env.reset(seed=42)

    action = int(np.flatnonzero(env.action_masks())[0])
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert env.observation_space.contains(next_obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
def test_crown_jewel_is_randomized_and_not_entry_host():
    targets = set()

    for seed in range(20):
        env = make_env()
        env.reset(seed=seed)
        assert 1 <= env.crown_jewel < env.n_hosts
        targets.add(env.crown_jewel)

    assert len(targets) > 1


def test_crown_jewel_identity_is_hidden_in_observation():
    env = make_env()
    obs, _ = env.reset(seed=42)

    target = env.crown_jewel
    env.discovered.add(target)
    obs = env._obs()

    # Feature index 7 is the crown-jewel indicator in the node feature vector.
    target_feature_index = target * 10 + 7
    assert obs[target_feature_index] == 0.0


def test_non_target_exfiltration_is_a_valid_failed_attempt():
    env = make_env()
    env.reset(seed=42)

    non_target = next(
        node for node in range(env.n_hosts)
        if node != env.crown_jewel
    )
    env.discovered.add(non_target)
    env.compromised.add(non_target)
    env.privileged.add(non_target)

    action = 4 * env.config.max_nodes + non_target
    _, _, terminated, _, info = env.step(action)

    assert not terminated
    assert info["action_result"] == "exfiltration_attempt_failed"

def test_scan_mask_rejects_exhausted_scan():
    env = make_env()
    env.reset(seed=42)

    # Mark every neighbour of the entry host as discovered.
    for neighbour in env.graph.neighbors(0):
        env.discovered.add(neighbour)

    mask = env.action_masks()
    scan_action = 0  # scan action for host 0

    assert not mask[scan_action]
