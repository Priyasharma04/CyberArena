import networkx as nx
import pytest

from cyberarena.topology import (
    TOPOLOGY_FAMILIES,
    generate_topology,
)


@pytest.mark.parametrize("family", TOPOLOGY_FAMILIES)
@pytest.mark.parametrize("n_hosts", [2, 10, 20, 30])
def test_topology_family_is_connected_and_valid(family, n_hosts):
    graph = generate_topology(n_hosts, seed=123, family=family)

    assert graph.number_of_nodes() == n_hosts
    assert nx.is_connected(graph)
    assert graph.graph["family"] == family

    crown_jewels = [
        node for node, data in graph.nodes(data=True)
        if data["is_crown_jewel"]
    ]

    assert len(crown_jewels) == 1
    assert crown_jewels[0] != 0

    for node, data in graph.nodes(data=True):
        assert "services" in data
        assert "vulnerabilities" in data
        assert "credential_posture" in data
        assert data["is_entry"] == (node == 0)

    for _, _, data in graph.edges(data=True):
        assert isinstance(data["firewall"], bool)


@pytest.mark.parametrize("family", TOPOLOGY_FAMILIES)
def test_topology_generation_is_reproducible(family):
    first = generate_topology(20, seed=77, family=family)
    second = generate_topology(20, seed=77, family=family)

    assert set(first.edges()) == set(second.edges())

    for node in first.nodes:
        assert first.nodes[node] == second.nodes[node]


def test_unknown_topology_family_is_rejected():
    with pytest.raises(ValueError):
        generate_topology(10, seed=1, family="unknown")


def test_mixed_configuration_is_accepted():
    from cyberarena.config import ArenaConfig

    config = ArenaConfig(topology_family="mixed")
    config.validate()
