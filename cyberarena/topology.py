from __future__ import annotations

import random
import networkx as nx

SERVICES = ("web", "ssh", "database", "file", "api")
VULNS = ("outdated_service", "weak_auth", "misconfigured_acl")

TOPOLOGY_FAMILIES = (
    "enterprise",
    "erdos_renyi",
    "watts_strogatz",
    "barabasi_albert",
)


def _connect_components(graph: nx.Graph, rng: random.Random) -> None:
    """Connect components using randomly selected inter-component edges."""
    components = [list(c) for c in nx.connected_components(graph)]
    if len(components) <= 1:
        return

    rng.shuffle(components)
    for left, right in zip(components, components[1:]):
        graph.add_edge(rng.choice(left), rng.choice(right))


def _generate_structure(n_hosts: int, seed: int, family: str) -> nx.Graph:
    rng = random.Random(seed)

    if family == "enterprise":
        graph = nx.Graph()
        graph.add_nodes_from(range(n_hosts))

        # Random parent-linked spanning tree.
        for node in range(1, n_hosts):
            graph.add_edge(node, rng.randrange(0, node))

        # Add redundant enterprise links.
        candidates = [
            (u, v)
            for u in range(n_hosts)
            for v in range(u + 1, n_hosts)
            if not graph.has_edge(u, v)
        ]
        rng.shuffle(candidates)
        graph.add_edges_from(candidates[:max(1, n_hosts // 5)])

    elif family == "erdos_renyi":
        # Sparse random graph, conditioned to be connected.
        probability = min(0.35, max(0.10, 2.8 / max(1, n_hosts - 1)))
        graph = nx.erdos_renyi_graph(n_hosts, probability, seed=seed)
        _connect_components(graph, rng)

    elif family == "watts_strogatz":
        # Small-world graph: local ring connections plus rewiring.
        if n_hosts < 3:
            graph = nx.path_graph(n_hosts)
            graph.graph["family"] = family
            return graph
        k = min(4, n_hosts - 1)
        if k % 2:
            k -= 1
        k = max(2, k)
        graph = nx.watts_strogatz_graph(
            n_hosts, k, 0.15, seed=seed
        )
        _connect_components(graph, rng)

    elif family == "barabasi_albert":
        # Preferential attachment creates high-degree hub nodes.
        m = min(2, n_hosts - 1)
        graph = nx.barabasi_albert_graph(n_hosts, m, seed=seed)

    else:
        raise ValueError(
            f"Unknown topology family {family!r}. "
            f"Choose from {TOPOLOGY_FAMILIES} or 'mixed'."
        )

    # Keep host IDs contiguous and stable for the simulator.
    graph = nx.convert_node_labels_to_integers(
        graph, first_label=0, ordering="sorted"
    )
    graph.graph["family"] = family
    return graph


def generate_topology(
    n_hosts: int,
    seed: int | None = None,
    family: str = "enterprise",
) -> nx.Graph:
    """Generate a connected graph with abstract enterprise metadata."""
    if n_hosts < 2:
        raise ValueError("A topology needs at least two hosts")

    if family not in TOPOLOGY_FAMILIES:
        raise ValueError(
            f"Unknown topology family: {family}. "
            f"Available families: {TOPOLOGY_FAMILIES}"
        )

    actual_seed = 7 if seed is None else int(seed)
    rng = random.Random(actual_seed)

    graph = _generate_structure(n_hosts, actual_seed, family)

    # Abstract firewall metadata only; no real network traffic is generated.
    # A firewalled edge is not traversable by the attacker, so a firewall is
    # only kept if every host stays reachable over the remaining edges.
    # (Without this, ~5% of episodes were unwinnable by construction.)
    # The same random draw is consumed per edge as before, so the rest of the
    # generated metadata is unchanged for a given seed.
    for u, v in graph.edges:
        graph.edges[u, v]["firewall"] = False
    for u, v in list(graph.edges):
        if rng.random() < 0.10:
            graph.edges[u, v]["firewall"] = True
            open_edges = [
                (a, b) for a, b, d in graph.edges(data=True)
                if not d.get("firewall", False)
            ]
            probe = nx.Graph()
            probe.add_nodes_from(graph.nodes)
            probe.add_edges_from(open_edges)
            if not nx.is_connected(probe):
                graph.edges[u, v]["firewall"] = False

    # Randomize the protected target independently of its node ID.
    crown = rng.randrange(1, n_hosts)

    for node in graph.nodes:
        services = rng.sample(SERVICES, k=rng.randint(1, 3))
        vulns = rng.sample(VULNS, k=rng.randint(0, 2))

        graph.nodes[node].update({
            "services": services,
            "vulnerabilities": vulns,
            "subnet": rng.randint(0, max(1, n_hosts // 5)),
            "is_entry": node == 0,
            "is_crown_jewel": node == crown,
            "patched": False,
            "isolated": False,
            "honeypot": False,
            "credential_posture": rng.choice(
                ("weak", "standard", "strong")
            ),
            "credential_rotated": False,
        })

    # The crown jewel has a database service.
    graph.nodes[crown]["services"] = sorted(
        set(graph.nodes[crown]["services"] + ["database"])
    )

    return graph
