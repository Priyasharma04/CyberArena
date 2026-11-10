from fastapi.testclient import TestClient
from cyberarena.api import app

client = TestClient(app)

url = (
    "/ws/episode/gnn_ppo"
    "?seed=7"
    "&defender_mode=learned"
    "&topology_family=mixed"
    "&stochastic=true"
)

steps = 0
defender_events = 0
completed = False

print("Starting Graph PPO + Learned Defender test...")

with client.websocket_connect(url) as ws:
    while True:
        message = ws.receive_json()
        event_type = message.get("type")

        if event_type == "error":
            raise RuntimeError(message["message"])

        if event_type == "started":
            print("Simulation started:", message["agent"])
            print("Defender:", message["defender"])
            print("Topology:", message["topology_family"])

        elif event_type == "step":
            steps += 1

            if message.get("defender_event"):
                defender_events += 1

            # The hidden target must not be exposed during simulation.
            assert "target_node" not in message
            for node in message["topology"]["nodes"]:
                assert "crown_jewel" not in node

        elif event_type == "complete":
            completed = True

            print("\nSIMULATION COMPLETED")
            print("Success:", message["success"])
            print("Steps:", message["steps"])
            print("Reward:", message["reward"])
            print("Discovered:", message["discovered"])
            print("Compromised:", message["compromised"])
            print("Detected:", message["detected"])
            print("Attack path:", message["attack_path"])
            print("Defender actions:", len(message["defender_actions"]))
            print("Defender decisions:", len(message["defender_decisions"]))
            print("Target revealed only at completion:", message["target_node"])
            break

assert completed, "Simulation did not complete."
assert steps > 0, "No live step events received."

print("\nLIVE INTEGRATION TEST PASSED")
print("Live events:", steps)
print("Defender interventions:", defender_events)
