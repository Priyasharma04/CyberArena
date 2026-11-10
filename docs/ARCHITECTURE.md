# Architecture and State Model

## Environment

`CyberArenaEnv` is a Gymnasium environment that simulates an enterprise network as a connected graph. Each episode generates hosts with metadata, service labels, vulnerability labels, subnet identifiers, abstract firewall edges, and credential posture. A crown-jewel database is assigned to the final node.

All attack and defence operations are simulated state transitions. They modify environment metadata and do not execute real exploits, scan external networks, or interact with production systems.

## Partial Observation

The observation contains a fixed-size node-feature matrix and adjacency matrix. Features and edges are populated only for discovered hosts and discovered-to-discovered edges. Undiscovered host metadata and links remain zero.

The action mask avoids exposing whether an undiscovered neighbor exists: scanning remains available from a compromised, non-isolated host even when the action discovers nothing.

## Action Encoding

There are five action families, each with one action slot per padded host index:

- Scan
- Exploit
- Escalate
- Move
- Exfiltrate

An action is encoded as:

`action_type_index * max_nodes + host_index`

All state changes occur inside the Python simulator.

## Reward Modes

- `shaped`: Intermediate discovery, compromise, and privilege rewards, together with a terminal crown-jewel reward.
- `sparse`: Terminal crown-jewel reward and step/detection costs, without intermediate progress bonuses.
- `potential`: Sparse terminal reward plus `gamma * Phi(s') - Phi(s)`, where Phi is the fraction of compromised hosts and terminal potential is zero.

The potential function is intentionally simple and observable from simulator state. It is a research ablation, not a claim that it is the only or best shaping function.

## Defensive Actions

The budgeted defender can perform the following simulated actions:

- Isolate a compromised host.
- Rotate weak credentials.
- Patch a discovered host.
- Deploy a honeypot.

These actions modify simulator metadata only.

CyberArena supports multiple defensive strategies:

- **No defence:** The attacker operates without defender intervention.
- **Rule-based defender:** A scripted baseline that selects defensive actions using predefined rules.
- **Learned defender v6:** A tabular Q-learning policy that selects defensive actions using a compact state representation.

The v6 state representation incorporates confirmed-compromise telemetry, initial-access alert information, pre-action response timing, and attack-path-aware patching. Host selection for patching uses a separate heuristic; the learned defender itself is not a graph neural network.

## Learned Defender v6

The learned defender is trained against a mixture of attacker policies, including random, greedy, PPO, and GNN-PPO agents, under the configured alert-visibility setting.

The v6 implementation uses:

- A compact tabular Q-learning state representation.
- Alert-based defensive visibility.
- Pre-action response timing.
- Confirmed-compromise telemetry.
- Attack-path-aware patch targeting.
- A limited defensive action budget.

The trained policy is stored in `models/learned_defender_v6_alerts.json` and loaded by the runtime when the learned defender mode is selected.

## Attacker Policies

The simulation supports baseline and learned attacker policies:

- Random
- Greedy
- Oracle
- PPO
- GNN-PPO

The oracle policy has access to full simulator knowledge and should be interpreted as a full-information reference rather than a directly comparable partially observed attacker.

## Live Simulation API

The FastAPI backend streams simulation updates over WebSockets. Events include:

- `started`: Initial topology, metrics, and simulation configuration.
- `step`: Attacker action, defender response, new alerts, topology, metrics, and risk history.
- `complete`: Final outcome, reward, step count, detection information, and defender activity.

The frontend consumes these events to update the network visualization and simulation dashboard.

## Runtime Checkpoints

The current runtime loads the following model files:

- `models/ppo_mixed_30k.zip`
- `models/gnn_ppo_mixed_30k.zip`
- `models/learned_defender_v6_alerts.json`

These checkpoints allow the configured policies to be loaded without retraining.