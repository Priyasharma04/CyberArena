# CyberArena

**Enterprise Attack-Path Discovery and Adaptive Defense Using Reinforcement Learning**

CyberArena is a graph-based cybersecurity research simulator that models an abstract enterprise network and studies how attacker strategies and defensive policies interact over time. It combines partially observable attack-path discovery, reinforcement-learning-based attackers, and an adaptive Q-learning defender in a controlled simulation environment.

All attack and defence actions are simulated state transitions over network metadata. CyberArena does not execute exploits, scan external systems, or interact with production networks.

## Overview

An enterprise environment is represented as a graph of hosts and network connections. The attacker begins with an initial foothold and must discover and traverse the network to reach a protected objective. A defender operates with a limited intervention budget and responds using available telemetry.

The project includes:

- Procedurally generated enterprise network topologies.
- Partially observable attacker environments and legal-action masking.
- Random, greedy and oracle baseline attackers.
- PPO and graph-aware PPO attacker policies.
- Rule-based and learned defensive policies.
- A v6 alert-aware tabular Q-learning defender.
- A FastAPI backend and interactive React dashboard.
- Multi-seed training and held-out evaluation workflows.

## Architecture

```text
                 React + TypeScript Dashboard
                            |
                       REST / WebSocket
                            |
                       FastAPI Backend
                            |
                    Shared Runtime Layer
                     /              \
              Attacker Agents     Defender
             /    |    |    \         |
          Random Greedy Oracle PPO   Rule / Q-learning
                     \    |    /       |
                      Simulation Engine
                              |
                    Gymnasium Environment
                              |
                    NetworkX Enterprise Graph
```

### Main components

| Component | Responsibility |
|---|---|
| `cyberarena/env.py` | Gymnasium environment, observations, actions and simulation transitions |
| `cyberarena/topology.py` | Procedural network topology generation |
| `cyberarena/baselines.py` | Random, greedy and full-knowledge oracle attackers |
| `cyberarena/rl.py` | Reinforcement-learning attacker implementations |
| `cyberarena/defender.py` | Rule-based and learned defensive policies |
| `cyberarena/runtime.py` | Shared attacker/defender construction and action selection |
| `cyberarena/evaluate.py` | Episode execution and evaluation metrics |
| `cyberarena/api.py` | FastAPI endpoints and live simulation interface |
| `frontend/` | React dashboard and network visualisation |
| `scripts/` | Training, evaluation and research command-line workflows |
| `tests/` | Environment, topology, audit and containment tests |

For more detail, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md).

## Simulation design

### Network environment

The environment represents hosts as graph nodes and network connectivity as graph edges. Host metadata includes abstract service labels, vulnerability labels, credentials and defensive status.

Supported topology families include:

- Enterprise
- Erdős–Rényi
- Watts–Strogatz
- Barabási–Albert
- Mixed topology selection

The attacker has partial knowledge of the network. Its observation is populated as hosts and connections are discovered rather than exposing the complete graph.

### Attacker actions

The simulator supports five abstract action families:

- **Scan:** discover connected hosts.
- **Exploit:** attempt to compromise a discovered host.
- **Escalate:** attempt privilege escalation.
- **Move:** attempt lateral movement.
- **Exfiltrate:** attempt the simulated objective.

Action masking prevents invalid or unavailable actions. These actions only change simulator state; they do not execute real-world exploit code.

### Attacker policies

| Policy | Description |
|---|---|
| Random | Selects from legal actions |
| Greedy | Uses heuristic action priorities |
| Oracle | Full-knowledge reference planner; an information-rich upper-bound-style comparison, not a fair partially observable baseline |
| PPO | Maskable Proximal Policy Optimization |
| GNN-PPO | PPO with graph-aware feature extraction |

### Defensive policies

The defender operates with a limited action budget. Available interventions include host isolation, patching, credential rotation and honeypot deployment.

CyberArena includes a rule-based defender and a learned tabular Q-learning defender. The v6 policy uses alert-aware state information, confirmed-compromise telemetry, pre-action response timing and attack-path-aware patch targeting. Host selection is handled separately from the tabular state-value policy; the learned defender is not a graph neural network.

## v6 learned defender

The final learned defender checkpoint is:

`models/learned_defender_v6_alerts.json`

### Training configuration

| Setting | Value |
|---|---:|
| Training episodes | 6,000 |
| Training seeds | 10,000–15,999 |
| Attacker policies | Random, greedy, oracle, PPO, GNN-PPO |
| Topology selection | Mixed |
| Defender budget | 5 |
| Defender interval | 4 |
| Alert visibility | Enabled |
| State version | 6 |
| Q-table states | 216 |
| Final epsilon | 0.05 |

The v6 training run uses a reward signal that penalises newly compromised hosts and intervention costs, penalises loss of the crown-jewel objective, and rewards verified prevention in matched no-defender runs.

Training summary: [`results/learned_defender_v6_training_summary.json`](results/learned_defender_v6_training_summary.json).

## Held-out evaluation

The defender comparison evaluates three modes:

- `off`: no defender.
- `rule`: rule-based defender.
- `learned_v6`: v6 learned defender.

The evaluation uses matched attacker/seed pairs across five attacker policies, with 500 episodes per attacker and mixed topology selection.

### Pooled results

| Metric | No defender | Rule-based | Learned v6 |
|---|---:|---:|---:|
| Attacker success rate | 95.44% | 7.36% | 7.36% |
| Mean compromised hosts | 12.727 | 3.585 | 3.318 |
| Mean defender interventions | 0 | 3.001 | 2.626 |

The v6 defender reduced mean compromised hosts and used fewer interventions than the rule-based policy in this evaluation. The pooled attacker success rate was the same for both defensive policies; therefore, these results do not establish a success-rate advantage for v6.

The evaluation is limited to mixed topology settings. The supplied evaluation data does not provide a validated crown-jewel-loss comparison or a breakdown by topology family and network size.

Evaluation files:

- [`results/defender_comparison.csv`](results/defender_comparison.csv)
- [`results/defender_comparison_summary.json`](results/defender_comparison_summary.json)

Training and evaluation are separate: the held-out evaluation uses seeds 20,000–20,499 for each attacker, while training uses seeds 10,000–15,999.

## Technology stack

**Backend and simulation**
- Python 3.11
- FastAPI
- Gymnasium
- NetworkX
- NumPy
- Pydantic

**Reinforcement learning**
- PyTorch
- Stable-Baselines3
- SB3-Contrib (MaskablePPO)

**Frontend**
- React 18
- TypeScript
- Vite
- Cytoscape.js

**Testing and analysis**
- Pytest
- Pandas
- Matplotlib

## Getting started

### Requirements

- Python 3.11
- Node.js 20 LTS
- npm

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/CyberArena.git
cd CyberArena
```

### 2. Set up the Python environment

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[rl,dev]"
```

The `rl` extra installs the reinforcement-learning dependencies required for PPO and GNN-PPO. The repository includes the v6 defender and attacker checkpoints used by the configured runtime.

### 3. Run tests

```powershell
pytest -q
```

### 4. Start the backend

```powershell
uvicorn cyberarena.api:app --reload
```

Open:

- API documentation: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

### 5. Start the frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

### 6. Train the v6 defender

To reproduce the v6 training workflow:

```powershell
python scripts/train_learned_defender.py --episodes 6000 --visibility alerts --out models/learned_defender_v6_alerts.json
```

This command starts a new training run and writes a checkpoint and training outputs. It is not required just to run the existing checkpoint.

## Repository structure

```text
CyberArena/
├── cyberarena/          # Simulation, agents, defender, API and runtime
├── frontend/            # React + TypeScript dashboard
├── configs/             # Experiment configuration files
├── scripts/             # Training and evaluation scripts
├── tests/               # Automated tests
├── models/              # Required attacker and v6 defender checkpoints
├── results/             # Selected v6 training and evaluation outputs
├── docs/                # Architecture and research protocol
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Scope and responsible use

CyberArena is an abstract research simulator. Its attack actions operate on simulated host metadata and graph state. It is not a penetration-testing tool and should not be connected to production networks.

The separate SiteGuard component is a passive HTTP/TLS audit utility intended for authorised targets. Only assess systems for which you have explicit permission.

## Limitations

- The reported held-out evaluation uses mixed topology selection; generalisation across individual topology families and network sizes requires additional evaluation.
- The oracle attacker has full network knowledge and is an information-rich reference, not a like-for-like partially observable baseline.
- The pooled success-rate results do not demonstrate a statistically established success-rate improvement for the v6 defender over the rule-based policy.
- The evaluation data currently does not provide a crown-jewel-loss comparison.
- Results are specific to the simulator configuration and should not be interpreted as real-world security effectiveness.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Research protocol](docs/RESEARCH_PROTOCOL.md)
