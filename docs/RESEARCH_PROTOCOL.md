# CyberArena research protocol

## Research question
Can a policy trained on partial observations of procedurally generated enterprise graphs generalize to unseen topology sizes, and does training against a defender improve attack/defense outcomes compared with static policies?

## Hypotheses
- H1: graph-structured policies will exhibit a smaller performance drop on held-out topology sizes than a fixed-vector MLP policy, when trained with comparable budgets.
- H2: action masking will improve sample efficiency where most host-action pairs are invalid.
- H3: defender interventions will reduce attacker success and increase time-to-compromise, subject to a fixed defensive budget.
- H4: reward shaping will improve learning speed; potential-based shaping should preserve the optimal policy under its assumptions.

## Experimental controls
- Fix topology seeds and report every seed used.
- Separate training topology sizes (10, 20, 30) from held-out sizes (50, 75, 100).
- Use identical environment settings and training budgets when comparing agents.
- Report mean, standard deviation, and preferably 95% bootstrap confidence intervals.
- Do not tune hyperparameters on held-out topology sizes.
- Keep oracle/full-information results clearly labelled as an information-rich reference, not a partially observable peer.

## Metrics
- Crown-jewel success rate.
- Mean steps to success (also report censored episode length for failures).
- Cumulative reward.
- Hosts discovered and compromised.
- Detection rate.
- Generalization gap: in-range success rate minus held-out success rate.
- Sample efficiency: success rate/reward against environment timesteps.
- Defender efficiency: reduction in success rate per defensive action spent.

## Ablations
1. DQN vs masked PPO vs graph-encoder masked PPO.
2. Sparse vs shaped vs potential-based rewards.
3. Defender disabled vs budgeted scripted defender vs trained/self-play defender.
4. Training and testing on matched sizes vs held-out sizes.
5. Action masking enabled vs disabled where the algorithm supports a fair comparison.

## Current implementation status
The repository currently provides the abstract Gymnasium environment, graph generator, random/greedy/oracle baselines, a budgeted scripted defender, multi-seed baseline evaluation, topology-size and reward-mode study runners, optional DQN/PPO/GNN-PPO training, a FastAPI/WebSocket service, and a React/Cytoscape local dashboard.

The defender is currently rule-based, not a learned policy. The provided study runners evaluate heuristic baselines; they do not by themselves establish the hypotheses above. A proper co-evolution experiment still needs a defender observation/action policy, alternating or simultaneous training, checkpointed self-play rounds, and held-out evaluation. Likewise, the GNN-PPO and DQN runs must be trained and compared empirically before making claims about superiority.

## Safety and validity
All vulnerabilities and actions are abstract labels and probabilistic simulator transitions. No exploit payloads, packet transmission, port scanning, or external target integration are implemented. The simulator's results are not evidence of real-world exploitability.
