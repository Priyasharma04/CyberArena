import { useEffect, useRef, useState } from 'react';
import cytoscape, { Core } from 'cytoscape';

const API = 'http://127.0.0.1:8000';

type TopologyNode = {
  id: string;
  label: string;
  status: string;
  discovered: boolean;
  compromised: boolean;
  privileged: boolean;
  isolated: boolean;
  patched: boolean;
  honeypot: boolean;
};

type Topology = {
  nodes: TopologyNode[];
  edges: { id: string; source: string; target: string }[];
  node_count?: number;
  edge_count?: number;
};

type LiveMetrics = {
  step: number;
  reward: number;
  discovered: number;
  compromised: number;
  privileged: number;
  detected: boolean;
  defender_budget_remaining: number | null;
};

type CompletedEpisode = {
  type?: string;
  status?: 'completed' | 'stalled' | 'error';
  agent: string;
  defender: string;
  success: boolean;
  stalled?: boolean;
  outcome?: string;
  termination_reason?: string;
  steps: number;
  reward: number;
  detected: boolean | number;
  discovered: number;
  compromised: number;
  attack_path: number[];
  target_node: number;
  topology: Topology;
  logs: unknown[];
  defender_actions: Record<string, unknown>[];
  defender_decisions: Record<string, unknown>[];
};

type ModelInfo = {
  attackers: Record<string, { available: boolean; type: string }>;
  defenders: Record<string, boolean>;
};

type EvaluationMetric = {
  episodes: number;
  success_rate: number;
  success_ci95_low?: number;
  success_ci95_high?: number;
  mean_steps: number;
  std_steps?: number;
  mean_reward: number;
  std_reward?: number;
  detection_rate?: number;
  mean_detection_events?: number;
  mean_compromised?: number;
  mean_discovered?: number;
  mean_defender_interventions?: number;
  stall_rate?: number;
  outcomes?: Record<string, number>;
};

type EvaluationConfig = {
  agents: string[];
  seed_start: number;
  seed_end: number;
  episodes_per_agent: number;
  defender_mode: string;
  topology_family: string;
  stochastic: boolean;
  min_hosts?: number;
  max_hosts?: number;
  max_steps?: number;
  reward_mode?: string;
  defender_budget?: number;
  defender_interval?: number;
  defender_visibility?: string;
  defender_first_response?: string;
  stall_window?: number;
};

type EvaluationResponse = {
  config: EvaluationConfig;
  summary: Record<string, EvaluationMetric>;
  episodes: unknown[];
};

type Activity = 'idle' | 'simulation' | 'evaluation';

const agentLabels: Record<string, string> = {
  gnn_ppo: 'Graph PPO',
  ppo: 'Standard PPO',
  greedy: 'Greedy heuristic',
  random: 'Random baseline',
  oracle: 'Full-knowledge oracle',
};

const defenderLabels: Record<string, string> = {
  off: 'No defender',
  rule: 'Rule-based defender',
  learned: 'Learned Q-learning defender (v6)',
};

const topologyLabels: Record<string, string> = {
  mixed: 'Mixed enterprise',
  enterprise: 'Enterprise',
  erdos_renyi: 'Random network',
  watts_strogatz: 'Small-world',
  barabasi_albert: 'Scale-free',
};

const outcomeLabels: Record<string, string> = {
  objective_reached: 'OBJECTIVE REACHED',
  contained_by_defender: 'CONTAINED BY DEFENDER',
  attacker_stalled: 'ATTACKER STALLED',
  time_limit: 'TIME LIMIT REACHED',
  terminated: 'MISSION TERMINATED',
};

function eventDescription(event: unknown): string {
  if (typeof event === 'string') return event;

  if (event && typeof event === 'object') {
    const item = event as Record<string, unknown>;
    const action = String(item.action ?? 'Action');
    const host = item.host == null ? '' : ` H${item.host}`;
    const result = item.result == null ? '' : ` — ${item.result}`;
    return `${action}${host}${result}`;
  }

  return 'Simulation action performed';
}

function defenderDescription(event: Record<string, unknown>): string {
  const action = String(event.action ?? 'intervention');
  const host = event.host == null ? '' : ` on H${event.host}`;
  const reason = event.reason == null ? '' : ` — ${event.reason}`;
  return `${action}${host}${reason}`;
}

function formatPercent(value: number | undefined): string {
  return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—';
}

function formatNumber(value: number | undefined, digits = 1): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? value.toFixed(digits)
    : '—';
}

function outcomeDescription(episode: CompletedEpisode): string {
  if (episode.outcome && outcomeLabels[episode.outcome]) {
    return outcomeLabels[episode.outcome];
  }
  if (episode.success) return 'OBJECTIVE REACHED';
  if (episode.stalled) return 'ATTACKER STALLED';
  return 'OBJECTIVE NOT REACHED';
}

export default function App() {
  const graphRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const [health, setHealth] = useState('checking');
  const [models, setModels] = useState<ModelInfo | null>(null);

  const [agent, setAgent] = useState('gnn_ppo');
  const [defender, setDefender] = useState('learned');
  const [topologyFamily, setTopologyFamily] = useState('mixed');
  const [stochastic, setStochastic] = useState(true);
  const [seed, setSeed] = useState(7);

  const [busy, setBusy] = useState(false);
  const [activity, setActivity] = useState<Activity>('idle');
  const [episode, setEpisode] = useState<CompletedEpisode | null>(null);
  const [liveMetrics, setLiveMetrics] = useState<LiveMetrics | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null);

  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState('');
  const [connection, setConnection] = useState('disconnected');

  useEffect(() => {
    let active = true;

    async function checkBackend() {
      try {
        const [healthResponse, modelsResponse] = await Promise.all([
          fetch(`${API}/health`),
          fetch(`${API}/api/models`),
        ]);

        if (!healthResponse.ok || !modelsResponse.ok) {
          throw new Error('Backend health check failed.');
        }

        const modelData = (await modelsResponse.json()) as ModelInfo;
        if (!active) return;

        setHealth('connected');
        setModels(modelData);

        if (!modelData.attackers?.gnn_ppo?.available) {
          setAgent('ppo');
        }
      } catch {
        if (active) setHealth('offline');
      }
    }

    checkBackend();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!graphRef.current) return;

    const cy = cytoscape({
      container: graphRef.current,
      elements: [],
      minZoom: 0.25,
      maxZoom: 2.5,
      style: [
        {
          selector: 'node',
          style: {
            label: 'data(label)',
            'background-color': '#64748b',
            color: '#e2e8f0',
            'text-valign': 'center',
            'text-halign': 'center',
            'font-size': 10,
            'font-weight': 700,
            width: 38,
            height: 38,
            'border-width': 1,
            'border-color': '#94a3b8',
            'text-outline-width': 2,
            'text-outline-color': '#0c1625',
          },
        },
        {
          selector: 'node.compromised',
          style: {
            'background-color': '#dc5252',
            'border-color': '#ff9292',
            'border-width': 3,
          },
        },
        {
          selector: 'node.discovered',
          style: {
            'background-color': '#bd913d',
            'border-color': '#f5cb75',
          },
        },
        {
          selector: 'node.privileged',
          style: {
            'background-color': '#8c61bd',
            'border-color': '#d8b4fe',
            'border-width': 3,
          },
        },
        {
          selector: 'node.isolated',
          style: {
            'background-color': '#475569',
            'border-color': '#f87171',
            'border-style': 'dashed',
            'border-width': 3,
          },
        },
        {
          selector: 'node.patched',
          style: {
            'background-color': '#267b65',
            'border-color': '#6ee7b7',
          },
        },
        {
          selector: 'node.honeypot',
          style: {
            'border-color': '#f59e0b',
            'border-width': 4,
          },
        },
        {
          selector: 'edge',
          style: {
            width: 2,
            'line-color': '#40536d',
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'target-arrow-color': '#40536d',
            'arrow-scale': 0.7,
          },
        },
      ],
      layout: { name: 'cose', animate: false },
    });

    cyRef.current = cy;
    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, []);

  function drawTopology(topology: Topology) {
    const cy = cyRef.current;
    if (!cy || !topology) return;

    const nodes = (topology.nodes ?? []).map((node) => ({
      data: {
        id: String(node.id),
        label: node.label || `H${node.id}`,
      },
      classes: [
        node.status,
        node.compromised ? 'compromised' : '',
        node.privileged ? 'privileged' : '',
        node.isolated ? 'isolated' : '',
        node.patched ? 'patched' : '',
        node.honeypot ? 'honeypot' : '',
        node.status === 'discovered' ? 'discovered' : '',
      ]
        .filter(Boolean)
        .join(' '),
    }));

    const edges = (topology.edges ?? []).map((edge) => ({
      data: {
        id: edge.id,
        source: String(edge.source),
        target: String(edge.target),
      },
    }));

    cy.elements().remove();
    cy.add([...nodes, ...edges]);
    cy.layout({
      name: 'cose',
      animate: false,
      fit: true,
      padding: 35,
      nodeRepulsion: () => 5000,
      idealEdgeLength: () => 100,
    }).run();
  }

  function addLog(line: string) {
    setLogs((previous) => [...previous.slice(-149), line]);
  }

  function runEpisode() {
    if (busy || health !== 'connected') return;

    wsRef.current?.close();
    setBusy(true);
    setActivity('simulation');
    setError('');
    setLogs([]);
    setEpisode(null);
    setEvaluation(null);
    setLiveMetrics(null);
    setConnection('connecting');

    const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const params = new URLSearchParams({
      seed: String(seed),
      defender_mode: defender,
      topology_family: topologyFamily,
      stochastic: String(stochastic),
      defender_visibility: 'alerts',
    });

    const socket = new WebSocket(
      `${scheme}://127.0.0.1:8000/ws/episode/${agent}?${params.toString()}`
    );
    wsRef.current = socket;

    socket.onopen = () => {
      setConnection('connected');
      addLog('Connected to CyberArena simulation engine.');
      addLog(`Attacker: ${agentLabels[agent]}`);
      addLog(`Defender: ${defenderLabels[defender]}`);
    };

    socket.onmessage = (message) => {
      try {
        const data = JSON.parse(message.data);

        if (data.type === 'error') {
          setError(data.message || 'Simulation error.');
          setBusy(false);
          setActivity('idle');
          setConnection('error');
          socket.close();
          return;
        }

        if (data.type === 'started') {
          drawTopology(data.topology);
          addLog(`Simulation started — seed ${data.seed}.`);
          addLog(`Topology family: ${data.topology_family}.`);
          if (data.initial_defender_event) {
            addLog(`[DEFENDER] ${defenderDescription(data.initial_defender_event)}`);
          }
          if (Array.isArray(data.alerts)) {
            for (const alert of data.alerts) {
              addLog(`[ALERT] H${alert.host}: ${alert.message}`);
            }
          }
        }

        if (data.type === 'step') {
          drawTopology(data.topology);
          setLiveMetrics(data.metrics);
          addLog(`[STEP ${data.step}] ${eventDescription(data.event)}`);

          if (data.defender_event) {
            addLog(`[DEFENDER] ${defenderDescription(data.defender_event)}`);
          }
          if (Array.isArray(data.new_alerts)) {
            for (const alert of data.new_alerts) {
              const status = alert.confirmed_compromise ? 'CONFIRMED' : 'ALERT';
              addLog(`[${status}] H${alert.host}: ${alert.message}`);
            }
          }
        }

        if (data.type === 'complete') {
          drawTopology(data.topology);
          setEpisode(data as CompletedEpisode);
          setBusy(false);
          setActivity('idle');
          setConnection(data.status === 'stalled' ? 'stalled' : 'completed');

          const completed = data as CompletedEpisode;
          const label = outcomeDescription(completed);
          addLog(
            completed.success
              ? 'MISSION COMPLETE — Crown jewel reached.'
              : `SIMULATION ENDED — ${label.replace(/_/g, ' ')}.`
          );
          if (completed.termination_reason) {
            addLog(`Termination reason: ${completed.termination_reason}.`);
          }
          addLog(
            `Final statistics: ${data.discovered} discovered, ${data.compromised} compromised.`
          );
          socket.close();
        }
      } catch {
        setError('Could not process a simulation event.');
        setBusy(false);
        setActivity('idle');
        setConnection('error');
        socket.close();
      }
    };

    socket.onerror = () => {
      setError('Live connection failed. Check that the API is running.');
      setBusy(false);
      setActivity('idle');
      setConnection('error');
    };

    socket.onclose = () => {
      setBusy(false);
      if (wsRef.current === socket) {
        wsRef.current = null;
        setActivity('idle');
      }
    };
  }

  function stopEpisode() {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setBusy(false);
    setActivity('idle');
    setConnection('stopped');
    addLog('Simulation stopped by operator.');
  }

  async function runEvaluation() {
    if (busy || health !== 'connected') return;

    setBusy(true);
    setActivity('evaluation');
    setError('');
    setEvaluation(null);

    try {
      const response = await fetch(`${API}/api/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          agents: ['random', 'greedy', 'oracle'],
          seeds: 10,
          seed_start: seed,
          defender_mode: defender,
          topology_family: topologyFamily,
          stochastic,
          defender_visibility: 'alerts',
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || data.error || `Evaluation failed (${response.status}).`
        );
      }

      if (!data.summary || !data.config) {
        throw new Error('Evaluation response is missing config or summary.');
      }

      setEvaluation(data as EvaluationResponse);
      setConnection('evaluation-complete');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Evaluation failed.');
      setConnection('error');
    } finally {
      setBusy(false);
      setActivity('idle');
    }
  }

  const discovered = liveMetrics?.discovered ?? episode?.discovered ?? 0;
  const compromised = liveMetrics?.compromised ?? episode?.compromised ?? 0;
  const currentStep = liveMetrics?.step ?? episode?.steps ?? 0;
  const budget = liveMetrics?.defender_budget_remaining;
  const simulatorLabel =
    activity === 'simulation'
      ? 'RUNNING'
      : activity === 'evaluation'
        ? 'EVALUATING'
        : episode
          ? episode.success
            ? 'MISSION COMPLETE'
            : episode.stalled
              ? 'STALLED'
              : 'MISSION ENDED'
          : 'READY';

  const latestOutcome = episode ? outcomeDescription(episode) : null;
  const threatDetected = liveMetrics
    ? liveMetrics.detected
      ? 'Yes'
      : 'No'
    : episode
      ? episode.detected
        ? 'Yes'
        : 'No'
      : '—';

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">AUTONOMOUS CYBERSECURITY SIMULATION</div>
          <h1>
            Cyber<span>Arena</span>
          </h1>
          <p>Enterprise attack-path discovery &amp; adaptive defense</p>
        </div>

        <div className={`status ${health}`}>
          <i />
          API {health}
        </div>
      </header>

      <section className="hero-strip">
        <div>
          <div className="eyebrow">SIMULATION COMMAND CENTER</div>
          <h2>Network operations overview</h2>
          <p>
            Deploy autonomous agents, observe attack behaviour and evaluate
            defensive responses in a controlled environment.
          </p>
        </div>
        <div className="hero-tag">
          <span className="live-dot" />
          SIMULATOR {simulatorLabel}
        </div>
      </section>

      <section className="stats-grid">
        <StatCard label="Discovered hosts" value={discovered} icon="◈" />
        <StatCard label="Compromised hosts" value={compromised} icon="⌘" />
        <StatCard label="Simulation steps" value={currentStep} icon="↗" />
        <StatCard label="Defender budget" value={budget ?? '—'} icon="⬡" />
      </section>

      <section className="panel control-panel">
        <div className="section-heading">
          <div>
            <div className="eyebrow">MISSION CONFIGURATION</div>
            <h2>Configure simulation</h2>
          </div>
          <span className={`connection-pill ${connection}`}>{connection}</span>
        </div>

        <div className="control-grid">
          <div className="field">
            <label>AI attacker</label>
            <select value={agent} onChange={(e) => setAgent(e.target.value)}>
              {Object.entries(agentLabels).map(([value, label]) => (
                <option
                  key={value}
                  value={value}
                  disabled={
                    value in (models?.attackers ?? {}) &&
                    !models?.attackers[value]?.available
                  }
                >
                  {label}
                </option>
              ))}
            </select>
          </div>

          <div className="field">
            <label>Defensive system</label>
            <select
              value={defender}
              onChange={(e) => setDefender(e.target.value)}
            >
              {Object.entries(defenderLabels).map(([value, label]) => (
                <option
                  key={value}
                  value={value}
                  disabled={models?.defenders[value] === false}
                >
                  {label}
                </option>
              ))}
            </select>
          </div>

          <div className="field">
            <label>Network topology</label>
            <select
              value={topologyFamily}
              onChange={(e) => setTopologyFamily(e.target.value)}
            >
              {Object.entries(topologyLabels).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </div>

          <div className="field">
            <label>Random seed</label>
            <input
              type="number"
              min="0"
              value={seed}
              onChange={(e) => setSeed(Math.max(0, Number(e.target.value) || 0))}
            />
          </div>
        </div>

        <div className="control-bottom">
          <label className="toggle">
            <input
              type="checkbox"
              checked={stochastic}
              onChange={(e) => setStochastic(e.target.checked)}
            />
            Stochastic AI policy
          </label>

          <div className="action-buttons">
            <button
              className="primary-button"
              disabled={busy || health !== 'connected'}
              onClick={runEpisode}
            >
              {activity === 'simulation' ? 'Simulation running…' : '▶  Launch simulation'}
            </button>

            {activity === 'simulation' && (
              <button className="stop-button" onClick={stopEpisode}>
                Stop
              </button>
            )}

            <button
              className="secondary-button"
              disabled={busy || health !== 'connected'}
              onClick={runEvaluation}
            >
              {activity === 'evaluation' ? 'Evaluating…' : 'Evaluate baselines'}
            </button>
          </div>
        </div>
      </section>

      {error && <div className="error">{error}</div>}

      <section className="main-grid">
        <article className="panel graph-panel">
          <div className="section-heading">
            <div>
              <div className="eyebrow">LIVE NETWORK VISUALIZATION</div>
              <h2>Enterprise topology</h2>
            </div>
            <span className="node-count">{discovered} visible nodes</span>
          </div>

          <div ref={graphRef} className="graph" />

          <div className="legend">
            <span><i className="legend-discovered" /> Discovered</span>
            <span><i className="legend-compromised" /> Compromised</span>
            <span><i className="legend-privileged" /> Privileged</span>
            <span><i className="legend-patched" /> Patched</span>
            <span><i className="legend-isolated" /> Isolated</span>
            <span><i className="legend-honeypot" /> Honeypot</span>
          </div>
        </article>

        <article className="panel episode-panel">
          <div className="eyebrow">MISSION STATUS</div>
          <h2>Simulation intelligence</h2>

          <div className="mission-status">
            <span className="mission-label">Latest outcome</span>
            <strong
              className={
                episode
                  ? episode.success
                    ? 'outcome-success'
                    : 'outcome-failed'
                  : busy && activity === 'simulation'
                    ? 'outcome-pending'
                    : 'outcome-pending'
              }
            >
              {episode
                ? latestOutcome
                : activity === 'simulation'
                  ? 'IN PROGRESS'
                  : 'AWAITING MISSION'}
            </strong>
          </div>

          <div className="metric-list">
            <Metric label="Attacker" value={agentLabels[agent]} />
            <Metric label="Defender" value={defenderLabels[defender]} />
            <Metric label="Steps executed" value={currentStep} />
            <Metric
              label="Total reward"
              value={
                liveMetrics
                  ? liveMetrics.reward.toFixed(2)
                  : episode
                    ? episode.reward.toFixed(2)
                    : '—'
              }
            />
            <Metric label="Hosts discovered" value={discovered} />
            <Metric label="Hosts compromised" value={compromised} />
            <Metric label="Threat detected" value={threatDetected} />
          </div>

          {episode && (
            <div className="completion-card">
              <div className="eyebrow">FINAL OBJECTIVE</div>
              <p>
                {episode.success
                  ? `The attacker reached the target in ${episode.steps} steps.`
                  : episode.outcome === 'contained_by_defender'
                    ? `The defender contained the attacker after ${episode.steps} steps.`
                    : episode.stalled
                      ? `The attacker stalled after ${episode.steps} steps.`
                      : `The target was not reached within ${episode.steps} steps.`}
              </p>
              <span>Target: Host {episode.target_node}</span>
              {episode.termination_reason && (
                <span>Reason: {episode.termination_reason}</span>
              )}
            </div>
          )}
        </article>
      </section>

      <section className="panel results">
        <div className="section-heading">
          <div>
            <div className="eyebrow">LIVE TELEMETRY</div>
            <h2>Attacker and defender activity</h2>
          </div>
          <span className="node-count">{logs.length} events</span>
        </div>

        <div className="terminal">
          {logs.length ? (
            logs.map((line, index) => (
              <div
                key={`${index}-${line}`}
                className={
                  line.includes('[DEFENDER]')
                    ? 'defender-log'
                    : line.includes('MISSION COMPLETE')
                      ? 'success-log'
                      : line.includes('STALLED')
                        ? 'error-log'
                        : ''
                }
              >
                {line}
              </div>
            ))
          ) : (
            <span className="empty">
              Simulation events will appear here after launch.
            </span>
          )}
        </div>
      </section>

      <section className="panel results">
        <div className="eyebrow">PERFORMANCE EVALUATION</div>
        <h2>Baseline agent comparison</h2>

        {evaluation ? (
          <>
            <div className="evaluation-meta">
              <span>
                Defender: {defenderLabels[evaluation.config.defender_mode] ?? evaluation.config.defender_mode}
              </span>
              <span>
                Topology: {topologyLabels[evaluation.config.topology_family] ?? evaluation.config.topology_family}
              </span>
              <span>
                Seeds: {evaluation.config.seed_start}–{evaluation.config.seed_end}
              </span>
              <span>
                Episodes per agent: {evaluation.config.episodes_per_agent}
              </span>
              <span>
                Policy: {evaluation.config.stochastic ? 'Stochastic' : 'Deterministic'}
              </span>
            </div>

            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Agent</th>
                    <th>Success rate</th>
                    <th>Mean steps</th>
                    <th>Mean reward</th>
                    <th>Detection rate</th>
                    <th>Stall rate</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(evaluation.summary).map(([name, metric]) => (
                    <tr key={name}>
                      <td>{agentLabels[name] || name}</td>
                      <td>
                        {formatPercent(metric.success_rate)}
                        {typeof metric.success_ci95_low === 'number' &&
                          typeof metric.success_ci95_high === 'number' && (
                            <small className="table-subtext">
                              95% CI {formatPercent(metric.success_ci95_low)}–
                              {formatPercent(metric.success_ci95_high)}
                            </small>
                          )}
                      </td>
                      <td>{formatNumber(metric.mean_steps)}</td>
                      <td>{formatNumber(metric.mean_reward, 2)}</td>
                      <td>{formatPercent(metric.detection_rate)}</td>
                      <td>{formatPercent(metric.stall_rate)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <p className="empty">
            Run baseline evaluation to compare the heuristic agents under the
            currently selected defender, topology and seed range.
          </p>
        )}
      </section>

      <footer>
        CYBERARENA · AI-DRIVEN NETWORK SIMULATION · ABSTRACT ENVIRONMENT ONLY
      </footer>
    </main>
  );
}

function StatCard({
  label,
  value,
  icon,
}: {
  label: string;
  value: string | number;
  icon: string;
}) {
  return (
    <article className="panel stat-card">
      <div className="stat-icon">{icon}</div>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
    </article>
  );
}

function Metric({
  label,
  value,
}: {
  label: string;
  value: string | number;
}) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
