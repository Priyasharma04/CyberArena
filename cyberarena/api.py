from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import ArenaConfig
from .defender import LearnedDefender
from .env import CyberArenaEnv
from .evaluate import run_episode, run_evaluation, write_evaluation
from .runtime import (  # noqa: F401  (re-exported for scripts / tests)
    AGENTS,
    ALL_AGENTS,
    BASELINE_AGENTS,
    DEFENDER_MODES,
    LEARNED_DEFENDER_PATH,
    MODEL_PATHS,
    RL_AGENTS,
    TOPOLOGY_CHOICES,
    StallMonitor,
    classify_outcome,
    create_attacker,
    create_defender,
    load_attacker_model,
    select_action,
    stall_event,
)
from .siteguard_api import router as siteguard_router

# NOTE: torch / stable-baselines3 are imported lazily inside
# runtime.load_attacker_model, so the API (baselines, defenders, SiteGuard,
# evaluation) starts even when the optional RL stack is not installed.


# =========================================================
# APPLICATION
# =========================================================

app = FastAPI(
    title="CyberArena API",
    version="1.3.0",
    description=(
        "Interactive AI-powered enterprise network "
        "attack and adaptive defense simulator."
    ),
)

_default_origins = (
    "http://localhost:5173,http://127.0.0.1:5173,"
    "http://localhost:4173,http://127.0.0.1:4173"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get(
            "CYBERARENA_CORS_ORIGINS", _default_origins
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(siteguard_router)

TOPOLOGY_FAMILIES = set(TOPOLOGY_CHOICES)


# =========================================================
# MISSION INTELLIGENCE / ACTION SELECTION
# =========================================================

def build_live_action_mask(env: CyberArenaEnv) -> np.ndarray:
    """Legal actions minus actions that proved unproductive this episode.

    Thin wrapper kept for backward compatibility: the logic now lives in
    CyberArenaEnv.live_action_mask() so that the evaluator, the trainer and the
    live API all share exactly the same rules (exploit/escalate/move/exfiltrate
    failures are all tracked there)."""
    return env.live_action_mask()


def has_live_action(env: CyberArenaEnv, mask: np.ndarray | None = None) -> bool:
    """True when at least one real (non-fallback) action remains."""
    return env.has_live_action(mask)


def select_live_action(
    agent_name: str,
    attacker: Any,
    obs: np.ndarray,
    env: CyberArenaEnv,
    stochastic: bool,
    attempted_targets: set[int],
    mission_assist: bool = False,
) -> tuple[int | None, Any]:
    """
    Select an attacker action under the shared live action mask.

    Mission assistance is optional and disabled by default. It tests
    privileged database hosts first, which biases toward the (hidden)
    objective, so assisted runs must not be compared with unassisted ones.
    """
    action_mask = env.live_action_mask()

    if not env.has_live_action(action_mask):
        return None, {
            "status": "stalled",
            "reason": "no_unfailed_legal_actions",
            "message": "No legal actions remain after failed-action filtering.",
        }

    if mission_assist:
        candidates = sorted(
            node
            for node in env.privileged
            if node not in attempted_targets
            and not env.graph.nodes[node].get("isolated", False)
            and not env.graph.nodes[node].get("patched", False)
            and "database" in env.graph.nodes[node].get("services", [])
        )

        if candidates:
            node = candidates[0]
            action = 4 * env.config.max_nodes + node

            if action_mask[action]:
                attempted_targets.add(node)
                intelligence = {
                    "type": "database_asset_identified",
                    "host": node,
                    "message": (
                        f"Privileged database host identified: "
                        f"H{node}. Testing the mission objective."
                    ),
                    "source": "mission_intelligence_layer",
                }
                return action, intelligence

    return select_action(
        agent_name, attacker, obs, env, action_mask, stochastic
    ), None


# =========================================================
# REQUEST SCHEMAS
# =========================================================

class EvaluationRequest(BaseModel):
    """Same knobs as a live mission, so results are comparable."""

    agents: list[str] = Field(
        default_factory=lambda: ["random", "greedy", "oracle"]
    )
    seeds: int = Field(default=10, ge=1, le=100)   # episodes per agent
    seed_start: int = Field(default=0, ge=0)
    topology_family: str = "enterprise"
    defender_mode: str | None = None                # off | rule | learned
    defender_enabled: bool = False                  # legacy: True => rule
    stochastic: bool = True
    defender_visibility: str = "alerts"
    save: bool = False                              # also write results/*.csv


# =========================================================
# HEALTH AND MODEL INFORMATION
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "cyberarena",
        "version": "1.3.0",
    }


@app.get("/api/models")
def available_models():
    return {
        "attackers": {
            name: {
                "available": (
                    Path(MODEL_PATHS[name]).is_file()
                    if name in RL_AGENTS
                    else True
                ),
                "type": (
                    "reinforcement_learning"
                    if name in RL_AGENTS
                    else "baseline"
                ),
            }
            for name in sorted(ALL_AGENTS)
        },

        "defenders": {
            "off": True,
            "rule": True,
            "learned": LEARNED_DEFENDER_PATH.is_file(),
        },

        "topologies": sorted(TOPOLOGY_FAMILIES),

        "features": {
            "live_simulation": True,
            "mission_intelligence": True,
            "siteguard": True,
            "host_risk_scoring": True,
            "incident_history": True,
            "live_risk_metrics": True,
        },
    }


# =========================================================
# EVALUATION
# =========================================================

@app.post("/api/evaluate")
def evaluate_runs(request: EvaluationRequest):
    """
    Run an evaluation under an explicit, echoed configuration.

    Response: {"config": {...}, "summary": {agent: metrics}, "episodes": [...]}
    """
    unknown = set(request.agents) - ALL_AGENTS
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown attacker(s): {sorted(unknown)}",
        )

    mode = request.defender_mode or (
        "rule" if request.defender_enabled else "off"
    )
    if mode not in DEFENDER_MODES:
        raise HTTPException(status_code=400, detail="Invalid defender mode.")
    if request.topology_family not in TOPOLOGY_FAMILIES:
        raise HTTPException(status_code=400, detail="Invalid topology family.")
    if request.defender_visibility not in {"full", "alerts"}:
        raise HTTPException(status_code=400, detail="Invalid defender visibility.")

    try:
        result = run_evaluation(
            tuple(request.agents),
            range(request.seed_start, request.seed_start + request.seeds),
            defender_mode=mode,
            topology_family=request.topology_family,
            stochastic=request.stochastic,
            config=ArenaConfig(defender_visibility=request.defender_visibility),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if request.save:
        write_evaluation(result, "results/api_evaluation.csv")

    return result


@app.get("/api/episode/{agent}")
def episode(
    agent: str,
    seed: int = 7,
    defender: bool = False,
    defender_mode: str | None = None,
    topology_family: str = "enterprise",
    stochastic: bool = True,
):
    if agent not in ALL_AGENTS:
        raise HTTPException(status_code=400, detail=f"Unknown attacker: {agent}")
    if topology_family not in TOPOLOGY_FAMILIES:
        raise HTTPException(status_code=400, detail="Invalid topology family.")

    mode = defender_mode or ("rule" if defender else "off")
    if mode not in DEFENDER_MODES:
        raise HTTPException(status_code=400, detail="Invalid defender mode.")

    try:
        return run_episode(
            agent,
            seed,
            config=ArenaConfig(
                topology_family=topology_family,
                defender_visibility="alerts",
            ),
            defender_mode=mode,
            stochastic=stochastic,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# =========================================================
# INCIDENT AND RISK HELPERS
# =========================================================

def get_host_alerts(env: CyberArenaEnv):
    """Return the most recent alert for each host."""

    alerts = {}

    for incident in env.detection_events:
        host = int(incident["host"])
        alerts[host] = incident

    return alerts


def get_compromise_metrics(env: CyberArenaEnv):
    isolated = {
        node
        for node in env.compromised
        if env.graph.nodes[node].get("isolated", False)
    }

    active = set(env.compromised) - isolated

    return {
        "active_compromised": len(active),
        "isolated_compromised": len(isolated),
    }


# =========================================================
# NETWORK SERIALIZATION
# =========================================================

def build_topology_snapshot(
    env: CyberArenaEnv,
):
    """
    Return discovered network information only.

    Risk information is exposed only for discovered hosts.
    The crown-jewel identity remains hidden during simulation.
    """

    nodes = []
    host_alerts = get_host_alerts(env)

    for node_id in sorted(env.discovered):
        data = env.graph.nodes[node_id]
        alert = host_alerts.get(node_id)

        if data.get("isolated", False):
            status = "isolated"
        elif data.get("patched", False):
            status = "patched"
        elif node_id in env.privileged:
            status = "privileged"
        elif node_id in env.compromised:
            status = "compromised"
        else:
            status = "discovered"

        nodes.append({
            "id": str(node_id),
            "label": f"Host {node_id}",
            "status": status,
            "discovered": True,
            "compromised": (
                node_id in env.compromised
            ),
            "privileged": (
                node_id in env.privileged
            ),
            "isolated": bool(
                data.get("isolated", False)
            ),
            "patched": bool(
                data.get("patched", False)
            ),
            "honeypot": bool(
                data.get("honeypot", False)
            ),
            "risk_score": round(
                float(env.risk_scores.get(node_id, 0.0)),
                2,
            ),
            "alerted": node_id in env.detected_hosts,
            "alert_severity": (
                alert["severity"] if alert else None
            ),
            "last_detection": alert,
        })

    edges = [
        {
            "id": f"{u}-{v}",
            "source": str(u),
            "target": str(v),
        }
        for u, v in env.graph.edges
        if (
            u in env.discovered
            and v in env.discovered
        )
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


# =========================================================
# LIVE METRICS
# =========================================================

def build_metrics(
    env: CyberArenaEnv,
    total_reward: float,
    defense=None,
):
    compromise = get_compromise_metrics(env)
    behavioural_events = [
        event for event in env.detection_events
        if event.get("signal") != "initial_access" or event.get("confirmed_compromise")
    ]

    # Aggregate risk only across discovered hosts.
    visible_risks = [
        float(env.risk_scores.get(node, 0.0))
        for node in env.discovered
    ]

    risk_total = sum(visible_risks)
    max_host_risk = (
        max(visible_risks)
        if visible_risks
        else 0.0
    )

    return {
        "step": int(env.steps),
        "reward": round(float(total_reward), 4),

        "discovered": len(env.discovered),
        "compromised": len(env.compromised),
        "active_compromised": compromise[
            "active_compromised"
        ],
        "isolated_compromised": compromise[
            "isolated_compromised"
        ],
        "privileged": len(env.privileged),

        "detected": bool(behavioural_events),
        "detection_events": len(behavioural_events),
        "alerted_hosts": len(env.detected_hosts),

        "risk_total": round(risk_total, 2),
        "max_host_risk": round(max_host_risk, 2),

        "defender_budget_remaining": (
            defense.budget if defense is not None else None
        ),
        "defender_decisions": (
            len(getattr(defense, "decisions", []))
            if defense is not None
            else 0
        ),
        "defender_interventions": (
            len(getattr(defense, "actions", []))
            if defense is not None
            else 0
        ),
    }


def build_risk_snapshot(metrics: dict):
    """Compact time-series point for dashboard charts."""

    return {
        "step": metrics["step"],
        "risk_total": metrics["risk_total"],
        "max_host_risk": metrics["max_host_risk"],
        "detection_events": metrics["detection_events"],
        "compromised": metrics["compromised"],
        "active_compromised": metrics["active_compromised"],
        "isolated_compromised": metrics["isolated_compromised"],
    }


# =========================================================
# LIVE SIMULATION WEBSOCKET
# =========================================================

@app.websocket("/ws/episode/{agent}")
async def stream_episode(
    websocket: WebSocket,
    agent: str,
    seed: int = 7,
    defender: bool = False,
    defender_mode: str = "off",
    topology_family: str = "enterprise",
    stochastic: bool = True,
    mission_assist: bool = False,
    defender_visibility: str = "alerts",
):
    await websocket.accept()

    env = None

    try:
        # -------------------------------------------------
        # VALIDATE CONFIGURATION
        # -------------------------------------------------

        if agent not in ALL_AGENTS:
            await websocket.send_json({
                "type": "error",
                "message": f"Unknown attacker: {agent}",
            })
            return

        if defender_mode not in set(DEFENDER_MODES):
            await websocket.send_json({
                "type": "error",
                "message": "Invalid defender mode.",
            })
            return

        if topology_family not in TOPOLOGY_FAMILIES:
            await websocket.send_json({
                "type": "error",
                "message": "Invalid topology family.",
            })
            return
        if defender_visibility not in {"full", "alerts"}:
            await websocket.send_json({
                "type": "error",
                "message": "Invalid defender visibility.",
            })
            return

        # Backward compatibility.
        if defender and defender_mode == "off":
            defender_mode = "rule"

        # -------------------------------------------------
        # INITIALIZE ENVIRONMENT
        # -------------------------------------------------

        config = ArenaConfig(
            seed=seed,
            topology_family=topology_family,
            defender_visibility=defender_visibility,
        )

        env = CyberArenaEnv(config)

        obs, _ = env.reset(seed=seed)

        attacker = create_attacker(agent, env)

        defense = create_defender(
            defender_mode,
            seed,
            env.config.defender_budget,
            env.config.defender_visibility,
        )

        total_reward = 0.0
        attempted_targets: set[int] = set()

        # Episode-level history for dashboard charts.
        risk_history = []

        # Live simulation stall detection (shared with the evaluator).
        stalled = False
        stalled_reason = None
        monitor = StallMonitor(env, defense)
        initial_defender_event = defense.respond(env) if defense is not None else None

        terminated = False
        truncated = False
        info = {"success": False}

        run_config = {
            "agent": agent,
            "defender": defender_mode,
            "seed": seed,
            "topology_family": topology_family,
            "stochastic": stochastic,
            "mission_assist": mission_assist,
            "max_steps": env.config.max_steps,
            "defender_interval": env.config.defender_interval,
            "defender_visibility": env.config.defender_visibility,
            "defender_first_response": "pre_action_step_0",
        }

        # -------------------------------------------------
        # SEND START EVENT
        # -------------------------------------------------

        initial_metrics = build_metrics(env, total_reward, defense)

        risk_history.append(build_risk_snapshot(initial_metrics))

        await websocket.send_json({
            "type": "started",
            "status": "running",
            **run_config,
            "topology": build_topology_snapshot(env),
            "metrics": initial_metrics,
            "risk_history": risk_history.copy(),
            "alerts": env.detection_events.copy(),
            "initial_defender_event": initial_defender_event,
        })

        # -------------------------------------------------
        # SIMULATION LOOP
        # -------------------------------------------------

        while True:
            # ATTACKER DECISION

            action, mission_intelligence = select_live_action(
                agent_name=agent,
                attacker=attacker,
                obs=obs,
                env=env,
                stochastic=stochastic,
                attempted_targets=attempted_targets,
                mission_assist=mission_assist,
            )

            # -------------------------------------------------
            # HANDLE STALLED ATTACKER (no legal action remains)
            # -------------------------------------------------

            if action is None:
                stalled = True
                stalled_reason = (
                    mission_intelligence.get(
                        "reason", "no_unfailed_legal_actions"
                    )
                    if isinstance(mission_intelligence, dict)
                    else "no_unfailed_legal_actions"
                )

                terminated = False
                truncated = True
                info = {"success": False}

                attacker_event = stall_event(env, stalled_reason)
                new_alerts = []

            else:
                previous_detection_count = len(env.detection_events)

                # ENVIRONMENT STEP
                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += float(reward)

                attacker_event = (
                    env.logs[-1]
                    if env.logs
                    else {
                        "step": env.steps,
                        "action": "unknown",
                        "result": "unknown",
                    }
                )

                new_alerts = [
                    incident.copy()
                    for incident in env.detection_events[
                        previous_detection_count:
                    ]
                ]
                confirmed_event = info.get("detection_event")
                if (
                    confirmed_event
                    and confirmed_event.get("confirmed_compromise")
                    and not any(
                        int(item.get("host", -1)) == int(confirmed_event.get("host", -2))
                        and item.get("confirmed_compromise")
                        for item in new_alerts
                    )
                ):
                    new_alerts.append(confirmed_event.copy())

            # DEFENDER RESPONSE

            defender_event = None

            if (
                defense is not None
                and env.steps % env.config.defender_interval == 0
                and not terminated
                and not truncated
            ):
                defender_event = defense.respond(env)

            # -------------------------------------------------
            # NO-PROGRESS WATCHDOG
            # -------------------------------------------------

            if (
                action is not None
                and not terminated
                and not truncated
                and monitor.update()
            ):
                stalled = True
                stalled_reason = (
                    f"no_network_progress_for_{monitor.window}_steps"
                )
                truncated = True
                info = dict(info)
                info["success"] = False
                attacker_event = stall_event(env, stalled_reason)

            # CURRENT NETWORK AND METRICS

            topology = build_topology_snapshot(env)

            metrics = build_metrics(env, total_reward, defense)

            risk_history.append(build_risk_snapshot(metrics))

            # STREAM LIVE STEP

            await websocket.send_json({
                "type": "step",
                "status": "running",
                "step": env.steps,
                "event": attacker_event,
                "defender_event": defender_event,
                "mission_intelligence": mission_intelligence,
                "new_alerts": new_alerts,
                "topology": topology,
                "metrics": metrics,
                "risk_history": risk_history.copy(),
            })

            # -------------------------------------------------
            # EPISODE COMPLETION
            # -------------------------------------------------

            if terminated or truncated:
                success = bool(info.get("success", False))

                outcome = classify_outcome(
                    success, stalled, truncated, defense
                )

                defender_actions = (
                    defense.actions.copy() if defense is not None else []
                )

                defender_decisions = (
                    defense.decisions.copy()
                    if isinstance(defense, LearnedDefender)
                    else []
                )

                final_metrics = build_metrics(env, total_reward, defense)

                await websocket.send_json({
                    "type": "complete",
                    # running | completed | stalled -> drives the dashboard badge
                    "status": "stalled" if stalled else "completed",
                    "agent": agent,
                    "defender": defender_mode,
                    "config": run_config,
                    "success": success,
                    "stalled": bool(stalled),
                    # objective_reached | contained_by_defender |
                    # attacker_stalled | time_limit
                    "outcome": outcome,
                    "termination_reason": (
                        stalled_reason if stalled else outcome
                    ),
                    "steps": int(env.steps),
                    "reward": round(total_reward, 4),

                    "detected": final_metrics["detected"],
                    "detection_events": final_metrics["detection_events"],
                    "alerted_hosts": len(env.detected_hosts),
                    "discovered": len(env.discovered),
                    "compromised": len(env.compromised),
                    "active_compromised": final_metrics["active_compromised"],
                    "isolated_compromised": final_metrics["isolated_compromised"],
                    "privileged": len(env.privileged),

                    "risk_total": final_metrics["risk_total"],
                    "max_host_risk": final_metrics["max_host_risk"],

                    "attack_path": list(env.last_path),

                    # Revealed only after the episode ends.
                    "target_node": int(env.crown_jewel),

                    "topology": topology,
                    "metrics": final_metrics,
                    "logs": env.logs.copy(),

                    "detection_events_history": env.detection_events.copy(),
                    "risk_history": risk_history.copy(),

                    "defender_actions": defender_actions,
                    "defender_decisions": defender_decisions,

                    "mission_assisted": mission_assist,
                    "mission_target_attempts": sorted(attempted_targets),
                })

                break

    except WebSocketDisconnect:
        return

    except Exception as exc:
        try:
            await websocket.send_json({
                "type": "error",
                "status": "error",
                "message": str(exc),
            })
        except Exception:
            pass

    finally:
        if env is not None:
            env.close()

        try:
            await websocket.close()
        except Exception:
            pass


# =========================================================
# DEFENDER COMPARISON
# =========================================================

@app.get("/api/defender-comparison")
def defender_comparison():
    summary_path = Path(
        "results/defender_v3_comparison_summary.json"
    )

    if not summary_path.is_file():
        return {
            "available": False,
            "message": (
                "Run scripts/evaluate_defenders.py first."
            ),
        }

    return {
        "available": True,
        "results": json.loads(
            summary_path.read_text(encoding="utf-8")
        ),
    }

