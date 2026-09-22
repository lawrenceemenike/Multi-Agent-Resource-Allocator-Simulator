"""Integration tests for Milestone 5: Adversarial Stress Testing & Anomaly Metrics."""
import pytest
import numpy as np
from maras.schemas.contracts import ResourceType, PersonaType
from maras.environment.economy_env import EconomyEnv
from maras.metrics.econometrics import calculate_gini, calculate_hhi, detect_anomalies
from maras.agents.adversary import AdversaryAgent


def test_gini_and_hhi_mathematical_properties():
    """Verify Gini and HHI calculations across uniform, monopolistic, and edge case distributions."""
    # Perfect equality: 4 agents with $100 each
    assert calculate_gini([100.0, 100.0, 100.0, 100.0]) == pytest.approx(0.0, abs=1e-4)

    # Pure monopoly: 1 agent has 100% of the resource
    assert calculate_hhi({"agent_0": 100.0, "agent_1": 0.0, "agent_2": 0.0}) == pytest.approx(10000.0, abs=1e-4)

    # 4 equal competitors (25% each): HHI = 4 * (25^2) = 2500
    assert calculate_hhi([25.0, 25.0, 25.0, 25.0]) == pytest.approx(2500.0, abs=1e-4)

    # Anomaly alerts
    alerts = detect_anomalies({"ENERGY": 5500.0, "LABOR": 1800.0}, hhi_cornering_threshold=4000.0)
    assert len(alerts) == 1
    assert "ADV-02 ALERT" in alerts[0]
    assert "ENERGY" in alerts[0]


def test_adv02_cornering_triggers_hhi_anomaly_alert():
    """Integration Test: Assert ADV-02 alert triggers when adversary corners energy supply (HHI > 4,000)."""
    env = EconomyEnv(num_agents=6, max_steps=50, seed=42)
    obs, _ = env.reset()

    # Assign agent_0 as an adversarial cornering agent with massive capital
    adv_id = "agent_0"
    env.personas[adv_id] = PersonaType.ADVERSARY
    env.accounts[adv_id].persona = PersonaType.ADVERSARY
    env.accounts[adv_id].cash = 50000.0  # Massive war-chest to corner the energy supply

    adversary = AdversaryAgent(
        agent_id=adv_id,
        attack_mode="ADV-02",
        target_resource=ResourceType.ENERGY.value,
    )

    alert_triggered = False
    max_hhi_observed = 0.0

    for step in range(30):
        actions = {}
        for agent_id in env.agents:
            if agent_id == adv_id:
                # Adversary executes continuous aggressive bidding on Energy
                actions[agent_id] = adversary.get_continuous_action(env.resources, env.accounts[agent_id])
            else:
                # Normal background agents selling inventory / normal random actions
                actions[agent_id] = np.random.uniform(-1.0, 1.0, size=env.action_spaces[agent_id].shape).astype(np.float32)

        obs, rewards, terminations, truncations, infos = env.step(actions)

        # Inspect latest step log
        last_log = env.step_history[-1]
        energy_hhi = last_log.hhi_by_resource.get(ResourceType.ENERGY.value, 0.0)
        max_hhi_observed = max(max_hhi_observed, energy_hhi)

        if energy_hhi > 4000.0 and len(last_log.anomaly_alerts) > 0:
            alert_triggered = True
            break

    assert max_hhi_observed > 4000.0, f"Energy HHI did not breach 4,000: Max HHI = {max_hhi_observed}"
    assert alert_triggered is True, "ADV-02 Anomaly alert failed to trigger"
