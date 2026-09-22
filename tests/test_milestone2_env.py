"""Unit tests for Milestone 2: PettingZoo Environment & Personas."""
import pytest
import numpy as np
from maras.schemas.contracts import ResourceType
from maras.environment.economy_env import EconomyEnv, ProductionEngine, ActionProjector


def test_production_engines():
    """Test Cobb-Douglas and CES mathematical properties."""
    inputs = {
        ResourceType.ENERGY.value: 8.0,
        ResourceType.LABOR.value: 8.0,
        ResourceType.RAW_MATERIALS.value: 8.0,
    }
    alphas = {
        ResourceType.ENERGY.value: 1/3,
        ResourceType.LABOR.value: 1/3,
        ResourceType.RAW_MATERIALS.value: 1/3,
    }

    # Cobb-Douglas: (8^(1/3)) * (8^(1/3)) * (8^(1/3)) = 2 * 2 * 2 = 8.0
    cd_out = ProductionEngine.cobb_douglas(inputs, alphas, tfp_a=1.0)
    assert cd_out == pytest.approx(8.0, abs=1e-4)

    # CES with rho -> 0 or rho = 0.5
    ces_out = ProductionEngine.ces(inputs, alphas, rho=0.5, gamma=1.0, tfp_a=1.0)
    assert ces_out > 0.0

    # Zero input should result in zero production
    zero_inputs = {ResourceType.ENERGY.value: 0.0, ResourceType.LABOR.value: 8.0}
    assert ProductionEngine.cobb_douglas(zero_inputs, alphas) == 0.0


def test_500_steps_random_actions_resource_conservation_and_bounds():
    """Unit Test: Run 500 steps with random actions;

    verify strict resource conservation sum(Delta Resources) <= Q_supply and zero out-of-bounds orders.
    """
    env = EconomyEnv(num_agents=6, max_steps=500, production_type="cobb_douglas", seed=42)
    obs, infos = env.reset(seed=42)

    total_steps_executed = 0

    for step in range(500):
        # Sample completely unconstrained random continuous actions
        actions = {
            agent: np.random.uniform(-10.0, 10.0, size=env.action_spaces[agent].shape).astype(np.float32)
            for agent in env.agents
        }

        # Snapshot inventories before step
        inv_before = {}
        for res in env.resources:
            inv_before[res] = sum(acc.get_inventory(res) for acc in env.accounts.values())
        cash_before = sum(acc.cash for acc in env.accounts.values())

        # Step the environment
        obs, rewards, terminations, truncations, infos = env.step(actions)
        total_steps_executed += 1

        # Check: Zero out-of-bounds accounts
        for agent_id, acc in env.accounts.items():
            assert acc.cash >= -1e-9, f"Agent {agent_id} has negative cash: {acc.cash}"
            for res in env.resources:
                assert acc.get_inventory(res) >= -1e-9, f"Agent {agent_id} has negative {res}: {acc.get_inventory(res)}"

        # Check: Zero out-of-bounds orders in this step's logs
        last_log = env.step_history[-1]
        for res, clr_res in last_log.clearing_results.items():
            for trade in clr_res.trades:
                assert trade.price > 0, f"Non-positive trade price: {trade.price}"
                assert trade.quantity > 0, f"Non-positive trade quantity: {trade.quantity}"

        # Check: Strict Resource Conservation
        # Endowments injected:
        # Producers (2 agents) get +5 Energy, +5 Raw Materials
        # Consumers (2 agents) get +5 Labor
        # Production converts inputs -> Finished Goods
        # Total change in any raw resource <= injected supply
        for res in env.resources:
            inv_after_res = sum(acc.get_inventory(res) for acc in env.accounts.values())
            delta_res = inv_after_res - inv_before[res]
            # Max possible injected supply per step
            max_injected_supply = 20.0  # Safe upper bound for single step injection
            if res != ResourceType.FINISHED_GOODS.value:
                # Delta resource <= injected supply (since consumption only reduces inventory)
                assert delta_res <= max_injected_supply + 1e-4, (
                    f"Resource {res} violated conservation: delta={delta_res} > max_supply={max_injected_supply}"
                )

        # Check total cash conservation across trades (cash only transfers between buyers and sellers)
        cash_after = sum(acc.cash for acc in env.accounts.values())
        assert cash_after == pytest.approx(cash_before, abs=1e-4), "Cash was not conserved during trading"

        if all(truncations.values()) or not env.agents:
            obs, infos = env.reset()

    assert total_steps_executed == 500
