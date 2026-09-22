"""PettingZoo Parallel Economy Environment with Cobb-Douglas / CES Production and Continuous Action Projections."""
from __future__ import annotations

import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
from typing import Dict, List, Tuple, Optional, Any

from maras.schemas.contracts import (
    Order,
    OrderType,
    ResourceType,
    PersonaType,
    AgentAccount,
    AuctionClearingResult,
    EpisodeStepLog,
)
from maras.environment.auction import DoubleAuctionEngine


class ProductionEngine:
    """Implements Cobb-Douglas and CES (Constant Elasticity of Substitution) production functions."""

    @staticmethod
    def cobb_douglas(
        inputs: Dict[str, float],
        alphas: Dict[str, float],
        tfp_a: float = 1.0,
    ) -> float:
        """Y = A * Prod(X_i ^ alpha_i) where sum(alpha_i) <= 1."""
        output = tfp_a
        for res_id, alpha in alphas.items():
            x_i = max(0.0, inputs.get(res_id, 0.0))
            if x_i <= 1e-9 and alpha > 0:
                return 0.0
            output *= (x_i ** alpha)
        return float(output)

    @staticmethod
    def ces(
        inputs: Dict[str, float],
        alphas: Dict[str, float],
        rho: float = 0.5,
        gamma: float = 1.0,
        tfp_a: float = 1.0,
    ) -> float:
        """Y = A * (sum(alpha_i * X_i^rho)) ^ (gamma / rho) where rho <= 1, rho != 0."""
        if abs(rho) < 1e-6:
            # Limit as rho -> 0 is Cobb-Douglas
            return ProductionEngine.cobb_douglas(inputs, alphas, tfp_a)
        
        inner_sum = 0.0
        for res_id, alpha in alphas.items():
            x_i = max(0.0, inputs.get(res_id, 0.0))
            if x_i > 0:
                inner_sum += alpha * (x_i ** rho)
        
        if inner_sum <= 0:
            return 0.0
        return float(tfp_a * (inner_sum ** (gamma / rho)))


class ActionProjector:
    """Projects unconstrained continuous neural network actions to valid, bounded auction orders.

    Uses log-uniform price scaling and softmax budget allocations.
    """

    def __init__(
        self,
        resources: List[str],
        price_min: float = 0.5,
        price_max: float = 50.0,
    ):
        self.resources = resources
        self.num_res = len(resources)
        self.price_min = price_min
        self.price_max = price_max
        self.log_p_min = np.log(price_min)
        self.log_p_max = np.log(price_max)

    @property
    def action_dim(self) -> int:
        # For each resource:
        # [bid_price_norm, bid_budget_logit, ask_price_norm, ask_qty_logit]
        # plus 1 extra logit for cash reservation (hold cash)
        return self.num_res * 4 + 1

    def project(
        self,
        agent_id: str,
        account: AgentAccount,
        raw_action: np.ndarray,
        timestamp: int = 0,
    ) -> List[Order]:
        """Maps raw neural network action [-inf, inf] or [-1, 1] to strictly feasible Orders."""
        orders: List[Order] = []
        raw_action = np.nan_to_num(raw_action, nan=0.0, posinf=5.0, neginf=-5.0)

        # Slice logits
        idx = 0
        bid_prices_raw = raw_action[idx : idx + self.num_res]
        idx += self.num_res
        bid_budget_logits = raw_action[idx : idx + self.num_res]
        idx += self.num_res
        ask_prices_raw = raw_action[idx : idx + self.num_res]
        idx += self.num_res
        ask_qty_logits = raw_action[idx : idx + self.num_res]
        idx += self.num_res
        cash_hold_logit = raw_action[idx] if idx < len(raw_action) else 0.0

        # 1. Log-uniform price projections: mapping [-1, 1] -> [P_min, P_max]
        # Using tanh to clamp raw prices smoothly into [-1, 1]
        norm_bid_p = (np.tanh(bid_prices_raw) + 1.0) / 2.0  # [0, 1]
        bid_prices = np.exp(self.log_p_min + norm_bid_p * (self.log_p_max - self.log_p_min))

        norm_ask_p = (np.tanh(ask_prices_raw) + 1.0) / 2.0  # [0, 1]
        ask_prices = np.exp(self.log_p_min + norm_ask_p * (self.log_p_max - self.log_p_min))

        # 2. Softmax budget allocation across bids + cash reserve
        # Bounded within available cash balance
        all_budget_logits = np.append(bid_budget_logits, cash_hold_logit)
        # Shift for numerical stability
        exp_logits = np.exp(all_budget_logits - np.max(all_budget_logits))
        budget_shares = exp_logits / (np.sum(exp_logits) + 1e-12)

        cash_available = max(0.0, account.cash)
        for i, res_id in enumerate(self.resources):
            bid_cash_allocated = cash_available * budget_shares[i]
            p_bid = float(bid_prices[i])
            q_bid = bid_cash_allocated / max(p_bid, 1e-6)
            if q_bid > 1e-6:
                orders.append(
                    Order(
                        agent_id=agent_id,
                        resource_id=res_id,
                        order_type=OrderType.BID,
                        price=round(p_bid, 4),
                        quantity=round(float(q_bid), 6),
                        timestamp=timestamp,
                    )
                )

        # 3. Sigmoid sell fraction allocations for asks
        # Bounded within available resource inventories
        for i, res_id in enumerate(self.resources):
            inv = account.get_inventory(res_id)
            if inv > 1e-6:
                sell_fraction = 1.0 / (1.0 + np.exp(-ask_qty_logits[i]))  # sigmoid in [0, 1]
                q_ask = inv * sell_fraction
                p_ask = float(ask_prices[i])
                if q_ask > 1e-6:
                    orders.append(
                        Order(
                            agent_id=agent_id,
                            resource_id=res_id,
                            order_type=OrderType.ASK,
                            price=round(p_ask, 4),
                            quantity=round(float(q_ask), 6),
                            timestamp=timestamp,
                        )
                    )

        return orders


class EconomyEnv(ParallelEnv):
    """Multi-Agent PettingZoo Parallel Environment for resource allocation and production economics."""

    metadata = {"render_modes": ["human"], "name": "maras_economy_v1"}

    def __init__(
        self,
        num_agents: int = 6,
        resources: Optional[List[str]] = None,
        max_steps: int = 100,
        reward_mode: str = "competitive",  # "competitive", "cooperative", "mixed"
        coop_lambda: float = 0.5,
        production_type: str = "cobb_douglas",  # "cobb_douglas" or "ces"
        seed: Optional[int] = None,
    ):
        super().__init__()
        if resources is None:
            self.resources = [
                ResourceType.ENERGY.value,
                ResourceType.LABOR.value,
                ResourceType.RAW_MATERIALS.value,
                ResourceType.FINISHED_GOODS.value,
            ]
        else:
            self.resources = resources

        self._num_agents = num_agents
        self.max_steps = max_steps
        self.reward_mode = reward_mode
        self.coop_lambda = coop_lambda
        self.production_type = production_type

        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = copy.copy(self.possible_agents)

        # Personas assignment
        self.personas: Dict[str, PersonaType] = {}
        for i, agent_id in enumerate(self.possible_agents):
            if i % 3 == 0:
                self.personas[agent_id] = PersonaType.PRODUCER
            elif i % 3 == 1:
                self.personas[agent_id] = PersonaType.CONSUMER
            else:
                self.personas[agent_id] = PersonaType.SPECULATOR

        self.action_projector = ActionProjector(self.resources)
        self.auction_engines = {res: DoubleAuctionEngine(res) for res in self.resources}

        # Spaces
        # Obs: [cash, inv_0..inv_k, persona_onehot_4, last_clearing_prices_k, last_clearing_volumes_k]
        self.num_res = len(self.resources)
        obs_dim = 1 + self.num_res + 4 + self.num_res + self.num_res
        self.observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Box(
                low=-5.0, high=5.0, shape=(self.action_projector.action_dim,), dtype=np.float32
            )
            for agent in self.possible_agents
        }

        # Internal state
        self.accounts: Dict[str, AgentAccount] = {}
        self.current_step = 0
        self.last_clearing_prices: Dict[str, float] = {res: 10.0 for res in self.resources}
        self.last_clearing_volumes: Dict[str, float] = {res: 0.0 for res in self.resources}
        self.step_history: List[EpisodeStepLog] = []

        # Production inputs & output target
        self.production_inputs = [
            ResourceType.ENERGY.value,
            ResourceType.LABOR.value,
            ResourceType.RAW_MATERIALS.value,
        ]
        self.production_output = ResourceType.FINISHED_GOODS.value
        self.prod_alphas = {
            ResourceType.ENERGY.value: 0.3,
            ResourceType.LABOR.value: 0.3,
            ResourceType.RAW_MATERIALS.value: 0.3,
        }

        # Supply injections per step to verify conservation bounds
        self.endowment_supply_per_step: Dict[str, float] = {
            ResourceType.ENERGY.value: 20.0,
            ResourceType.LABOR.value: 20.0,
            ResourceType.RAW_MATERIALS.value: 20.0,
            ResourceType.FINISHED_GOODS.value: 0.0,
        }

        self._rng = np.random.default_rng(seed)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.agents = copy.copy(self.possible_agents)
        self.current_step = 0
        self.step_history.clear()

        # Initialize agent accounts with initial endowments
        self.accounts = {}
        for agent_id in self.possible_agents:
            persona = self.personas[agent_id]
            if persona == PersonaType.PRODUCER:
                cash = 500.0
                inv = {
                    ResourceType.ENERGY.value: 10.0,
                    ResourceType.LABOR.value: 10.0,
                    ResourceType.RAW_MATERIALS.value: 10.0,
                    ResourceType.FINISHED_GOODS.value: 5.0,
                }
            elif persona == PersonaType.CONSUMER:
                cash = 800.0
                inv = {
                    ResourceType.ENERGY.value: 2.0,
                    ResourceType.LABOR.value: 15.0,
                    ResourceType.RAW_MATERIALS.value: 0.0,
                    ResourceType.FINISHED_GOODS.value: 0.0,
                }
            else:  # SPECULATOR
                cash = 1000.0
                inv = {res: 5.0 for res in self.resources}

            self.accounts[agent_id] = AgentAccount(
                agent_id=agent_id,
                cash=cash,
                inventory=inv,
                persona=persona,
            )

        self.last_clearing_prices = {res: 10.0 for res in self.resources}
        self.last_clearing_volumes = {res: 0.0 for res in self.resources}

        obs = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return obs, infos

    def _get_obs(self, agent_id: str) -> np.ndarray:
        acc = self.accounts[agent_id]
        # Persona one-hot (4 dims)
        persona_onehot = np.zeros(4, dtype=np.float32)
        persona_map = {
            PersonaType.PRODUCER: 0,
            PersonaType.CONSUMER: 1,
            PersonaType.SPECULATOR: 2,
            PersonaType.ADVERSARY: 3,
        }
        persona_onehot[persona_map.get(acc.persona, 0)] = 1.0

        inv_vec = np.array([acc.get_inventory(res) for res in self.resources], dtype=np.float32)
        prices_vec = np.array([self.last_clearing_prices.get(res, 0.0) for res in self.resources], dtype=np.float32)
        volumes_vec = np.array([self.last_clearing_volumes.get(res, 0.0) for res in self.resources], dtype=np.float32)

        return np.concatenate(
            ([float(acc.cash)], inv_vec, persona_onehot, prices_vec, volumes_vec)
        ).astype(np.float32)

    def get_global_state(self) -> np.ndarray:
        """Returns concatenated centralized state for MAPPO centralized critic."""
        state_parts = []
        for agent_id in self.possible_agents:
            state_parts.append(self._get_obs(agent_id))
        return np.concatenate(state_parts).astype(np.float32)

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        self.current_step += 1

        # 1. Inject natural endowments per step to producers/workers
        for i, agent_id in enumerate(self.agents):
            acc = self.accounts[agent_id]
            if acc.persona == PersonaType.PRODUCER:
                acc.inventory[ResourceType.ENERGY.value] = (
                    acc.inventory.get(ResourceType.ENERGY.value, 0.0) + 5.0
                )
                acc.inventory[ResourceType.RAW_MATERIALS.value] = (
                    acc.inventory.get(ResourceType.RAW_MATERIALS.value, 0.0) + 5.0
                )
            elif acc.persona == PersonaType.CONSUMER:
                acc.inventory[ResourceType.LABOR.value] = (
                    acc.inventory.get(ResourceType.LABOR.value, 0.0) + 5.0
                )

        # 2. Project actions to bounded orders
        all_orders: List[Order] = []
        for agent_id, raw_act in actions.items():
            if agent_id in self.accounts:
                agent_orders = self.action_projector.project(
                    agent_id=agent_id,
                    account=self.accounts[agent_id],
                    raw_action=raw_act,
                    timestamp=self.current_step,
                )
                all_orders.extend(agent_orders)

        # 3. Apply pre-clearing escrow scaling
        valid_orders = DoubleAuctionEngine.apply_escrow_scaling(self.accounts, all_orders)

        # 4. Clear auction across all commodity markets
        clearing_results: Dict[str, AuctionClearingResult] = {}
        for res in self.resources:
            engine = self.auction_engines[res]
            res_result = engine.clear_market(valid_orders, self.accounts, timestamp=self.current_step)
            clearing_results[res] = res_result
            if res_result.clearing_price is not None:
                self.last_clearing_prices[res] = res_result.clearing_price
            self.last_clearing_volumes[res] = res_result.clearing_volume

        # 5. Production & Consumption phase
        production_outputs: Dict[str, float] = {}
        utilities: Dict[str, float] = {}

        for agent_id, acc in self.accounts.items():
            if acc.persona == PersonaType.PRODUCER:
                # Use fraction of available inputs to produce finished goods
                inputs_used = {
                    res: min(acc.get_inventory(res), 3.0) for res in self.production_inputs
                }
                # Deduct consumed inputs
                for res, amt in inputs_used.items():
                    acc.inventory[res] = max(0.0, acc.inventory.get(res, 0.0) - amt)

                # Compute production output
                if self.production_type == "ces":
                    y_out = ProductionEngine.ces(inputs_used, self.prod_alphas, rho=0.5, gamma=1.0)
                else:
                    y_out = ProductionEngine.cobb_douglas(inputs_used, self.prod_alphas)

                acc.inventory[self.production_output] = (
                    acc.inventory.get(self.production_output, 0.0) + y_out
                )
                production_outputs[agent_id] = y_out
                # Producer utility: production profit + cash growth
                utilities[agent_id] = float(y_out * 10.0 + (acc.cash * 0.01))

            elif acc.persona == PersonaType.CONSUMER:
                # Consumes finished goods and household electricity/energy for utility
                fg_avail = acc.get_inventory(ResourceType.FINISHED_GOODS.value)
                consumed_fg = min(fg_avail, 4.0)
                acc.inventory[ResourceType.FINISHED_GOODS.value] = max(0.0, fg_avail - consumed_fg)

                energy_avail = acc.get_inventory(ResourceType.ENERGY.value)
                consumed_energy = min(energy_avail, 2.0)
                acc.inventory[ResourceType.ENERGY.value] = max(0.0, energy_avail - consumed_energy)

                # Diminishing marginal utility of goods and power U(c, e) = sqrt(c) + 0.5*sqrt(e)
                u_c = float(np.sqrt(consumed_fg) * 12.0)
                u_e = float(np.sqrt(consumed_energy) * 6.0)
                utilities[agent_id] = float(u_c + u_e + (acc.cash * 0.01))

            else:  # SPECULATOR / ADVERSARY
                # Utility is net wealth: Cash + sum(Inventory * Price)
                total_wealth = acc.cash + sum(
                    acc.get_inventory(r) * self.last_clearing_prices.get(r, 10.0) for r in self.resources
                )
                utilities[agent_id] = float(total_wealth * 0.02)

        # 6. Rewards according to schedule
        rewards: Dict[str, float] = {}
        social_welfare = float(sum(utilities.values()) / max(1, len(self.agents)))

        for agent_id in self.agents:
            u_i = utilities.get(agent_id, 0.0)
            if self.reward_mode == "competitive":
                rewards[agent_id] = u_i
            elif self.reward_mode == "cooperative":
                rewards[agent_id] = social_welfare
            elif self.reward_mode == "mixed":
                rewards[agent_id] = self.coop_lambda * social_welfare + (1.0 - self.coop_lambda) * u_i
            else:
                rewards[agent_id] = u_i

        # 7. Econometrics and Anomaly Metrics
        wealths = [
            acc.cash + sum(acc.get_inventory(r) * self.last_clearing_prices.get(r, 10.0) for r in self.resources)
            for acc in self.accounts.values()
        ]
        from maras.metrics.econometrics import calculate_gini, calculate_hhi, detect_anomalies
        gini_wealth = calculate_gini(wealths)

        hhi_by_res = {}
        for res in self.resources:
            res_holdings = {aid: acc.get_inventory(res) for aid, acc in self.accounts.items()}
            hhi_by_res[res] = calculate_hhi(res_holdings)

        alerts = detect_anomalies(hhi_by_res, hhi_cornering_threshold=6500.0)

        # 8. Termination & Truncation
        truncated = self.current_step >= self.max_steps
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}

        # Step log
        log_entry = EpisodeStepLog(
            step=self.current_step,
            clearing_results=clearing_results,
            agent_accounts={aid: copy.deepcopy(acc) for aid, acc in self.accounts.items()},
            gini_wealth=gini_wealth,
            hhi_by_resource=hhi_by_res,
            anomaly_alerts=alerts,
            production_output=production_outputs,
            social_welfare=social_welfare,
        )
        self.step_history.append(log_entry)

        if truncated:
            self.agents = []

        obs = {agent: self._get_obs(agent) for agent in self.agents if agent in self.accounts}
        infos = {
            agent: {
                "utility": utilities.get(agent, 0.0),
                "cash": self.accounts[agent].cash if agent in self.accounts else 0.0,
            }
            for agent in actions.keys()
        }

        return obs, rewards, terminations, truncations, infos
