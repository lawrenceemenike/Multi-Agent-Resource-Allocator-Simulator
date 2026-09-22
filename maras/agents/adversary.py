"""Adversarial stress testing agents: ADV-01 (Spoofing) and ADV-02 (Cornering)."""
from __future__ import annotations

import numpy as np
from typing import List, Optional
from maras.schemas.contracts import Order, OrderType, ResourceType, AgentAccount, PersonaType


class AdversaryAgent:
    """Implements ADV-01 (Spoofing) and ADV-02 (Cornering) adversarial attack strategies."""

    def __init__(
        self,
        agent_id: str,
        attack_mode: str = "ADV-02",  # "ADV-01" (Spoofing) or "ADV-02" (Cornering)
        target_resource: str = ResourceType.ENERGY.value,
    ):
        self.agent_id = agent_id
        self.attack_mode = attack_mode
        self.target_resource = target_resource

    def get_action_orders(
        self,
        account: AgentAccount,
        resources: List[str],
        current_prices: dict[str, float],
        timestamp: int = 0,
    ) -> List[Order]:
        """Generates direct adversarial orders bypassing standard random noise."""
        orders: List[Order] = []

        if self.attack_mode == "ADV-02":
            # Cornering: Commit 100% of available cash to bid aggressively on target resource at high price
            cash = max(0.0, account.cash)
            if cash > 1.0:
                target_p = float(current_prices.get(self.target_resource, 10.0) * 1.5)
                # Maximize quantity to corner all available supply
                q_bid = cash / max(target_p, 1e-4)
                orders.append(
                    Order(
                        agent_id=self.agent_id,
                        resource_id=self.target_resource,
                        order_type=OrderType.BID,
                        price=round(target_p, 4),
                        quantity=round(q_bid, 6),
                        timestamp=timestamp,
                    )
                )

        elif self.attack_mode == "ADV-01":
            # Spoofing: Place massive shadow asks or bids at extreme price bounds
            target_p = float(current_prices.get(self.target_resource, 10.0) * 3.0)
            inv = account.get_inventory(self.target_resource)
            if inv > 0.1:
                orders.append(
                    Order(
                        agent_id=self.agent_id,
                        resource_id=self.target_resource,
                        order_type=OrderType.ASK,
                        price=round(target_p, 4),
                        quantity=round(inv * 0.9, 6),
                        timestamp=timestamp,
                    )
                )

        return orders

    def get_continuous_action(
        self,
        resources: List[str],
        account: AgentAccount,
    ) -> np.ndarray:
        """Constructs an unconstrained continuous action vector configured to trigger the attack."""
        num_res = len(resources)
        # Vector structure: [bid_p * k, bid_w * k, ask_p * k, ask_w * k, cash_hold]
        action = np.zeros(num_res * 4 + 1, dtype=np.float32)

        if self.attack_mode == "ADV-02":
            # Find target resource index
            if self.target_resource in resources:
                target_idx = resources.index(self.target_resource)
                # Max bid price (+5.0)
                action[target_idx] = 5.0
                # Max budget allocation for target resource (+10.0)
                action[num_res + target_idx] = 10.0
                # Zero cash reservation
                action[-1] = -10.0
                # Do not sell any target resource (min ask fraction -10.0)
                action[num_res * 3 + target_idx] = -10.0

        elif self.attack_mode == "ADV-01":
            if self.target_resource in resources:
                target_idx = resources.index(self.target_resource)
                # Max ask price (+5.0)
                action[num_res * 2 + target_idx] = 5.0
                # Max sell quantity (+10.0)
                action[num_res * 3 + target_idx] = 10.0

        return action
