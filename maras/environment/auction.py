"""Deterministic Double Auction Engine with Pre-Clearing Escrow and Pro-Rata Tie-Breaking."""
from __future__ import annotations

import copy
from typing import Dict, List, Tuple, Optional
from maras.schemas.contracts import (
    Order,
    OrderType,
    Trade,
    AgentAccount,
    AuctionClearingResult,
)


class DoubleAuctionEngine:
    """Executes deterministic discrete uniform-price double auctions with pre-clearing escrow scaling."""

    def __init__(self, resource_id: str):
        self.resource_id = resource_id

    @staticmethod
    def apply_escrow_scaling(
        accounts: Dict[str, AgentAccount],
        orders: List[Order],
    ) -> List[Order]:
        """Scales down orders proportionally if total commitments exceed available cash or inventory.

        Guarantees:
        1. Buyer total bid liabilities (sum(P * Q)) <= available cash.
        2. Seller total ask quantities (sum(Q)) <= available resource inventory.
        """
        valid_orders: List[Order] = []

        # Group bids and asks by agent
        bids_by_agent: Dict[str, List[Order]] = {}
        asks_by_agent: Dict[str, Dict[str, List[Order]]] = {}

        for order in orders:
            if order.quantity <= 1e-9 or order.price <= 0:
                continue
            if order.order_type == OrderType.BID:
                bids_by_agent.setdefault(order.agent_id, []).append(copy.deepcopy(order))
            elif order.order_type == OrderType.ASK:
                asks_by_agent.setdefault(order.agent_id, {}).setdefault(order.resource_id, []).append(
                    copy.deepcopy(order)
                )

        # Scale bids across all commodities based on agent cash
        for agent_id, agent_bids in bids_by_agent.items():
            account = accounts.get(agent_id)
            if not account or account.cash <= 0:
                continue

            total_cash_liability = sum(b.price * b.quantity for b in agent_bids)
            scale = 1.0
            if total_cash_liability > account.cash:
                scale = account.cash / max(total_cash_liability, 1e-12)

            for b in agent_bids:
                scaled_qty = b.quantity * scale
                if scaled_qty > 1e-9:
                    b.quantity = scaled_qty
                    valid_orders.append(b)

        # Scale asks per commodity based on agent inventory
        for agent_id, res_dict in asks_by_agent.items():
            account = accounts.get(agent_id)
            if not account:
                continue
            for res_id, agent_asks in res_dict.items():
                curr_inv = account.get_inventory(res_id)
                if curr_inv <= 0:
                    continue
                total_ask_qty = sum(a.quantity for a in agent_asks)
                scale = 1.0
                if total_ask_qty > curr_inv:
                    scale = curr_inv / max(total_ask_qty, 1e-12)

                for a in agent_asks:
                    scaled_qty = a.quantity * scale
                    if scaled_qty > 1e-9:
                        a.quantity = scaled_qty
                        valid_orders.append(a)

        return valid_orders

    def clear_market(
        self,
        orders: List[Order],
        accounts: Dict[str, AgentAccount],
        timestamp: int = 0,
    ) -> AuctionClearingResult:
        """Clears the double auction deterministically for a single resource.

        Matches bids and asks using uniform clearing price and pro-rata tie breaking.
        Directly settles cash and inventory balances in the provided accounts dict.
        """
        # Step 1: Filter orders for this resource
        res_orders = [o for o in orders if o.resource_id == self.resource_id and o.quantity > 1e-9]

        bids = [copy.deepcopy(o) for o in res_orders if o.order_type == OrderType.BID]
        asks = [copy.deepcopy(o) for o in res_orders if o.order_type == OrderType.ASK]

        # Step 2: Build supply and demand curves
        # Bids sorted descending by price, asks sorted ascending by price
        bids.sort(key=lambda x: (-x.price, x.agent_id))
        asks.sort(key=lambda x: (x.price, x.agent_id))

        demand_curve: List[Tuple[float, float]] = []
        cum_d = 0.0
        for b in bids:
            cum_d += b.quantity
            demand_curve.append((b.price, cum_d))

        supply_curve: List[Tuple[float, float]] = []
        cum_s = 0.0
        for a in asks:
            cum_s += a.quantity
            supply_curve.append((a.price, cum_s))

        if not bids or not asks or bids[0].price < asks[0].price:
            return AuctionClearingResult(
                resource_id=self.resource_id,
                clearing_price=None,
                clearing_volume=0.0,
                trades=[],
                bids_unfilled=bids,
                asks_unfilled=asks,
                demand_curve=demand_curve,
                supply_curve=supply_curve,
            )

        # Step 3: Find maximum total clearable volume and clearing price
        # Form discrete price intervals from all unique prices
        all_prices = sorted(list(set([b.price for b in bids] + [a.price for a in asks])))

        # Find maximum crossing quantity
        # For any price P, demand D(P) is sum of bids with price >= P
        # supply S(P) is sum of asks with price <= P
        # Clearable volume at price P is min(D(P), S(P))
        best_p = None
        max_vol = 0.0
        for p in all_prices:
            d_p = sum(b.quantity for b in bids if b.price >= p - 1e-7)
            s_p = sum(a.quantity for a in asks if a.price <= p + 1e-7)
            vol = min(d_p, s_p)
            if vol > max_vol + 1e-7:
                max_vol = vol
                best_p = p
            elif abs(vol - max_vol) <= 1e-7 and best_p is not None:
                # If tied on volume, choose midpoint
                best_p = (best_p + p) / 2.0

        if max_vol <= 1e-9 or best_p is None:
            return AuctionClearingResult(
                resource_id=self.resource_id,
                clearing_price=None,
                clearing_volume=0.0,
                trades=[],
                bids_unfilled=bids,
                asks_unfilled=asks,
                demand_curve=demand_curve,
                supply_curve=supply_curve,
            )

        clearing_price = round(best_p, 4)
        total_clearing_volume = max_vol

        # Step 4: Allocate cleared volume pro-rata among qualified buyers and sellers
        # Buyer allocation
        in_money_bids = [b for b in bids if b.price > clearing_price + 1e-7]
        at_money_bids = [b for b in bids if abs(b.price - clearing_price) <= 1e-7]

        in_money_bid_vol = sum(b.quantity for b in in_money_bids)
        rem_bid_vol = max(0.0, total_clearing_volume - in_money_bid_vol)
        at_money_bid_total = sum(b.quantity for b in at_money_bids)

        buyer_fills: List[Tuple[Order, float]] = []
        for b in in_money_bids:
            buyer_fills.append((b, b.quantity))
        for b in at_money_bids:
            if at_money_bid_total > 1e-9:
                fill = rem_bid_vol * (b.quantity / at_money_bid_total)
                buyer_fills.append((b, min(b.quantity, fill)))

        # Seller allocation
        in_money_asks = [a for a in asks if a.price < clearing_price - 1e-7]
        at_money_asks = [a for a in asks if abs(a.price - clearing_price) <= 1e-7]

        in_money_ask_vol = sum(a.quantity for a in in_money_asks)
        rem_ask_vol = max(0.0, total_clearing_volume - in_money_ask_vol)
        at_money_ask_total = sum(a.quantity for a in at_money_asks)

        seller_fills: List[Tuple[Order, float]] = []
        for a in in_money_asks:
            seller_fills.append((a, a.quantity))
        for a in at_money_asks:
            if at_money_ask_total > 1e-9:
                fill = rem_ask_vol * (a.quantity / at_money_ask_total)
                seller_fills.append((a, min(a.quantity, fill)))

        # Pair buyer fills and seller fills into trades
        trades: List[Trade] = []
        s_idx = 0
        s_rem = seller_fills[0][1] if seller_fills else 0.0

        for b_order, b_fill in buyer_fills:
            b_rem = b_fill
            while b_rem > 1e-9 and s_idx < len(seller_fills):
                s_order, _ = seller_fills[s_idx]
                match_qty = min(b_rem, s_rem)
                rounded_match_qty = round(match_qty, 6)
                if rounded_match_qty > 1e-6:
                    trades.append(
                        Trade(
                            buyer_id=b_order.agent_id,
                            seller_id=s_order.agent_id,
                            resource_id=self.resource_id,
                            price=clearing_price,
                            quantity=rounded_match_qty,
                            timestamp=timestamp,
                        )
                    )
                b_rem -= match_qty
                s_rem -= match_qty
                if s_rem <= 1e-9:
                    s_idx += 1
                    if s_idx < len(seller_fills):
                        s_rem = seller_fills[s_idx][1]

        # Step 5: Settle trades directly in accounts
        for t in trades:
            buyer_acc = accounts.get(t.buyer_id)
            seller_acc = accounts.get(t.seller_id)
            cost = t.price * t.quantity

            if buyer_acc:
                buyer_acc.cash = max(0.0, buyer_acc.cash - cost)
                buyer_acc.inventory[self.resource_id] = (
                    buyer_acc.inventory.get(self.resource_id, 0.0) + t.quantity
                )

            if seller_acc:
                seller_acc.cash += cost
                seller_acc.inventory[self.resource_id] = max(
                    0.0, seller_acc.inventory.get(self.resource_id, 0.0) - t.quantity
                )

        # Unfilled orders calculation
        filled_bid_quantities: Dict[str, float] = {}
        filled_ask_quantities: Dict[str, float] = {}
        for t in trades:
            filled_bid_quantities[t.buyer_id] = filled_bid_quantities.get(t.buyer_id, 0.0) + t.quantity
            filled_ask_quantities[t.seller_id] = filled_ask_quantities.get(t.seller_id, 0.0) + t.quantity

        bids_unfilled: List[Order] = []
        for b in bids:
            rem = b.quantity - filled_bid_quantities.get(b.agent_id, 0.0)
            if rem > 1e-6:
                unfilled_b = copy.deepcopy(b)
                unfilled_b.quantity = rem
                bids_unfilled.append(unfilled_b)

        asks_unfilled: List[Order] = []
        for a in asks:
            rem = a.quantity - filled_ask_quantities.get(a.agent_id, 0.0)
            if rem > 1e-6:
                unfilled_a = copy.deepcopy(a)
                unfilled_a.quantity = rem
                asks_unfilled.append(unfilled_a)

        return AuctionClearingResult(
            resource_id=self.resource_id,
            clearing_price=clearing_price,
            clearing_volume=round(total_clearing_volume, 6),
            trades=trades,
            bids_unfilled=bids_unfilled,
            asks_unfilled=asks_unfilled,
            demand_curve=demand_curve,
            supply_curve=supply_curve,
        )
