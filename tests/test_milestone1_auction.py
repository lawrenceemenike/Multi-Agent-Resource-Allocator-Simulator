"""Unit tests for Milestone 1: Deterministic Auction & Escrow Validation."""
import pytest
from maras.schemas.contracts import (
    Order,
    OrderType,
    ResourceType,
    AgentAccount,
    PersonaType,
)
from maras.environment.auction import DoubleAuctionEngine


def test_pro_rata_tie_breaking_identical_limit_prices():
    """Unit Test: 2 buyers bidding identical limit prices above available supply

    receive proportional allocations.
    """
    engine = DoubleAuctionEngine(resource_id=ResourceType.ENERGY.value)

    # Accounts
    accounts = {
        "buyer_1": AgentAccount(
            agent_id="buyer_1",
            cash=1000.0,
            inventory={ResourceType.ENERGY.value: 0.0},
            persona=PersonaType.CONSUMER,
        ),
        "buyer_2": AgentAccount(
            agent_id="buyer_2",
            cash=1000.0,
            inventory={ResourceType.ENERGY.value: 0.0},
            persona=PersonaType.CONSUMER,
        ),
        "seller_1": AgentAccount(
            agent_id="seller_1",
            cash=100.0,
            inventory={ResourceType.ENERGY.value: 100.0},
            persona=PersonaType.PRODUCER,
        ),
    }

    # Buyer 1 bids 60 units @ $10.0, Buyer 2 bids 40 units @ $10.0 (Total demand = 100)
    # Seller offers 50 units @ $10.0 (Supply = 50 < Demand = 100)
    orders = [
        Order(
            agent_id="buyer_1",
            resource_id=ResourceType.ENERGY.value,
            order_type=OrderType.BID,
            price=10.0,
            quantity=60.0,
        ),
        Order(
            agent_id="buyer_2",
            resource_id=ResourceType.ENERGY.value,
            order_type=OrderType.BID,
            price=10.0,
            quantity=40.0,
        ),
        Order(
            agent_id="seller_1",
            resource_id=ResourceType.ENERGY.value,
            order_type=OrderType.ASK,
            price=10.0,
            quantity=50.0,
        ),
    ]

    # Pre-clearing escrow scaling
    valid_orders = DoubleAuctionEngine.apply_escrow_scaling(accounts, orders)
    assert len(valid_orders) == 3

    # Clear market
    result = engine.clear_market(valid_orders, accounts)

    assert result.clearing_price == 10.0
    assert result.clearing_volume == pytest.approx(50.0, abs=1e-4)

    # Buyer 1 should receive 60 / (60 + 40) * 50 = 30 units
    # Buyer 2 should receive 40 / (60 + 40) * 50 = 20 units
    b1_inv = accounts["buyer_1"].get_inventory(ResourceType.ENERGY.value)
    b2_inv = accounts["buyer_2"].get_inventory(ResourceType.ENERGY.value)
    s1_inv = accounts["seller_1"].get_inventory(ResourceType.ENERGY.value)

    assert b1_inv == pytest.approx(30.0, abs=1e-4)
    assert b2_inv == pytest.approx(20.0, abs=1e-4)
    assert s1_inv == pytest.approx(50.0, abs=1e-4)

    # Cash balances check
    assert accounts["buyer_1"].cash == pytest.approx(1000.0 - (30.0 * 10.0), abs=1e-4)
    assert accounts["buyer_2"].cash == pytest.approx(1000.0 - (20.0 * 10.0), abs=1e-4)
    assert accounts["seller_1"].cash == pytest.approx(100.0 + (50.0 * 10.0), abs=1e-4)


def test_zero_negative_cash_balances_and_escrow_scaling():
    """Unit Test: Zero negative cash balances possible even with extreme overbidding."""
    engine = DoubleAuctionEngine(resource_id=ResourceType.ENERGY.value)

    # Buyer only has $50 in cash, but bids for 100 units @ $10 ($1,000 liability)
    accounts = {
        "insolvent_buyer": AgentAccount(
            agent_id="insolvent_buyer",
            cash=50.0,
            inventory={ResourceType.ENERGY.value: 0.0},
        ),
        "seller": AgentAccount(
            agent_id="seller",
            cash=0.0,
            inventory={ResourceType.ENERGY.value: 100.0},
        ),
    }

    orders = [
        Order(
            agent_id="insolvent_buyer",
            resource_id=ResourceType.ENERGY.value,
            order_type=OrderType.BID,
            price=10.0,
            quantity=100.0,
        ),
        Order(
            agent_id="seller",
            resource_id=ResourceType.ENERGY.value,
            order_type=OrderType.ASK,
            price=10.0,
            quantity=100.0,
        ),
    ]

    valid_orders = DoubleAuctionEngine.apply_escrow_scaling(accounts, orders)
    # The buyer's bid must be scaled down to 50 / 10 = 5 units
    bids = [o for o in valid_orders if o.agent_id == "insolvent_buyer"]
    assert len(bids) == 1
    assert bids[0].quantity == pytest.approx(5.0, abs=1e-4)

    result = engine.clear_market(valid_orders, accounts)
    assert result.clearing_volume == pytest.approx(5.0, abs=1e-4)
    assert accounts["insolvent_buyer"].cash >= 0.0
    assert accounts["insolvent_buyer"].cash == pytest.approx(0.0, abs=1e-4)
    assert accounts["insolvent_buyer"].get_inventory(ResourceType.ENERGY.value) == pytest.approx(5.0, abs=1e-4)


def test_seller_inventory_escrow_scaling():
    """Unit Test: Seller cannot sell more inventory than physically possessed."""
    engine = DoubleAuctionEngine(resource_id=ResourceType.LABOR.value)

    accounts = {
        "buyer": AgentAccount(
            agent_id="buyer",
            cash=500.0,
            inventory={ResourceType.LABOR.value: 0.0},
        ),
        "seller": AgentAccount(
            agent_id="seller",
            cash=0.0,
            inventory={ResourceType.LABOR.value: 10.0},  # Only 10 units
        ),
    }

    # Seller attempts to sell 50 units @ $5
    orders = [
        Order(
            agent_id="buyer",
            resource_id=ResourceType.LABOR.value,
            order_type=OrderType.BID,
            price=5.0,
            quantity=20.0,
        ),
        Order(
            agent_id="seller",
            resource_id=ResourceType.LABOR.value,
            order_type=OrderType.ASK,
            price=5.0,
            quantity=50.0,
        ),
    ]

    valid_orders = DoubleAuctionEngine.apply_escrow_scaling(accounts, orders)
    asks = [o for o in valid_orders if o.agent_id == "seller"]
    assert len(asks) == 1
    assert asks[0].quantity == pytest.approx(10.0, abs=1e-4)

    result = engine.clear_market(valid_orders, accounts)
    assert result.clearing_volume == pytest.approx(10.0, abs=1e-4)
    assert accounts["seller"].get_inventory(ResourceType.LABOR.value) == pytest.approx(0.0, abs=1e-4)
    assert accounts["seller"].cash == pytest.approx(50.0, abs=1e-4)
