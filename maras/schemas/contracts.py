"""Pydantic contracts and schemas for the Multi-Agent Resource Allocation Simulator."""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


class OrderType(str, Enum):
    BID = "BID"  # Buy order
    ASK = "ASK"  # Sell order


class ResourceType(str, Enum):
    ENERGY = "ENERGY"
    LABOR = "LABOR"
    RAW_MATERIALS = "RAW_MATERIALS"
    COMPUTE = "COMPUTE"
    FINISHED_GOODS = "FINISHED_GOODS"


class PersonaType(str, Enum):
    PRODUCER = "PRODUCER"          # Focuses on buying raw inputs, converting via production function, selling finished goods
    CONSUMER = "CONSUMER"          # Focuses on purchasing finished goods to maximize consumption utility
    SPECULATOR = "SPECULATOR"      # Exploits price discrepancies across turns/markets
    ADVERSARY = "ADVERSARY"        # Executes market manipulation attacks (Spoofing / Cornering)


class Order(BaseModel):
    agent_id: str
    resource_id: str
    order_type: OrderType
    price: float = Field(gt=0, description="Limit price (must be strictly positive)")
    quantity: float = Field(ge=0, description="Order quantity (>= 0)")
    timestamp: int = 0

    @field_validator("price", "quantity")
    @classmethod
    def validate_finite(cls, v: float) -> float:
        if v != v or v == float("inf") or v == float("-inf"):
            raise ValueError("Price and quantity must be finite real numbers")
        return float(v)


class Trade(BaseModel):
    buyer_id: str
    seller_id: str
    resource_id: str
    price: float = Field(gt=0)
    quantity: float = Field(gt=0)
    timestamp: int = 0


class AgentAccount(BaseModel):
    agent_id: str
    cash: float = Field(ge=0.0, description="Current unreserved cash balance")
    inventory: Dict[str, float] = Field(default_factory=dict, description="Current unreserved resource inventories")
    persona: PersonaType = PersonaType.PRODUCER

    def get_inventory(self, resource_id: str) -> float:
        return max(0.0, float(self.inventory.get(resource_id, 0.0)))


class AuctionClearingResult(BaseModel):
    resource_id: str
    clearing_price: Optional[float] = None
    clearing_volume: float = 0.0
    trades: List[Trade] = Field(default_factory=list)
    bids_unfilled: List[Order] = Field(default_factory=list)
    asks_unfilled: List[Order] = Field(default_factory=list)
    demand_curve: List[Tuple[float, float]] = Field(default_factory=list, description="List of (price, cumulative_qty)")
    supply_curve: List[Tuple[float, float]] = Field(default_factory=list, description="List of (price, cumulative_qty)")


class MarketSnapshot(BaseModel):
    step: int
    resource_id: str
    clearing_price: Optional[float]
    clearing_volume: float
    total_bids_volume: float
    total_asks_volume: float
    hhi: float = 0.0


class EpisodeStepLog(BaseModel):
    step: int
    clearing_results: Dict[str, AuctionClearingResult]
    agent_accounts: Dict[str, AgentAccount]
    gini_wealth: float = 0.0
    hhi_by_resource: Dict[str, float] = Field(default_factory=dict)
    anomaly_alerts: List[str] = Field(default_factory=list)
    production_output: Dict[str, float] = Field(default_factory=dict)
    social_welfare: float = 0.0
