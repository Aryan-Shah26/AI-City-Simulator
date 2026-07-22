"""Pydantic data models for agent config, runtime state, and world events."""
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field


class Financials(BaseModel):
    cash: float
    land_owned: int = 0
    bread: int = 0


class AgentConfig(BaseModel):
    """Static config loaded from agents.json — matches the external agent schema."""
    id: str
    name: str
    age: int
    gender: str
    personality_traits: list[str]
    background: str
    financials: Financials
    current_location: str
    llm: str  # "provider:model", e.g. "groq:openai/gpt-oss-20b"
    long_term_goal: str
    short_term_goal: str


class ActionType(str, Enum):
    TRADE = "trade"
    STEAL = "steal"
    POST_BOARD = "post_board"
    CONVERSE = "converse"
    WORK = "work"
    MOVE = "move"
    IDLE = "idle"


class AgentAction(BaseModel):
    action_type: ActionType
    target_agent_id: str | None = None
    content: str | None = None            # dialogue / board post text
    is_deceptive: bool = False            # ground-truth self-flag, hidden from other agents
    offer: dict[str, float] | None = None  # trade deltas e.g. {"cash": 10, "bread": -2}
    reasoning: str = ""                    # private justification, never shown to others


class MemoryEvent(BaseModel):
    tick: int
    description: str
    witnessed_by: list[str] = Field(default_factory=list)


class BoardPost(BaseModel):
    tick: int
    author_id: str
    content: str
    is_bounty: bool = False


class AgentRuntimeState(BaseModel):
    """Mutable state layered on top of AgentConfig, owned by the World during simulation."""
    reputation: float = 50.0
    opinions: dict[str, float] = Field(default_factory=dict)  # agent_id -> trust score, -100..100
    short_term_memory: list[MemoryEvent] = Field(default_factory=list)
    long_term_memory: str = ""
    last_plan_tick: int = 0
    current_plan: str = ""