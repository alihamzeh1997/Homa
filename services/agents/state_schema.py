from typing import Dict, Optional, Annotated, List
from typing_extensions import TypedDict

from services.agents.agents_schema import (
    NewsSummary,
    MarketContext,
    PortfolioContext,
    AgentSignal,
    DeskManagerDecision,
    MoneyManagerDecision,
    PsychEvaluation,
    CTODecision,
)


def merge_signals(
    existing: Dict[str, AgentSignal],
    new: Dict[str, AgentSignal],
) -> Dict[str, AgentSignal]:
    """LangGraph reducer: merges concurrent analyst signals without overwriting."""
    if not existing:
        return new
    return {**existing, **new}


def _keep_last_3(existing: list, new: list) -> list:
    """Appends new entries and trims to the last 3."""
    return ((existing or []) + (new or []))[-3:]


def _merge_analyst_history(
    existing: Dict[str, List[dict]],
    new: Dict[str, List[dict]],
) -> Dict[str, List[dict]]:
    """Per-analyst history: appends each agent's new entry and keeps last 3."""
    result = dict(existing or {})
    for agent, entries in (new or {}).items():
        result[agent] = (result.get(agent, []) + entries)[-3:]
    return result


class GraphState(TypedDict):
    # --- Graph Inputs ---
    symbol: str
    wallet_address: Optional[str]

    # --- Data Context (populated by hl_input_node) ---
    news: Optional[NewsSummary]
    market_data: Optional[MarketContext]
    portfolio: Optional[PortfolioContext]

    # --- Sentinel Router Flags ---
    skip_workflow: bool
    is_emergency: bool

    # --- MoE Analyst Panel (parallel; reducer merges 7 concurrent writes) ---
    agent_signals: Annotated[Dict[str, AgentSignal], merge_signals]

    # --- Sequential Pipeline Decisions ---
    desk_decision: Optional[DeskManagerDecision]
    money_management: Optional[MoneyManagerDecision]
    psych_evaluation: Optional[PsychEvaluation]
    final_decision: Optional[CTODecision]

    # --- Execution Result ---
    execution_status: Optional[str]

    # --- Memory (persisted across runs via checkpointer) ---
    sentinel_history: Annotated[List[dict], _keep_last_3]
    analyst_signal_history: Annotated[Dict[str, List[dict]], _merge_analyst_history]
    desk_history: Annotated[List[dict], _keep_last_3]
    cto_history: Annotated[List[dict], _keep_last_3]
