from typing import Dict, Optional, Annotated
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
