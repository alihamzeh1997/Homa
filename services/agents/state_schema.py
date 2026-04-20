from typing import Dict, Any, List, Optional, Literal, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import operator

from services.agents.agents_schema import NewsSummary, MarketContext, PortfolioContext

# ---------------------------------------------------------
# 1. AGENT & DECISION SCHEMAS (Contracts for the LLMs)
# ---------------------------------------------------------
class AgentSignal(BaseModel):
    agent_name: Literal["DeepSeek", "Qwen", "GPT", "Claude", "Gemini"]
    signal: Literal["BUY", "SELL", "HOLD", "CLOSE", "MODIFY"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str

class RiskAssessment(BaseModel):
    is_approved: bool
    max_position_size: Optional[float]
    required_stop_loss: Optional[float]
    risk_warnings: List[str]

class CTODecision(BaseModel):
    final_action: Literal["HOLD", "CLOSE", "OPEN", "MODIFY"]
    execution_price: Optional[float]
    size: Optional[float]
    rationale: str

# ---------------------------------------------------------
# 2. CUSTOM REDUCERS FOR PARALLEL EXECUTION
# ---------------------------------------------------------
def merge_signals(existing: Dict[str, AgentSignal], new: Dict[str, AgentSignal]) -> Dict[str, AgentSignal]:
    """Reducer to safely merge signals from parallel LLM nodes."""
    if not existing:
        return new
    # Merge dictionaries without overwriting
    return {**existing, **new}

# ---------------------------------------------------------
# 3. THE MASTER GRAPH STATE
# ---------------------------------------------------------
class GraphState(TypedDict):
    """
    The central state object passed through the LangGraph workflow.
    """
    # --- Initialization Inputs ---
    symbol: str
    wallet_address: str
    
    # --- Context / Data Nodes (Parallel) ---
    news: Optional[NewsSummary]
    market_data: Optional[MarketContext]
    portfolio: Optional[PortfolioContext]
    
    # --- Sentinel Node (Router) ---
    is_emergency: bool
    skip_workflow: bool
    
    # --- AI Analysts (Parallel Ensemble) ---
    # The Annotated[..., merge_signals] tells LangGraph how to handle the parallel outputs
    agent_signals: Annotated[Dict[str, AgentSignal], merge_signals]
    
    # --- Desk Manager & Risk (Aggregation) ---
    desk_consensus: Optional[Literal["BULLISH", "BEARISH", "NEUTRAL", "CONFLICTED"]]
    risk_assessment: Optional[RiskAssessment]
    psych_evaluation: Optional[str] # Or a Pydantic model if you want strict JSON from the Psych Agent
    
    # --- CTO / Final Execution ---
    final_decision: Optional[CTODecision]