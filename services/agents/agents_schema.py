from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from datetime import datetime

from services.schema import Position, IntradayIndicator, HTFIndicator

# ---------- NEWS & SENTIMENT ----------
class Article(BaseModel):
    title: str
    body: str
    source: str
    published_at: datetime
    raw_sentiment: float = Field(..., ge=0.0, le=1.0)

class NewsSummary(BaseModel):
    article_count: int
    average_sentiment: float = Field(..., ge=0.0, le=1.0)
    market_bias: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    top_headlines: List[str]  # Max 3-5 titles for the LLM agents to read
    last_fetched: datetime    # Crucial for cache/TTL tracking

# ---------- MARKET & PORTFOLIO CONTEXT (LLM INPUTS) ----------
class PortfolioContext(BaseModel):
    total_return_pct: Optional[float]  # Changed to Optional
    available_cash: float
    account_value: float
    sharpe_ratio: Optional[float]      # Changed to Optional
    open_positions: List[Position]

class MarketContext(BaseModel):
    symbol: str
    current_price: float
    funding_rate: Optional[float]      # Changed to Optional
    open_interest: Optional[float]     # Changed to Optional
    
    intraday_series: List[IntradayIndicator]
    htf_series: List[HTFIndicator]

# ----------Sentinel Output Schema (Router Contract)----------
class SentinelDecision(BaseModel):
    decision: Literal["SKIP", "EMERGENCY", "NORMAL"] = Field(
        description="Must be exactly one of: 'SKIP', 'EMERGENCY', or 'NORMAL'"
    )
    reasoning: str = Field(
        description="A brief 1-2 sentence explanation of why this decision was made based on the data."
    )

# ---------- Analyst Output Schema (MoE Panel Contract) ----------
class AgentSignal(BaseModel):
    """
    The standardized output contract for each LLM Analyst in the Mixture of Experts panel.
    """
    agent_name: Literal["DeepSeek", "MiniMax", "Gemini", "Grok", "Kimi", "Qwen", "GPT"] = Field(
        description="The identity of the analyst generating this signal."
    )
    action: Literal["BUY", "SELL", "HOLD", "CLOSE", "MODIFY"] = Field(
        description="The primary action recommended by the analyst."
    )
    side: Optional[Literal["LONG", "SHORT"]] = Field(
        default=None,
        description="Required if action is BUY or SELL. Specifies the direction of the trade."
    )
    asset_size: Optional[float] = Field(
        default=None,
        description="Required if action is BUY or SELL. The size of the position in the base asset (e.g., BTC, ETH)."
    )
    usdc_size: Optional[float] = Field(
        default=None,
        description="Required if action is BUY or SELL. The size of the position in terms of USDC."
    )
    leverage: Optional[int] = Field(
        default=None,
        ge=0,
        le=40,
        description="Required if action is BUY or SELL. Leverage multiplier between 0 and 40."
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="Required if action is BUY or SELL. The exact price for the stop loss."
    )
    take_profit: Optional[float] = Field(
        default=None,
        description="Required if action is BUY or SELL. The exact price for taking profit."
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Confidence level in the signal from 0.0 (none) to 1.0 (absolute)."
    )
    reasoning: str = Field(
        description="The justification for the trade proposition, particularly explaining the entry, size, stop-loss, and take-profit targets if suggesting a BUY or SELL."
    )

# ---------- Desk Manager & Risk Schema (Consensus Contract) ----------
class DeskManagerDecision(BaseModel):
    """
    The consensus and risk management output from the Desk Manager node.
    It aggregates the 7 MoE signals and outputs a finalized, risk-adjusted trade plan.
    """
    consensus_bias: Literal["BULLISH", "BEARISH", "NEUTRAL", "CONFLICTED"] = Field(
        description="The overall market bias derived from the 7 MoE analysts."
    )
    recommended_action: Literal["BUY", "SELL", "HOLD", "CLOSE", "MODIFY", "ABORT"] = Field(
        description="The final aggregated action recommended to the CTO for execution."
    )
    approved_side: Optional[Literal["LONG", "SHORT"]] = Field(
        default=None,
        description="Risk-approved trade direction, required if action is BUY or SELL."
    )
    approved_asset_size: Optional[float] = Field(
        default=None,
        description="Risk-adjusted base asset size (e.g., BTC amount)."
    )
    approved_usdc_size: Optional[float] = Field(
        default=None,
        description="Risk-adjusted USDC position size."
    )
    approved_leverage: Optional[int] = Field(
        default=None,
        ge=0,
        le=40,
        description="Risk-adjusted leverage multiplier. Must be conservative."
    )
    approved_stop_loss: Optional[float] = Field(
        default=None,
        description="The final, safest consensus stop-loss price."
    )
    approved_take_profit: Optional[float] = Field(
        default=None,
        description="The final, optimized consensus take-profit price."
    )
    risk_warnings: List[str] = Field(
        default_factory=list,
        description="A list of any risk flags, divergences between analysts, or sizing warnings."
    )
    reasoning: str = Field(
        description="Detailed explanation of how the consensus was reached (e.g., '5 voted BUY, 2 voted HOLD') and why the specific trade parameters were chosen."
    )

# ---------- Money Manager Schema (Capital Allocation) ----------
class MoneyManagerDecision(BaseModel):
    """
    The final capital allocation and portfolio risk decision. 
    Evaluates the Desk Manager's proposed trade against actual margin and drawdown limits.
    """
    decision: Literal["APPROVED", "REJECTED", "ADJUSTED", "FAILED"] = Field(
        description="APPROVED: pass; ADJUSTED: modified for safety; REJECTED: violates rules; FAILED: system/API error."
    )
    final_asset_size: Optional[float] = Field(
        default=None,
        description="The final approved size in the base asset (e.g., BTC). Null if REJECTED."
    )
    final_usdc_size: Optional[float] = Field(
        default=None,
        description="The final approved size in USDC. Null if REJECTED."
    )
    final_leverage: Optional[int] = Field(
        default=None,
        ge=0,
        le=40,
        description="The final approved leverage multiplier. Null if REJECTED."
    )
    final_stop_loss: Optional[float] = Field(
        default=None,
        description="The final validated stop loss price. Null if REJECTED."
    )
    final_take_profit: Optional[float] = Field(
        default=None,
        description="The final validated take profit price. Null if REJECTED."
    )
    portfolio_health_status: Literal["HEALTHY", "WARNING", "CRITICAL"] = Field(
        description="An assessment of the current portfolio margin, drawdown, and exposure."
    )
    reasoning: str = Field(
        description="Explanation of the sizing math, why parameters were adjusted, or why the trade was rejected."
    )