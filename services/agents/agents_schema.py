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