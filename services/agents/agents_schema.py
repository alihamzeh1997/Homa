from pydantic import BaseModel, Field
from typing import List, Literal
from datetime import datetime

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