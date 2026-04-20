from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class Wallet(BaseModel):
    account_value: float
    total_notional_position: float
    total_raw_usd: float
    total_margin_used: float

    maintenance_margin: float
    withdrawable: float

    timestamp: int

class Position(BaseModel):
    symbol: str
    size: float
    entry_price: float
    pnl: float

    direction: str  # "long" / "short"
    trade_value: float

    stop_loss: Optional[float]
    take_profit: Optional[float]

    created_at: Optional[datetime]

class MarketData(BaseModel):
    symbol: str
    price: float


class OrderBookLevel(BaseModel):
    price: float
    size: float


class OrderBook(BaseModel):
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]


class Candle(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class FundingRate(BaseModel):
    symbol: str
    rate: float
    timestamp: datetime


class OpenInterest(BaseModel):
    symbol: str
    value: float

class IntradayIndicator(BaseModel):
    timestamp: datetime
    close: float

    ema_20: float | None
    rsi_7: float | None
    rsi_14: float | None

    macd: float | None
    macd_signal: float | None


class HTFIndicator(BaseModel):
    timestamp: datetime
    close: float

    ema_20: float | None
    ema_50: float | None
    ema_cross: bool | None

    rsi: float | None

    macd: float | None
    macd_signal: float | None

    atr: float | None