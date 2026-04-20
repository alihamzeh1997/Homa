from hyperliquid.info import Info
from datetime import datetime, timedelta
import time
from langchain_openai import data

from schema import (
    MarketData,
    OrderBook,
    OrderBookLevel,
    Candle,
    FundingRate,
    OpenInterest,
)


VALID_INTERVALS = {"1m", "5m", "15m", "1h", "4h", "1d"}

INTERVAL_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


class MarketService:
    def __init__(self, testnet: bool = True):
        base_url = (
            "https://api.hyperliquid-testnet.xyz"
            if testnet
            else "https://api.hyperliquid.xyz"
        )
        self.info = Info(base_url=base_url)

    # ---------- PRICE ----------
    def get_price(self, symbol: str) -> MarketData:
        data = self.info.all_mids()
        return MarketData(symbol=symbol, price=float(data[symbol]))

    # ---------- ORDER BOOK ----------
    def get_orderbook(self, symbol: str, depth: int = 10):
        data = self.info.l2_snapshot(symbol)

        bids = [
            OrderBookLevel(price=float(level["px"]), size=float(level["sz"]))
            for level in data["levels"][0][:depth]
        ]

        asks = [
            OrderBookLevel(price=float(level["px"]), size=float(level["sz"]))
            for level in data["levels"][1][:depth]
        ]

        return OrderBook(symbol=symbol, bids=bids, asks=asks)

    # ---------- CANDLES ----------
    def get_candles(
    self,
    symbol: str,
    interval: str = "1m",
    limit: int = 100,
        ) -> list[Candle]:

        if interval not in VALID_INTERVALS:
            raise ValueError(f"Invalid interval: {interval}")

        minutes = INTERVAL_MINUTES[interval]

        end = datetime.utcnow()
        start = end - timedelta(minutes=minutes * limit)

        data = self.info.candles_snapshot(
            symbol,
            interval,
            int(start.timestamp() * 1000),
            int(end.timestamp() * 1000),
        )

        candles = [
            Candle(
                timestamp=datetime.fromtimestamp(c["t"] / 1000),
                open=float(c["o"]),
                high=float(c["h"]),
                low=float(c["l"]),
                close=float(c["c"]),
                volume=float(c["v"]),
            )
            for c in data
        ]

        return candles

    # ---------- FUNDING ----------
    def get_funding_rate(self, symbol: str):
        meta, asset_ctxs = self.info.meta_and_asset_ctxs()

        for i, asset_meta in enumerate(meta["universe"]):
            if asset_meta["name"] == symbol:
                asset = asset_ctxs[i]

                return [
                    FundingRate(
                        symbol=symbol,
                        rate=float(asset["funding"]),
                        timestamp=datetime.utcnow(),
                    )
]

        raise ValueError(f"No funding data for {symbol}")

    # ---------- OPEN INTEREST ----------
    def get_open_interest(
    self,
    symbol: str,
    limit: int = 1,
    ) -> list[OpenInterest]:

        meta, asset_ctxs = self.info.meta_and_asset_ctxs()

        for i, asset_meta in enumerate(meta["universe"]):
            if asset_meta["name"] == symbol:
                asset = asset_ctxs[i]

                value = float(asset["openInterest"])

                return [
                    OpenInterest(symbol=symbol, value=value)
                ] * limit

        raise ValueError(f"No OI data for {symbol}")

