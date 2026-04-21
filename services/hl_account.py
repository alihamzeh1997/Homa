from hyperliquid.info import Info
from datetime import datetime
from typing import List, Optional

from services.schema import Wallet, Position


class HyperliquidService:
    def __init__(self, address: str, testnet: bool = True):
        self.address = address
        self.base_url = (
            "https://api.hyperliquid-testnet.xyz"
            if testnet
            else "https://api.hyperliquid.xyz"
        )
        self.info = Info(base_url=self.base_url)

    # ---------- WALLET ----------
    def get_wallet(self) -> Wallet:
        data = self.info.user_state(self.address)
        ms = data["marginSummary"]

        return Wallet(
            account_value=float(ms["accountValue"]),
            total_notional_position=float(ms["totalNtlPos"]),
            total_raw_usd=float(ms["totalRawUsd"]),
            total_margin_used=float(ms["totalMarginUsed"]),
            maintenance_margin=float(data["crossMaintenanceMarginUsed"]),
            withdrawable=float(data["withdrawable"]),
            timestamp=data["time"],
        )

    # ---------- INTERNAL HELPERS ----------

    def _get_frontend_orders(self):
        return self.info.frontend_open_orders(self.address)

    def _extract_sl_tp(self, orders, symbol):
        sl, tp = None, None

        for o in orders:
            if o["coin"] != symbol or not o.get("isTrigger"):
                continue

            px = float(o["triggerPx"])

            if o["orderType"] == "Take Profit Market":
                tp = px
            elif o["orderType"] == "Stop Market":
                sl = px

        return sl, tp

    def _extract_entry_time(self, fills, symbol):
        relevant = [
            f for f in fills
            if f["coin"] == symbol and "Open" in f["dir"]
        ]

        if not relevant:
            return None

        first = sorted(relevant, key=lambda x: x["time"])[0]
        return datetime.fromtimestamp(first["time"] / 1000)

    def _extract_direction(self, fills, symbol):
        for f in fills:
            if f["coin"] == symbol and "Open" in f["dir"]:
                return "long" if "Long" in f["dir"] else "short"
        return None

    # ---------- POSITIONS ----------
    def get_positions(self) -> List[Position]:
        state = self.info.user_state(self.address)
        fills = self.info.user_fills(self.address)
        orders = self._get_frontend_orders()

        positions = []

        for p in state["assetPositions"]:
            pos = p["position"]

            symbol = pos["coin"]
            size = float(pos["szi"])
            entry_price = float(pos["entryPx"])

            direction = self._extract_direction(fills, symbol)
            created_at = self._extract_entry_time(fills, symbol)
            sl, tp = self._extract_sl_tp(orders, symbol)

            positions.append(
                Position(
                    symbol=symbol,
                    size=size,
                    entry_price=entry_price,
                    pnl=float(pos["unrealizedPnl"]),
                    direction=direction,
                    trade_value=float(pos["positionValue"]),
                    stop_loss=sl,
                    take_profit=tp,
                    created_at=created_at,
                )
            )

        return positions