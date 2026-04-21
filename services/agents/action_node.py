import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from loguru import logger
import eth_account
from eth_account.signers.local import LocalAccount

from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

from services.agents.state_schema import GraphState

load_dotenv()

PRIVATE_KEY = os.getenv("HYPERLIQUID_PRIVATE_KEY")
# If using an API Agent, put the main account address here in .env
MAIN_WALLET_ADDRESS = os.getenv("HL_WALLET_ADDRESS") 


# ---------------------------------------------------------
# CLIENT HELPERS
# ---------------------------------------------------------
def _get_exchange() -> Exchange:
    account: LocalAccount = eth_account.Account.from_key(PRIVATE_KEY)
    # If a MAIN_WALLET_ADDRESS is provided, we act as an API agent for it (vault_address)
    return Exchange(account, constants.TESTNET_API_URL, account_address=MAIN_WALLET_ADDRESS)


def _get_info() -> Info:
    return Info(constants.TESTNET_API_URL, skip_ws=True)


# ---------------------------------------------------------
# HYPERLIQUID FORMATTERS
# ---------------------------------------------------------
def to_hl_price(px: float) -> float:
    return float(f"{px:.5g}")

def to_hl_size(sz: float, sz_decimals: int) -> float:
    return round(float(sz), sz_decimals)


# ---------------------------------------------------------
# STATE FETCH HELPER (NEW)
# ---------------------------------------------------------
def _get_position(info: Info, address: str, asset: str):
    """Safely fetches user state and logs exactly what Hyperliquid sees."""
    user_state = info.user_state(address)
    positions = user_state.get("assetPositions", [])
    
    active_coins = [p["position"]["coin"] for p in positions]
    logger.debug(f"[STATE] Queried Address: {address}")
    logger.debug(f"[STATE] Open Positions Found: {active_coins}")
    
    for p in positions:
        if p["position"]["coin"] == asset:
            return p
    return None


# ---------------------------------------------------------
# ORDER BUILDER
# ---------------------------------------------------------
def build_tpsl_orders(decision, current_price: float, sz_decimals: int) -> List[Dict]:
    is_buy = decision.side == "LONG"
    sz = to_hl_size(decision.asset_size, sz_decimals)
    asset = decision.asset
    slippage = 0.05
    raw_entry_px = current_price * (1 + slippage) if is_buy else current_price * (1 - slippage)
    entry_px = to_hl_price(raw_entry_px)

    orders = [{
        "coin": asset,
        "is_buy": is_buy,
        "sz": sz,
        "limit_px": entry_px,
        "order_type": {"limit": {"tif": "Ioc"}},
        "reduce_only": False,
    }]

    if decision.stop_loss:
        sl_trigger = to_hl_price(decision.stop_loss)
        raw_sl_limit = sl_trigger * (0.95 if is_buy else 1.05)
        orders.append({
            "coin": asset,
            "is_buy": not is_buy,
            "sz": sz,
            "limit_px": to_hl_price(raw_sl_limit),
            "order_type": {"trigger": {"isMarket": True, "triggerPx": sl_trigger, "tpsl": "sl"}},
            "reduce_only": True,
        })

    if decision.take_profit:
        tp_trigger = to_hl_price(decision.take_profit)
        raw_tp_limit = tp_trigger * (0.95 if is_buy else 1.05)
        orders.append({
            "coin": asset,
            "is_buy": not is_buy,
            "sz": sz,
            "limit_px": to_hl_price(raw_tp_limit),
            "order_type": {"trigger": {"isMarket": True, "triggerPx": tp_trigger, "tpsl": "tp"}},
            "reduce_only": True,
        })

    return orders


# ---------------------------------------------------------
# RESPONSE VALIDATION
# ---------------------------------------------------------
def validate_result(result):
    if not result:
        return False, "No response"
    statuses = result.get("response", {}).get("data", {}).get("statuses", [])
    errors = [s for s in statuses if isinstance(s, dict) and "error" in s]
    if errors:
        return False, errors
    return True, statuses


# ---------------------------------------------------------
# ACTION NODE
# ---------------------------------------------------------
async def action_node(state: GraphState) -> Dict[str, Any]:
    decision = state.get("final_decision")
    symbol = state.get("symbol", "UNKNOWN")

    if not decision or decision.final_action in ["HOLD", "ABORT"]:
        return {"execution_status": "SKIPPED"}

    try:
        exchange = _get_exchange()
        info = _get_info()
        asset = decision.asset or symbol
        target_address = MAIN_WALLET_ADDRESS or exchange.wallet.address

        meta = info.meta()
        asset_meta = next(a for a in meta["universe"] if a["name"] == asset)
        sz_decimals = asset_meta["szDecimals"]
        current_price = float(info.all_mids()[asset])

        # -------------------------------------------------
        # EXECUTE TRADE
        # -------------------------------------------------
        if decision.final_action == "EXECUTE_TRADE":
            if decision.leverage:
                exchange.update_leverage(leverage=decision.leverage, name=asset, is_cross=True)

            orders = build_tpsl_orders(decision, current_price, sz_decimals)
            result = exchange.bulk_orders(order_requests=orders, grouping="normalTpsl")
            
            ok, details = validate_result(result)
            if not ok:
                logger.error(f"[FAILED] {details}")
                return {"execution_status": "FAILED", "details": details}

            logger.success("[SUCCESS] Trade executed")

        # -------------------------------------------------
        # MODIFY POSITION
        # -------------------------------------------------
        elif decision.final_action == "MODIFY_POSITION":
            if decision.leverage:
                logger.debug(f"[LEVERAGE] Updating to {decision.leverage}x")
                exchange.update_leverage(leverage=decision.leverage, name=asset, is_cross=True)

            pos = _get_position(info, target_address, asset)
            if not pos:
                logger.warning(f"[MODIFY] No position found for {asset}")
                return {"execution_status": "NO_POSITION"}

            # FIX: Hyperliquid uses 'szi' for size string
            real_size = abs(float(pos["position"]["szi"]))
            logger.debug(f"[POSITION] Real size detected={real_size}")

            open_orders = info.open_orders(target_address)
            cancels = [{"coin": asset, "oid": o["oid"]} for o in open_orders if o["coin"] == asset]
            if cancels:
                exchange.bulk_cancel(cancels)
            decision.asset_size = real_size
            orders = build_tpsl_orders(decision, current_price, sz_decimals)
            
            result = exchange.bulk_orders(order_requests=orders[1:], grouping="positionTpsl")
            ok, details = validate_result(result)
            if not ok:
                return {"execution_status": "FAILED"}

            logger.success("[SUCCESS] Modified")

        # -------------------------------------------------
        # CLOSE POSITION
        # -------------------------------------------------
        elif decision.final_action == "CLOSE_POSITION":
            pos = _get_position(info, target_address, asset)
            if not pos:
                logger.warning(f"[CLOSE] No position found for {asset}")
                return {"execution_status": "NO_POSITION"}

            result = exchange.market_close(asset)
            logger.debug(f"[CLOSE] result: {result}")
            ok, details = validate_result(result)
            if not ok:
                logger.error(f"[CLOSE] Failed: {details}")
                return {"execution_status": "FAILED"}

            open_orders = info.open_orders(target_address)
            cancels = [{"coin": asset, "oid": o["oid"]} for o in open_orders if o["coin"] == asset]
            if cancels:
                exchange.bulk_cancel(cancels)
            logger.success("[SUCCESS] Closed")

        return {"execution_status": "SUCCESS"}

    except Exception as e:
        logger.exception(f"[CRASH] {str(e)}")
        return {"execution_status": f"FAILED: {str(e)}"}