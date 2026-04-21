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


# ---------------------------------------------------------
# CLIENT HELPERS
# ---------------------------------------------------------
def _get_exchange() -> Exchange:
    account: LocalAccount = eth_account.Account.from_key(PRIVATE_KEY)
    return Exchange(account, constants.TESTNET_API_URL)


def _get_info() -> Info:
    return Info(constants.TESTNET_API_URL, skip_ws=True)


# ---------------------------------------------------------
# CORE BUILDER (IMPORTANT)
# ---------------------------------------------------------
def build_tpsl_orders(decision) -> List[Dict]:
    """
    Build grouped TP/SL order payload
    """
    is_buy = decision.side == "LONG"
    sz = decision.asset_size
    asset = decision.asset

    logger.debug(f"[BUILD] Creating TP/SL structure | asset={asset}, size={sz}")

    # ENTRY (market-style IOC)
    entry = {
        "coin": asset,
        "is_buy": is_buy,
        "sz": sz,
        "limit_px": 0,
        "order_type": {"limit": {"tif": "Ioc"}},
        "reduce_only": False,
    }

    orders = [entry]

    # STOP LOSS
    if decision.stop_loss:
        sl_price = decision.stop_loss * (0.995 if is_buy else 1.005)

        orders.append({
            "coin": asset,
            "is_buy": not is_buy,
            "sz": sz,
            "limit_px": sl_price,
            "order_type": {
                "trigger": {
                    "isMarket": True,
                    "triggerPx": decision.stop_loss,
                    "tpsl": "sl",
                }
            },
            "reduce_only": True,
        })

        logger.debug(f"[BUILD] SL added | trigger={decision.stop_loss}, exec={sl_price}")

    # TAKE PROFIT
    if decision.take_profit:
        tp_price = decision.take_profit * (0.995 if is_buy else 1.005)

        orders.append({
            "coin": asset,
            "is_buy": not is_buy,
            "sz": sz,
            "limit_px": tp_price,
            "order_type": {
                "trigger": {
                    "isMarket": True,
                    "triggerPx": decision.take_profit,
                    "tpsl": "tp",
                }
            },
            "reduce_only": True,
        })

        logger.debug(f"[BUILD] TP added | trigger={decision.take_profit}, exec={tp_price}")

    return orders


# ---------------------------------------------------------
# EXECUTION NODE
# ---------------------------------------------------------
async def action_node(state: GraphState) -> Dict[str, Any]:
    decision = state.get("final_decision")
    symbol = state.get("symbol", "UNKNOWN")

    logger.info(f"[ACTION NODE] Received decision: {decision}")

    # ---------------------------
    # VALIDATION
    # ---------------------------
    if not decision or decision.final_action in ["HOLD", "ABORT"]:
        logger.warning("[SKIP] No executable action")
        return {"execution_status": "SKIPPED"}

    if not PRIVATE_KEY:
        logger.error("[DRY RUN] No private key — printing decision")
        logger.info(decision.model_dump_json(indent=2))
        return {"execution_status": "SIMULATED"}

    try:
        exchange = _get_exchange()
        info = _get_info()

        asset = decision.asset or symbol
        logger.info(f"[EXECUTION] Action={decision.final_action} | Asset={asset}")

        # -------------------------------------------------
        # EXECUTE TRADE (CORRECT IMPLEMENTATION)
        # -------------------------------------------------
        if decision.final_action == "EXECUTE_TRADE":

            # STEP 1 — leverage
            if decision.leverage:
                logger.debug(f"[LEVERAGE] Setting {decision.leverage}x")
                exchange.update_leverage(asset, decision.leverage, "cross")

            # STEP 2 — build grouped orders
            orders = build_tpsl_orders(decision)

            logger.debug(f"[ORDERS] Built payload:")
            for i, o in enumerate(orders):
                logger.debug(f"   Order {i+1}: {o}")

            # STEP 3 — execute atomically
            logger.info("[SUBMIT] Sending grouped TP/SL order...")
            result = exchange.bulk_orders(
                order_requests=orders,
                grouping="normalTpsl"
            )

            logger.info(f"[RESULT] {result}")

            if result.get("status") != "ok":
                logger.error("[FAIL] Order rejected")
                return {"execution_status": "FAILED", "details": result}

            logger.success("[SUCCESS] Trade + TP/SL placed atomically")

        # -------------------------------------------------
        # CLOSE POSITION
        # -------------------------------------------------
        elif decision.final_action == "CLOSE_POSITION":

            logger.info("[CLOSE] Closing position...")
            result = exchange.market_close(asset)

            logger.info(f"[RESULT] {result}")

            if result.get("status") == "ok":
                logger.success("[SUCCESS] Position closed")
                exchange.cancel_all(asset)
                logger.debug("[CLEANUP] Cancelled all remaining orders")
            else:
                logger.error("[FAIL] Close failed")

        # -------------------------------------------------
        # MODIFY POSITION
        # -------------------------------------------------
        elif decision.final_action == "MODIFY_POSITION":

            logger.info("[MODIFY] Modifying position...")

            if decision.leverage:
                exchange.update_leverage(asset, decision.leverage, "cross")

            # ⚠️ MUST fetch real position size
            user_state = info.user_state(exchange.wallet.address)
            positions = user_state.get("assetPositions", [])

            pos = next((p for p in positions if p["position"]["coin"] == asset), None)

            if not pos:
                logger.error("[MODIFY] No open position found")
                return {"execution_status": "FAILED"}

            real_size = abs(float(pos["position"]["s"]))
            logger.debug(f"[POSITION] Real size detected: {real_size}")

            # cancel existing triggers only (still basic version)
            exchange.cancel_all(asset)

            # rebuild TP/SL
            decision.asset_size = real_size
            orders = build_tpsl_orders(decision)

            logger.debug("[REBUILD] New TP/SL orders:")
            for o in orders:
                logger.debug(o)

            result = exchange.bulk_orders(
                order_requests=orders[1:],  # only TP/SL, no entry
                grouping="positionTpsl"
            )

            logger.info(f"[RESULT] {result}")
            logger.success("[SUCCESS] Position modified")

        return {"execution_status": "SUCCESS"}

    except Exception as e:
        logger.exception(f"[CRASH] {str(e)}")
        return {"execution_status": f"FAILED: {str(e)}"}