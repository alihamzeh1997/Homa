import os
import time
from typing import Dict, Any
from dotenv import load_dotenv
from loguru import logger
from eth_account.signers.local import LocalAccount
import eth_account

from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

from services.agents.state_schema import GraphState

load_dotenv()

# ---------------------------------------------------------
# 1. INITIALIZE HYPERLIQUID CLIENTS
# ---------------------------------------------------------
# Hyperliquid requires a private key to sign transactions directly on their L1.
PRIVATE_KEY = os.getenv("HYPERLIQUID_PRIVATE_KEY")
if not PRIVATE_KEY:
    logger.warning("No HYPERLIQUID_PRIVATE_KEY found. Action node will run in Dry-Run/Simulation mode.")

def _get_exchange() -> Exchange:
    """Helper to initialize the Hyperliquid Exchange client."""
    account: LocalAccount = eth_account.Account.from_key(PRIVATE_KEY)
    # Use constants.TESTNET_API_URL if you are testing!
    return Exchange(account, constants.TESTNET_API_URL)

def _get_info() -> Info:
    return Info(constants.TESTNET_API_URL, skip_ws=True)

# ---------------------------------------------------------
# 2. ASYNC ACTION (EXECUTION) NODE
# ---------------------------------------------------------
async def action_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph Node: The Execution Layer.
    Reads the CTODecision and translates it into direct Hyperliquid API calls.
    """
    decision = state.get("final_decision")
    symbol = state.get("symbol", "UNKNOWN")
    
    if not decision or decision.final_action in ["HOLD", "ABORT"]:
        logger.info(f"🛑 Execution Node: Action is {decision.final_action if decision else 'None'}. No trades placed.")
        return {"execution_status": "SKIPPED"}

    if not PRIVATE_KEY:
        logger.error("🛑 Cannot execute trade: Missing Private Key. Printing payload instead.")
        logger.info(f"DRY RUN PAYLOAD: {decision.model_dump_json(indent=2)}")
        return {"execution_status": "SIMULATED"}

    try:
        exchange = _get_exchange()
        info = _get_info()
        
        asset = decision.asset or symbol
        is_buy = True if decision.side == "LONG" else False
        sz = decision.asset_size

        logger.info(f"🚀 EXECUTING: {decision.final_action} on {asset}")

        # --- 1. EXECUTE_TRADE (Open Position) ---
        if decision.final_action == "EXECUTE_TRADE":
            # Set Leverage First
            if decision.leverage:
                logger.debug(f"Setting leverage to {decision.leverage}x cross for {asset}...")
                exchange.update_leverage(asset, decision.leverage, "cross")
            
            # Fetch current price to calculate slippage (required for market_open)
            # Market state gives us the oracle/mark price
            ctx = info.meta_and_asset_ctxs()[1]
            asset_ctx = next((item for item in ctx if item["coin"] == asset), None)
            current_px = float(asset_ctx["oraclePx"]) if asset_ctx else None

            # Open the market position (defaults to 1% slippage)
            logger.debug(f"Opening Market {decision.side} for {sz} {asset}...")
            order_result = exchange.market_open(
                coin=asset, 
                is_buy=is_buy, 
                sz=sz, 
                px=current_px, 
                slippage=0.01 
            )
            
            if order_result["status"] == "ok":
                logger.success(f"✅ Position Opened Successfully!")
                
                # Place Stop Loss if provided
                if decision.stop_loss:
                    logger.debug(f"Placing Stop Loss at {decision.stop_loss}...")
                    exchange.order(
                        asset, 
                        not is_buy, # Opposite side to close
                        sz, 
                        decision.stop_loss, 
                        order_type={"trigger": {"isMarket": True, "triggerPx": decision.stop_loss, "tpsl": "sl"}}, 
                        reduce_only=True
                    )
                    
                # Place Take Profit if provided
                if decision.take_profit:
                    logger.debug(f"Placing Take Profit at {decision.take_profit}...")
                    exchange.order(
                        asset, 
                        not is_buy, 
                        sz, 
                        decision.take_profit, 
                        order_type={"trigger": {"isMarket": True, "triggerPx": decision.take_profit, "tpsl": "tp"}}, 
                        reduce_only=True
                    )
            else:
                logger.error(f"❌ Failed to open position: {order_result}")

        # --- 2. CLOSE_POSITION ---
        elif decision.final_action == "CLOSE_POSITION":
             logger.debug(f"Closing market position for {asset}...")
             # market_close automatically figures out the size and places a reduce-only order
             close_result = exchange.market_close(asset)
             if close_result["status"] == "ok":
                 logger.success(f"✅ Position Closed Successfully!")
                 # Important: Cancel all resting SL/TP trigger orders for this coin
                 exchange.cancel_all(asset) 
             else:
                 logger.error(f"❌ Failed to close position: {close_result}")

        # --- 3. MODIFY_POSITION ---
        elif decision.final_action == "MODIFY_POSITION":
             logger.info(f"Modifying position for {asset}...")
             # Step A: Update leverage if requested
             if decision.leverage:
                 exchange.update_leverage(asset, decision.leverage, "cross")
             
             # Step B: If there are new SL/TP, we cancel the old ones and place new ones
             if decision.stop_loss or decision.take_profit:
                 logger.debug("Canceling old trigger orders to place new SL/TP...")
                 exchange.cancel_all(asset) 
                 time.sleep(1) # Brief pause for exchange to process cancels
                 
                 # Note: In a production bot, you must fetch the exact open position size here 
                 # instead of relying purely on decision.asset_size to ensure perfect closure.
                 sz_to_close = sz if sz else 0 # Fallback
                 
                 if decision.stop_loss:
                     exchange.order(asset, not is_buy, sz_to_close, decision.stop_loss, 
                                  {"trigger": {"isMarket": True, "triggerPx": decision.stop_loss, "tpsl": "sl"}}, reduce_only=True)
                 if decision.take_profit:
                     exchange.order(asset, not is_buy, sz_to_close, decision.take_profit, 
                                  {"trigger": {"isMarket": True, "triggerPx": decision.take_profit, "tpsl": "tp"}}, reduce_only=True)
             
             logger.success(f"✅ Position Modified.")

        return {"execution_status": "SUCCESS"}

    except Exception as e:
        logger.error(f"❌ Execution Node Crash: {e}")
        return {"execution_status": f"FAILED: {str(e)}"}