import asyncio
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional

# Import your action node from the services folder
from services.agents.action_node import action_node

# ---------------------------------------------------------
# 1. MOCK SCHEMA
# ---------------------------------------------------------
# The action_node.py expects a Pydantic model for 'final_decision'.
# Since we are bypassing the rest of the LangGraph, we create a 
# mock class here that matches the attributes used in action_node.py
class MockCTODecision(BaseModel):
    final_action: str
    asset: str
    side: str
    asset_size: float
    leverage: Optional[int] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

# ---------------------------------------------------------
# 2. MAIN TEST RUNNER
# ---------------------------------------------------------
async def main():
    # Load environment variables (to get HYPERLIQUID_PRIVATE_KEY)
    load_dotenv()
    
    if not os.getenv("HYPERLIQUID_PRIVATE_KEY"):
        print("🚨 ERROR: HYPERLIQUID_PRIVATE_KEY is missing from your .env file!")
        return

    # Set the asset we want to test with.
    # IMPORTANT: Ensure your testnet wallet has USDC to trade.
    symbol = "BTC"

    print("🚀 Starting Hyperliquid Action Node Test...\n")

    # ==========================================
    # TEST 1: OPEN A POSITION
    # ==========================================
    print("--- 🔵 TEST 1: OPENING POSITION ---")
    
    # We construct a mock decision to go LONG on BTC
    open_decision = MockCTODecision(
        final_action="EXECUTE_TRADE",
        asset=symbol,
        side="LONG",
        asset_size=0.00015,  # Adjust size based on Hyperliquid minimums
        leverage=5,        # Set 5x leverage
        stop_loss=75000.0, # Example SL (Make sure it's lower than current testnet price!)
        take_profit=77000.0 # Example TP (Make sure it's higher than current testnet price!)
    )
    
    # We package it into the mock GraphState dictionary format that action_node expects
    state_open = {
        "symbol": symbol,
        "final_decision": open_decision
    }
    
    # Run the action node
    result1 = await action_node(state_open)
    print(f"Result (Open): {result1}\n")
    
    # Pause for 10 seconds to let the exchange update and process the orders
    print("⏳ Waiting 10 seconds for exchange to process...\n")
    await asyncio.sleep(10)

    # ==========================================
    # TEST 2: MODIFY THE POSITION
    # ==========================================
    print("--- 🟡 TEST 2: MODIFYING POSITION ---")
    
    # Now we simulate the bot deciding to change the Stop Loss and Take Profit
    modify_decision = MockCTODecision(
        final_action="MODIFY_POSITION",
        asset=symbol,
        side="LONG",
        asset_size=0.00015,  # action_node fetches real size dynamically, but we provide it anyway
        leverage=10,        # Change leverage to 10x
        stop_loss=74000.0, # Move Stop Loss up
        take_profit=76400.0 # Move Take Profit up
    )

    state_modify = {
        "symbol": symbol,
        "final_decision": modify_decision
    }

    result2 = await action_node(state_modify)
    print(f"Result (Modify): {result2}\n")
    
    # Pause again
    print("⏳ Waiting 10 seconds for exchange to process...\n")
    await asyncio.sleep(10)

    # ==========================================
    # TEST 3: CLOSE THE POSITION
    # ==========================================
    print("--- 🔴 TEST 3: CLOSING POSITION ---")
    
    # Finally, we tell the node to close everything and cancel remaining orders
    close_decision = MockCTODecision(
        final_action="CLOSE_POSITION",
        asset=symbol,
        side="LONG",       # Arbitrary, not used for closing
        asset_size=0.0     # The action_node handles size automatically on market_close
    )

    state_close = {
        "symbol": symbol,
        "final_decision": close_decision
    }

    result3 = await action_node(state_close)
    print(f"Result (Close): {result3}\n")
    
    print("✅ Testing Complete!")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())