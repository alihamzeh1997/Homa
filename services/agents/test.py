import os
from dotenv import load_dotenv
from loguru import logger
from hyperliquid.info import Info
from hyperliquid.utils import constants

# ---------------------------------------------------------
# 1. THE FUNCTION TO TEST
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
# 2. THE TEST RUNNER
# ---------------------------------------------------------
def main():
    load_dotenv()
    
    # Ensure we have a wallet address to check. 
    # (If using an API agent, this needs to be the MAIN wallet address, not the API key address)
    address = os.getenv("HL_WALLET_ADDRESS") 
    
    if not address:
        # Fallback if MAIN_WALLET_ADDRESS isn't in your .env
        print("🚨 MAIN_WALLET_ADDRESS not found in .env. Attempting to use a hardcoded fallback...")
        address = "0x_YOUR_WALLET_ADDRESS_HERE" # <-- PASTE YOUR ACTUAL ADDRESS HERE IF NEEDED
        
        if address == "0x_YOUR_WALLET_ADDRESS_HERE":
            print("❌ Please paste your public wallet address into the script or .env to test.")
            return

    asset = "BTC"
    
    print(f"🔍 Checking Testnet state for {asset} on wallet {address}...\n")
    
    # Initialize the read-only Info client for Testnet
    info = Info(constants.TESTNET_API_URL, skip_ws=True)
    
    # Run the function
    result = _get_position(info, address, asset)
    
    # Print the raw data
    print("\n==========================================")
    print("✅ FINAL RESULT RETURNED BY FUNCTION:")
    print("==========================================")
    
    if result:
        print(f"Found active position for {asset}!")
        print(f"Raw Dictionary Data:\n{result}\n")
        
        # Pull out the size string just to double check it isn't "0.0"
        size = result.get("position", {}).get("szi", "UNKNOWN")
        print(f"👉 The raw size (szi) is: {size}")
    else:
        print(f"❌ Returned None. No {asset} position exists in the assetPositions array.")

if __name__ == "__main__":
    main()