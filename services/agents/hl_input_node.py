import os
from typing import Dict, Any
from dotenv import load_dotenv

from services.hl_market import MarketService
from services.hl_account import HyperliquidService
from services.indicators import build_multi_tf_features
from services.agents.agents_schema import MarketContext, PortfolioContext

# Load environment variables from the .env file
load_dotenv()

market_service = MarketService(testnet=True)

def market_data_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph Node: Fetches live market data, computes multi-timeframe indicators,
    and aggregates portfolio state to prepare the context for LLM Analysts.
    """
    symbol = state.get("symbol", "BTC")
    
    # ---------------------------------------------------------
    # 1. LOAD WALLET ADDRESS FROM .ENV
    # ---------------------------------------------------------
    wallet_address = os.getenv("HL_WALLET_ADDRESS") or state.get("wallet_address")
    if not wallet_address:
        raise ValueError("CRITICAL: WALLET_ADDRESS not found in .env file or graph state.")
    
    account_service = HyperliquidService(address=wallet_address, testnet=True)

    # ---------------------------------------------------------
    # 2. FETCH LIVE MARKET METRICS
    # ---------------------------------------------------------
    price_data = market_service.get_price(symbol)
    
    # Use None instead of 0.0 to prevent LLM bias on missing data
    funding_data = market_service.get_funding_rate(symbol)
    funding_rate = funding_data[0].rate if funding_data else None
    
    oi_data = market_service.get_open_interest(symbol)
    open_interest = oi_data[0].value if oi_data else None

    # ---------------------------------------------------------
    # 3. FETCH CANDLES & COMPUTE MULTI-TIMEFRAME INDICATORS
    # ---------------------------------------------------------
    intraday_candles = market_service.get_candles(symbol, interval="3m", limit=100)
    htf_candles = market_service.get_candles(symbol, interval="4h", limit=100)

    features = build_multi_tf_features(intraday_candles, htf_candles)

    intraday_series = features["intraday"][-20:]
    htf_series = features["higher_tf"][-10:]

    # ---------------------------------------------------------
    # 4. FETCH PORTFOLIO & RISK DATA
    # ---------------------------------------------------------
    wallet = account_service.get_wallet()
    positions = account_service.get_positions()

    # ---------------------------------------------------------
    # 5. CONSTRUCT THE LLM SCHEMAS
    # ---------------------------------------------------------
    market_context = MarketContext(
        symbol=symbol,
        current_price=price_data.price,
        funding_rate=funding_rate,
        open_interest=open_interest,
        intraday_series=intraday_series,
        htf_series=htf_series
    )

    portfolio_context = PortfolioContext(
        total_return_pct=None,
        sharpe_ratio=None,
        available_cash=wallet.withdrawable,
        account_value=wallet.account_value,
        open_positions=positions
    )

    return {
        "market_data": market_context,
        "portfolio": portfolio_context
    }