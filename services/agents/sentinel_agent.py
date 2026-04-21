import os
from typing import Dict, Any
from datetime import datetime, timezone
from dotenv import load_dotenv
from loguru import logger

from langchain_openai import ChatOpenAI
from services.agents.state_schema import GraphState
from services.agents.agents_schema import SentinelDecision

load_dotenv()

# Model names for the Sentinel with fallback routing
_SentinelModel = "deepseek/deepseek-v3.2"
_SentinelModelFallback = "openai/gpt-5.4-nano"

# ---------------------------------------------------------
# 1. INITIALIZE LLMS WITH FALLBACK ROUTING
# ---------------------------------------------------------
primary_llm = ChatOpenAI(
    model=_SentinelModel,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.1 
)
primary_structured = primary_llm.with_structured_output(SentinelDecision)

fallback_llm = ChatOpenAI(
    model=_SentinelModelFallback,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.1
)
fallback_structured = fallback_llm.with_structured_output(SentinelDecision)

robust_llm = primary_structured.with_fallbacks([fallback_structured])

# ---------------------------------------------------------
# 2. THE ASYNC SENTINEL NODE LOGIC
# ---------------------------------------------------------
async def sentinel_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph Node: The gatekeeper. Evaluates market context to determine
    if the workflow should proceed, skip, or trigger an emergency bypass.
    Executes asynchronously to unblock the main event loop.
    """
    symbol = state.get("symbol", "BTC")
    market = state.get("market_data")
    news = state.get("news")
    portfolio = state.get("portfolio")

    # Capture the exact current time in UTC
    current_time_utc = datetime.now(timezone.utc).strftime("%A, %Y-%m-%d %H:%M:%S UTC")

    logger.debug(f"🛡️ Evaluating Sentinel constraints for {symbol} at {current_time_utc}...")

    # Safely extract HTF (last 10 bars) and LTF (last 20 bars)
    htf_data_str = str([h.model_dump() for h in market.htf_series[-10:]]) if market and market.htf_series else "N/A"
    ltf_data_str = str([i.model_dump() for i in market.intraday_series[-20:]]) if market and market.intraday_series else "N/A"

    prompt_text = f"""
    You are the Sentinel Risk Manager for an automated {symbol} trading desk.
    Your job is to classify the current market regime and decide if the trading desk should operate.
    
    CURRENT TIME: {current_time_utc}
    
    CURRENT MARKET DATA:
    - Price: {market.current_price if market else 'N/A'}
    - Funding Rate: {market.funding_rate if market else 'N/A'}
    - Open Interest: {market.open_interest if market else 'N/A'}
    - HTF Trend — 4h candles, last 10 bars: {htf_data_str}
    - LTF Pullback — 3m candles, last 20 bars: {ltf_data_str}
    
    LATEST NEWS SUMMARY:
    - Articles Count: {news.article_count if news else 0}
    - Average Sentiment (0=Bearish, 1=Bullish): {news.average_sentiment if news else 0.5}
    - Top Headlines: {news.top_headlines if news else 'None'}
    
    CURRENT PORTFOLIO EXPOSURE:
    - Open Positions: {len(portfolio.open_positions) if portfolio else 0}
    
    DECISION RULES:
    1. Output 'SKIP' if the market is choppy, volatility is dead, and news is irrelevant. (Goal: Save API costs).
    2. Output 'EMERGENCY' if there is extreme breaking news, a flash crash/pump, or extreme funding rate anomalies. (Goal: Protect capital).
    3. Output 'NORMAL' if there is healthy volatility and a clear trend/pullback structure. (Goal: Trade).
    
    Take the current time and day of the week into account when assessing volatility expectations.
    """

    try:
        result: SentinelDecision = await robust_llm.ainvoke(prompt_text)
        
        if result.decision == "EMERGENCY":
            logger.warning(f"🚨 SENTINEL TRIGGERED EMERGENCY | Reason: {result.reasoning}")
            return {"skip_workflow": False, "is_emergency": True}
        
        elif result.decision == "SKIP":
            logger.info(f"⏭️ SENTINEL SKIPPING WORKFLOW | Reason: {result.reasoning}")
            return {"skip_workflow": True, "is_emergency": False}
            
        else: # NORMAL
            logger.info(f"✅ SENTINEL APPROVED NORMAL WORKFLOW | Reason: {result.reasoning}")
            return {"skip_workflow": False, "is_emergency": False}

    except Exception as e:
        logger.error(f"⚠️ Sentinel & Fallbacks Failed: {e}. Defaulting to NORMAL workflow.")
        return {"skip_workflow": False, "is_emergency": False}