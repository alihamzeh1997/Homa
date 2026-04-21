import os
from typing import Dict, Any
from datetime import datetime, timezone
from dotenv import load_dotenv
from loguru import logger

from langchain_openai import ChatOpenAI
from services.agents.state_schema import GraphState
from services.agents.agents_schema import CTODecision

load_dotenv()

# We use the highest reasoning models available for the ultimate decision maker
_CTOModel = "deepseek/deepseek-v3.2"
_CTOModelFallback = "google/gemini-3-flash-preview"

# ---------------------------------------------------------
# 1. CTO PROMPT TEMPLATE
# ---------------------------------------------------------
CTO_PROMPT = """
You are the Chief Trading Officer (CTO) of an elite quantitative hedge fund.
You have ultimate execution authority. Your subordinate agents have provided their reports.

TARGET ASSET: {symbol}
CURRENT TIME (UTC): {current_time}

--- MARKET & NEWS CONTEXT ---
Current Price: {current_price}
Funding Rate: {funding_rate}
Open Interest: {open_interest}
HTF Trend Context — 4h candles, last 10 bars: {htf_data}
LTF Pullback Context — 3m candles, last 20 bars: {ltf_data}
Average News Sentiment (0-1): {average_sentiment}
Top Headlines: {headlines}

--- PORTFOLIO STATUS ---
Available Cash (USDC): {available_cash}
Account Value: {account_value}
Open Positions: {open_positions}

--- 1. DESK MANAGER REPORT (Trade Proposal) ---
{desk_report}

--- 2. MONEY MANAGER REPORT (Risk & Capital Audit) ---
{money_report}

--- 3. PSYCHOLOGIST REPORT (Behavioral Audit) ---
{psych_report}

--- YOUR TASK ---
You must review the inputs from your team and the current market context. You have the authority to accept the recommendations, adjust the parameters, or completely overrule your subordinates.

Determine the absolute final parameters for the exchange API.
1. Decide the final_action (EXECUTE_TRADE, CLOSE_POSITION, MODIFY_POSITION, HOLD, ABORT). All entries are MARKET orders.
2. If executing, specify the exact asset_size, leverage (1-40), side, and stop_loss/take_profit.
3. If modifying or closing, explicitly state the 'asset' (e.g., BTC) to identify which position to adjust, and provide the new SL/TP or size adjustments.
4. Define clear 'invalidation_criteria' (e.g., '4H close below 60k invalidates setup').
5. If you override any of your subordinates, list them in 'agents_overruled' and justify your dictatorship in the 'reasoning' field.
"""

# ---------------------------------------------------------
# 2. ASYNC CTO NODE
# ---------------------------------------------------------
async def cto_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph Node: The Ultimate Arbiter. Reviews Market Data, Desk, Money, 
    and Psych reports to generate the absolute final execution payload.
    Includes robust fallback logic.
    """
    symbol = state.get("symbol", "UNKNOWN")
    market = state.get("market_data")
    news = state.get("news")
    portfolio = state.get("portfolio")
    
    desk = state.get("desk_decision")
    money = state.get("money_management")
    psych = state.get("psych_evaluation")

    current_time_utc = datetime.now(timezone.utc).strftime("%A, %Y-%m-%d %H:%M:%S UTC")
    logger.debug(f"👔 CTO is reviewing reports and market data for {symbol} at {current_time_utc}...")

    # Safely extract and stringify Market, News, and Portfolio context
    htf_data = str([h.model_dump() for h in market.htf_series[-10:]]) if market and market.htf_series else "N/A"
    ltf_data = str([i.model_dump() for i in market.intraday_series[-20:]]) if market and market.intraday_series else "N/A"
    headlines = str(news.top_headlines) if news and news.top_headlines else "N/A"
    open_pos = str([p.model_dump() for p in portfolio.open_positions]) if portfolio and portfolio.open_positions else "None"

    # Safely dump the subordinate reports (indent=2 makes it highly readable for the LLM)
    desk_str = desk.model_dump_json(indent=2) if desk else "No Desk Report."
    money_str = money.model_dump_json(indent=2) if money else "No Money Manager Report."
    psych_str = psych.model_dump_json(indent=2) if psych else "No Psych Report."

    # --- Robust LLM Setup ---
    primary_llm = ChatOpenAI(
        model=_CTOModel,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1,
        max_retries=1
    ).with_structured_output(CTODecision)

    fallback_llm = ChatOpenAI(
        model=_CTOModelFallback,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1,
        max_retries=2
    ).with_structured_output(CTODecision)

    robust_llm = primary_llm.with_fallbacks([fallback_llm])

    # Inject all variables into the prompt
    prompt_text = CTO_PROMPT.format(
        symbol=symbol,
        current_time=current_time_utc,
        current_price=market.current_price if market else "N/A",
        funding_rate=market.funding_rate if market else "N/A",
        open_interest=market.open_interest if market else "N/A",
        htf_data=htf_data,
        ltf_data=ltf_data,
        average_sentiment=news.average_sentiment if news else 0.5,
        headlines=headlines,
        available_cash=portfolio.available_cash if portfolio else "N/A",
        account_value=portfolio.account_value if portfolio else "N/A",
        open_positions=open_pos,
        desk_report=desk_str,
        money_report=money_str,
        psych_report=psych_str
    )

    try:
        # Await the robust decision
        result: CTODecision = await robust_llm.ainvoke(prompt_text)
        
        logger.success(f"🎯 CTO FINAL DECISION: {result.final_action}")
        if result.final_action in ["EXECUTE_TRADE", "MODIFY_POSITION"]:
             logger.info(f"   ↳ Target: {result.side} {result.asset_size} {result.asset} @ {result.leverage}x leverage")
             logger.info(f"   ↳ Invalidation: {result.invalidation_criteria}")
        
        if result.agents_overruled:
            logger.warning(f"⚡ CTO OVERRULED: {', '.join(result.agents_overruled)}")
            
        logger.info(f"📝 CTO Reasoning: {result.reasoning}")

        return {"final_decision": result}

    except Exception as e:
        logger.error(f"❌ CTO Failed: {e}. Issuing Emergency ABORT.")
        # Ultimate fail-safe: Do no harm.
        fallback_decision = CTODecision(
            final_action="ABORT",
            agents_overruled=[],
            reasoning=f"Critical system error in CTO node: {str(e)}. Defaulting to ABORT to protect capital."
        )
        return {"final_decision": fallback_decision}