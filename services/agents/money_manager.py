import os
from typing import Dict, Any
from datetime import datetime, timezone
from dotenv import load_dotenv
from loguru import logger

from langchain_openai import ChatOpenAI
from services.agents.state_schema import GraphState
from services.agents.agents_schema import MoneyManagerDecision

load_dotenv()

# --- Fallback Configuration ---
_MoneyManagerPrimary = "deepseek/deepseek-v3.2"
_MoneyManagerFallback = "google/gemini-3-flash-preview"

# ---------------------------------------------------------
# 1. MONEY MANAGER PROMPT
# ---------------------------------------------------------
MONEY_MANAGER_PROMPT = """
You are the Chief Financial Officer and Money Manager for an institutional crypto trading desk.
Your job is to perform a final audit on the trade proposed by the Desk Manager.

TARGET ASSET: {symbol}
CURRENT TIME (UTC): {current_time}

--- PORTFOLIO STATUS ---
Available Cash (USDC): {available_cash}
Account Value: {account_value}
Current Price: {current_price}

--- CURRENT EXPOSURE ---
The following positions are already open in the account:
{open_positions}

--- PROPOSED TRADE (FROM DESK MANAGER) ---
Action: {action}
Side: {side}
Proposed USDC Size: {proposed_usdc_size}
Proposed Leverage: {proposed_leverage}
Proposed Stop Loss: {proposed_sl}
Proposed Take Profit: {proposed_tp}

--- MANDATORY RISK RULES ---
1. MAX POSITION SIZE: You are strictly forbidden from putting more than 10% of the total Account Value into a single position.
2. NET EXPOSURE: Consider the 'CURRENT EXPOSURE'. If opening this trade makes the total exposure to {symbol} exceed 15% of account value, you must ADJUST or REJECT.
3. RISK-TO-REWARD (R:R): The trade must have a minimum R:R of 1:1.5. 
4. MARGIN CHECK: Ensure (Size / Leverage) < Available Cash.

--- YOUR TASK ---
... (rest of the task) ...
"""

# ---------------------------------------------------------
# 2. ASYNC MONEY MANAGER NODE
# ---------------------------------------------------------
async def money_manager_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph Node: The final risk vault with fallback redundancy. 
    Enforces the 10% max balance rule and 1:1.5 R:R ratio.
    """
    symbol = state.get("symbol", "UNKNOWN")
    portfolio = state.get("portfolio")
    market = state.get("market_data")
    desk_decision = state.get("desk_decision")

    # Format open positions for the prompt
    if portfolio and portfolio.open_positions:
        # Use model_dump() to turn Pydantic objects into readable dictionaries
        open_positions_str = str([p.model_dump() for p in portfolio.open_positions])
    else:
        open_positions_str = "No open positions."

    # Exit early if no trade was proposed
    if not desk_decision or desk_decision.recommended_action not in ["BUY", "SELL"]:
        return {"money_management": None}

    current_time_utc = datetime.now(timezone.utc).strftime("%A, %Y-%m-%d %H:%M:%S UTC")
    logger.debug(f"💰 Money Manager auditing {symbol} trade proposal...")

    # --- Robust LLM Setup ---
    primary_llm = ChatOpenAI(
        model=_MoneyManagerPrimary,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0
    ).with_structured_output(MoneyManagerDecision)

    fallback_llm = ChatOpenAI(
        model=_MoneyManagerFallback,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0
    ).with_structured_output(MoneyManagerDecision)

    robust_llm = primary_llm.with_fallbacks([fallback_llm])

    prompt_text = MONEY_MANAGER_PROMPT.format(
        symbol=symbol,
        current_time=current_time_utc,
        available_cash=portfolio.available_cash if portfolio else 0,
        account_value=portfolio.account_value if portfolio else 0,
        current_price=market.current_price if market else "N/A",
        open_positions=open_positions_str,
        action=desk_decision.recommended_action,
        side=desk_decision.approved_side,
        proposed_usdc_size=desk_decision.approved_usdc_size,
        proposed_leverage=desk_decision.approved_leverage,
        proposed_sl=desk_decision.approved_stop_loss,
        proposed_tp=desk_decision.approved_take_profit
    )

    try:
        # Await the robust call
        result: MoneyManagerDecision = await robust_llm.ainvoke(prompt_text)
        
        logger.info(f"💵 MONEY DECISION: {result.decision} | Final Size: ${result.final_usdc_size}")
        return {"money_management": result}

    except Exception as e:
        logger.error(f"❌ Money Manager & Fallback Failed: {e}")
        # Explicitly return a FAILED decision state
        return {
            "money_management": MoneyManagerDecision(
                decision="FAILED",
                portfolio_health_status="CRITICAL",
                reasoning=f"Critical system failure: Both primary and fallback LLMs failed to respond. {str(e)}"
            )
        }