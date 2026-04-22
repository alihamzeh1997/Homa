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
Your sole job is to audit the trade proposed by the Desk Manager against hard risk rules.
You are not a trader — you do not have opinions on market direction. You only approve, adjust, or reject based on numbers.

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
Proposed Asset Size: {proposed_asset_size}
Proposed USDC Size: {proposed_usdc_size}
Proposed Leverage: {proposed_leverage}
Proposed Stop Loss: {proposed_sl}
Proposed Take Profit: {proposed_tp}

--- MANDATORY RISK RULES (ALL MUST PASS) ---

RULE 1 — STOP LOSS REQUIRED:
If proposed_stop_loss is null, REJECT immediately. A trade without a stop loss is never approved under any circumstances.

RULE 2 — MAX POSITION SIZE:
The Margin (Proposed Asset Size, Proposed USDC Size) must not exceed 10% of Account Value.
If it does, ADJUST the size down to meet this limit. Show your math.

RULE 3 — MARGIN CHECK:
Required margin = Proposed USDC Size.
This must be strictly less than Available Cash. If it exceeds it, ADJUST the size down.

RULE 4 — RISK-TO-REWARD (R:R):
Calculate explicitly:
  Risk per unit   = |current_price - proposed_stop_loss|
  Reward per unit = |proposed_take_profit - current_price|
  R:R ratio       = Reward / Risk
The minimum acceptable R:R is 1.5. If R:R < 1.5, REJECT. Do not adjust TP/SL — that is the trader's job, not yours.

RULE 5 — LEVERAGE CAP:
Leverage above 10x is only acceptable if the stop loss is within 2% of the current price.
  Max SL distance for high leverage = current_price × 0.02
  If proposed_leverage > 10 and |current_price - proposed_stop_loss| > max SL distance: ADJUST leverage down to 10x.

EXCHANGE CONSTRAINT — CRITICAL:
Hyperliquid supports only ONE position per asset per account.
If an open position for {symbol} already exists, this trade will ADD TO or REDUCE the existing position — not open a new one.
Factor the existing position size into your exposure calculations.

--- YOUR TASK ---
Go through each rule above in order. Show your calculation for each one.
Then output your final decision:

- APPROVED: all rules pass, no changes needed.
- ADJUSTED: one or more rules failed but you were able to fix the parameters. You MUST provide the corrected final_asset_size, final_usdc_size, and/or final_leverage. Do not just flag issues — fix them with exact numbers.
- REJECTED: a rule failed that cannot be fixed by adjusting size (e.g., R:R too low, no stop loss). State exactly which rule caused the rejection.
- FAILED: system or data error prevented the audit.

Also assess the overall portfolio_health_status:
- HEALTHY: margin usage < 30%, no rule violations.
- WARNING: margin usage 30-60%, or minor rule adjustments needed.
- CRITICAL: margin usage > 60%, or hard rejection triggered.
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

    current_price = market.current_price if market and market.current_price is not None else "N/A"

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
        current_price=current_price,
        open_positions=open_positions_str,
        action=desk_decision.recommended_action,
        side=desk_decision.approved_side,
        proposed_asset_size=desk_decision.approved_asset_size,
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