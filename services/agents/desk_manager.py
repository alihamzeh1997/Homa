import os
from typing import Dict, Any
from datetime import datetime, timezone
from dotenv import load_dotenv
from loguru import logger

from langchain_openai import ChatOpenAI
from services.agents.state_schema import GraphState
from services.agents.agents_schema import DeskManagerDecision

load_dotenv()

# We use highly capable models for the final risk aggregation
_DeskManagerModel = "deepseek/deepseek-v3.2"
_DeskManagerModelFallback = "google/gemini-3-flash-preview"

# ---------------------------------------------------------
# 1. SHARED PROMPT TEMPLATE
# ---------------------------------------------------------
DESK_MANAGER_PROMPT = """
You are the Chief Risk Officer and Desk Manager of a quantitative trading firm.
Your MoE (Mixture of Experts) panel of 7 elite analysts has just submitted their trading signals.

TARGET ASSET: {symbol}
CURRENT TIME (UTC): {current_time}

--- MARKET & NEWS CONTEXT ---
Current Price: {current_price}
Funding Rate: {funding_rate}
Open Interest: {open_interest}
HTF Trend Context: {htf_data}
Average News Sentiment (0-1): {average_sentiment}
Top Headlines: {headlines}

--- PORTFOLIO STATE (RISK LIMITS) ---
Available Cash (USDC): {available_cash}
Account Value: {account_value}
Current Open Positions: {open_positions}

--- ANALYST SIGNALS ---
{analyst_signals_summary}

--- YOUR TASK ---
1. Evaluate the consensus. What is the majority recommending (BUY, SELL, HOLD)? If the panel is heavily split (e.g., 3 BUY, 4 SELL), the bias is CONFLICTED.
2. If the consensus is to BUY or SELL, you must extract the safest, most logical trade parameters from the analysts' suggestions.
3. APPLY STRICT RISK MANAGEMENT:
   - Do not approve leverage higher than the most conservative reasonable suggestion.
   - Ensure the approved_usdc_size does not dangerously exceed the Available Cash (factoring in leverage).
   - Select the most logical stop_loss and take_profit based on the analysts' reasoning and current market context.
4. If the trade is too risky, or the analysts are completely conflicted, your recommended_action MUST be "ABORT" or "HOLD".
5. Log any severe disagreements between models in the 'risk_warnings' list.
"""

def _format_signals(signals: Dict[str, Any]) -> str:
    """Helper to format the 7 agent signals into a readable string for the LLM."""
    if not signals:
        return "No signals received."
    
    summary = ""
    for agent, signal_data in signals.items():
        summary += f"[{agent}] - Action: {signal_data.action} | Conf: {signal_data.confidence:.2f}\n"
        if signal_data.action in ["BUY", "SELL"]:
            summary += f"    Side: {signal_data.side}, Lev: {signal_data.leverage}x, Size: {signal_data.asset_size} / {signal_data.usdc_size} USDC\n"
            summary += f"    SL: {signal_data.stop_loss}, TP: {signal_data.take_profit}\n"
        summary += f"    Reasoning: {signal_data.reasoning}\n\n"
    return summary

# ---------------------------------------------------------
# 2. ASYNC DESK MANAGER NODE
# ---------------------------------------------------------
async def desk_manager_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph Node: The Consensus Aggregator. Reads the 7 MoE signals and market data,
    calculates the majority, applies risk limits, and outputs the final plan.
    Includes OpenRouter fallback routing for max reliability.
    """
    symbol = state.get("symbol", "UNKNOWN")
    portfolio = state.get("portfolio")
    agent_signals = state.get("agent_signals", {})
    market = state.get("market_data")
    news = state.get("news")

    current_time_utc = datetime.now(timezone.utc).strftime("%A, %Y-%m-%d %H:%M:%S UTC")

    logger.debug(f"⚖️ Desk Manager evaluating {len(agent_signals)} signals for {symbol} at {current_time_utc}...")

    # --- FALLBACK LLM SETUP ---
    # Primary setup (GPT-4o)
    primary_llm = ChatOpenAI(
        model=_DeskManagerModel,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1, 
        max_retries=1
    )
    primary_structured = primary_llm.with_structured_output(DeskManagerDecision)

    # Fallback setup (DeepSeek V3.2)
    fallback_llm = ChatOpenAI(
        model=_DeskManagerModelFallback,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1,
        max_retries=2
    )
    fallback_structured = fallback_llm.with_structured_output(DeskManagerDecision)

    # Combine them natively
    robust_llm = primary_structured.with_fallbacks([fallback_structured])

    # Safely extract Pydantic models for the prompt
    htf_data = str([h.model_dump() for h in market.htf_series]) if market and market.htf_series else "N/A"
    headlines = str(news.top_headlines) if news and news.top_headlines else "N/A"
    open_pos = str([p.model_dump() for p in portfolio.open_positions]) if portfolio and portfolio.open_positions else "None"
    signals_str = _format_signals(agent_signals)

    prompt_text = DESK_MANAGER_PROMPT.format(
        symbol=symbol,
        current_time=current_time_utc,
        current_price=market.current_price if market else "N/A",
        funding_rate=market.funding_rate if market else "N/A",
        open_interest=market.open_interest if market else "N/A",
        htf_data=htf_data,
        average_sentiment=news.average_sentiment if news else 0.5,
        headlines=headlines,
        available_cash=portfolio.available_cash if portfolio else "N/A",
        account_value=portfolio.account_value if portfolio else "N/A",
        open_positions=open_pos,
        analyst_signals_summary=signals_str
    )

    try:
        # ⚡ Await the robust async invocation (will automatically fallback if primary fails)
        result: DeskManagerDecision = await robust_llm.ainvoke(prompt_text)
        
        logger.info(f"📊 DESK CONSENSUS: {result.consensus_bias} | ACTION: {result.recommended_action}")
        if result.risk_warnings:
            for warning in result.risk_warnings:
                logger.warning(f"⚠️ Risk Flag: {warning}")

        return {"desk_decision": result}

    except Exception as e:
        logger.error(f"❌ Desk Manager AND Fallback Failed: {e}. Defaulting to ABORT.")
        # Fail-safe strict risk routing if OpenRouter completely goes down
        fallback_decision = DeskManagerDecision(
            consensus_bias="CONFLICTED",
            recommended_action="ABORT",
            risk_warnings=[f"Both Primary and Fallback LLMs failed: {str(e)}"],
            reasoning="System error during consensus aggregation. Aborting to protect capital."
        )
        return {"desk_decision": fallback_decision}