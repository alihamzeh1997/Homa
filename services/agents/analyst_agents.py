import os
from typing import Dict, Any
from dotenv import load_dotenv
from loguru import logger
from datetime import datetime, timezone

from langchain_openai import ChatOpenAI
from services.agents.state_schema import GraphState
from services.agents.agents_schema import AgentSignal

load_dotenv()

# ---------------------------------------------------------
# 1. MOE CONFIGURATION (OpenRouter Models)
# ---------------------------------------------------------
# Mapping the exact 7 analyst names to their OpenRouter model IDs
ANALYST_MODELS = {
    "DeepSeek": "deepseek/deepseek-v3.2",
    "MiniMax": "minimax/minimax-m2.5",
    "Gemini": "google/gemini-2.5-flash-lite",
    "Grok": "x-ai/grok-4.1-fast",
    "Kimi": "moonshotai/kimi-k2.5",
    "Qwen": "qwen/qwen3.5-plus-02-15",
    "GPT": "openai/gpt-5.4-mini"
}

# Enabled analysts for the MoE panel — can be toggled on/off
ENABLED_ANALYSTS = [
    "DeepSeek",
    # "MiniMax",
    "Gemini",
    # "Grok",
    # "Kimi",
    # "Qwen",
    "GPT",
]

# Node names are the lowercase analyst keys — used by graph.py for wiring
ANALYST_NODES = [name.lower() for name in ENABLED_ANALYSTS]

# ---------------------------------------------------------
# 2. SHARED PROMPT TEMPLATE
# ---------------------------------------------------------
ANALYST_PROMPT = """
You are {agent_name}, an elite quantitative trading analyst on a multi-model ensemble desk.
Analyze the complete multi-timeframe market context, portfolio state, and news summary to formulate a trading plan.

TARGET ASSET: {symbol}
CURRENT TIME (UTC): {current_time}

--- MARKET DATA ---
Current Price: {current_price}
Funding Rate: {funding_rate}
Open Interest: {open_interest}

High Timeframe (HTF) Regime — 4h candles, last 100 bars:
{htf_data}

Low Timeframe (LTF) Pullback Data — 3m candles, last 100 bars:
{ltf_data}

--- NEWS & SENTIMENT ---
Articles Analyzed: {article_count}
Average Sentiment (0=Bearish, 1=Bullish): {average_sentiment}
Top Headlines: {headlines}

--- PORTFOLIO STATE ---
Available Cash (USDC): {available_cash}
Account Value: {account_value}
Open Positions: {open_positions}

--- YOUR TASK ---
Based strictly on the data above, provide your trading signal. Take the current time and day of the week into account when assessing liquidity and volatility expectations (e.g., weekend chop vs. weekday volume).

EXCHANGE CONSTRAINT — CRITICAL:
Hyperliquid supports only ONE position per asset per account. If the portfolio already shows an open position for {symbol},
you CANNOT open a second independent trade. A BUY or SELL order would simply add to or reduce the existing position unintentionally.
Therefore:
- If a position is already open, prefer MODIFY (to adjust size, SL, or TP) or CLOSE (to exit entirely) or HOLD.
- Only recommend BUY or SELL if there is NO open position for {symbol}, or if you explicitly intend to add to or flip the existing one — and justify that clearly.

1. IF YOU BUY OR SELL:
   - You MUST specify side ("LONG" or "SHORT").
   - You MUST specify asset_size and/or usdc_size.
   - You MUST specify leverage (0 to 40).
   - You MUST specify exact stop_loss and take_profit prices.

2. JUSTIFICATION (CRITICAL):
   - In the 'reasoning' field, you MUST deeply justify your decision. Explain *why* the HTF and LTF data align, why the news supports your bias, and the exact logical justification for your chosen stop-loss and take-profit levels. Do not just state the numbers; defend them.

Always provide your confidence level (0.0 to 1.0). If you HOLD, CLOSE, or MODIFY, you may leave trade execution fields as null, but you still must justify the inaction/action.

LAST CTO DECISION (what the final decision-maker concluded last cycle):
{cto_memory}

YOUR LAST 3 SIGNALS (oldest first) — use these to avoid repeating mistakes and stay consistent unless the market has changed:
{memory}

Respond ONLY with a valid JSON object matching this schema — no markdown, no explanation outside the JSON:
{{
  "agent_name": "<your name>", # It's either DeepSeek, MiniMax, Gemini, Grok, Kimi, Qwen, or GPT — but do not self-name creatively. Always return the agent_name field with the correct name of you are.
  "action": "BUY | SELL | HOLD | CLOSE | MODIFY",
  "side": "LONG | SHORT | null",
  "asset_size": <float or null>, # If you say BUY or SELL You must specify the asset size in units (e.g. 0.001)
  "usdc_size": <float or null>, # If you say BUY or SELL You must specify the size in USDC (e.g. 50.0)
  "leverage": <int or null>, # If you say BUY or SELL You must specify the leverage (e.g. 5)
  "stop_loss": <float or null>, # If you say BUY or SELL or MODIFY You must specify the stop loss price (e.g. 19500.0)
  "take_profit": <float or null>, # If you say BUY or SELL or MODIFY You must specify the take profit price (e.g. 20000.0)
  "confidence": <float 0.0-1.0>,
  "reasoning": "<detailed justification>"
}}
"""

def _build_analyst_prompt(state: GraphState, agent_name: str, memory_str: str = "  No previous signals yet.", cto_memory_str: str = "  No previous CTO decision yet.") -> str:
    """Helper to safely format the prompt strings from GraphState."""
    current_time_utc = datetime.now(timezone.utc).strftime("%A, %Y-%m-%d %H:%M:%S UTC")
    symbol = state.get("symbol", "UNKNOWN")
    market = state.get("market_data")
    news = state.get("news")
    portfolio = state.get("portfolio")

    # Safe extraction of the FULL Pydantic model history
    htf_data = str([h.model_dump() for h in market.htf_series]) if market and market.htf_series else "N/A"
    ltf_data = str([i.model_dump() for i in market.intraday_series]) if market and market.intraday_series else "N/A"
    headlines = str(news.top_headlines) if news and news.top_headlines else "N/A"
    open_pos = str([p.model_dump() for p in portfolio.open_positions]) if portfolio and portfolio.open_positions else "None"

    return ANALYST_PROMPT.format(
        agent_name=agent_name,
        symbol=symbol,
        current_time=current_time_utc,
        current_price=market.current_price if market else "N/A",
        funding_rate=market.funding_rate if market else "N/A",
        open_interest=market.open_interest if market else "N/A",
        htf_data=htf_data,
        ltf_data=ltf_data,
        article_count=news.article_count if news else 0,
        average_sentiment=news.average_sentiment if news else 0.5,
        headlines=headlines,
        available_cash=portfolio.available_cash if portfolio else "N/A",
        account_value=portfolio.account_value if portfolio else "N/A",
        open_positions=open_pos,
        memory=memory_str,
        cto_memory=cto_memory_str,
    )

# ---------------------------------------------------------
# 3. ASYNC BASE ANALYST INVOKER
# ---------------------------------------------------------
async def invoke_analyst(state: GraphState, agent_name: str) -> Dict[str, Any]:
    """Asynchronous base function to instantiate an LLM, prompt it, and return the structured schema."""
    logger.debug(f"🧠 Waking up Analyst: {agent_name}...")
    
    model_id = ANALYST_MODELS.get(agent_name)
    
    llm = ChatOpenAI(
        model=model_id,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2, 
        max_retries=2
    )
    
    raw_llm = llm.with_structured_output(None, method="json_mode")

    # Inject last 3 signals from memory
    past = (state.get("analyst_signal_history") or {}).get(agent_name, [])
    if past:
        memory_str = "\n".join(
            f"  [Run -{len(past)-i}] Action: {h['action']} | Confidence: {h['confidence']} | Reasoning: {h['reasoning'][:300]}"
            for i, h in enumerate(past)
        )
    else:
        memory_str = "  No previous signals yet."

    # Inject last CTO decision
    cto_history = state.get("cto_history") or []
    if cto_history:
        last_cto = cto_history[-1]
        cto_memory_str = f"  Action: {last_cto.get('final_action')} | Confidence: {last_cto.get('confidence')} | Reasoning: {last_cto.get('reasoning', '')[:300]}"
    else:
        cto_memory_str = "  No previous CTO decision yet."

    prompt_text = _build_analyst_prompt(state, agent_name, memory_str, cto_memory_str)

    try:
        raw: dict = await raw_llm.ainvoke(prompt_text)
        raw["agent_name"] = agent_name  # always override — models name themselves creatively
        result = AgentSignal(**raw)
        logger.success(f"✅ {agent_name} Signal: {result.action} (Conf: {result.confidence:.2f})")

        return {
            "agent_signals": {agent_name: result},
            "analyst_signal_history": {agent_name: [result.model_dump()]},
        }
        
    except Exception as e:
        logger.error(f"❌ Analyst {agent_name} failed: {e}")
        # Fallback to HOLD to prevent the whole graph from crashing
        fallback_signal = AgentSignal(
            agent_name=agent_name, 
            action="HOLD", 
            confidence=0.0, 
            reasoning=f"API Error or timeout: {str(e)}. Defaulting to HOLD to preserve capital."
        )
        return {"agent_signals": {agent_name: fallback_signal}}

# ---------------------------------------------------------
# 4. ASYNC PARALLEL NODE WRAPPERS FOR LANGGRAPH
# ---------------------------------------------------------
# LangGraph natively handles async nodes seamlessly.

async def deepseek_node(state: GraphState) -> Dict[str, Any]: 
    return await invoke_analyst(state, "DeepSeek")

async def minimax_node(state: GraphState) -> Dict[str, Any]: 
    return await invoke_analyst(state, "MiniMax")

async def gemini_node(state: GraphState) -> Dict[str, Any]: 
    return await invoke_analyst(state, "Gemini")

async def grok_node(state: GraphState) -> Dict[str, Any]: 
    return await invoke_analyst(state, "Grok")

async def kimi_node(state: GraphState) -> Dict[str, Any]: 
    return await invoke_analyst(state, "Kimi")

async def qwen_node(state: GraphState) -> Dict[str, Any]: 
    return await invoke_analyst(state, "Qwen")

async def gpt_node(state: GraphState) -> Dict[str, Any]: 
    return await invoke_analyst(state, "GPT")