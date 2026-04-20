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
    "Qwen": "qwen/qwen3.6-plus",
    "GPT": "openai/gpt-5.4-mini"
}

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

High Timeframe (HTF) Regime (Full History):
{htf_data}

Low Timeframe (LTF) Pullback Data (Full History):
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

1. IF YOU BUY OR SELL:
   - You MUST specify side ("LONG" or "SHORT").
   - You MUST specify asset_size and/or usdc_size.
   - You MUST specify leverage (0 to 40).
   - You MUST specify exact stop_loss and take_profit prices.
   
2. JUSTIFICATION (CRITICAL):
   - In the 'reasoning' field, you MUST deeply justify your decision. Explain *why* the HTF and LTF data align, why the news supports your bias, and the exact logical justification for your chosen stop-loss and take-profit levels. Do not just state the numbers; defend them.

Always provide your confidence level (0.0 to 1.0). If you HOLD, CLOSE, or MODIFY, you may leave trade execution fields as null, but you still must justify the inaction/action.
"""

def _build_analyst_prompt(state: GraphState, agent_name: str) -> str:
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
        open_positions=open_pos
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
    
    structured_llm = llm.with_structured_output(AgentSignal)
    prompt_text = _build_analyst_prompt(state, agent_name)
    
    try:
        # Await the async invocation to allow true parallel non-blocking execution
        result: AgentSignal = await structured_llm.ainvoke(prompt_text)
        logger.success(f"✅ {agent_name} Signal: {result.action} (Conf: {result.confidence:.2f})")
        
        return {"agent_signals": {agent_name: result}}
        
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