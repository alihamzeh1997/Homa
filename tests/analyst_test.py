"""
Analyst Agent Test Suite
========================
Tests all analyst nodes with mocked LLM responses (no real API calls).

HOW TO USE:
  - Run all:        python -m pytest tests/analyst_test.py -v
  - Run one test:   python -m pytest tests/analyst_test.py::test_analyst_buy_signal -v

CONFIGURE WHICH ANALYSTS TO TEST:
  Edit the ANALYSTS_TO_TEST list below.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from services.agents.analyst_agents import invoke_analyst, _build_analyst_prompt
from services.agents.agents_schema import AgentSignal, MarketContext, PortfolioContext, NewsSummary
from services.schema import IntradayIndicator, HTFIndicator, Position

# ---------------------------------------------------------
# CONFIGURE: which analysts to include in the test run
# Comment out any you want to skip
# ---------------------------------------------------------
ANALYSTS_TO_TEST = [
    # "DeepSeek",
    # "MiniMax",
    # "Gemini",
    # "Grok",
    # "Kimi",
    "Qwen",
    # "GPT",
]


# ---------------------------------------------------------
# MOCK DATA
# ---------------------------------------------------------

def make_mock_state() -> dict:
    """Builds a realistic but fully mocked GraphState."""

    intraday = [
        IntradayIndicator(
            timestamp=datetime(2026, 4, 21, 10, i * 3, tzinfo=timezone.utc),
            close=76000.0 + i * 10,
            ema_20=75900.0 + i * 8,
            rsi_7=55.0 + i * 0.5,
            rsi_14=52.0 + i * 0.3,
            macd=12.5,
            macd_signal=11.0,
        )
        for i in range(20)
    ]

    htf = [
        HTFIndicator(
            timestamp=datetime(2026, 4, 20, (i * 4) % 24, 0, tzinfo=timezone.utc),
            close=74000.0 + i * 300,
            ema_20=73500.0 + i * 250,
            ema_50=72000.0 + i * 200,
            ema_cross=True,
            rsi=58.0,
            macd=80.0,
            macd_signal=70.0,
            atr=450.0,
        )
        for i in range(10)
    ]

    market = MarketContext(
        symbol="BTC",
        current_price=76200.0,
        funding_rate=0.0001,
        open_interest=980_000_000.0,
        intraday_series=intraday,
        htf_series=htf,
    )

    news = NewsSummary(
        article_count=12,
        average_sentiment=0.72,
        market_bias="BULLISH",
        top_headlines=[
            "Bitcoin breaks $76K resistance with strong volume",
            "Institutional inflows hit Q1 record for BTC ETFs",
            "Fed signals rate pause, risk assets rally",
        ],
        last_fetched=datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc),
    )

    portfolio = PortfolioContext(
        total_return_pct=4.2,
        sharpe_ratio=1.8,
        available_cash=480.0,
        account_value=511.66,
        open_positions=[],
    )

    return {
        "symbol": "BTC",
        "market_data": market,
        "news": news,
        "portfolio": portfolio,
    }


def make_mock_signal(agent_name: str, action: str = "BUY") -> AgentSignal:
    """Returns a realistic mocked AgentSignal."""
    if action == "BUY":
        return AgentSignal(
            agent_name=agent_name,
            action="BUY",
            side="LONG",
            asset_size=0.00015,
            usdc_size=11.43,
            leverage=5,
            stop_loss=74500.0,
            take_profit=78000.0,
            confidence=0.82,
            reasoning="HTF shows EMA cross and RSI momentum. LTF confirms pullback entry. News bullish.",
        )
    else:  # HOLD
        return AgentSignal(
            agent_name=agent_name,
            action="HOLD",
            side=None,
            asset_size=None,
            usdc_size=None,
            leverage=None,
            stop_loss=None,
            take_profit=None,
            confidence=0.4,
            reasoning="Mixed signals. Waiting for clearer confirmation.",
        )


# ---------------------------------------------------------
# HELPER: patch the LLM inside invoke_analyst
# ---------------------------------------------------------

def mock_llm_returning(signal: AgentSignal):
    """Returns a context manager that patches ChatOpenAI to return the given signal."""
    mock_structured = AsyncMock()
    mock_structured.ainvoke = AsyncMock(return_value=signal)

    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_structured

    return patch(
        "services.agents.analyst_agents.ChatOpenAI",
        return_value=mock_llm_instance,
    )


# ---------------------------------------------------------
# TEST 1: Prompt builds without crashing
# ---------------------------------------------------------

def test_prompt_builds_correctly():
    """Checks that the prompt template fills in correctly with mock data."""
    state = make_mock_state()
    prompt = _build_analyst_prompt(state, "DeepSeek")

    assert "BTC" in prompt
    assert "76200.0" in prompt
    assert "Bitcoin breaks $76K resistance" in prompt
    assert "480.0" in prompt
    print("\n--- PROMPT PREVIEW (first 500 chars) ---")
    print(prompt[:500])


# ---------------------------------------------------------
# TEST 2: Each analyst returns a BUY signal correctly
# ---------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("analyst", ANALYSTS_TO_TEST)
async def test_analyst_buy_signal(analyst):
    """Each analyst should return a structured BUY signal from the mocked LLM."""
    state = make_mock_state()
    mock_signal = make_mock_signal(analyst, action="BUY")

    with mock_llm_returning(mock_signal):
        result = await invoke_analyst(state, analyst)

    assert "agent_signals" in result, "Result must contain 'agent_signals'"
    assert analyst in result["agent_signals"], f"'{analyst}' key missing from agent_signals"

    signal = result["agent_signals"][analyst]
    assert signal.action == "BUY"
    assert signal.side == "LONG"
    assert signal.confidence == 0.82
    assert signal.stop_loss == 74500.0
    assert signal.take_profit == 78000.0
    print(f"\n[{analyst}] OK — action={signal.action}, confidence={signal.confidence}")


# ---------------------------------------------------------
# TEST 3: Each analyst falls back to HOLD on LLM failure
# ---------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("analyst", ANALYSTS_TO_TEST)
async def test_analyst_fallback_on_error(analyst):
    """If the LLM throws an exception, the analyst should return HOLD with confidence=0."""
    state = make_mock_state()

    mock_structured = AsyncMock()
    mock_structured.ainvoke = AsyncMock(side_effect=Exception("Simulated API timeout"))

    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_structured

    with patch("services.agents.analyst_agents.ChatOpenAI", return_value=mock_llm_instance):
        result = await invoke_analyst(state, analyst)

    signal = result["agent_signals"][analyst]
    assert signal.action == "HOLD"
    assert signal.confidence == 0.0
    assert "API Error" in signal.reasoning
    print(f"\n[{analyst}] Fallback OK — action={signal.action}")


# ---------------------------------------------------------
# TEST 4: Analyst works even with missing optional data
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_analyst_with_missing_news_and_portfolio():
    """Analyst should not crash if news or portfolio context is None."""
    state = {
        "symbol": "BTC",
        "market_data": make_mock_state()["market_data"],
        "news": None,
        "portfolio": None,
    }
    analyst = ANALYSTS_TO_TEST[0]
    mock_signal = make_mock_signal(analyst, action="HOLD")

    with mock_llm_returning(mock_signal):
        result = await invoke_analyst(state, analyst)

    assert result["agent_signals"][analyst].action == "HOLD"
    print(f"\nMissing data test OK for {analyst}")


# ---------------------------------------------------------
# TEST 5: LIVE — real API call, prints full LLM output
# Run with: uv run pytest tests/analyst_test.py::test_analyst_live -v -s
# ---------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("analyst", ANALYSTS_TO_TEST)
async def test_analyst_live(analyst):
    """Makes a REAL API call to OpenRouter. No mocks. Prints the full LLM signal."""
    state = make_mock_state()
    result = await invoke_analyst(state, analyst)

    signal = result["agent_signals"][analyst]

    print(f"\n{'='*60}")
    print(f"ANALYST : {signal.agent_name}")
    print(f"ACTION  : {signal.action}")
    print(f"SIDE    : {signal.side}")
    print(f"SIZE    : {signal.asset_size} BTC / {signal.usdc_size} USDC")
    print(f"LEVERAGE: {signal.leverage}x")
    print(f"SL      : {signal.stop_loss}")
    print(f"TP      : {signal.take_profit}")
    print(f"CONF    : {signal.confidence}")
    print(f"REASONING:\n{signal.reasoning}")
    print(f"{'='*60}")

    assert signal.action in ["BUY", "SELL", "HOLD", "CLOSE", "MODIFY"]
    assert 0.0 <= signal.confidence <= 1.0


# ---------------------------------------------------------
# STANDALONE RUNNER (no pytest needed)
# ---------------------------------------------------------

if __name__ == "__main__":
    async def run():
        print("=" * 50)
        print("ANALYST MOCK TEST — STANDALONE RUN")
        print("=" * 50)

        # Test prompt
        test_prompt_builds_correctly()

        # Test each analyst
        for analyst in ANALYSTS_TO_TEST:
            state = make_mock_state()
            signal = make_mock_signal(analyst, action="BUY")
            with mock_llm_returning(signal):
                result = await invoke_analyst(state, analyst)
            sig = result["agent_signals"][analyst]
            print(f"[{analyst}] action={sig.action} | confidence={sig.confidence} | side={sig.side}")

        print("\nAll tests passed.")

    asyncio.run(run())
