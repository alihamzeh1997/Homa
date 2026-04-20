from langgraph.graph import StateGraph, START, END

from services.agents.state_schema import GraphState
from services.agents.hl_input_node import market_data_node
from services.agents.news_agent import news_node
from services.agents.sentinel_agent import sentinel_node
from services.agents.analyst_agents import (
    ANALYST_NODES,
    deepseek_node,
    minimax_node,
    gemini_node,
    grok_node,
    kimi_node,
    qwen_node,
    gpt_node,
)
from services.agents.desk_manager import desk_manager_node
from services.agents.money_manager import money_manager_node
from services.agents.psych_agent import psych_node
from services.agents.cto_agent import cto_node
from services.agents.action_node import action_node

def _route_sentinel(state: GraphState) -> list[str]:
    """
    Fan-out router after the Sentinel:
      SKIP      → terminate; no analysts invoked (saves API cost).
      EMERGENCY → jump straight to CTO; skip MoE panel and risk pipeline.
      NORMAL    → fan-out to all 7 MoE analysts in parallel.
    """
    if state.get("skip_workflow"):
        return [END]
    if state.get("is_emergency"):
        return ["cto"]
    return ANALYST_NODES


def build_graph() -> StateGraph:
    builder = StateGraph(GraphState)

    # ------------------------------------------------------------------ nodes
    builder.add_node("market_data", market_data_node)
    builder.add_node("news", news_node)
    builder.add_node("sentinel", sentinel_node)

    builder.add_node("deepseek", deepseek_node)
    builder.add_node("minimax", minimax_node)
    builder.add_node("gemini", gemini_node)
    builder.add_node("grok", grok_node)
    builder.add_node("kimi", kimi_node)
    builder.add_node("qwen", qwen_node)
    builder.add_node("gpt", gpt_node)

    builder.add_node("desk_manager", desk_manager_node)
    builder.add_node("money_manager", money_manager_node)
    builder.add_node("psych", psych_node)
    builder.add_node("cto", cto_node)
    builder.add_node("action", action_node)

    # ------------------------------------------------------------------ edges

    # Entry: market data and news fetch run in parallel; sentinel waits for both
    builder.add_edge(START, "market_data")
    builder.add_edge(START, "news")
    builder.add_edge("market_data", "sentinel")
    builder.add_edge("news", "sentinel")

    # Sentinel: conditional fan-out (SKIP / EMERGENCY / NORMAL)
    builder.add_conditional_edges(
        "sentinel",
        _route_sentinel,
        ANALYST_NODES + ["cto", END],
    )

    # Analyst fan-in: all 7 analysts feed into Desk Manager.
    # LangGraph waits for every parallel branch to complete before
    # advancing to desk_manager (merge_signals reducer aggregates the signals).
    for name in ANALYST_NODES:
        builder.add_edge(name, "desk_manager")

    # desk_manager fans out to money_manager and psych in parallel; cto waits for both
    builder.add_edge("desk_manager", "money_manager")
    builder.add_edge("desk_manager", "psych")
    builder.add_edge("money_manager", "cto")
    builder.add_edge("psych", "cto")
    builder.add_edge("cto", "action")
    builder.add_edge("action", END)

    return builder.compile()


# Module-level compiled graph — import and invoke directly:
#   from services.agents.graph import graph
#   result = await graph.ainvoke({"symbol": "BTC", "agent_signals": {}})
graph = build_graph()
