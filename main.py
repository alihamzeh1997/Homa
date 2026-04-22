import streamlit as st
import asyncio
import threading
import time
from datetime import datetime, timezone
from collections import deque
import json

# Import your compiled LangGraph
from services.agents.graph import graph

# ---------------------------------------------------------
# 1. HELPER: SERIALIZE PYDANTIC FOR STREAMLIT UI
# ---------------------------------------------------------
def serialize_for_ui(obj):
    """Recursively converts Pydantic models and complex objects into dicts for st.json()"""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif isinstance(obj, list):
        return [serialize_for_ui(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: serialize_for_ui(v) for k, v in obj.items()}
    return obj

# ---------------------------------------------------------
# 2. BACKGROUND BOT RUNNER (EVERY 3 MINS)
# ---------------------------------------------------------
async def _run_one_cycle(shared_traces):
    """Single bot execution cycle: runs the full LangGraph pipeline once."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    symbol = "BTC"

    current_trace = {"timestamp": timestamp, "symbol": symbol, "steps": []}
    shared_traces.appendleft(current_trace)

    try:
        initial_state = {"symbol": symbol, "agent_signals": {}}
        config = {"configurable": {"thread_id": f"homa-bot-{symbol}"}}

        async for chunk in graph.astream(initial_state, config=config):
            for node_name, state_update in chunk.items():
                safe_data = serialize_for_ui(state_update)
                current_trace["steps"].append({"node": node_name, "data": safe_data})

        # After all nodes finish, grab the full accumulated state (includes all history)
        final_state = graph.get_state(config)
        if final_state and final_state.values:
            history_snapshot = {
                k: serialize_for_ui(v)
                for k, v in final_state.values.items()
                if k in ("sentinel_history", "analyst_signal_history", "desk_history", "cto_history")
            }
            current_trace["steps"].append({"node": "📚 FULL MEMORY SNAPSHOT", "data": history_snapshot})

    except Exception as e:
        current_trace["steps"].append({"node": "🚨 CRITICAL ERROR", "data": {"error": str(e)}})


# ---------------------------------------------------------
# 3. THREAD MANAGEMENT (STREAMLIT SAFE)
# ---------------------------------------------------------
@st.cache_resource
def get_shared_state():
    """
    Ensures the background thread and memory deque are only created ONCE,
    even when Streamlit reruns the script on UI interactions.
    """
    state = {
        "traces": deque(maxlen=100),
        "thread_started": False
    }
    return state

shared_state = get_shared_state()

if not shared_state["thread_started"]:
    def run_async_loop():
        """
        Runs forever: one fresh asyncio.run() per cycle so every iteration
        gets a clean event loop and executor — prevents 'cannot schedule new
        futures after shutdown' errors between cycles.
        """
        while True:
            try:
                asyncio.run(_run_one_cycle(shared_state["traces"]))
            except Exception as e:
                pass  # errors are captured inside _run_one_cycle; this is a safety net
            time.sleep(180)

    bot_thread = threading.Thread(target=run_async_loop, daemon=True)
    bot_thread.start()
    shared_state["thread_started"] = True

# ---------------------------------------------------------
# 4. STREAMLIT UI: THE LOG VIEWER
# ---------------------------------------------------------
st.set_page_config(page_title="Homa Bot Logs", layout="wide", page_icon="📈")

st.title("📈 Homa Trading Bot - Live Executions")
st.markdown("The bot is running in the background, executing the LangGraph pipeline every 3 minutes. Click below to fetch the latest traces.")

# Manual refresh button (Streamlit reruns the script, fetching the latest data from the deque)
if st.button("🔄 Refresh Logs"):
    st.rerun()

st.divider()

if not shared_state["traces"]:
    st.info("⏳ Waiting for the first bot execution to start... (Click refresh in a few seconds)")

# Iterate over the traces in the deque (already sorted newest to oldest)
for i, trace in enumerate(list(shared_state["traces"])):
    
    # 1st Level Accordion: The Execution Run
    # Auto-expand only the most recent execution
    with st.expander(f"🚀 Execution: {trace['timestamp']} | Target: {trace['symbol']}", expanded=(i == 0)):
        
        if not trace["steps"]:
            st.warning("Execution in progress or failed to start...")
            
        # Group parallel executions nicely or just display them in sequence
        for step in trace["steps"]:
            node_name = step["node"]
            node_data = step["data"]
            
            # 2nd Level Accordion: The Node
            with st.expander(f"⚙️ Node: {node_name.upper()}"):
                
                # 3rd Level Accordion: Node Data (Input/Output details)
                with st.expander("📦 Output State Data", expanded=True):
                    st.json(node_data)