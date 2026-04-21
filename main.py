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
async def bot_loop(shared_traces):
    """The async loop that triggers the graph every 3 minutes."""
    while True:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        symbol = "BTC"  # You can extend this to iterate over multiple symbols later
        
        # Create a new trace dictionary for this execution
        current_trace = {
            "timestamp": timestamp,
            "symbol": symbol,
            "steps": []
        }
        
        # Appendleft so the newest execution is always at the top
        shared_traces.appendleft(current_trace)
        
        try:
            initial_state = {"symbol": symbol, "agent_signals": {}}
            
            # .astream() yields the state updates outputted by each node as they finish
            async for chunk in graph.astream(initial_state):
                for node_name, state_update in chunk.items():
                    safe_data = serialize_for_ui(state_update)
                    
                    # Store the node's output data
                    current_trace["steps"].append({
                        "node": node_name,
                        "data": safe_data
                    })
                    
        except Exception as e:
            current_trace["steps"].append({
                "node": "🚨 CRITICAL ERROR",
                "data": {"error": str(e)}
            })
        
        # Sleep for 3 minutes (180 seconds) before triggering the next cycle
        await asyncio.sleep(180)

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
        "traces": deque(maxlen=100),  # Keep only the last 100 traces
        "thread_started": False
    }
    return state

shared_state = get_shared_state()

if not shared_state["thread_started"]:
    # Spin up the background bot in a separate daemon thread
    def run_async_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(bot_loop(shared_state["traces"]))

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