import os
from typing import Dict, Any
from loguru import logger

from services.agents.state_schema import GraphState
from services.agents.agents_schema import PsychEvaluation

# ---------------------------------------------------------
# ASYNC PSYCHO AGENT NODE (PLACEHOLDER)
# ---------------------------------------------------------
async def psych_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph Node: The Psychologist Agent.
    Currently a pass-through placeholder. Future implementations will evaluate
    the MoE panel for groupthink, emotional bias, or irrational exuberance.
    """
    logger.debug("🧠 Psych Agent triggered: Currently functioning as a pass-through placeholder.")
    
    # Return a dummy skipped evaluation to satisfy the state schema
    dummy_eval = PsychEvaluation(
        status="SKIPPED",
        notes="Psych evaluation logic is not yet defined. Proceeding to execution."
    )
    
    return {"psych_evaluation": dummy_eval}