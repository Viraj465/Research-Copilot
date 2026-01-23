import os
import logging
from agents.state import OverallState
from langchain_core.prompts import ChatPromptTemplate
from agents.agentSchema import DirectionAdvisorSchema
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from prompts.prompts_template import DIRECTION_ADVISOR_SYSTEM_PROMPT
from utils.llm_factory import LLMFactory

logger = logging.getLogger(__name__)

def direction_advisor_node(state: OverallState) -> OverallState:
    """
    Direction Advisor Agent - Synthesis and strategic guidance
    
    Uses: All hierarchical levels + all specialist outputs
    Note: market_intelligence removed from the pipeline
    """

    print("\n 📈 Direction Advisor Agent 🚀")
    try:
        paper_analysis = state.get("paper_analysis") or {}
        web_research = state.get("web_research") or {}
        sota_tracker = state.get("sota_tracker") or {}
        comparative_analysis = state.get("comparative_analysis") or {}

        hierarchical = paper_analysis.get("hierarchical_summaries", [])

        level3 = hierarchical[2]["summary"] if len(hierarchical) > 2 else ""
        level2 = hierarchical[1]["summary"] if len(hierarchical) > 1 else ""
        level1 = hierarchical[0]["summary"] if len(hierarchical) > 0 else ""

        logger.info("Using all hierarchical levels: Level 3, Level 2, Level 1")
        contributions = paper_analysis.get("contributions", [])
        limitations = paper_analysis.get("limitations", [])
        domain_tags = paper_analysis.get("domain_tags", [])
        
        # Get key insights from web research
        key_players = web_research.get("key_players", [])
        trend_signals = web_research.get("trend_signals", {})

        llm = LLMFactory.get_llm(
            agent="direction_advisor", 
            temperature=0.1,
            llm_config=state.get("llm_config")
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", DIRECTION_ADVISOR_SYSTEM_PROMPT),
            ("user", """Level 3 (Executive): {level3}
                        Level 2 (Intermediate): {level2}
                        Contributions: {contributions}
                        Limitations: {limitations}
                        Domain: {domains}
                        Key Players: {key_players}
                        Trends: {trends}
                        SOTA: {sota}
                        Comparative: {comparative}

                        Provide strategic direction using the DirectionAdvisorSchema tool.
                        IMPORTANT: You must ONLY call the tool 'DirectionAdvisorSchema'. Do not output markdown text.
                    """)])

        structured_llm = llm.with_structured_output(DirectionAdvisorSchema)
        
        analysis = structured_llm.invoke(prompt.format_messages(
            level3=level3[:3000],
            level2=level2[:3000],
            contributions=", ".join([str(c) for c in contributions[:5]]),
            limitations=", ".join([str(l) for l in limitations[:5]]),
            domains=", ".join([str(d) for d in domain_tags[:5]]),
            key_players=", ".join([str(p) for p in key_players[:5]]) if key_players else "N/A",
            trends=str(trend_signals)[:2000] if trend_signals else "N/A",
            sota=str(sota_tracker.get("sota_tracker_summary", ""))[:2000] if sota_tracker else "N/A",
            comparative=str(comparative_analysis.get("comparative_analysis_summary", ""))[:2000] if comparative_analysis else "N/A"
        ))

        direction_advisor_result = analysis.dict()

        print(f" ✅ Direction Advisor Complete")
        print(f"   Gaps: {len(analysis.gaps_analysis_results)}")
        print(f"   Directions: {len(analysis.future_directions_results)}")

        # Return properly formatted state update (key fix for direction_advisor being null)
        return {"direction_advisor": direction_advisor_result}
        
    except Exception as e:
        print(f" ❌ Error in direction advisor: {str(e)}")
        import traceback
        traceback.print_exc()
        errors = state.get("errors", [])
        errors.append({"agent": "direction_advisor", "error": str(e)})
        return {"errors": errors}
