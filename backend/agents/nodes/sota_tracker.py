import os
from agents.state import OverallState
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from datetime import datetime
from agents.agentSchema import SOTATrackerSchema
from prompts.prompts_template import SOTA_TRACKER_SYSTEM_PROMPT
from utils.llm_factory import LLMFactory

def sota_tracker_agent(state: OverallState) -> OverallState:
    """
    SOTA Tracker Agent
    
    Uses: Level 1 (detailed) + Methodology + Results section summaries
    Why: Needs full technical depth for proper SOTA comparison
    """
    
    print("\n🏆 SOTA TRACKER AGENT")
    
    try:
        paper_analysis = state.get("paper_analysis") or {}
        web_research = state.get("web_research") or {}
        
        # Access Level 1 (detailed) summary
        hierarchical = paper_analysis.get("hierarchical_summaries", [])
        
        if len(hierarchical) >= 1:
            level1_summary = hierarchical[0]["summary"]
            print(f"   Using Level 1: {len(level1_summary)} chars")
        else:
            level1_summary = "No detailed summary"
        
        # Access section summaries for technical depth
        section_summaries = paper_analysis.get("section_summaries", {})
        
        methodology_summary = section_summaries.get("Methodology", "")
        results_summary = section_summaries.get("Results", "")
        
        print(f"   Methodology: {len(methodology_summary)} chars")
        print(f"   Results: {len(results_summary)} chars")
        
        domain_tags = paper_analysis.get("domain_tags", [])


        # --- Ensure API keys are present in llm_config ---
        llm_config = state.get("llm_config") or {}
        api_keys = llm_config.get("api_keys") or {}
        state_api_keys = state.get("api_keys") or {}
        api_keys.update(state_api_keys)
        llm_config["api_keys"] = api_keys
        # ---
        llm = LLMFactory.get_llm(
            agent="sota_tracker", 
            temperature=0.1,
            llm_config=llm_config
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", SOTA_TRACKER_SYSTEM_PROMPT),
            ("user", """Level 1 (Detailed): {level1}

Methodology: {methodology}
Results: {results}

Domain: {domains}
Trends: {trends}

Provide SOTA analysis using the SOTATrackerSchema tool.
IMPORTANT: You must call the tool 'SOTATrackerSchema' with your analysis. Do not output markdown text.""")
        ])
        
        structured_llm = llm.with_structured_output(SOTATrackerSchema)
        chain = prompt | structured_llm
        
        analysis = chain.invoke({
            "level1": level1_summary[:2000],
            "methodology": methodology_summary[:1000],
            "results": results_summary[:1000],
            "domains": ", ".join([str(d) for d in domain_tags[:3]]),
            "trends": str(web_research.get("trend_signals") or {})[:500]
        })
        
        if not analysis.sota_tracker_date:
            analysis.sota_tracker_date = datetime.now().strftime("%Y-%m-%d")
        
        sota_tracker = analysis.dict()
        
        print(f"✅ SOTA TRACKER COMPLETE")
        print(f"   Title: {analysis.sota_tracker_title}")
        print(f"   Status: {analysis.sota_tracker_status}")
        
        return {"sota_tracker": sota_tracker}
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append({"agent": "sota_tracker", "error": str(e)})
        return {"errors": state["errors"]}