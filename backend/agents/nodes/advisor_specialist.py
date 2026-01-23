import os
from typing import List, Dict, Any
from agents.state import OverallState
from agents.tools.web_search import tavily_search
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from agents.agentSchema import SearchQualityAssessment, SpecialistRoutingDecision
from prompts.prompts_template import ADVISOR_SPECIALIST_ROUTING_SYSTEM_PROMPT
from utils.llm_factory import LLMFactory

def advisor_specialist_agent(state: OverallState) -> OverallState:
    """
    Advisor Specialist Agent - Quality control & routing
    
    Uses from paper analysis:
    - Level 3 (executive) for quick assessment
    - Level 2 (intermediate) for detailed understanding
    - Section summaries for gap detection
    
    Process:
    1. Assess search quality
    2. Fill gaps if needed (additional searches)
    3. Make routing decisions for specialists
    """
    
    print("\n" + "="*70)
    print("🤔 ADVISOR SPECIALIST AGENT (Self-Reflection + ReAct)")
    print("="*70)
    
    try:
        paper_analysis = state.get("paper_analysis") or {}
        web_research = state.get("web_research") or {}
        
        if not paper_analysis:
            raise ValueError("Missing paper analysis")
        
        # If web research failed or is missing, use fallback routing
        if not web_research or not web_research.get("retrieval_results"):
            print("⚠️ Web research data missing or incomplete, using fallback routing...")
            return {
                "next_agents": ["sota_tracker", "comparative_analysis"],
                "advisor_metadata": {
                    "quality_assessment": {"search_quality_score": 0.0, "is_sufficient": False, "reasoning": "Web research failed"},
                    "routing_decision": {"needed_specialists": ["sota_tracker", "comparative_analysis"], "specialist_priorities": {}}
                }
            }
        
        # Access hierarchical summaries
        print("\n🔹 Accessing hierarchical summaries...")
        
        hierarchical = paper_analysis.get("hierarchical_summaries", [])
        
        if len(hierarchical) >= 3:
            level3 = hierarchical[2]["summary"]
            level2 = hierarchical[1]["summary"]
            print(f"   ✓ Level 3: {len(level3)} chars")
            print(f"   ✓ Level 2: {len(level2)} chars")
        else:
            raise ValueError("Missing hierarchical summaries")
        
        domain_tags = paper_analysis.get("domain_tags", [])
        contributions = paper_analysis.get("contributions", [])


        # --- Ensure API keys are present in llm_config ---
        llm_config = state.get("llm_config") or {}
        api_keys = llm_config.get("api_keys") or {}
        state_api_keys = state.get("api_keys") or {}
        api_keys.update(state_api_keys)
        llm_config["api_keys"] = api_keys
        # ---
        llm = LLMFactory.get_llm(
            agent="advisor_specialist", 
            temperature=0,
            llm_config=llm_config
        )
        
        # Get Tavily API Key from state.api_keys (priority) or llm_config (fallback)
        tavily_api_key = None
        if state.get("api_keys"):
            tavily_api_key = state.get("api_keys").get("tavily")
        
        if not tavily_api_key:
            llm_config = state.get("llm_config") or {}
            tavily_api_key = (llm_config.get("api_keys") or {}).get("tavily")
        
        print("\n🔹 STEP 1: Assessing search quality...")
        
        quality_prompt = ChatPromptTemplate.from_messages([
            ("system", """Evaluate search quality. Is coverage sufficient?

Consider: domain coverage, recency, diversity, depth, completeness.

If score < 0.7, suggest specific queries to fill gaps."""),
            ("user", """Domain: {domains}
Contributions: {contributions}
Level 2 Context: {level2}

Search Results: {result_count} items
Key Players: {players}

Assess quality:""")
        ])
        
        structured_llm = llm.with_structured_output(SearchQualityAssessment)
        chain = quality_prompt | structured_llm
        
        assessment = chain.invoke({
            "domains": ", ".join([str(d) for d in domain_tags[:3]]),
            "contributions": " | ".join([str(c) for c in contributions[:2]]),
            "level2": level2[:1000],
            "result_count": len(web_research.get("retrieval_results", [])),
            "players": ", ".join([str(p) for p in (web_research.get("key_players") or [])[:5]])
        })
        
        print(f"   ✓ Quality: {assessment.search_quality_score:.2f}")
        print(f"   ✓ Sufficient: {assessment.is_sufficient}")
        
        # Step 2: Fill gaps if needed
        if not assessment.is_sufficient and assessment.additional_queries_needed:
            print(f"\n🔹 STEP 2: Filling gaps ({len(assessment.additional_queries_needed)} searches)...")
            
            additional_results = []
            for query in assessment.additional_queries_needed[:3]:
                print(f"      🔎 {query}")
                results = tavily_search(query, max_results=3, api_key=tavily_api_key)
                additional_results.extend(results)
            
            # Update web research
            current_results = web_research.get("retrieval_results", [])
            current_results.extend(additional_results)
            web_research["retrieval_results"] = current_results
            state["web_research"] = web_research
            
            print(f"   ✓ Added {len(additional_results)} results")
        else:
            print("\n🔹 STEP 2: Search sufficient, proceeding...")
        
        # Step 3: Routing decision
        print("\n🔹 STEP 3: Making routing decisions...")
        
        routing_prompt = ChatPromptTemplate.from_messages([
            ("system", ADVISOR_SPECIALIST_ROUTING_SYSTEM_PROMPT),
            ("user", """Level 3 (Executive): {level3}
                        Level 2 (Intermediate): {level2}

                        Domain: {domains}
                        Novelty: {novelty}
                        Trends: {trends}

                        Which specialists?
                    """)])
        
        structured_llm_routing = llm.with_structured_output(SpecialistRoutingDecision)
        chain_routing = routing_prompt | structured_llm_routing
        
        decision = chain_routing.invoke({
            "level3": level3[:500],
            "level2": level2[:1000],
            "domains": ", ".join([str(d) for d in domain_tags[:3]]),
            "novelty": paper_analysis.get("novelty", "Unknown"),
            "trends": str(web_research.get("trend_signals") or {})[:300]
        })
        
        print(f"   ✓ Routing: {', '.join([str(s) for s in decision.needed_specialists])}")
        print(f"   ✓ Priorities: {decision.specialist_priorities}")
        
        # Prepare state updates (return only updated fields)
        state_update = {
            "next_agents": decision.needed_specialists,
            "advisor_metadata": {
                "quality_assessment": assessment.dict(),
                "routing_decision": decision.dict()
            }
        }
        
        if not assessment.is_sufficient and assessment.additional_queries_needed:
            state_update["web_research"] = web_research
        
        print("\n✅ ADVISOR COMPLETE")
        print(f"   Quality: {assessment.search_quality_score:.2f}")
        print(f"   Specialists: {len(decision.needed_specialists)}")
        
        return state_update
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        errors = state.get("errors", [])
        errors.append({"agent": "advisor_specialist", "error": str(e)})
        # Fallback routing (market_intelligence removed)
        return {
            "errors": errors,
            "next_agents": ["sota_tracker", "comparative_analysis"]
        }