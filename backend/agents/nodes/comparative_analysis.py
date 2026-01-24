import os
import logging
from typing import List, Dict, Any
from agents.state import OverallState
from agents.tools.web_search import tavily_search
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from agents.agentSchema import ComparativeAnalysisSchema
from prompts.prompts_template import COMPARATIVE_AGENT_SYSTEM_PROMPT
from utils.llm_factory import LLMFactory
from utils.safe_structured_output import safe_structured_invoke, create_empty_schema_instance

logger = logging.getLogger(__name__)

REACT_COMPARATIVE_PROMPT = """You are a comparative analysis research assistant using ReAct (Reasoning + Acting).

CRITICAL GROUNDING RULES:
- ONLY compare against papers/methods you find evidence for in search results
- DO NOT invent paper names, methods, or benchmarks that are not in the search results
- If you cannot find comparison data, say "insufficient data" rather than guessing

Task: Find and compare related work, methods, and results for the given paper.

Respond with:
THOUGHT: Your reasoning about what comparison info is needed
ACTION: 'web_search' or 'FINISH'
ACTION_INPUT: Specific search query (for web_search) or summary (for FINISH)

Search for:
1. Direct citations mentioned in the paper
2. Competing methods/approaches in the same domain
3. Benchmark comparisons and results
4. Similar papers from key authors

Finish when you have enough data to make grounded comparisons (at least 5-8 related papers/methods found)."""


def comparative_analysis_node(state: OverallState) -> OverallState:
    """
    Comparative Analysis Agent - Uses ReAct pattern for evidence-based comparison
    
    Process:
    1. Extract citations and related work from paper analysis
    2. Use ReAct loop to search for comparison data
    3. Ground all comparisons in actual search results
    4. Generate structured comparative analysis
    """

    print("\n" + "="*70)
    print("⚖️ COMPARATIVE ANALYSIS AGENT (ReAct + Citation Grounding)")
    print("="*70)
    
    try:
        paper_analysis = state.get("paper_analysis") or {}
        web_research = state.get("web_research") or {}

        # Extract paper context
        paper_title = paper_analysis.get("paper_title", "Unknown Paper")
        section_summaries = paper_analysis.get("section_summaries") or {}
        citations = paper_analysis.get("citations") or []
        related_papers = paper_analysis.get("related_papers") or []
        domain_tags = paper_analysis.get("domain_tags") or []
        contributions = paper_analysis.get("contributions") or []
        methodology = section_summaries.get("Methodology", "")
        results = section_summaries.get("Results", "")
        
        # Get existing web research results
        existing_results = web_research.get("retrieval_results") or []
        
        print(f"   📄 Paper: {paper_title[:60]}...")
        print(f"   📚 Citations: {len(citations)}")
        print(f"   🔗 Related papers: {len(related_papers)}")
        print(f"   🌐 Existing web results: {len(existing_results)}")


        # --- Ensure API keys are present in llm_config ---
        llm_config = state.get("llm_config") or {}
        api_keys = llm_config.get("api_keys") or {}
        state_api_keys = state.get("api_keys") or {}
        api_keys.update(state_api_keys)
        llm_config["api_keys"] = api_keys
        # ---
        llm = LLMFactory.get_llm(
            agent="comparative_analysis", 
            temperature=0.1,
            llm_config=llm_config
        )

        # Get Tavily API Key from state.api_keys (priority) or llm_config (fallback)
        tavily_api_key = None
        if state.get("api_keys"):
            tavily_api_key = state.get("api_keys").get("tavily")
        
        if not tavily_api_key:
            llm_config = state.get("llm_config") or {}
            tavily_api_key = (llm_config.get("api_keys") or {}).get("tavily")

        # =====================================================
        # REACT LOOP: Search for comparison data
        # =====================================================
        print("\n🔹 Starting ReAct Loop for comparison data...")
        
        react_steps = []
        comparison_results = list(existing_results)  # Start with existing results
        max_iterations = 5
        min_comparison_papers = 5
        
        # Build context from paper
        context = f"""
        Paper Title: {paper_title}
        Domain: {', '.join(domain_tags[:3])}
        Key Contributions: {', '.join([str(c)[:100] for c in contributions[:2]])}
        Citations (sample): {', '.join([str(c) for c in citations[:5]])}
        Related Papers: {', '.join([str(r) for r in related_papers[:3]])}
        """
        
        for iteration in range(max_iterations):
            print(f"\n   🔄 Iteration {iteration + 1}/{max_iterations}")
            
            thought_prompt = ChatPromptTemplate.from_messages([
                ("system", REACT_COMPARATIVE_PROMPT),
                ("user", """Context: {context}
                
                            Previous steps: {previous}
                            Comparison data collected: {count} sources
                            
                            What comparison info do we still need?
                        """)])
            
            previous = "\n".join([f"Step {i+1}: {s['thought'][:80]}" for i, s in enumerate(react_steps)])
            
            response = llm.invoke(thought_prompt.format_messages(
                context=context,
                previous=previous if previous else "None",
                count=len(comparison_results)
            ))
            
            response_text = response.content
            
            # Parse ReAct response
            thought = ""
            action = "FINISH"
            action_input = ""
            
            if "THOUGHT:" in response_text:
                thought = response_text.split("THOUGHT:")[1].split("ACTION:")[0].strip()
            if "ACTION:" in response_text:
                action = response_text.split("ACTION:")[1].split("ACTION_INPUT:")[0].strip()
            if "ACTION_INPUT:" in response_text:
                action_input = response_text.split("ACTION_INPUT:")[1].strip()
            
            print(f"      💭 THOUGHT: {thought[:80]}...")
            print(f"      ⚡ ACTION: {action}")
            
            observation = ""
            
            # Force search if too few results
            if "FINISH" in action.upper() and len(comparison_results) < min_comparison_papers:
                action = "web_search"
                if not action_input:
                    # Search for cited papers or related work
                    if citations:
                        action_input = f"{citations[0]} paper comparison"
                    else:
                        action_input = f"{paper_title} related work comparison"
            
            if "web_search" in action.lower() and action_input:
                print(f"      🔎 Search: {action_input}")
                search_results = tavily_search(action_input, max_results=5, api_key=tavily_api_key)
                
                # De-duplicate results
                seen = set()
                for r in (search_results or []):
                    url = (r.get("url") or "").strip()
                    title = (r.get("title") or "").strip()
                    key = (url or title).lower()
                    if key and key not in seen:
                        seen.add(key)
                        comparison_results.append(r)
                
                observation = f"Found {len(search_results or [])} results"
                print(f"      👁️ {observation}")
                
            elif "FINISH" in action.upper():
                observation = "Comparison data collection complete"
                print(f"      ✓ {observation}")
                react_steps.append({
                    "thought": thought,
                    "action": action,
                    "action_input": action_input,
                    "observation": observation
                })
                break
            
            react_steps.append({
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": observation
            })
            
            if len(comparison_results) >= 15:
                print("      ⚠️ Reached result limit, proceeding to analysis...")
                break
        
        print(f"\n   ✓ ReAct: {len(react_steps)} steps, {len(comparison_results)} comparison sources")

        # =====================================================
        # GENERATE GROUNDED COMPARATIVE ANALYSIS
        # =====================================================
        print("\n🔹 Generating grounded comparative analysis...")
        
        # Format search results for comparison
        import json
        results_json = json.dumps(comparison_results[:12], ensure_ascii=False, default=str)
        
        # Check if we have enough comparison data
        if len(comparison_results) < 2:
            print("⚠️ Insufficient comparison data, generating placeholder analysis...")
            comparative_analysis = {
                "comparative_analysis_results": [],
                "comparative_analysis_summary": f"Insufficient comparison data available. Only {len(comparison_results)} source(s) found.",
                "comparative_analysis_recommendation": "More research needed to establish meaningful comparisons.",
                "comparative_analysis_status": "Incomplete - insufficient data",
                "comparative_analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "comparative_analysis_author": "ResearchCopilot",
                "comparative_analysis_title": f"Comparative Analysis - {paper_title}",
                "comparative_analysis_publication": "N/A",
                "react_steps": react_steps,
                "sources_used": len(comparison_results)
            }
            return {"comparative_analysis": comparative_analysis}
        
        aggregation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are performing comparative analysis based on ACTUAL search results.

CRITICAL GROUNDING RULES:
- Use ONLY papers/methods that appear in the Search Results
- DO NOT invent paper titles, author names, or benchmark numbers
- If comparing methods, cite the source from Search Results
- For fields where no data is found, use empty lists [] or "N/A" strings
- Every comparison claim must be backed by evidence from Search Results

Provide comprehensive comparison based on:
1. Methodology differences (from sources)
2. Results/performance (from sources)  
3. Strengths/weaknesses (inferred from sources)
4. Potential impact (based on evidence)

IMPORTANT: You MUST use the ComparativeAnalysisSchema tool to respond. Fill all required fields."""),
            ("user", """Paper being analyzed:
Title: {paper_title}
Methodology: {methodology}
Results: {results}
Domain: {domains}

Search Results (JSON): {results_json}

Generate comparative analysis grounded ONLY in the search results above:""")
        ])
        
        # Use safe_structured_invoke with automatic JSON repair for robust LLM handling
        aggregation_messages = aggregation_prompt.format_messages(
            paper_title=paper_title,
            methodology=str(methodology)[:800],
            results=str(results)[:800],
            domains=", ".join([str(d) for d in domain_tags[:3]]),
            results_json=results_json
        )
        
        analysis = safe_structured_invoke(
            llm=llm,
            schema=ComparativeAnalysisSchema,
            messages=aggregation_messages,
            max_retries=2,
            retry_delay=1.0,
            fallback_value=create_empty_schema_instance(ComparativeAnalysisSchema)
        )

        if not analysis.comparative_analysis_date:
            analysis.comparative_analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        comparative_analysis = analysis.model_dump()
        
        # Add ReAct trace for transparency
        comparative_analysis["react_steps"] = react_steps
        comparative_analysis["sources_used"] = len(comparison_results)

        print(f"\n✅ Comparative Analysis Complete")
        print(f"   Title: {analysis.comparative_analysis_title}")
        print(f"   Sources: {len(comparison_results)}")
        print(f"   ReAct steps: {len(react_steps)}")

        return {"comparative_analysis": comparative_analysis}
        
    except Exception as e:
        print(f"❌ Error in comparative analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        errors = state.get("errors", [])
        errors.append({"agent": "comparative_analysis", "error": str(e)})
        return {"errors": errors}
