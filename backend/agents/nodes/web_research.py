import os
from typing import List, Dict, Any
from agents.state import OverallState
from agents.tools.web_search import tavily_search
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from agents.agentSchema import WebResearchSchema, ReActStep
# Alias for structured output
WebResearchOutput = WebResearchSchema
from prompts.prompts_template import WEB_RESEARCH_SYSTEM_PROMPT
from dotenv import load_dotenv
from utils.llm_factory import LLMFactory

load_dotenv()

def web_research_agent(state: OverallState) -> OverallState:
    """
    Web Research Agent - Uses ReAct pattern with web search
    
    Uses from paper analysis:
    - Level 3 (executive summary) for quick context
    - domain_tags for targeted search queries
    - contributions to find related work
    - research_gaps to search for solutions
    """
    
    print("\n" + "="*70)
    print("🔍 WEB RESEARCH AGENT (ReAct + Web Search)")
    print("="*70)
    
    try:
        paper_analysis = state.get("paper_analysis", {})
        
        if not paper_analysis:
            raise ValueError("No paper analysis available")
        
        # Extract context from SAS+HAS output
        print("\n🔹 Extracting context from paper analysis...")
        
        hierarchical_summaries = paper_analysis.get("hierarchical_summaries", [])
        if len(hierarchical_summaries) >= 3:
            executive_summary = hierarchical_summaries[2]["summary"]
            print(f"   ✓ Using Level 3: {len(executive_summary)} chars")
        else:
            executive_summary = "No executive summary"
        
        domain_tags = paper_analysis.get("domain_tags", []) or []
        contributions = paper_analysis.get("contributions", []) or []
        research_gaps = paper_analysis.get("research_gaps", []) or []
        paper_title = paper_analysis.get("paper_title") or ""
        authors = paper_analysis.get("authors", []) or []
        citations = paper_analysis.get("citations", []) or []
        related_papers = paper_analysis.get("related_papers", []) or []
        
        print(f"   ✓ Domain: {', '.join(domain_tags[:3])}")
        
        llm = LLMFactory.get_llm(
            agent="web_research", 
            temperature=0,
            llm_config=state.get("llm_config")
        )
        
        # Get Tavily API Key from state.api_keys (priority) or llm_config (fallback)
        tavily_api_key = None
        if state.get("api_keys"):
            tavily_api_key = state.get("api_keys").get("tavily")
        
        if not tavily_api_key:
            llm_config = state.get("llm_config") or {}
            tavily_api_key = (llm_config.get("api_keys") or {}).get("tavily")
        
        print("\n🔹 Starting ReAct Loop...")
        
        react_steps = []
        all_search_results = []
        max_iterations = 8
        
        context = f"""
        Executive Summary: {executive_summary[:500]}
        Domain: {', '.join(domain_tags[:3])}
        Contributions: {', '.join(contributions[:2])}
        Research Gaps: {', '.join([str(g) for g in research_gaps[:2]])}
        Paper Title: {paper_title[:200]}
        Authors: {', '.join([str(a) for a in authors[:5]])}
        Citations (sample): {', '.join([str(c) for c in citations[:5]])}
        Related Papers (sample): {', '.join([str(r) for r in related_papers[:5]])}
        """
        
        # =====================================================
        # PHASE 1: Search for cited papers specifically
        # =====================================================
        print("\n🔹 Phase 1: Searching for cited papers...")
        citation_results = []
        if citations:
            for citation in citations[:3]:  # Search first 3 citations
                print(f"      📚 Searching: {citation[:50]}...")
                results = tavily_search(f"{citation} paper", max_results=2, api_key=tavily_api_key)
                citation_results.extend(results or [])
            print(f"   ✓ Found {len(citation_results)} citation-related results")
            all_search_results.extend(citation_results)
        
        # =====================================================
        # PHASE 2: ReAct loop for broader research
        # =====================================================
        print("\n🔹 Phase 2: ReAct Loop for broader research...")
        min_results = 15
        for iteration in range(max_iterations):
            print(f"\n   🔄 Iteration {iteration + 1}/{max_iterations}")
            
            thought_prompt = ChatPromptTemplate.from_messages([
                ("system", WEB_RESEARCH_SYSTEM_PROMPT),
                ("user", """Context: {context}

                            Previous: {previous}
                            Results so far: {count}

                            What next?
                        """)])
            
            previous = "\n".join([f"Step {i+1}: {s['thought'][:80]}" for i, s in enumerate(react_steps)])
            
            response = llm.invoke(thought_prompt.format_messages(
                context=context,
                previous=previous if previous else "None",
                count=len(all_search_results)
            ))
            
            response_text = response.content
            
            # Parse response
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
            
            # If the model tries to finish too early, force at least some searching
            if "FINISH" in action.upper() and len(all_search_results) < min_results:
                action = "web_search"
                if not action_input:
                    action_input = f"{paper_title} related work" if paper_title else "recent work " + " ".join(domain_tags[:3])

            if "web_search" in action.lower() and action_input:
                print(f"      🔎 Search: {action_input}")
                # Tavily (web-only)
                tavily_results = tavily_search(action_input, max_results=7, api_key=tavily_api_key)

                # De-dupe by URL/title (Tavily sometimes returns duplicates across queries)
                seen = set()
                deduped = []
                for r in (tavily_results or []):
                    url = (r.get("url") or r.get("URL") or "").strip() if isinstance(r, dict) else ""
                    title = (r.get("title") or "").strip() if isinstance(r, dict) else ""
                    key = (url or title).lower()
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    deduped.append(r)

                all_search_results.extend(deduped)
                observation = f"Found {len(deduped)} results"
                print(f"      👁️ {observation}")
                
            elif "FINISH" in action.upper():
                observation = "Search complete"
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
            
            if len(all_search_results) >= 30:
                break
        
        print(f"\n   ✓ ReAct: {len(react_steps)} steps, {len(all_search_results)} results")
        
        # Aggregate results
        print("\n🔹 Aggregating results...")
        
        aggregation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are extracting structured information from WEB SEARCH RESULTS.

CRITICAL GROUNDING RULES:
- Use ONLY the provided Results to populate every field.
- Do NOT invent names, papers, benchmarks, metrics, or trends not explicitly present in Results.
- If you cannot find evidence in Results, return empty lists/dicts or 'N/A' strings (do not guess).
- key_players MUST be names/entities that appear verbatim in Results titles/snippets.

You must return the output using the 'WebResearchSchema' structure. Do not invent other tools."""),
            ("user", "Context (for query intent only): {context}\n\nHere are the Search Results:\n{results_json}\n\nExtract the information now:"),
        ])
        
        structured_llm = llm.with_structured_output(WebResearchOutput)
        chain = aggregation_prompt | structured_llm
        
        import json
        results_json = json.dumps(all_search_results[:10], ensure_ascii=False)

        output = chain.invoke({
            "context": context[:2000],
            "results_json": results_json
        })
        
        output_dict = output.dict()
        output_dict["react_steps"] = react_steps
        output_dict["retrieval_results"] = all_search_results
        
        # =====================================================
        # PHASE 3: Extract candidate papers from search results
        # =====================================================
        candidate_papers = []
        for result in all_search_results:
            title = result.get("title", "") if isinstance(result, dict) else ""
            url = result.get("url", "") if isinstance(result, dict) else ""
            # Check if it looks like a paper (arxiv, semanticscholar, pdf, paper keywords)
            if any(kw in url.lower() for kw in ["arxiv", "semanticscholar", "acm.org", "ieee.org", ".pdf", "paper"]) or \
               any(kw in title.lower() for kw in ["paper", "study", "research", "learning", "neural", "model"]):
                candidate_papers.append({
                    "title": title,
                    "url": url,
                    "snippet": result.get("content", "")[:200] if isinstance(result, dict) else ""
                })
        
        # De-duplicate candidate papers
        seen_titles = set()
        unique_candidates = []
        for paper in candidate_papers:
            if paper["title"].lower() not in seen_titles:
                seen_titles.add(paper["title"].lower())
                unique_candidates.append(paper)
        
        output_dict["candidate_papers"] = unique_candidates[:15]
        print(f"\n   📄 Identified {len(unique_candidates)} candidate related papers")
        
        print("\n✅ WEB RESEARCH COMPLETE")
        print(f"   Steps: {len(react_steps)}")
        print(f"   Results: {len(all_search_results)}")
        print(f"   Players: {', '.join(output_dict.get('key_players', [])[:3])}")
        
        return {"web_research": output_dict}
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append({"agent": "web_research", "error": str(e)})
        return {"errors": state["errors"]}