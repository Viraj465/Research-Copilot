import os
from typing import List, Dict, Any
from agents.state import OverallState
from agents.tools.SAS_HAS_processor import SASHASProcessor
from agents.tools.pdf_loader import load_paper_from_path, load_pdf_from_url
from langchain_core.prompts import ChatPromptTemplate
import logging
from prompts.prompts_template import (REASONING_QUESTIONS, REASONING_SYSTEM_PROMPT)
from utils.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


def paper_analysis_node(state: OverallState) -> OverallState:
    """
    Paper Analysis Agent - Processes paper with SAS+HAS + Reasoning RAG
    
    This is the ONLY agent that uses SAS+HAS processor.
    Creates hierarchical summaries that all other agents consume.
    """
    logger.info("=" * 70)
    print("Analyzing paper... 📰")
    logger.info("=" * 70)

    try:
        paper_path = state.get("paper_path")
        paper_url = state.get("paper_url")

        print(f"STEP 1:🔃 Loading paper...")
        if not paper_path and not paper_url:
            raise ValueError("Invalid input: paper path or URL is required")
        elif paper_path:
            paper_content = load_paper_from_path(paper_path)
        else:
            paper_content = load_pdf_from_url(paper_url)


        print(f"STEP 2: 🧠 Running SAS+HAS Processor...")
        # --- Ensure API keys are present in llm_config ---
        llm_config = state.get("llm_config") or {}
        api_keys = llm_config.get("api_keys") or {}
        state_api_keys = state.get("api_keys") or {}
        api_keys.update(state_api_keys)
        llm_config["api_keys"] = api_keys
        # ---
        processor = SASHASProcessor(llm_config=llm_config)
        sas_has_analysis = processor.process_paper(paper_content)

        print(f"      - Level 1: {len(sas_has_analysis.hierarchical_summaries[0].summary)} chars")

        logger.info(f"STEP 3: Running Reasoning RAG...")
        
        llm = LLMFactory.get_llm(
            agent="paper_analysis",
            temperature=float(os.getenv("GROQ_TEMPERATURE", "0.3")),
            llm_config=state.get("llm_config")
        )

        reasoning_traces = []

        for rq in REASONING_QUESTIONS:
            evidence = []
            for section_name, section_summary in sas_has_analysis.section_summaries.items():
                for relevant_section in rq["relevant_sections"]:
                    if relevant_section.lower() in section_name.lower():
                        evidence.append(f"[{section_name}]: {section_summary[:300]}")
                        break
            if not evidence:
                continue
            
            reasoning_prompt = ChatPromptTemplate.from_messages([
                ("system", REASONING_SYSTEM_PROMPT),
                ("user", "Question: {question}\n\nEvidence:\n{evidence}\n\nContext: {context}\n\nProvide reasoning:")
            ]) 

            reasoning_response = llm.invoke(reasoning_prompt.format_messages(
                question=rq['question'],
                evidence="\n\n".join(evidence[:5]),
                context=sas_has_analysis.hierarchical_summaries[1].summary[:1000]
            ))

            reasoning_traces.append({
                "question": rq['question'],
                "evidence": evidence[:3],
                "reasoning": reasoning_response.content,
                "conclusion": reasoning_response.content[:200]
            })
        print(f"\n   ✓  Reasoning: {len(reasoning_traces)} steps.")

        research_gaps = []
        for trace in reasoning_traces:
            if "limitation" in trace["question"].lower() or "gap" in trace["question"].lower():
                research_gaps.append(trace["conclusion"])
        
        hierarchical_summaries_dict = [
            {"level": h.level,
            "summary": h.summary,
            "key_contributions": h.key_contributions,
            "scope": h.scope
            }
            for h in sas_has_analysis.hierarchical_summaries
        ]

        final_analysis = {
            "hierarchical_summaries": hierarchical_summaries_dict,
            "section_summaries": sas_has_analysis.section_summaries,
            "reasoning_traces": reasoning_traces,
            "paper_title": sas_has_analysis.paper_title,
            "authors": sas_has_analysis.authors,
            "abstract_summary": sas_has_analysis.abstract_summary,
            "contributions": sas_has_analysis.contributions,
            "methodology": sas_has_analysis.methodology,
            "datasets": sas_has_analysis.datasets,
            "experiments": sas_has_analysis.experiments,
            "results": sas_has_analysis.results,
            "limitations": sas_has_analysis.limitations,
            "future_work": sas_has_analysis.future_work,
            "research_gaps": research_gaps,
            "technical_depth": sas_has_analysis.technical_depth,
            "novelty": sas_has_analysis.novelty,
            "domain_tags": sas_has_analysis.domain_tags,
            "code_resources": sas_has_analysis.code_resources,
            "related_papers": sas_has_analysis.related_papers,
            "citations": sas_has_analysis.citations,
            "relevance_score": sas_has_analysis.relevance_score,
            "quality_score": sas_has_analysis.quality_score,
            "total_sections": sas_has_analysis.total_sections
        }

        state["paper_analysis"] = final_analysis

        logger.info("=" * 70)
        print("✅ Paper Analysis Complete")
        print(f"Title: {sas_has_analysis.paper_title}")
        print(f"Contributions: {sas_has_analysis.contributions}")
        print(f"Domains: {', '.join([str(d) for d in sas_has_analysis.domain_tags[:3]])}")
        logger.info("=" * 70)

        return {"paper_analysis": final_analysis}
    except Exception as e:
        print(f" ❌ Error in paper analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append({"agent": "paper_analysis", "error": str(e)})
        return {"errors": state["errors"]}