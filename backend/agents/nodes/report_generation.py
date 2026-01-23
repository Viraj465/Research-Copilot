import os
from agents.state import OverallState
from langchain_core.prompts import ChatPromptTemplate
from agents.agentSchema import ReportGenerationSchema
from prompts.prompts_template import REPORT_GENERATION_SYSTEM_PROMPT
from dotenv import load_dotenv
import logging
from utils.llm_factory import LLMFactory

logger = logging.getLogger(__name__)
load_dotenv()


def report_generation_node(state: OverallState) -> OverallState:
    """
    Report Generation Agent - Creates comprehensive final report
    
    Uses: Everything from all agents (market_intelligence removed)
    """
    print("\n 📝 Report Generation Agent 🚀")
    try:
        paper_analysis = state.get("paper_analysis", {})
        web_research = state.get("web_research", {})
        sota_tracker = state.get("sota_tracker", {})
        comparative_analysis = state.get("comparative_analysis", {})
        direction_advisor = state.get("direction_advisor", {})

        hierarchical = paper_analysis.get("hierarchical_summaries", [])

        level_3 = hierarchical[2]["summary"] if len(hierarchical) > 2 else ""
        level_2 = hierarchical[1]["summary"] if len(hierarchical) > 1 else ""
        level_1 = hierarchical[0]["summary"] if len(hierarchical) > 0 else ""

        logger.info(" - Synthesizing all analyses...")

        llm = LLMFactory.get_llm(
            agent="report_generation", 
            temperature=0.1,
            llm_config=state.get("llm_config")
        )

        # Web research insights
        web_section = ""
        if web_research:
            key_players = web_research.get("key_players", [])[:5]
            candidate_papers = web_research.get("candidate_papers", [])[:5]
            web_section = f"""WEB RESEARCH INSIGHTS:
                            - Key Players: {', '.join(key_players) if key_players else 'N/A'}
                            - Related Papers Found: {len(candidate_papers)}
                            - Trend Signals: {str(web_research.get('trend_signals', {}))[:200]}"""

        sota_section = ""
        if sota_tracker:
            sota_section = f"""SOTA TRACKING:
                            - Summary: {sota_tracker.get('sota_tracker_summary', 'N/A')[:300]}
                            - Recommendation: {sota_tracker.get('sota_tracker_recommendation', 'N/A')[:200]}"""

        comparative_section = ""
        if comparative_analysis:
            sources_used = comparative_analysis.get('sources_used', 0)
            comparative_section = f"""COMPARATIVE ANALYSIS (ReAct-based):
                            - Summary: {comparative_analysis.get('comparative_analysis_summary', 'N/A')[:300]}
                            - Recommendation: {comparative_analysis.get('comparative_analysis_recommendation', 'N/A')[:200]}
                            - Sources Used: {sources_used}"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", REPORT_GENERATION_SYSTEM_PROMPT + """

IMPORTANT OUTPUT FORMAT:
You MUST use the ReportGenerationSchema tool to respond. 

For dictionary fields (sota_overview, comparative_analysis, etc.), use this format:
- sota_overview: {{"summary": "text here", "key_points": ["point1", "point2"]}}
- comparative_analysis: {{"summary": "text here", "key_comparisons": ["comp1", "comp2"]}}
- trend_analysis: {{"trends": ["trend1", "trend2"], "insights": "text here"}}
- ecosystem_map: {{"key_players": ["player1", "player2"], "technologies": ["tech1", "tech2"]}}
- recommendations: {{"recommendations": ["rec1", "rec2"], "priority": "text here"}}
- future_directions: {{"directions": ["dir1", "dir2"], "opportunities": ["opp1", "opp2"]}}

For export_formats, use: ["markdown", "json", "pdf"]

Do NOT return plain text - use the structured schema!"""),
            ("user", """HIERARCHICAL SUMMARIES:
                        Level 3 (Executive): {level3}
                        Level 2 (Intermediate): {level2}
                        Level 1 (Detailed): {level1}

                        {web_section}

                        {sota_section}

                        {comparative_section}

                        STRATEGIC DIRECTION:
                        {direction}

                        Create comprehensive report using the ReportGenerationSchema:
                    """)])
        
        structured_llm = llm.with_structured_output(ReportGenerationSchema, method="function_calling", strict=True)
        chain = prompt | structured_llm
        
        try:
            report = chain.invoke({
                "level3": level_3,
                "level2": level_2[:1500],
                "level1": level_1[:1000],
                "web_section": web_section,
                "sota_section": sota_section,
                "comparative_section": comparative_section,
                "direction": str(direction_advisor)[:500]
            })
        except Exception as e:
            logger.error(f"Structured output failed: {e}")
            print(f"⚠️ Structured output failed, creating fallback report...")
            # Create fallback report
            from pydantic import ValidationError
            report = ReportGenerationSchema(
                executive_summary=level_3[:1000] if level_3 else "Report generation encountered an error.",
                research_findings=level_2[:1000] if level_2 else "Limited findings available.",
                technical_landscape=str(web_section)[:800] if web_section else "Technical landscape analysis pending.",
                sota_overview={"summary": str(sota_tracker.get("sota_tracker_summary", "N/A"))[:500], "key_points": []},
                comparative_analysis={"summary": str(comparative_analysis.get("comparative_analysis_summary", "N/A"))[:500], "key_comparisons": []},
                trend_analysis={"trends": [], "insights": "Trend analysis pending"},
                ecosystem_map={"key_players": web_research.get("key_players", [])[:5] if web_research else [], "technologies": []},
                recommendations={"recommendations": [], "priority": "Further analysis recommended"},
                future_directions={"directions": [], "opportunities": []},
                export_formats=["markdown", "json"]
            )

        report_result = report.dict()
        
        logger.info("\n" + "=" *70)
        print(f"\n   ✓ Report Generation Complete:")
        logger.info("\n" + "=" *70)
        print("\n 📄 Executive Summary:")      
        logger.info("\n" + "=" *70)
        print(report.executive_summary[:500])
        logger.info("\n" + "=" *70)
        print(f"Export formats: {', '.join(report.export_formats)}")
        logger.info("\n" + "=" *70)

        # Return proper state update format for LangGraph
        return {"report_generation": report_result}
        
    except Exception as e:
        print(f"Error in Report Generation: {e}")
        logger.error(f"Error in Report Generation: {e}")
        import traceback
        traceback.print_exc()
        errors = state.get("errors", [])
        errors.append({"agent": "report_generation", "error": str(e)})
        return {"errors": errors}
