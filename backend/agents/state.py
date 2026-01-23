import os
from typing import Annotated, Literal, TypedDict, List, Any, Optional, Dict
from langgraph.graph.message import MessagesState

class AgentState(TypedDict):
    paper_url: str
    paper_path: str # from supabase storage bucket url

class PaperAnalysisState(TypedDict):
    metadata: Dict[str, Any]
    abstract_summary: str
    section_summaries: Dict[str, str]
    contributions: List[str]
    methodology: Dict[str, Any]
    datasets: List[str]
    experiments: List[str]
    results: Dict[str, Any]
    limitations: List[str]
    technical_depth: str
    novelty: str
    code_resources: Dict[str, Any]
    figures: List[Any]
    tables: List[Any]
    citations: List[str]
    related_papers: List[str]
    domain_tags: List[str]
    relevance_score: float

class WebResearchState(TypedDict):
    topic_interpretation: str
    related_terms: List[str]
    query_intent: str
    retrieval_results: List[Dict[str, Any]]
    aggregated_sources: Dict[str, List[Dict[str, Any]]]
    snippet_highlights: List[str]
    credibility_scores: Dict[str, float]
    trend_signals: Dict[str, Any]
    key_players: List[str]
    candidate_papers: List[str]

class MarketIntelligenceState(TypedDict): # for market intelligence agent (Optional)
    market_size: float # Market size
    market_growth: float # Market growth
    market_share: float # Market share
    market_leader: str # Market leader
    market_follower: str # Market follower
    market_opportunity: str # Market opportunity
    market_threat: str # Market threat
    market_risk: str # Market risk

class SOTATrackerAgent(TypedDict): # for SOTA tracker agent (Optional)
    sota_tracker_results: List[Dict[str, Any]]
    sota_tracker_summary: str
    sota_tracker_recommendation: str
    sota_tracker_status: str
    sota_tracker_date: str
    sota_tracker_author: str
    sota_tracker_title: str
    sota_tracker_publication: str

class ComparativeAnalysisAgent(TypedDict): # for comparative analysis agent 
    comparative_analysis_results: List[Dict[str, Any]]
    comparative_analysis_summary: str
    comparative_analysis_recommendation: str
    comparative_analysis_status: str
    comparative_analysis_date: str
    comparative_analysis_author: str
    comparative_analysis_title: str
    comparative_analysis_publication: str

class DirectionAdvisorAgent(TypedDict): # for direction advisor agent 
    gaps_analysis_results: List[Dict[str, Any]]
    future_directions_results: List[Dict[str, Any]]
    future_references_results: List[Dict[str, Any]]

class ReportGenerationAgent(TypedDict):
    executive_summary: str
    research_findings: str
    technical_landscape: str
    sota_overview: Dict[str, Any]
    comparative_analysis: Dict[str, Any]
    trend_analysis: Dict[str, Any]
    ecosystem_map: Dict[str, Any]
    recommendations: Dict[str, Any]
    future_directions: Dict[str, Any]
    market_insights: Optional[Dict[str, Any]]
    export_formats: List[str]  # e.g ["markdown", "pdf", "slide", "notion"]


class OverallState(TypedDict):
    paper_path: str
    paper_url: Optional[str]

    paper_analysis: Optional[PaperAnalysisState]
    web_research: Optional[WebResearchState]
    # market_intelligence: Optional[MarketIntelligenceState]  # REMOVED from pipeline
    sota_tracker: Optional[SOTATrackerAgent]
    comparative_analysis: Optional[ComparativeAnalysisAgent]
    direction_advisor: Optional[DirectionAdvisorAgent]
    report_generation: Optional[ReportGenerationAgent]
    api_keys: Optional[Dict[str, str]]  # For storing user-provided API keys

    active_agents: Optional[List[str]]
    errors: Optional[List[Dict[str, str]]]
    next_agents: Optional[List[str]]
    advisor_metadata: Optional[Dict[str, Any]]
    llm_config: Optional[Dict[str, Any]]

# class OverallState(TypedDict):
#     paper_path: str
    
#     paper_analysis: Optional[PaperAnalysisState]
#     web_research: Optional[WebResearchState]
#     market_intelligence: Optional[MarketIntelligenceState]
#     sota_tracker: Optional[SOTATrackerAgent]
#     comparative_analysis: Optional[ComparativeAnalysisAgent]
#     direction_advisor: Optional[DirectionAdvisorAgent]
#     report_generation: Optional[ReportGenerationAgent]
