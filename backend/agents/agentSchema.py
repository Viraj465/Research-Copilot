import os
from pydantic import BaseModel, Field
from  typing import TypedDict, List, Dict, Any, Optional

class AgentSchema(BaseModel):
    paper_url: str = Field(..., description="The URL of the research paper to analyze.")
    paper_path: str = Field(..., description="The path to the paper from Supabase storage bucket URL.")

class PaperAnalysisSchema(BaseModel):
    abstract_summary: str = Field(..., description="A concise summary of the paper's abstract.")
    section_summaries: Dict[str, str] = Field(..., description="A dictionary of section names and their summaries.")
    contributions: List[str] = Field(..., description="A list of the paper's contributions.")
    methodology: Dict[str, Any] = Field(..., description="A dictionary of the paper's methodology.")
    datasets: List[str] = Field(..., description="A list of the paper's datasets.")
    experiments: List[str] = Field(..., description="A list of the paper's experiments.")
    results: Dict[str, Any] = Field(..., description="A dictionary of the paper's results.")
    limitations: List[str] = Field(..., description="A list of the paper's limitations.")
    technical_depth: str = Field(..., description="A description of the paper's technical depth.")
    novelty: str = Field(..., description="A description of the paper's novelty.")
    code_resources: Dict[str, Any] = Field(..., description="A dictionary of the paper's code resources.")
    related_papers: List[str] = Field(..., description="A list of the paper's related papers.")
    domain_tags: List[str] = Field(..., description="A list of the paper's domain tags.")
    relevance_score: float = Field(..., description="A float value representing the paper's relevance score.")
    citations: List[str] = Field(..., description="A list of the paper's citations.")

class WebResearchSchema(BaseModel):
    topic_interpretation: str = Field(..., description="The interpretation of the research topic.")
    related_terms: List[str] = Field(..., description="A list of related terms and keywords.")
    query_intent: str = Field(..., description="The intent behind the research query.")
    retrieval_results: List[Dict[str, Any]] = Field(..., description="A list of retrieval results from web research.")
    aggregated_sources: Dict[str, List[str]] = Field(..., description="Aggregated source URLs grouped by category (e.g., {'Blogs': ['url1', 'url2'], 'Research Papers': ['url3', 'url4']}).")
    snippet_highlights: List[str] = Field(..., description="A list of highlighted snippets from sources.")
    credibility_scores: Dict[str, float] = Field(..., description="Credibility scores for each source.")
    trend_signals: Dict[str, Any] = Field(..., description="Signals indicating current trends in the field.")
    key_players: List[str] = Field(..., description="A list of key players and researchers in the field.")
    candidate_papers: List[str] = Field(..., description="A list of candidate papers for further analysis.")

class MarketIntelligenceSchema(BaseModel):
    market_size: float = Field(..., description="The estimated market size.")
    market_growth: float = Field(..., description="The market growth rate.")
    market_share: float = Field(..., description="The market share percentage.")
    market_leader: str = Field(..., description="The leading company or entity in the market.")
    market_follower: str = Field(..., description="The following company or entity in the market.")
    market_opportunity: str = Field(..., description="Identified market opportunities.")
    market_threat: str = Field(..., description="Identified market threats.")
    market_risk: str = Field(..., description="Assessed market risks.")

class SOTATrackerSchema(BaseModel):
    sota_tracker_results: List[Dict[str, Any]] = Field(..., description="Results from state-of-the-art tracking.")
    sota_tracker_summary: str = Field(..., description="Summary of SOTA tracking findings.")
    sota_tracker_recommendation: str = Field(..., description="Recommendations based on SOTA analysis.")
    sota_tracker_status: str = Field(..., description="Current status of SOTA tracking.")
    sota_tracker_date: str = Field(..., description="Date of the SOTA tracking analysis.")
    sota_tracker_author: str = Field(..., description="Author of the SOTA analysis.")
    sota_tracker_title: str = Field(..., description="Title of the SOTA tracking report.")
    sota_tracker_publication: str = Field(..., description="Publication details of the SOTA analysis.")

class ComparativeAnalysisSchema(BaseModel):
    comparative_analysis_results: List[Dict[str, Any]] = Field(..., description="Results from comparative analysis.")
    comparative_analysis_summary: str = Field(..., description="Summary of comparative analysis findings.")
    comparative_analysis_recommendation: str = Field(..., description="Recommendations based on comparative analysis.")
    comparative_analysis_status: str = Field(..., description="Current status of comparative analysis.")
    comparative_analysis_date: str = Field(..., description="Date of the comparative analysis.")
    comparative_analysis_author: str = Field(..., description="Author of the comparative analysis.")
    comparative_analysis_title: str = Field(..., description="Title of the comparative analysis report.")
    comparative_analysis_publication: str = Field(..., description="Publication details of the comparative analysis.")

class DirectionAdvisorSchema(BaseModel):
    gaps_analysis_results: List[Dict[str, Any]] = Field(..., description="Results from research gaps analysis.")
    future_directions_results: List[Dict[str, Any]] = Field(..., description="Identified future research directions.")
    future_references_results: List[Dict[str, Any]] = Field(..., description="Future reference materials and resources.")

class ReportGenerationSchema(BaseModel):
    executive_summary: str = Field(..., description="Executive summary of the research report.")
    research_findings: str = Field(..., description="Key research findings and insights.")
    technical_landscape: str = Field(..., description="Overview of the technical landscape.")
    sota_overview: Dict[str, Any] = Field(default_factory=dict, description="Overview of state-of-the-art technologies. Use {'summary': 'text', 'key_points': []} format.")
    comparative_analysis: Dict[str, Any] = Field(default_factory=dict, description="Comparative analysis results. Use {'summary': 'text', 'key_comparisons': []} format.")
    trend_analysis: Dict[str, Any] = Field(default_factory=dict, description="Analysis of current trends. Use {'trends': [], 'insights': 'text'} format.")
    ecosystem_map: Dict[str, Any] = Field(default_factory=dict, description="Map of the research ecosystem. Use {'key_players': [], 'technologies': []} format.")
    recommendations: Dict[str, Any] = Field(default_factory=dict, description="Strategic recommendations. Use {'recommendations': [], 'priority': 'text'} format.")
    future_directions: Dict[str, Any] = Field(default_factory=dict, description="Suggested future research directions. Use {'directions': [], 'opportunities': []} format.")
    market_insights: Optional[Dict[str, Any]] = Field(None, description="Optional market intelligence insights.")
    export_formats: List[str] = Field(default=["markdown", "json"], description="List of export formats (e.g., markdown, pdf, slide, notion).")

class ReActStep(BaseModel):
    thought: str = Field(..., description="Current thinking")
    action: str = Field(..., description="Action: search or finish")
    action_input: str = Field(default="", description="Search query or summary")
    observation: str = Field(default="", description="Result of action")

class SearchQualityAssessment(BaseModel):
    search_quality_score: float = Field(..., description="Quality 0-1")
    coverage_gaps: List[str] = Field(default=[], description="Missing areas")
    additional_queries_needed: List[str] = Field(default=[], description="Queries to fill gaps")
    is_sufficient: bool = Field(..., description="Is search sufficient")
    reasoning: str = Field(..., description="Assessment reasoning")

class SpecialistRoutingDecision(BaseModel):
    needed_specialists: List[str] = Field(..., description="Specialists to invoke")
    specialist_priorities: Dict[str, str] = Field(..., description="Priority levels")
    routing_reasoning: str = Field(..., description="Why these specialists")
    research_focus_areas: List[str] = Field(default=[], description="Focus areas")