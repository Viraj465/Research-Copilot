from typing import Literal, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from agents.state import OverallState
from agents.nodes.paper_analysis import paper_analysis_node
from agents.nodes.web_research import web_research_agent
from agents.nodes.advisor_specialist import advisor_specialist_agent
from agents.nodes.sota_tracker import sota_tracker_agent
from agents.nodes.comparative_analysis import comparative_analysis_node
from agents.nodes.direction_advisor import direction_advisor_node
from agents.nodes.report_generation import report_generation_node


def should_run_specialists(state: OverallState) -> List[str]:
    """
    Routing function to determine which specialist agents to run.
    Based on advisor_specialist's routing decision or fallback to all specialists.
    
    Returns list of node names to run in parallel.
    """
    next_agents = state.get("next_agents", [])
    
    if not next_agents:
        # Default: run both specialist agents (market_intelligence removed)
        return ["sota_tracker", "comparative_analysis"]
    
    # Map agent names to node names (market_intelligence removed)
    agent_mapping = {
        "sota_tracker": "sota_tracker",
        "comparative_analysis": "comparative_analysis"
    }
    
    return [agent_mapping.get(agent, agent) for agent in next_agents if agent in agent_mapping]


def create_research_graph() -> CompiledStateGraph:
    """
    Creates and compiles the Research Copilot agent graph.
    
    The graph follows this sequential flow with a parallel branch:
    1. Paper Analysis (SAS/HAS + Reasoning RAG)
    2. Web Research (ReAG pattern)
    3. Advisor Specialist (Quality control + routing)
    4. Parallel Specialists:
       - Market Intelligence
       - SoTA Tracker
       - Comparative Analysis
    5. Direction Advisor (Synthesis)
    6. Report Generation (Final output)
    """
    
    # Initialize the graph with OverallState
    graph = StateGraph(OverallState)
    
    # =============================================
    # Add all agent nodes
    # =============================================
    
    # Core analysis nodes
    graph.add_node("paper_analysis", paper_analysis_node)
    graph.add_node("web_research", web_research_agent)
    graph.add_node("advisor_specialist", advisor_specialist_agent)
    
    # Specialist nodes (run in parallel) - market_intelligence removed
    graph.add_node("sota_tracker", sota_tracker_agent)
    graph.add_node("comparative_analysis", comparative_analysis_node)
    
    # Aggregation and output nodes
    graph.add_node("direction_advisor", direction_advisor_node)
    graph.add_node("report_generation", report_generation_node)
    
    # =============================================
    # Define edges (flow connections)
    # =============================================
    
    # Entry point: START → Paper Analysis
    graph.add_edge(START, "paper_analysis")
    
    # Sequential flow: Paper Analysis → Web Research → Advisor
    graph.add_edge("paper_analysis", "web_research")
    graph.add_edge("web_research", "advisor_specialist")
    
    # Conditional branching: Advisor → Specialist Agents (parallel)
    # market_intelligence removed
    graph.add_conditional_edges(
        "advisor_specialist",
        should_run_specialists,
        {
            "sota_tracker": "sota_tracker",
            "comparative_analysis": "comparative_analysis"
        }
    )
    
    # All specialists converge → Direction Advisor
    graph.add_edge("sota_tracker", "direction_advisor")
    graph.add_edge("comparative_analysis", "direction_advisor")
    
    # Final flow: Direction Advisor → Report Generation → END
    graph.add_edge("direction_advisor", "report_generation")
    graph.add_edge("report_generation", END)
    
    # =============================================
    # Compile and return the graph
    # =============================================
    
    compiled_graph = graph.compile()
    
    return compiled_graph


def create_research_graph_with_checkpointer(checkpointer) -> CompiledStateGraph:
    """
    Creates the Research Copilot graph with a checkpointer for state persistence.
    
    Args:
        checkpointer: A LangGraph checkpointer (e.g., MemorySaver, SqliteSaver)
        
    Returns:
        CompiledStateGraph with checkpointing enabled
    """
    graph = StateGraph(OverallState)
    
    # Add all nodes (market_intelligence removed)
    graph.add_node("paper_analysis", paper_analysis_node)
    graph.add_node("web_research", web_research_agent)
    graph.add_node("advisor_specialist", advisor_specialist_agent)
    graph.add_node("sota_tracker", sota_tracker_agent)
    graph.add_node("comparative_analysis", comparative_analysis_node)
    graph.add_node("direction_advisor", direction_advisor_node)
    graph.add_node("report_generation", report_generation_node)
    
    # Define edges
    graph.add_edge(START, "paper_analysis")
    graph.add_edge("paper_analysis", "web_research")
    graph.add_edge("web_research", "advisor_specialist")
    
    # market_intelligence removed from conditional edges
    graph.add_conditional_edges(
        "advisor_specialist",
        should_run_specialists,
        {
            "sota_tracker": "sota_tracker",
            "comparative_analysis": "comparative_analysis"
        }
    )
    
    graph.add_edge("sota_tracker", "direction_advisor")
    graph.add_edge("comparative_analysis", "direction_advisor")
    graph.add_edge("direction_advisor", "report_generation")
    graph.add_edge("report_generation", END)
    
    # Compile with checkpointer
    compiled_graph = graph.compile(checkpointer=checkpointer)
    
    return compiled_graph


# =============================================
# Graph instance for direct import
# =============================================
research_graph = create_research_graph()


# =============================================
# Utility functions
# =============================================

def run_research_pipeline(
    paper_path: str = None,
    paper_url: str = None,
    config: dict = None
) -> dict:
    """
    Run the complete research pipeline on a paper.
    
    Args:
        paper_path: Local path to the paper PDF
        paper_url: URL to the paper PDF
        config: Optional configuration dict for the graph run
        
    Returns:
        Final state with all analysis results
    """
    if not paper_path and not paper_url:
        raise ValueError("Either paper_path or paper_url must be provided")
    
    initial_state: OverallState = {
        "paper_path": paper_path or "",
        "paper_url": paper_url or "",
        "paper_analysis": None,
        "web_research": None,
        "sota_tracker": None,
        "comparative_analysis": None,
        "direction_advisor": None,
        "report_generation": None,
        "active_agents": [],
        "errors": [],
        "next_agents": [],
        "advisor_metadata": {}
    }
    
    # Run the graph
    final_state = research_graph.invoke(initial_state, config=config or {})
    
    return final_state


def stream_research_pipeline(
    paper_path: str = None,
    paper_url: str = None,
    config: dict = None
):
    """
    Stream the research pipeline execution for real-time updates.
    
    Args:
        paper_path: Local path to the paper PDF
        paper_url: URL to the paper PDF
        config: Optional configuration dict
        
    Yields:
        State updates from each node
    """
    if not paper_path and not paper_url:
        raise ValueError("Either paper_path or paper_url must be provided")
    
    initial_state: OverallState = {
        "paper_path": paper_path or "",
        "paper_url": paper_url or "",
        "paper_analysis": None,
        "web_research": None,
        "sota_tracker": None,
        "comparative_analysis": None,
        "direction_advisor": None,
        "report_generation": None,
        "active_agents": [],
        "errors": [],
        "next_agents": [],
        "advisor_metadata": {}
    }
    
    # Stream the graph execution
    for event in research_graph.stream(initial_state, config=config or {}):
        yield event


def get_graph_visualization():
    """
    Get a visual representation of the graph for debugging.
    
    Returns:
        Graph visualization as ASCII or Mermaid diagram
    """
    try:
        return research_graph.get_graph().draw_mermaid()
    except Exception:
        return research_graph.get_graph().print_ascii()


if __name__ == "__main__":
    # Print graph visualization for debugging
    print("Research Copilot Agent Graph")
    print("=" * 50)
    
    try:
        mermaid = get_graph_visualization()
        print(mermaid)
    except Exception as e:
        print(f"Could not generate visualization: {e}")
    
    print("\nGraph nodes:", list(research_graph.get_graph().nodes.keys()))
    print("\nReady to process research papers!")
