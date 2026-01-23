"""
Research Copilot API
A chat-like interface for analyzing research papers using multi-agent system.
"""

import os
import uuid
import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, HttpUrl
from dotenv import load_dotenv

# LangGraph imports
from langgraph.checkpoint.memory import MemorySaver

# Local imports
from agents.graph import create_research_graph_with_checkpointer, stream_research_pipeline
from agents.state import OverallState
from agents.nodes.deep_dive import deep_dive_agent, DeepDiveRequest, DeepDiveResponse
from utils.auth import get_current_user, require_auth, get_user_id, get_user_email
from utils.supabase_store import SupabaseStore

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("🚀 Research Copilot API Starting...")


# =============================================================================
# Application State & Storage
# =============================================================================

# Global session store
session_store = SupabaseStore()


def check_session_access(session: Dict, user_data: Optional[Dict]) -> bool:
    """
    Check if a user has access to a session.
    
    Rules:
    - If session has no user_id (anonymous), anyone can access
    - If session has user_id, only that user can access
    - If auth is not configured, allow all access
    
    Args:
        session: Session dict
        user_data: Current user data from JWT
        
    Returns:
        True if access is allowed, False otherwise
    """
    session_user_id = session.get("user_id")
    
    # Anonymous session - anyone can access
    if not session_user_id:
        return True
    
    # No user logged in - can only access anonymous sessions
    if not user_data:
        return False
    
    # Check if user owns the session
    current_user_id = get_user_id(user_data)
    return session_user_id == current_user_id


# =============================================================================
# Pydantic Models
# =============================================================================

class AgentConfig(BaseModel):
    """Configuration for a specific agent."""
    provider: str = Field(..., description="LLM provider (groq, google)")
    model: str = Field(..., description="Model name")

class LLMConfig(BaseModel):
    """Configuration for the LLM provider."""
    provider: str = Field(..., description="Default LLM provider (groq, google)")
    model: str = Field(..., description="Default Model name")
    api_keys: Optional[Dict[str, str]] = Field(None, description="Map of provider to API key")
    agents: Optional[Dict[str, AgentConfig]] = Field(None, description="Agent-specific overrides")

class StartSessionRequest(BaseModel):
    """Request to start a new research session."""
    paper_url: Optional[str] = Field(None, description="URL to the research paper PDF")
    llm_config: Optional[LLMConfig] = Field(None, description="LLM configuration")


class StartSessionResponse(BaseModel):
    """Response after starting a session."""
    session_id: str
    status: str
    message: str


class ChatMessage(BaseModel):
    """A chat message."""
    id: str
    role: str  # "user", "assistant", "system"
    content: str
    agent: Optional[str] = None
    timestamp: str


class SessionStatus(BaseModel):
    """Current session status."""
    session_id: str
    status: str  # "created", "processing", "completed", "error"
    current_agent: Optional[str]
    messages: List[ChatMessage]
    created_at: str
    updated_at: str


class AgentProgress(BaseModel):
    """Progress update from an agent."""
    agent: str
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


class ResearchReport(BaseModel):
    """Final research report."""
    session_id: str
    paper_title: Optional[str]
    executive_summary: Optional[str]
    contributions: List[str]
    methodology: Optional[Dict[str, Any]]
    results: Optional[Dict[str, Any]]
    sota_analysis: Optional[Dict[str, Any]]
    comparative_analysis: Optional[Dict[str, Any]]
    future_directions: List[Dict[str, Any]]
    recommendations: Optional[Dict[str, Any]]


# =============================================================================
# Lifespan & App Initialization
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("🚀 Research Copilot API Starting...")
    logger.info("📊 Multi-agent research pipeline ready")
    yield
    logger.info("👋 Research Copilot API Shutting down...")


app = FastAPI(
    title="Research Copilot API",
    description="A chat-like interface for analyzing research papers using a multi-agent AI system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Helper Functions
# =============================================================================

async def run_research_pipeline_async(session_id: str, paper_path: str = "", paper_url: str = "", llm_config: Dict = None):
    """Run the research pipeline asynchronously."""
    try:
        session_store.update_session(session_id, {"status": "processing"})
        
        # Create initial state (market_intelligence removed)
        initial_state: OverallState = {
            "paper_path": paper_path,
            "paper_url": paper_url,
            "paper_analysis": None,
            "web_research": None,
            "sota_tracker": None,
            "comparative_analysis": None,
            "direction_advisor": None,
            "report_generation": None,
            "active_agents": [],
            "errors": [],
            "next_agents": [],
            "advisor_metadata": {},
            "llm_config": llm_config,
            "api_keys": llm_config.get("api_keys") if llm_config else None
        }
        
        # Create graph with checkpointer
        graph = create_research_graph_with_checkpointer(session_store.checkpointer)
        
        # Run the graph
        config = {"configurable": {"thread_id": session_id}}
        final_state = await asyncio.to_thread(graph.invoke, initial_state, config)
        
        # Update session with final state
        session_store.update_session(session_id, {
            "status": "completed",
            "state": final_state
        })
        
        # Add completion message
        session_store.add_message(
            session_id,
            "assistant",
            "✅ Research analysis complete! Your comprehensive report is ready.",
            agent="report_generation"
        )
        
        return final_state
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        session_store.update_session(session_id, {
            "status": "error",
            "errors": [str(e)]
        })
        session_store.add_message(
            session_id,
            "system",
            f"❌ Error during analysis: {str(e)}",
            agent="system"
        )
        raise


async def stream_pipeline_events(session_id: str, paper_path: str = "", paper_url: str = "", llm_config: Dict = None):
    """Generator for streaming pipeline events."""
    
    try:
        logger.info(f"⚙️ stream_pipeline_events started for session: {session_id}")
        logger.info(f"   Paper path: {paper_path}")
        logger.info(f"   Paper URL: {paper_url}")
        
        session_store.update_session(session_id, {"status": "processing"})
        logger.info(f"✅ Session status updated to: processing")
        
        # Initial message
        init_msg = {'type': 'start', 'message': 'Starting research analysis...', 'agent': 'system'}
        logger.info(f"📤 Yielding initial message")
        yield f"data: {json.dumps(init_msg)}\n\n"
        
        # Create initial state (market_intelligence removed)
        initial_state: OverallState = {
            "paper_path": paper_path,
            "paper_url": paper_url,
            "paper_analysis": None,
            "web_research": None,
            "sota_tracker": None,
            "comparative_analysis": None,
            "direction_advisor": None,
            "report_generation": None,
            "active_agents": [],
            "errors": [],
            "next_agents": [],
            "advisor_metadata": {},
            "llm_config": llm_config,
            "api_keys": llm_config.get("api_keys") if llm_config else None
        }
        
        # Create graph
        logger.info(f"📊 Creating research graph...")
        graph = create_research_graph_with_checkpointer(session_store.checkpointer)
        config = {"configurable": {"thread_id": session_id}}
        logger.info(f"✅ Graph created")
        
        # Agent messages for chat-like experience (market_intelligence removed)
        agent_messages = {
            "paper_analysis": "📰 Analyzing paper structure and extracting key insights...",
            "web_research": "🔍 Searching the web for related research and trends...",
            "advisor_specialist": "🤔 Evaluating research quality and planning specialist analysis...",
            "sota_tracker": "🏆 Tracking state-of-the-art developments...",
            "comparative_analysis": "⚖️ Comparing with related research papers (ReAct)...",
            "direction_advisor": "📈 Synthesizing findings and identifying future directions...",
            "report_generation": "📝 Generating your comprehensive research report..."
        }
        
        # Stream graph execution
        logger.info(f"🚀 Starting graph stream...")
        for event in graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_output in event.items():
                logger.info(f"📍 Agent event: {node_name}")
                
                # Update current agent
                session_store.update_session(session_id, {"current_agent": node_name})
                
                # Send agent start message
                message = agent_messages.get(node_name, f"Processing {node_name}...")
                session_store.add_message(session_id, "assistant", message, agent=node_name)
                
                agent_start_msg = {'type': 'agent_start', 'agent': node_name, 'message': message}
                logger.info(f"  📤 Yielding agent_start for {node_name}")
                yield f"data: {json.dumps(agent_start_msg)}\n\n"
                
                # Extract key info from node output for chat
                chat_update = extract_chat_update(node_name, node_output)
                if chat_update:
                    session_store.add_message(session_id, "assistant", chat_update, agent=node_name)
                    agent_update_msg = {'type': 'agent_update', 'agent': node_name, 'message': chat_update}
                    logger.info(f"  📤 Yielding agent_update for {node_name}")
                    yield f"data: {json.dumps(agent_update_msg)}\n\n"
                
                # Send agent complete
                agent_complete_msg = {'type': 'agent_complete', 'agent': node_name}
                logger.info(f"  📤 Yielding agent_complete for {node_name}")
                yield f"data: {json.dumps(agent_complete_msg)}\n\n"
                
                # Small delay for UX
                await asyncio.sleep(0.1)
        
        logger.info(f"✅ Graph stream completed")
        
        # Get final state
        final_state = graph.get_state(config).values
        session_store.update_session(session_id, {
            "status": "completed",
            "state": final_state,
            "current_agent": None
        })
        
        # Final message
        completion_message = "✅ Research analysis complete! Your comprehensive report is ready."
        session_store.add_message(session_id, "assistant", completion_message, agent="system")
        yield f"data: {json.dumps({'type': 'complete', 'message': completion_message})}\n\n"
        
    except Exception as e:
        logger.error(f"Stream error: {e}")
        error_message = f"❌ Error: {str(e)}"
        session_store.update_session(session_id, {"status": "error"})
        session_store.add_message(session_id, "system", error_message, agent="system")
        yield f"data: {json.dumps({'type': 'error', 'message': error_message})}\n\n"


def extract_chat_update(node_name: str, node_output: Dict) -> Optional[str]:
    """Extract a chat-friendly update from node output."""
    
    if node_name == "paper_analysis" and node_output.get("paper_analysis"):
        analysis = node_output["paper_analysis"]
        title = analysis.get("paper_title", "Unknown")
        contributions = analysis.get("contributions", [])
        domains = analysis.get("domain_tags", [])
        return f"📄 **Paper:** {title}\n\n**Key Contributions:** {len(contributions)} identified\n**Domains:** {', '.join(domains[:3])}"
    
    elif node_name == "web_research" and node_output.get("web_research"):
        research = node_output["web_research"]
        results_count = len(research.get("retrieval_results", []))
        key_players = research.get("key_players", [])[:3]
        return f"🌐 Found **{results_count}** relevant sources\n**Key Researchers:** {', '.join(key_players) if key_players else 'Analyzing...'}"
    
    elif node_name == "sota_tracker" and node_output.get("sota_tracker"):
        sota = node_output["sota_tracker"]
        status = sota.get("sota_tracker_status", "Analyzed")
        return f"🎯 **SOTA Status:** {status}"
    
    elif node_name == "comparative_analysis" and node_output.get("comparative_analysis"):
        comp = node_output["comparative_analysis"]
        title = comp.get("comparative_analysis_title", "Comparison Complete")
        sources = comp.get("sources_used", 0)
        return f"📊 **Comparative Analysis:** {title}\n**Sources compared:** {sources}"
    
    elif node_name == "direction_advisor" and node_output.get("direction_advisor"):
        direction = node_output["direction_advisor"]
        gaps = len(direction.get("gaps_analysis_results", []))
        directions = len(direction.get("future_directions_results", []))
        return f"🧭 Identified **{gaps}** research gaps and **{directions}** future directions"
    
    elif node_name == "report_generation" and node_output.get("report_generation"):
        report = node_output["report_generation"]
        formats = report.get("export_formats", [])
        return f"📋 Report generated! Available formats: {', '.join(formats)}"
    
    return None


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "Research Copilot API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "start_session_url": "POST /api/sessions/start-url",
            "start_session_upload": "POST /api/sessions/start-upload",
            "stream_analysis": "GET /api/sessions/{session_id}/stream",
            "get_session": "GET /api/sessions/{session_id}",
            "get_report": "GET /api/sessions/{session_id}/report",
            "get_messages": "GET /api/sessions/{session_id}/messages"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# -----------------------------------------------------------------------------
# Session Management
# -----------------------------------------------------------------------------

@app.post("/api/sessions/start-url", response_model=StartSessionResponse)
async def start_session_with_url(
    request: StartSessionRequest, 
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Start a new research session with a paper URL.
    
    This initiates the multi-agent analysis pipeline.
    Optional authentication - if provided, session will be associated with user.
    """
    if not request.paper_url:
        raise HTTPException(status_code=400, detail="paper_url is required")
    
    user_id = get_user_id(current_user)
    if user_id:
        logger.info(f"Creating session for authenticated user: {user_id}")
    
    session_id = str(uuid.uuid4())
    session = session_store.create_session(
        session_id=session_id,
        paper_url=request.paper_url,
        llm_config=request.llm_config.dict() if request.llm_config else None,
        user_id=user_id
    )
    
    # Add initial message
    session_store.add_message(
        session_id,
        "user",
        f"Analyze this paper: {request.paper_url}"
    )
    session_store.add_message(
        session_id,
        "assistant",
        "🚀 Starting research analysis pipeline...",
        agent="system"
    )
    
    return StartSessionResponse(
        session_id=session_id,
        status="created",
        message="Session created. Use /api/sessions/{session_id}/stream to start streaming analysis."
    )


@app.post("/api/sessions/start-upload", response_model=StartSessionResponse)
async def start_session_with_upload(
    file: UploadFile = File(...),
    llm_provider: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None),
    llm_api_key: Optional[str] = Form(None),
    llm_api_keys: Optional[str] = Form(None),
    llm_agents: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Start a new research session by uploading a paper PDF.
    Optional authentication - if provided, session will be associated with user.
    """
    logger.info(f"📤 File upload started: {file.filename}")
    logger.info(f"   File size: {file.size} bytes")
    
    if not file.filename.lower().endswith('.pdf'):
        logger.error(f"❌ Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file
    session_id = str(uuid.uuid4())
    upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, f"{session_id}_{file.filename}")
    logger.info(f"📁 Saving file to: {file_path}")
    
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    logger.info(f"✅ File saved successfully ({len(content)} bytes)")
    
    llm_config = None
    if llm_provider and llm_model:
        # Parse API keys if provided
        api_keys = {}
        if llm_api_keys:
            try:
                api_keys = json.loads(llm_api_keys)
                logger.info(f"   LLM Config: provider={llm_provider}, model={llm_model}")
            except json.JSONDecodeError:
                logger.warning("   Failed to parse llm_api_keys JSON")
        
        # Fallback to single key if provided
        if llm_api_key and llm_provider not in api_keys:
            api_keys[llm_provider] = llm_api_key
        
        # Parse agent overrides if provided
        agents = None
        if llm_agents:
            try:
                agents = json.loads(llm_agents)
            except json.JSONDecodeError:
                logger.warning("   Failed to parse llm_agents JSON")
            
        llm_config = {
            "provider": llm_provider,
            "model": llm_model,
            "api_keys": api_keys,
            "agents": agents
        }

    user_id = get_user_id(current_user)
    if user_id:
        logger.info(f"Creating session for authenticated user: {user_id}")
    
    session = session_store.create_session(
        session_id=session_id,
        paper_path=file_path,
        llm_config=llm_config,
        user_id=user_id
    )
    logger.info(f"✨ Session created: {session_id}")
    
    # Add initial messages
    session_store.add_message(
        session_id,
        "user",
        f"Analyze this paper: {file.filename}"
    )
    session_store.add_message(
        session_id,
        "assistant",
        "🚀 Starting research analysis pipeline...",
        agent="system"
    )
    
    logger.info(f"📤 Returning session response: {session_id}")
    return StartSessionResponse(
        session_id=session_id,
        status="created",
        message="Session created. Use /api/sessions/{session_id}/stream to start streaming analysis."
    )


@app.get("/api/sessions/{session_id}/stream")
async def stream_session_analysis(
    session_id: str,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Stream the research analysis as Server-Sent Events.
    
    This provides a chat-like experience with real-time updates from each agent.
    """
    logger.info(f"🔄 Stream endpoint called for session: {session_id}")
    
    session = session_store.get_session(session_id)
    if not session:
        logger.error(f"❌ Session not found: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check access permission
    if not check_session_access(session, current_user):
        raise HTTPException(
            status_code=403, 
            detail="You don't have permission to access this session"
        )
    
    logger.info(f"   Session status: {session['status']}")
    
    if session["status"] == "completed":
        logger.warning(f"❌ Session already completed: {session_id}")
        raise HTTPException(status_code=400, detail="Session already completed")
    
    if session["status"] == "processing":
        logger.warning(f"❌ Session already processing: {session_id}")
        raise HTTPException(status_code=400, detail="Session already processing")
    
    logger.info(f"📡 Starting stream pipeline for session: {session_id}")
    return StreamingResponse(
        stream_pipeline_events(
            session_id,
            paper_path=session.get("paper_path", ""),
            paper_url=session.get("paper_url", ""),
            llm_config=session.get("llm_config")
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/sessions/{session_id}", response_model=SessionStatus)
async def get_session_status(
    session_id: str,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Get the current status of a research session."""
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check access permission
    if not check_session_access(session, current_user):
        raise HTTPException(
            status_code=403, 
            detail="You don't have permission to access this session"
        )
    
    return SessionStatus(
        session_id=session["session_id"],
        status=session["status"],
        current_agent=session.get("current_agent"),
        messages=[ChatMessage(**msg) for msg in session.get("messages", [])],
        created_at=session["created_at"],
        updated_at=session["updated_at"]
    )


@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Get chat messages for a session."""
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check access permission
    if not check_session_access(session, current_user):
        raise HTTPException(
            status_code=403, 
            detail="You don't have permission to access this session"
        )
    
    messages = session.get("messages", [])
    total = len(messages)
    paginated = messages[offset:offset + limit]
    
    return {
        "session_id": session_id,
        "messages": [ChatMessage(**msg) for msg in paginated],
        "total": total,
        "limit": limit,
        "offset": offset
    }


@app.get("/api/sessions/{session_id}/report")
async def get_session_report(
    session_id: str,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Get the comprehensive research report for a completed session.
    """
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check access permission
    if not check_session_access(session, current_user):
        raise HTTPException(
            status_code=403, 
            detail="You don't have permission to access this session"
        )
    
    if session["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Report not ready. Current status: {session['status']}"
        )
    
    state = session.get("state", {})
    paper_analysis = state.get("paper_analysis", {})
    report_gen = state.get("report_generation", {})
    
    return {
        "session_id": session_id,
        "paper_title": paper_analysis.get("paper_title"),
        "authors": paper_analysis.get("authors", []),
        "executive_summary": report_gen.get("executive_summary"),
        "research_findings": report_gen.get("research_findings"),
        "technical_landscape": report_gen.get("technical_landscape"),
        "hierarchical_summaries": paper_analysis.get("hierarchical_summaries", []),
        "contributions": paper_analysis.get("contributions", []),
        "methodology": paper_analysis.get("methodology"),
        "results": paper_analysis.get("results"),
        "limitations": paper_analysis.get("limitations", []),
        "research_gaps": paper_analysis.get("research_gaps", []),
        "sota_analysis": state.get("sota_tracker"),
        "comparative_analysis": state.get("comparative_analysis"),
        "direction_advisor": state.get("direction_advisor"),
        "recommendations": report_gen.get("recommendations"),
        "future_directions": report_gen.get("future_directions"),
        "export_formats": report_gen.get("export_formats", []),
        "domain_tags": paper_analysis.get("domain_tags", []),
        "related_papers": paper_analysis.get("related_papers", []),
        "citations": paper_analysis.get("citations", [])
    }


@app.get("/api/sessions/{session_id}/export")
async def export_session_report(
    session_id: str,
    format: str = Query("markdown", description="Export format: markdown, json, text"),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Export the research report in various formats.
    
    Supported formats:
    - markdown: Full report in Markdown format
    - json: Structured JSON data
    - text: Plain text format
    """
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check access permission
    if not check_session_access(session, current_user):
        raise HTTPException(
            status_code=403, 
            detail="You don't have permission to access this session"
        )
    
    if session["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Report not ready. Current status: {session['status']}"
        )
    
    state = session.get("state", {})
    paper_analysis = state.get("paper_analysis", {})
    report_gen = state.get("report_generation", {})
    
    # Get report data
    report_data = {
        "paper_title": paper_analysis.get("paper_title", "Research Report"),
        "authors": paper_analysis.get("authors", []),
        "executive_summary": report_gen.get("executive_summary", ""),
        "research_findings": report_gen.get("research_findings", ""),
        "technical_landscape": report_gen.get("technical_landscape", ""),
        "contributions": paper_analysis.get("contributions", []),
        "methodology": paper_analysis.get("methodology", {}),
        "results": paper_analysis.get("results", {}),
        "sota_analysis": state.get("sota_tracker", {}),
        "comparative_analysis": state.get("comparative_analysis", {}),
        "direction_advisor": state.get("direction_advisor", {}),
        "future_directions": report_gen.get("future_directions", []),
        "recommendations": report_gen.get("recommendations", {}),
    }
    
    format_lower = format.lower()
    
    if format_lower == "json":
        return report_data
    
    elif format_lower == "markdown":
        # Generate Markdown
        md_content = f"""# {report_data['paper_title']}

**Authors:** {', '.join(str(a) for a in report_data['authors'])}

---

## Executive Summary

{report_data['executive_summary']}

---

## Research Findings

{report_data['research_findings']}

---

## Technical Landscape

{report_data['technical_landscape']}

---

## Key Contributions

{chr(10).join(f"- {c}" for c in report_data['contributions'])}

---

## SOTA Analysis

{json.dumps(report_data['sota_analysis'], indent=2)}

---

## Comparative Analysis

{json.dumps(report_data['comparative_analysis'], indent=2)}

---

## Future Directions

{chr(10).join(f"- {d}" for d in report_data['future_directions'])}

---

## Recommendations

{json.dumps(report_data['recommendations'], indent=2)}

---

*Generated by Research Copilot on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(
            content=md_content,
            media_type="text/markdown",
            headers={
                "Content-Disposition": f"attachment; filename={report_data['paper_title'][:50].replace(' ', '_')}.md"
            }
        )
    
    elif format_lower == "text":
        # Generate plain text
        text_content = f"""{report_data['paper_title']}
{'=' * len(report_data['paper_title'])}

Authors: {', '.join(str(a) for a in report_data['authors'])}

EXECUTIVE SUMMARY
-----------------
{report_data['executive_summary']}

RESEARCH FINDINGS
-----------------
{report_data['research_findings']}

TECHNICAL LANDSCAPE
-------------------
{report_data['technical_landscape']}

KEY CONTRIBUTIONS
-----------------
{chr(10).join(f"- {c}" for c in report_data['contributions'])}

FUTURE DIRECTIONS
-----------------
{chr(10).join(f"- {d}" for d in report_data['future_directions'])}

---
Generated by Research Copilot on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(
            content=text_content,
            media_type="text/plain",
            headers={
                "Content-Disposition": f"attachment; filename={report_data['paper_title'][:50].replace(' ', '_')}.txt"
            }
        )
    
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {format}. Use 'markdown', 'json', or 'text'"
        )


@app.delete("/api/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Delete a research session."""
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check access permission - only owner can delete
    if not check_session_access(session, current_user):
        raise HTTPException(
            status_code=403, 
            detail="You don't have permission to delete this session"
        )
    
    if session_store.delete_session(session_id):
        return {"message": "Session deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


# -----------------------------------------------------------------------------
# Deep Dive Chat
# -----------------------------------------------------------------------------

@app.post("/api/sessions/{session_id}/chat", response_model=DeepDiveResponse)
async def deep_dive_chat(
    session_id: str,
    request: DeepDiveRequest,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Deep Dive Chat about a specific field.
    
    Allows the user to ask follow-up questions about specific parts of the report.
    The agent uses ReAct + Web Search to answer.
    """
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check access permission
    if not check_session_access(session, current_user):
        raise HTTPException(
            status_code=403, 
            detail="You don't have permission to access this session"
        )
    
    if session["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail="Session must be completed before starting deep dive chat"
        )
        
    # Get field content from state
    state = session.get("state", {})
    
    # Map friendly field names to state locations
    # This mapping might need to be adjusted based on exact state structure
    field_content = None
    
    # Try to find the field in various state components
    if request.field in state:
        field_content = state[request.field]
    elif request.field in state.get("report_generation", {}):
        field_content = state["report_generation"][request.field]
    elif request.field in state.get("paper_analysis", {}):
        field_content = state["paper_analysis"][request.field]
    elif request.field == "sota_analysis":
        field_content = state.get("sota_tracker")
    elif request.field == "comparative_analysis":
        field_content = state.get("comparative_analysis")
    elif request.field == "direction_advisor":
        field_content = state.get("direction_advisor")
        
    if not field_content:
        # Fallback: try to find it anywhere
        logger.warning(f"Field {request.field} not found in standard locations")
        field_content = f"Field {request.field} not found in structured output. Proceeding with general context."
        
    # Add User Message to History
    session_store.add_deep_dive_message(
        session_id, 
        request.field, 
        "user", 
        request.message
    )
    
    # Get History
    history = session_store.get_deep_dive_history(session_id, request.field)
    
    # Run Agent
    try:
        response = await deep_dive_agent(
            session_id=session_id,
            field=request.field,
            field_content=field_content,
            user_message=request.message,
            chat_history=history,
            llm_config=session.get("llm_config")
        )
        
        # Add Assistant Response to History
        session_store.add_deep_dive_message(
            session_id, 
            request.field, 
            "assistant", 
            response.answer
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Deep dive error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/url")
async def analyze_paper_url(
    request: StartSessionRequest, 
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Quick endpoint to analyze a paper URL.
    Returns session_id and starts processing in background.
    Poll /api/sessions/{session_id} for status.
    """
    if not request.paper_url:
        raise HTTPException(status_code=400, detail="paper_url is required")
    
    user_id = get_user_id(current_user)
    session_id = str(uuid.uuid4())
    session_store.create_session(
        session_id=session_id, 
        paper_url=request.paper_url,
        llm_config=request.llm_config.dict() if request.llm_config else None,
        user_id=user_id
    )
    
    # Start pipeline in background
    background_tasks.add_task(
        run_research_pipeline_async,
        session_id,
        paper_url=request.paper_url,
        llm_config=request.llm_config.dict() if request.llm_config else None
    )
    
    return {
        "session_id": session_id,
        "status": "processing",
        "message": "Analysis started. Poll /api/sessions/{session_id} for updates."
    }


# -----------------------------------------------------------------------------
# List Sessions
# -----------------------------------------------------------------------------

@app.get("/api/sessions")
async def list_sessions(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    List research sessions accessible to the user.
    
    If authenticated, shows user's sessions.
    If not authenticated, shows anonymous sessions only.
    """
    sessions = list(session_store.sessions.values())
    
    # Filter by access permission
    user_id = get_user_id(current_user)
    if user_id:
        # Show only user's sessions
        sessions = [s for s in sessions if s.get("user_id") == user_id]
    else:
        # Show only anonymous sessions
        sessions = [s for s in sessions if not s.get("user_id")]
    
    if status:
        sessions = [s for s in sessions if s["status"] == status]
    
    # Sort by created_at descending
    sessions.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {
        "sessions": [
            {
                "session_id": s["session_id"],
                "status": s["status"],
                "paper_url": s.get("paper_url"),
                "paper_path": s.get("paper_path"),
                "current_agent": s.get("current_agent"),
                "created_at": s["created_at"],
                "updated_at": s["updated_at"],
                "message_count": len(s.get("messages", []))
            }
            for s in sessions[:limit]
        ],
        "total": len(sessions)
    }


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
