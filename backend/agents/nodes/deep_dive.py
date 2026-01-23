import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from prompts.prompts_template import DEEP_DIVE_SYSTEM_PROMPT
from agents.tools.web_search import tavily_search
from utils.llm_factory import LLMFactory

load_dotenv()

# =============================================================================
# Pydantic Models for Deep Dive
# =============================================================================

class DeepDiveRequest(BaseModel):
    """Request for a deep dive chat message."""
    field: str = Field(..., description="The field to chat about (e.g., 'research_findings')")
    message: str = Field(..., description="The user's message or question")

class Source(BaseModel):
    """A source used in the answer."""
    title: str
    url: str
    snippet: str

class DeepDiveResponse(BaseModel):
    """Response from the deep dive agent."""
    answer: str
    sources: List[Source] = []
    thought_process: List[str] = []
# =============================================================================
# Agent Implementation
# =============================================================================

async def deep_dive_agent(
    session_id: str,
    field: str,
    field_content: Any,
    user_message: str,
    chat_history: List[Dict[str, str]],
    llm_config: Optional[Dict[str, Any]] = None
) -> DeepDiveResponse:
    """
    Run the Deep Dive ReAct agent.
    
    Args:
        session_id: The session ID.
        field: The field being discussed.
        field_content: The content of the field from the report.
        user_message: The user's new message.
        chat_history: Previous messages in this thread.
        
    Returns:
        DeepDiveResponse with answer, sources, and thoughts.
    """
    
    print(f"\n🌊 DEEP DIVE AGENT: {field}")
    print(f"   User: {user_message}")
    
    llm = LLMFactory.get_llm(
        agent="deep_dive", 
        temperature=0,
        llm_config=llm_config
    )
    
    # Get Tavily API Key from llm_config
    tavily_api_key = None
    if llm_config:
        tavily_api_key = (llm_config.get("api_keys") or {}).get("tavily")
    
    # Format context
    import json
    context_str = json.dumps(field_content, indent=2, default=str)
    if len(context_str) > 10000:
        context_str = context_str[:10000] + "... (truncated)"
        
    # Format history
    history_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history[-5:]])
    
    react_steps = []
    sources = []
    max_iterations = 5
    
    # Initial Context Construction
    prompt_context = f"""
    **Field**: {field}
    **Field Content**:
    {context_str}
    
    **Chat History**:
    {history_str}
    
    **User Question**: {user_message}
    """
    
    for iteration in range(max_iterations):
        print(f"   🔄 Iteration {iteration + 1}")
        
        # Construct Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", DEEP_DIVE_SYSTEM_PROMPT),
            ("user", """
            Context: {context}
            
            Previous Steps: {previous_steps}
            
            What is your next Thought and Action?
            
            IMPORTANT: You must strictly follow this format:
            THOUGHT: [Your reasoning]
            ACTION: [web_search: query OR FINISH]
            """)
        ])
        
        previous_steps_str = "\n".join([
            f"Step {i+1}:\nTHOUGHT: {s['thought']}\nACTION: {s['action']}\nOBSERVATION: {s['observation']}"
            for i, s in enumerate(react_steps)
        ])
        
        # Invoke LLM
        response = await llm.ainvoke(prompt.format_messages(
            context=prompt_context,
            previous_steps=previous_steps_str or "None"
        ))
        
        response_text = response.content
        
        # Parse Response
        thought = ""
        action = "FINISH"
        action_input = ""
        
        if "THOUGHT:" in response_text:
            thought = response_text.split("THOUGHT:")[1].split("ACTION:")[0].strip()
        if "ACTION:" in response_text:
            parts = response_text.split("ACTION:")[1].strip()
            if ":" in parts:
                action, action_input = parts.split(":", 1)
                action = action.strip()
                action_input = action_input.strip()
            else:
                action = parts
        
        print(f"      💭 {thought[:80]}...")
        print(f"      ⚡ {action} {action_input}")
        
        # Execute Action
        observation = ""
        
        if action.lower() == "web_search":
            try:
                search_results = tavily_search(action_input, max_results=3, api_key=tavily_api_key)
                observation = json.dumps(search_results, indent=2)
                
                # Collect sources
                for res in search_results:
                    sources.append(Source(
                        title=res.get("title", "Unknown"),
                        url=res.get("url", "Unknown"),
                        snippet=res.get("content", "")[:200]
                    ))
                    
            except Exception as e:
                observation = f"Search failed: {str(e)}"
                
        elif action.upper() == "FINISH":
            # The thought is effectively the final answer, or we need one more generation
            # Let's assume the LLM provides the answer in the thought or we ask for it.
            # Actually, for FINISH, we should ask the LLM to generate the final response based on all steps.
            break
            
        else:
            observation = "Unknown action. Valid actions: web_search: <query>, FINISH"
            
        react_steps.append({
            "thought": thought,
            "action": f"{action}: {action_input}",
            "observation": observation
        })
    
    # Final Synthesis
    print("   📝 Synthesizing final answer...")
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful research assistant. Synthesize the final answer based on the research steps."),
        ("user", """
        User Question: {question}
        
        Research Steps:
        {steps}
        
        Provide a clear, direct answer to the user.
        """)
    ])
    
    steps_summary = "\n".join([
        f"Thought: {s['thought']}\nObservation: {s['observation'][:500]}..." 
        for s in react_steps
    ])
    
    final_response = await llm.ainvoke(final_prompt.format_messages(
        question=user_message,
        steps=steps_summary
    ))
    
    return DeepDiveResponse(
        answer=final_response.content,
        sources=sources,
        thought_process=[s['thought'] for s in react_steps]
    )
