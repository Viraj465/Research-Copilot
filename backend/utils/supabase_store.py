import logging
import uuid
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from langgraph.checkpoint.memory import MemorySaver
from utils.supabase_client import supabase_client

logger = logging.getLogger(__name__)

class SupabaseStore:
    """Supabase-backed session store for research sessions."""
    
    def __init__(self):
        self.checkpointer = MemorySaver()
        if not supabase_client:
            logger.error("❌ Supabase client not initialized. Store will fail.")

    def create_session(self, session_id: str, paper_path: str = "", paper_url: str = "", llm_config: Dict = None, user_id: str = None) -> Dict:
        """Create a new research session in Supabase."""
        session_data = {
            "id": session_id,
            "user_id": user_id,
            "paper_path": paper_path,
            "paper_url": paper_url,
            "status": "created",
            "llm_config": llm_config,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        try:
            # If user_id is None, we might have issues with RLS if the table requires it.
            # Assuming the table allows null user_id or we handle it. 
            # Based on schema: user_id UUID REFERENCES profiles(id) ON DELETE CASCADE NOT NULL
            # So user_id IS REQUIRED. 
            # If we are in anonymous mode, we might need a dummy user or handle it differently.
            # For now, let's proceed. If user_id is None, this insert might fail if RLS enforces it.
            
            # However, the schema says NOT NULL. So we MUST have a user_id.
            # If the user is not logged in (anonymous), we can't save to this table structure as is.
            # But wait, the current main.py allows anonymous sessions.
            # We might need to adjust the schema or require auth.
            # For this implementation, we will try to insert.
            
            if not user_id:
                 logger.warning("⚠️ Attempting to create session without user_id. This may fail due to schema constraints.")

            response = supabase_client.table("sessions").insert(session_data).execute()
            logger.info(f"✅ Session created in Supabase: {session_id}")
            
            # Return the session data structure expected by the app
            return {
                **session_data,
                "messages": [],
                "deep_dive_threads": {},
                "state": None,
                "errors": []
            }
        except Exception as e:
            logger.error(f"❌ Failed to create session in Supabase: {e}")
            # Fallback to in-memory structure for now to prevent crash, but log error
            return {
                **session_data,
                "messages": [],
                "deep_dive_threads": {},
                "state": None,
                "errors": ["Failed to save to database"]
            }

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session by ID from Supabase, including messages."""
        try:
            # Fetch session
            response = supabase_client.table("sessions").select("*").eq("id", session_id).execute()
            if not response.data:
                return None
            
            session = response.data[0]
            
            # Fetch messages
            msg_response = supabase_client.table("messages").select("*").eq("session_id", session_id).order("timestamp").execute()
            messages = msg_response.data if msg_response.data else []
            
            # Fetch deep dive messages
            dd_response = supabase_client.table("deep_dive_messages").select("*").eq("session_id", session_id).order("timestamp").execute()
            dd_messages = dd_response.data if dd_response.data else []
            
            # Organize deep dive threads
            threads = {}
            for msg in dd_messages:
                field = msg.get("field")
                if field not in threads:
                    threads[field] = []
                threads[field].append(msg)

            # Construct full session object
            return {
                "session_id": session["id"],
                "status": session["status"],
                "created_at": session["created_at"],
                "updated_at": session["updated_at"],
                "paper_path": session.get("paper_path"),
                "paper_url": session.get("paper_url"),
                "llm_config": session.get("llm_config"),
                "current_agent": None, # Not stored in DB currently, or could be added
                "messages": messages,
                "state": session.get("final_state"), # We store final state
                "errors": session.get("errors", []),
                "deep_dive_threads": threads,
                "user_id": session.get("user_id")
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get session from Supabase: {e}")
            return None

    def update_session(self, session_id: str, updates: Dict) -> Optional[Dict]:
        """Update session with new data."""
        try:
            # Map updates to DB columns
            db_updates = {}
            if "status" in updates:
                db_updates["status"] = updates["status"]
            if "state" in updates:
                db_updates["final_state"] = updates["state"] # Map state to final_state
            if "errors" in updates:
                db_updates["errors"] = updates["errors"]
            
            db_updates["updated_at"] = datetime.now().isoformat()
            
            if db_updates:
                supabase_client.table("sessions").update(db_updates).eq("id", session_id).execute()
            
            # Return updated session (fetch fresh)
            return self.get_session(session_id)
            
        except Exception as e:
            logger.error(f"❌ Failed to update session in Supabase: {e}")
            return None

    def add_message(self, session_id: str, role: str, content: str, agent: str = None):
        """Add a message to session history."""
        try:
            message_data = {
                "session_id": session_id,
                "role": role,
                "content": content,
                "agent": agent,
                "timestamp": datetime.now().isoformat()
            }
            
            response = supabase_client.table("messages").insert(message_data).execute()
            if response.data:
                return response.data[0]
            return message_data
            
        except Exception as e:
            logger.error(f"❌ Failed to add message to Supabase: {e}")
            return None

    def add_deep_dive_message(self, session_id: str, field: str, role: str, content: str):
        """Add a message to a deep dive thread."""
        try:
            message_data = {
                "session_id": session_id,
                "field": field,
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            
            response = supabase_client.table("deep_dive_messages").insert(message_data).execute()
            if response.data:
                return response.data[0]
            return message_data
            
        except Exception as e:
            logger.error(f"❌ Failed to add deep dive message to Supabase: {e}")
            return None

    def get_deep_dive_history(self, session_id: str, field: str) -> List[Dict]:
        """Get deep dive chat history for a field."""
        try:
            response = supabase_client.table("deep_dive_messages").select("*").eq("session_id", session_id).eq("field", field).order("timestamp").execute()
            return response.data if response.data else []
        except Exception as e:
            logger.error(f"❌ Failed to get deep dive history: {e}")
            return []

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        try:
            supabase_client.table("sessions").delete().eq("id", session_id).execute()
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete session: {e}")
            return False
