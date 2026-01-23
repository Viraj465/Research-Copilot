import sys
import os
import uuid
import logging

# Add backend directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.supabase_store import SupabaseStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_supabase_store():
    print("🚀 Testing SupabaseStore...")
    store = SupabaseStore()
    
    session_id = str(uuid.uuid4())
    print(f"1. Creating session: {session_id}")
    
    # Note: This might fail if user_id is required by RLS and we don't provide a valid one.
    # We'll try with a dummy UUID if needed, but ideally we need a real user ID from profiles.
    # For this test, we'll try without user_id first, as the code attempts to handle it.
    # If it fails, we know we need to handle auth/user_id more strictly.
    
    # Actually, let's try to find an existing user first if possible, or just try insert.
    # Since we can't easily query profiles without a known ID, we'll try the insert.
    
    try:
        # Attempt to create session (might fail if foreign key constraint on user_id is strict and RLS is active)
        # We will try to pass a nil UUID or just let it fail to see the error.
        # But wait, the schema says user_id is NOT NULL.
        # So we MUST provide a user_id.
        # Let's try to fetch a user from profiles if any exist.
        
        from utils.supabase_client import supabase_client
        users = supabase_client.table("profiles").select("id").limit(1).execute()
        user_id = users.data[0]['id'] if users.data else None
        
        if not user_id:
            print("⚠️ No users found in profiles table. Cannot test session creation with FK constraint.")
            print("Skipping session creation test.")
            return

        print(f"   Using user_id: {user_id}")
        
        session = store.create_session(session_id, user_id=user_id)
        if "errors" in session and session["errors"]:
             print(f"❌ Creation failed: {session['errors']}")
             return

        print("✅ Session created.")
        
        print("2. Adding message...")
        msg = store.add_message(session_id, "user", "Hello Supabase!")
        if msg:
            print("✅ Message added.")
        else:
            print("❌ Failed to add message.")
            
        print("3. Fetching session...")
        fetched = store.get_session(session_id)
        if fetched and len(fetched["messages"]) > 0:
            print("✅ Session fetched with messages.")
        else:
            print("❌ Failed to fetch session or messages missing.")
            
        print("4. Deleting session...")
        if store.delete_session(session_id):
            print("✅ Session deleted.")
        else:
            print("❌ Failed to delete session.")
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")

if __name__ == "__main__":
    test_supabase_store()
