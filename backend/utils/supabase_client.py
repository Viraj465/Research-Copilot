import os
from supabase import create_client, Client
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    logger.warning("⚠️ Supabase credentials not found. Some features may not work.")

def get_supabase_client() -> Client:
    """
    Get or create a Supabase client instance.
    Using service key for backend operations to bypass RLS when necessary,
    or to perform admin tasks.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
    
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Singleton instance
supabase_client = get_supabase_client() if SUPABASE_URL and SUPABASE_SERVICE_KEY else None
