"""
Authentication utilities for Supabase JWT verification.
"""

import os
import jwt
import logging
from typing import Optional, Dict
from fastapi import HTTPException, Security, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

# Security scheme
security = HTTPBearer(auto_error=False)


def verify_supabase_token(token: str) -> Optional[Dict]:
    """
    Verify a Supabase JWT token and return the decoded payload.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload if valid, None otherwise
    """
    if not SUPABASE_JWT_SECRET:
        logger.warning("⚠️ SUPABASE_JWT_SECRET not set. Auth verification disabled.")
        return None
    
    try:
        # Decode and verify the JWT token
        decoded = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256", "RS256"],
            audience="authenticated"
        )
        
        logger.debug(f"✅ Token verified for user: {decoded.get('sub')}")
        return decoded
        
    except jwt.ExpiredSignatureError:
        logger.warning("❌ Token has expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"❌ Invalid token: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Token verification error: {e}")
        return None


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> Optional[Dict]:
    """
    FastAPI dependency to get current authenticated user.
    
    This is optional - if no token is provided or auth is not configured,
    it returns None and allows the request to proceed (for backward compatibility).
    
    Returns:
        User data dict if authenticated, None otherwise
    """
    if not credentials:
        logger.debug("No auth credentials provided")
        return None
    
    token = credentials.credentials
    user_data = verify_supabase_token(token)
    
    return user_data


def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> Dict:
    """
    FastAPI dependency that REQUIRES authentication.
    
    Use this for endpoints that must be protected.
    Raises HTTPException if auth fails.
    
    Returns:
        User data dict
    """
    if not SUPABASE_JWT_SECRET:
        raise HTTPException(
            status_code=503,
            detail="Authentication is not configured on this server"
        )
    
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please sign in."
        )
    
    token = credentials.credentials
    user_data = verify_supabase_token(token)
    
    if not user_data:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired authentication token"
        )
    
    return user_data


def get_user_id(user_data: Optional[Dict]) -> Optional[str]:
    """
    Extract user ID from decoded token.
    
    Args:
        user_data: Decoded JWT payload
        
    Returns:
        User ID string or None
    """
    if not user_data:
        return None
    return user_data.get("sub")


def get_user_email(user_data: Optional[Dict]) -> Optional[str]:
    """
    Extract user email from decoded token.
    
    Args:
        user_data: Decoded JWT payload
        
    Returns:
        User email string or None
    """
    if not user_data:
        return None
    return user_data.get("email")
