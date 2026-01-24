"""
Authentication utilities for Supabase JWT verification.
"""

import os
import jwt
import logging
import requests
from typing import Optional, Dict
from fastapi import HTTPException, Security, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from jwt import PyJWKClient

load_dotenv()

logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

# Security scheme
security = HTTPBearer(auto_error=False)

# JWKS client for ES256 verification (cached)
_jwks_client: Optional[PyJWKClient] = None


def get_jwks_client() -> Optional[PyJWKClient]:
    """
    Get or create a cached JWKS client for the Supabase project.
    """
    global _jwks_client
    
    if _jwks_client is not None:
        return _jwks_client
    
    if not SUPABASE_URL:
        logger.warning("⚠️ SUPABASE_URL not set. Cannot create JWKS client.")
        return None
    
    try:
        # Supabase JWKS endpoint
        jwks_url = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"
        _jwks_client = PyJWKClient(jwks_url, cache_keys=True)
        logger.info(f"✅ JWKS client initialized for {jwks_url}")
        return _jwks_client
    except Exception as e:
        logger.error(f"❌ Failed to create JWKS client: {e}")
        return None


def verify_supabase_token(token: str) -> Optional[Dict]:
    """
    Verify a Supabase JWT token and return the decoded payload.
    
    Uses JWKS for ES256/RS256 verification, falls back to secret for HS256.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload if valid, None otherwise
    """
    if not SUPABASE_URL and not SUPABASE_JWT_SECRET:
        logger.warning("⚠️ No Supabase auth configuration. Auth verification disabled.")
        return None
    
    try:
        # First, peek at the token header to determine the algorithm
        unverified_header = jwt.get_unverified_header(token)
        algorithm = unverified_header.get("alg", "HS256")
        
        if algorithm in ["ES256", "RS256"]:
            # Use JWKS for asymmetric algorithms
            jwks_client = get_jwks_client()
            if not jwks_client:
                logger.error("❌ JWKS client not available for asymmetric verification")
                return None
            
            # Get the signing key from JWKS
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            
            decoded = jwt.decode(
                token,
                signing_key.key,
                algorithms=[algorithm],
                audience="authenticated"
            )
        else:
            # Use secret for symmetric algorithms (HS256)
            if not SUPABASE_JWT_SECRET:
                logger.warning("⚠️ SUPABASE_JWT_SECRET not set for HS256 verification.")
                return None
                
            decoded = jwt.decode(
                token,
                SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
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
