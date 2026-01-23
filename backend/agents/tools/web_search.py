import os
import requests
import time
from typing import List, Any, Dict

def tavily_search(query: str, max_results: int = 5, search_depth: str = "advanced", api_key: str = None) -> List[Dict[str, Any]]:
    """
    Search using Tavily API with improved error handling.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        search_depth: Search depth ("basic" or "advanced")
        api_key: Tavily API key (can be passed directly or via environment variable)
    
    Returns:
        List of search results or empty list if:
        - API key is missing or invalid
        - Network error occurs
        - Rate limit is hit
    """
    try:
        # Check if API key is provided
        if not api_key or api_key.strip() == "":
            # Try environment variable as fallback
            api_key = os.getenv("TAVILY_API_KEY")
            
        if not api_key or api_key.strip() == "":
            print("⚠️ Tavily API key not found. Set TAVILY_API_KEY in environment or provide via llm_config.")
            print("   Skipping web search...")
            return []
        
        url = 'https://api.tavily.com/search'
        headers = {"content-type": "application/json"}
        data = {
            "api_key": api_key.strip(),
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth
        }

        response = requests.post(url, json=data, headers=headers, timeout=15)
        
        # Sleep after each Tavily response to reduce 429 risk (configurable)
        try:
            sleep_s = float(os.getenv("TAVILY_SLEEP_SECS", "1.5"))
        except Exception:
            sleep_s = 1.5
        if sleep_s > 0:
            time.sleep(sleep_s)
        
        # Handle different error codes
        if response.status_code == 401:
            print("❌ Tavily API authentication failed (401 Unauthorized)")
            print("   Please check your TAVILY_API_KEY is valid")
            return []
        elif response.status_code == 429:
            print("⚠️ Tavily API rate limit exceeded (429). Skipping this search...")
            return []
        elif response.status_code != 200:
            print(f"⚠️ Tavily API error: {response.status_code} - {response.text[:200]}")
            return []
        
        response.raise_for_status()

        results = response.json().get("results", [])
        return results
        
    except requests.exceptions.Timeout:
        print(f"⚠️ Tavily search timed out for query: {query[:50]}...")
        return []
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Tavily search network error: {e}")
        return []
    except Exception as e:
        print(f"⚠️ Tavily search unexpected error: {e}")
        return []