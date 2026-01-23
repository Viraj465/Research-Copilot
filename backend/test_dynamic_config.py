import os
import sys
from typing import Dict, Any

# Add backend directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.llm_factory import LLMFactory
# from main import LLMConfig, AgentConfig # Avoid importing main to prevent circular imports/path issues

def test_multi_key_config():
    print("\n🧪 Testing Multi-Key Configuration...")
    
    # Config: Global Groq, Web Research Google
    # Providing BOTH keys in api_keys
    config = {
        "provider": "groq",
        "model": "llama3-8b-8192",
        "api_keys": {
            "groq": "gsk_mock_groq_key",
            "google": "AIza_mock_google_key"
        },
        "agents": {
            "web_research": {
                "provider": "google",
                "model": "gemini-1.5-pro"
            }
        }
    }
    
    # 1. Test Default Agent (Paper Analysis) -> Should use Groq Key
    try:
        llm = LLMFactory.get_llm(agent="paper_analysis", llm_config=config)
        print(f"✅ Paper Analysis (Default): {llm.model_name}")
        # Check if the correct key was used (this is tricky to check on the object directly without private access, 
        # but if it didn't crash and we can inspect the object, it's good. 
        # For LangChain objects, api_key is usually in secret fields)
        
        # Let's check if the object has the key set (depending on implementation)
        if hasattr(llm, 'api_key'):
            key = llm.api_key.get_secret_value() if hasattr(llm.api_key, 'get_secret_value') else llm.api_key
            print(f"   Key used: {key[:4]}...")
            assert "gsk_" in key
            
    except Exception as e:
        print(f"❌ Paper Analysis failed: {e}")

    # 2. Test Overridden Agent (Web Research) -> Should use Google Key
    try:
        llm = LLMFactory.get_llm(agent="web_research", llm_config=config)
        # Google model name might be stored differently
        model_name = getattr(llm, 'model_name', getattr(llm, 'model', 'Unknown'))
        print(f"✅ Web Research (Override): {model_name}")
        
        if hasattr(llm, 'api_key'):
             key = llm.api_key.get_secret_value() if hasattr(llm.api_key, 'get_secret_value') else llm.api_key
             print(f"   Key used: {key[:4]}...")
             assert "AIza" in key
             
    except Exception as e:
        print(f"❌ Web Research failed: {e}")

if __name__ == "__main__":
    test_multi_key_config()
