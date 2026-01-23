import os
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import logging

logger = logging.getLogger(__name__)

class LLMFactory:
    """
    LLM Factory with per-agent model configuration.
    
    Supports multiple model options per agent (user can choose via env vars).
    Both Groq and Google providers can be used simultaneously for different agents.
    """
    
    # Agent-specific model configurations (based on modelsInfo.txt)
    # Each agent can have multiple options - user chooses via AGENT_<NAME>_OPTION env var (0, 1, ...)
    AGENT_MODEL_CONFIG = {
        "paper_analysis": [
            {"provider": "groq", "model": "openai/gpt-oss-120b"},      
            {"provider": "groq", "model": "llama-3.3-70b-versatile"},
            {"provider": "google", "model": "gemini-3-flash"}
        ],
        "web_research": [
            {"provider": "google", "model": "gemini-2.5-flash-lite"}
        ],
        "advisor_specialist": [
            {"provider": "groq", "model": "openai/gpt-oss-20b"}
        ],
        "market_intelligence": [
            {"provider": "groq", "model": "openai/gpt-oss-20b"}
        ],
        "comparative_analysis": [
            {"provider": "groq", "model": "openai/gpt-oss-120b"}, 
            {"provider": "groq", "model": "llama-3.3-70b-versatile"},
            {"provider": "google", "model": "gemini-2.5-flash-lite"}
        ],
        "sota_tracker": [
            {"provider": "google", "model": "gemini-2.5-flash-lite"}
        ],
        "direction_advisor": [
            {"provider": "groq", "model": "openai/gpt-oss-120b"},
            {"provider": "groq", "model": "llama-3.3-70b-versatile"}
        ],
        "report_generation": [
            {"provider": "groq", "model": "openai/gpt-oss-120b"},
            {"provider": "groq", "model": "llama-3.3-70b-versatile"}, 
            {"provider": "google", "model": "gemini-3-flash"}
        ]
    }
    
    @staticmethod
    def get_llm(
        agent: str = None, 
        provider: str = None, 
        model: str = None, 
        temperature: float = 0.3, 
        max_retries: int = 5,
        llm_config: dict = None
    ):
        """
        Get an LLM instance based on agent name or explicit provider/model.
        
        Args:
            agent: Agent name (e.g., "paper_analysis", "web_research").
                   If provided, uses AGENT_MODEL_CONFIG for that agent.
            provider: Explicit provider override ("groq" or "google").
            model: Explicit model override.
            temperature: Temperature for generation.
            max_retries: Max retries for API calls.
            llm_config: Optional dict with keys 'provider', 'model', 'api_key' to override everything.
            
        Env var overrides (in priority order):
            1. llm_config (passed from request)
            2. AGENT_<AGENT_NAME>_PROVIDER and AGENT_<AGENT_NAME>_MODEL
            3. AGENT_<AGENT_NAME>_OPTION (0, 1, ... to select from multiple options)
            4. Uses AGENT_MODEL_CONFIG defaults
            5. Falls back to LLM_PROVIDER and GROQ_MODEL/GOOGLE_MODEL
        """
        
        api_key = None
        
        # 1. Check llm_config first (highest priority)
        if llm_config:
            # Check for agent-specific override
            if agent and llm_config.get("agents") and agent in llm_config["agents"]:
                agent_config = llm_config["agents"][agent]
                # Handle both dict and Pydantic object (if passed directly)
                if isinstance(agent_config, dict):
                    provider = agent_config.get("provider")
                    model = agent_config.get("model")
                else:
                    provider = agent_config.provider
                    model = agent_config.model
                logger.info(f"Agent '{agent}': Using specific config - {provider}/{model}")
            else:
                # Fallback to global config if no specific override
                provider = llm_config.get("provider") or provider
                model = llm_config.get("model") or model
            
            # Retrieve API key for the specific provider from the api_keys dict
            api_keys = llm_config.get("api_keys") or {}
            # Fallback to legacy 'api_key' if present (for backward compatibility or single key usage)
            api_key = api_keys.get(provider) or llm_config.get("api_key")
            
            if provider and not agent: # Log only if using global config
                 logger.info(f"Using global dynamic config - {provider}/{model}")

        # 2. If agent is specified and no provider yet, look up its configuration
        if not provider and agent:
            # Normalize agent name for env var lookup
            agent_env_key = agent.upper().replace("-", "_")
            
            # Check for explicit env var overrides first
            env_provider = os.getenv(f"AGENT_{agent_env_key}_PROVIDER")
            env_model = os.getenv(f"AGENT_{agent_env_key}_MODEL")
            
            if env_provider or env_model:
                provider = env_provider or provider
                model = env_model or model
                logger.info(f"Agent '{agent}': Using env override - {provider}/{model}")
            else:
                # Check if agent has configured options
                if agent in LLMFactory.AGENT_MODEL_CONFIG:
                    options = LLMFactory.AGENT_MODEL_CONFIG[agent]
                    
                    # Get user's option choice (default to 0)
                    option_idx = int(os.getenv(f"AGENT_{agent_env_key}_OPTION", "0"))
                    
                    # Clamp to valid range
                    if option_idx >= len(options):
                        logger.warning(f"Agent '{agent}': Invalid option {option_idx}, using 0")
                        option_idx = 0
                    
                    config = options[option_idx]
                    provider = config["provider"]
                    model = config["model"]
                    
                    logger.info(f"Agent '{agent}': Using option {option_idx} - {provider}/{model}")
                else:
                    logger.warning(f"Agent '{agent}' not in AGENT_MODEL_CONFIG, using fallback")
        
        # 3. Fallback to explicit params or global env vars
        if not provider:
            provider = os.getenv("LLM_PROVIDER", "groq").lower()
            
        if provider == "groq":
            if not api_key:
                api_key = os.getenv("GROQ_API_KEY")
            
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables or config")
            
            model_name = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
            
            logger.debug(f"Creating ChatGroq: {model_name}")
            return ChatGroq(
                model=model_name,
                api_key=api_key,
                temperature=temperature,
                max_retries=max_retries
            )
            
        elif provider == "google":
            if not api_key:
                api_key = os.getenv("GOOGLE_API_KEY")
                
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables or config")
                
            model_name = model or os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite")
            
            logger.debug(f"Creating ChatGoogleGenerativeAI: {model_name}")
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=temperature,
                max_retries=max_retries
            )
        
        elif provider == "openai":
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
                
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables or config")
                
            model_name = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
            
            logger.debug(f"Creating ChatOpenAI: {model_name}")
            return ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                temperature=temperature,
                max_retries=max_retries
            )
            
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'groq' or 'google' or 'openai'.")
