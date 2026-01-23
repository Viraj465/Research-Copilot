import os
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import logging

logger = logging.getLogger(__name__)

class LLMFactory:
    @staticmethod
    def get_llm(provider: str = None, model: str = None, temperature: float = 0.1, max_retries: int = 2):
        """
        Get an LLM instance based on the provider.
        
        Args:
            provider: "groq" or "google". If None, defaults to env var LLM_PROVIDER or "groq".
            model: Model name. If None, uses default for provider.
            temperature: Temperature for generation.
            max_retries: Max retries for API calls.
        """
        if not provider:
            provider = os.getenv("LLM_PROVIDER", "groq").lower()
            
        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found")
            
            model_name = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
            
            return ChatGroq(
                model=model_name,
                api_key=api_key,
                temperature=temperature,
                max_retries=max_retries
            )
            
        elif provider == "google":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found")
                
            model_name = model or os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite")
            
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=temperature,
                max_retries=max_retries
            )
            
        else:
            raise ValueError(f"Unsupported provider: {provider}")
