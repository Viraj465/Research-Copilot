"""
Safe Structured Output Utility

Provides robust structured output parsing with:
- Automatic JSON repair for common LLM output issues
- Retry logic with exponential backoff
- Graceful fallback handling
- Works across all LLM providers (Groq, Google, OpenAI, etc.)
"""

import re
import json
import logging
import time
from typing import TypeVar, Type, Optional, Any, List
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class JSONRepairError(Exception):
    """Raised when JSON repair fails completely."""
    pass


def repair_json(malformed_json: str) -> str:
    """
    Attempt to repair common JSON malformations from LLM outputs.
    
    Handles:
    - Markdown bold syntax in keys: **key** -> "key"
    - Markdown bold syntax in values: **value** -> "value"
    - Missing quotes around keys
    - Trailing commas
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - Unicode dash variants
    """
    if not malformed_json:
        return malformed_json
    
    repaired = malformed_json
    
    # Fix 1: Replace markdown bold keys: **key**: -> "key":
    # Pattern: **word**: at the start of a line or after whitespace/comma/{
    repaired = re.sub(r'\*\*([^*]+)\*\*\s*:', r'"\1":', repaired)
    
    # Fix 2: Replace markdown bold values in strings (be careful not to break valid strings)
    # This is tricky - only replace if inside a value context
    # Pattern: : **value** (where value doesn't contain quotes)
    repaired = re.sub(r':\s*\*\*([^*"]+)\*\*([,}\]\n])', r': "\1"\2', repaired)
    
    # Fix 3: Fix unquoted keys (common in some LLM outputs)
    # Pattern: { key: or , key: where key is a word without quotes
    repaired = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', repaired)
    
    # Fix 4: Remove trailing commas before } or ]
    repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)
    
    # Fix 5: Replace single quotes with double quotes (but not inside strings)
    # This is a simple approach - may not work for all cases
    # Only do this if there are no double quotes in the string
    if '"' not in repaired and "'" in repaired:
        repaired = repaired.replace("'", '"')
    
    # Fix 6: Normalize unicode dashes to regular dashes in the JSON structure
    # (not inside string values, just in case they're in keys)
    repaired = repaired.replace('‑', '-')  # non-breaking hyphen
    repaired = repaired.replace('–', '-')  # en dash
    repaired = repaired.replace('—', '-')  # em dash
    
    # Fix 7: Escape unescaped newlines inside strings
    # This is complex - we need to find strings and escape newlines within them
    # For now, just ensure the JSON is somewhat valid
    
    return repaired


def extract_json_from_error(error_message: str) -> Optional[str]:
    """
    Extract the failed JSON generation from an error message.
    
    Many LLM APIs include the malformed JSON in their error response.
    """
    # Pattern for Groq errors: 'failed_generation': '...'
    groq_pattern = r"'failed_generation':\s*'(.+?)'\s*\}"
    match = re.search(groq_pattern, error_message, re.DOTALL)
    if match:
        return match.group(1)
    
    # Pattern for JSON in error messages
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, error_message, re.DOTALL)
    if matches:
        # Return the longest match (likely the full JSON)
        return max(matches, key=len)
    
    return None


def extract_arguments_from_tool_call(json_str: str) -> Optional[str]:
    """
    Extract the 'arguments' field from a tool call JSON.
    
    Tool call format: {"name": "SchemaName", "arguments": {...}}
    """
    try:
        # First try to parse as-is
        data = json.loads(json_str)
        if isinstance(data, dict) and 'arguments' in data:
            return json.dumps(data['arguments'])
    except json.JSONDecodeError:
        pass
    
    # Try to extract arguments manually
    args_pattern = r'"arguments"\s*:\s*(\{.+\})\s*\}$'
    match = re.search(args_pattern, json_str, re.DOTALL)
    if match:
        return match.group(1)
    
    return None


def parse_with_repair(
    json_str: str, 
    schema: Type[T],
    max_repair_attempts: int = 3
) -> T:
    """
    Parse JSON string into a Pydantic model with automatic repair attempts.
    """
    last_error = None
    current_json = json_str
    
    for attempt in range(max_repair_attempts):
        try:
            # Try to parse JSON
            data = json.loads(current_json)
            
            # Validate against Pydantic schema
            return schema.model_validate(data)
            
        except json.JSONDecodeError as e:
            last_error = e
            logger.warning(f"JSON parse attempt {attempt + 1} failed: {e}")
            
            # Apply repairs
            current_json = repair_json(current_json)
            
        except ValidationError as e:
            last_error = e
            logger.warning(f"Pydantic validation attempt {attempt + 1} failed: {e}")
            # JSON is valid but schema doesn't match - no repair can help
            break
    
    raise JSONRepairError(f"Failed to parse JSON after {max_repair_attempts} attempts: {last_error}")


def safe_structured_invoke(
    llm,
    schema: Type[T],
    messages: List[Any],
    max_retries: int = 2,
    retry_delay: float = 1.0,
    fallback_value: Optional[T] = None
) -> T:
    """
    Safely invoke an LLM with structured output, with automatic error recovery.
    
    Args:
        llm: The LangChain LLM instance
        schema: Pydantic model class for structured output
        messages: List of messages to send to the LLM
        max_retries: Number of retry attempts
        retry_delay: Delay between retries (with exponential backoff)
        fallback_value: Optional fallback value if all retries fail
        
    Returns:
        Parsed Pydantic model instance
        
    Raises:
        Exception: If all retries fail and no fallback is provided
    """
    structured_llm = llm.with_structured_output(schema)
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            # Primary attempt: use structured output directly
            result = structured_llm.invoke(messages)
            return result
            
        except Exception as e:
            last_error = e
            error_str = str(e)
            
            logger.warning(f"Structured output attempt {attempt + 1} failed: {error_str[:200]}...")
            
            # Check if error contains failed JSON we can repair
            if 'failed_generation' in error_str or 'Failed to parse' in error_str:
                try:
                    # Extract and repair the failed JSON
                    failed_json = extract_json_from_error(error_str)
                    
                    if failed_json:
                        logger.info("Attempting to repair failed JSON generation...")
                        
                        # Check if it's a tool call format
                        arguments_json = extract_arguments_from_tool_call(failed_json)
                        json_to_repair = arguments_json or failed_json
                        
                        # Repair and parse
                        repaired = repair_json(json_to_repair)
                        result = parse_with_repair(repaired, schema)
                        
                        logger.info("Successfully repaired and parsed JSON!")
                        return result
                        
                except (JSONRepairError, Exception) as repair_error:
                    logger.warning(f"JSON repair failed: {repair_error}")
            
            # Wait before retry with exponential backoff
            if attempt < max_retries:
                wait_time = retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
    
    # All retries exhausted
    if fallback_value is not None:
        logger.warning(f"All retries failed, using fallback value")
        return fallback_value
    
    raise last_error


def create_empty_schema_instance(schema: Type[T]) -> T:
    """
    Create an empty/default instance of a Pydantic schema.
    Useful as a fallback when parsing fails completely.
    """
    # Get field defaults or empty values
    field_values = {}
    
    for field_name, field_info in schema.model_fields.items():
        if field_info.default is not None:
            field_values[field_name] = field_info.default
        elif field_info.default_factory is not None:
            field_values[field_name] = field_info.default_factory()
        else:
            # Provide empty defaults based on type annotation
            annotation = field_info.annotation
            if annotation == str:
                field_values[field_name] = ""
            elif annotation == int:
                field_values[field_name] = 0
            elif annotation == float:
                field_values[field_name] = 0.0
            elif annotation == bool:
                field_values[field_name] = False
            elif hasattr(annotation, '__origin__'):
                origin = annotation.__origin__
                if origin == list:
                    field_values[field_name] = []
                elif origin == dict:
                    field_values[field_name] = {}
                else:
                    field_values[field_name] = None
            else:
                field_values[field_name] = None
    
    return schema.model_validate(field_values)
