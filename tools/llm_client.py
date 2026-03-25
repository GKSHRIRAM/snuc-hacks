import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# Configuration
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()

if LLM_PROVIDER == "groq":
    BASE_URL = "https://api.groq.com/openai/v1"
    MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    API_KEY = os.environ.get("GROQ_API_KEY")
else:
    BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://10.31.4.21:11434/v1")
    MODEL = os.environ.get("OLLAMA_MODEL", "kimi-k2.5:cloud")
    API_KEY = "hackathon-key"

client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=300.0  # Increased timeout for slow local models like Kimi
)

async def llm_chat(
    messages: List[Dict[str, str]], 
    temperature: float = 0.2, 
    max_retries: int = 5, 
    json_mode: bool = False
) -> str:
    """
    Centralized LLM chat function for local Ollama.
    """
    kwargs: Dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
        
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(**kwargs)
            return str(response.choices[0].message.content).strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [Ollama] Final failure: {e}")
                raise RuntimeError(f"Local LLM failed after {max_retries} retries: {e}")
            
            wait_time = 2 * (attempt + 1)
            print(f"  [Ollama] Connection error: {e}. Retrying {attempt+1}/{max_retries} in {wait_time}s...")
            await asyncio.sleep(wait_time)
            
    return ""

async def llm_chat_json(
    messages: List[Dict[str, str]], 
    temperature: float = 0.2, 
    max_retries: int = 5
) -> Dict[str, Any]:
    """
    Helper for JSON-specific LLM calls.
    Attempts to parse result as JSON.
    """
    raw = await llm_chat(messages, temperature, max_retries, json_mode=True)
    
    # Strip markdown fences if present (some models still include them despite json_mode)
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    
    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        print(f"  [Ollama] JSON Parse Error: {e}")
        # Potentially try a fallback or fix here if needed
        raise ValueError(f"LLM did not return valid JSON. Raw: {raw[:200]}")
