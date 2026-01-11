# D:/kharagpur_hackathon/src/llm_api_fixed.py
import os
import google.generativeai as genai
import json
import time
from typing import Dict, Any

class GeminiAPIWrapper:
    """
    Robust Gemini API wrapper that NEVER silently falls back
    - Retries on failure
    - Validates JSON responses
    - Clear error messages
    - No fallback to crude extraction
    """
    
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment!")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.max_retries = 3
        self.timeout = 30  # seconds
    
    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """
        Call Gemini with JSON guarantee
        Retries on failure, never silently fails
        """
        
        # Force JSON output
        full_prompt = prompt + "\n\nReturn ONLY valid JSON. No markdown fences, no extra text."
        
        for attempt in range(self.max_retries):
            try:
                print(f"[GEMINI] API call attempt {attempt + 1}/{self.max_retries}...")
                
                response = self.model.generate_content(
                    full_prompt,
                    generation_config={
                        'temperature': 0.1,
                        'top_p': 0.95,
                        'top_k': 40,
                        'max_output_tokens': 2048,
                    }
                )
                
                # Extract text
                if not response.text:
                    raise ValueError("Empty response from API")
                
                raw_text = response.text.strip()
                print(f"[GEMINI] Raw response: {raw_text[:100]}...")
                
                # Clean markdown fences if present
                if raw_text.startswith('```'):
                    raw_text = raw_text.split('```')[1]
                    if raw_text.startswith('json'):
                        raw_text = raw_text[4:]
                    raw_text = raw_text.strip()
                
                # Parse JSON
                result = json.loads(raw_text)
                print(f"✅ API success: JSON parsed correctly")
                return result
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON parse error: {e}")
                print(f"Raw text: {raw_text[:200]}...")
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                print(f"❌ API error: {e}")
                time.sleep(2 ** attempt)
        
        # All retries failed - RAISE ERROR (don't fall back)
        raise RuntimeError(f"Gemini API failed after {self.max_retries} attempts. Check API key and connectivity.")

# Global instance
api_wrapper = GeminiAPIWrapper()
