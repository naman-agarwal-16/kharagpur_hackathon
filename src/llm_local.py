# D:/kharagpur_hackathon/src/llm_local.py
# [USE THIS VERSION FOR YOUR LAPTOP]

import requests
import json
import time
from typing import Dict, Any

class LocalLLMWrapper:
    def __init__(self, model: str = "llama3:8b-instruct-q4_0"):
        self.model = model
        self.base_url = "http://localhost:11434"
        
        # Force CPU mode (more stable on your hardware)
        self.cpu_only = True
        
        # Don't fail initialization if Ollama isn't running yet
        self.is_available = self._test_connection()
        if not self.is_available:
            print("[WARN] Ollama not running. Start with: ollama serve")
            print("[INFO] LLM will use fallback methods until Ollama is available")
    
    def _test_connection(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_content(self, prompt: str, max_retries: int = 3, 
                        temperature: float = 0.1, json_mode: bool = True) -> Dict[str, Any]:
        # Check if Ollama is available
        if not self.is_available:
            # Retry connection once
            self.is_available = self._test_connection()
            if not self.is_available:
                return {'content': None, 'raw_text': '', 'status': 'ollama_unavailable'}
        
        # Add JSON instruction
        if json_mode:
            prompt += "\n\nReturn ONLY valid JSON. No explanatory text, no markdown fences."
        
        for attempt in range(max_retries):
            try:
                print(f"[LOCAL LLM] Calling {self.model} (attempt {attempt + 1})...")
                
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "top_p": 0.95,
                            "top_k": 40,
                            "num_predict": 512,  # Limit output length (speed)
                            # CPU-specific options
                            "num_thread": 8,  # Use 8 threads (leave 4 for system)
                        }
                    },
                    timeout=180  # 3 minute timeout for CPU
                )
                
                if response.status_code == 200:
                    result_text = response.json()['response']
                    
                    # Clean up
                    result_text = result_text.replace('```json', '').replace('```', '').strip()
                    
                    if json_mode:
                        try:
                            parsed = json.loads(result_text)
                            return {'content': parsed, 'raw_text': result_text, 'status': 'success'}
                        except:
                            return {'content': result_text, 'raw_text': result_text, 'status': 'parse_error'}
                    else:
                        return {'content': result_text, 'raw_text': result_text, 'status': 'success'}
                
            except requests.exceptions.Timeout:
                print(f"  ⚠️  Timeout on attempt {attempt + 1}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
            
            time.sleep(2  ** attempt)  # Backoff
        
        return {'content': None, 'raw_text': '', 'status': 'failed'}

# Global instance
local_llm = LocalLLMWrapper()

# Quick test
def test_local_llm():
    print("Testing local LLM...")
    result = local_llm.generate_content("Respond with 'WORKING' only.", json_mode=False)
    if result['status'] == 'success':
        print(f"✓ Local LLM response: {result['content']}")
    else:
        print(f"✗ LLM failed: {result['status']}")

if __name__ == "__main__":
    test_local_llm()
