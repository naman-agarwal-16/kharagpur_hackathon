import requests
import json
import time
from typing import Dict, Any

class LocalLLMWrapper:
    def __init__(self, model: str = None):
        # Import config to get model name and host
        try:
            from config import OLLAMA_MODEL, OLLAMA_HOST
            self.model = model or OLLAMA_MODEL
            ollama_host = OLLAMA_HOST
        except:
            self.model = model or "llama3:8b-instruct-q4_0"
            ollama_host = "host.docker.internal"
        
        self.base_url = f"http://{ollama_host}:11434"
        
        # Test connection
        self.is_available = self._test_connection()
        if not self.is_available:
            print("=" * 70)
            print("[ERROR] Ollama not running!")
            print("=" * 70)
            print("To start Ollama:")
        """Test if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _ensure_model_available(self):
        """Generate content using Ollama"""
        
        # Check if Ollama is available
        if not self.is_available:
            self.is_available = self._test_connection()
            if not self.is_available:
                return {
                    'content': None, 
                    'raw_text': '', 
                    'status': 'ollama_unavailable',
                    'error': 'Ollama server not running. Start with: ollama serve'
                }
        
        # Add JSON instrucLM] Calling {self.model} (attempt {attempt + 1}/{max_retries})...")
                
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
                            "num_predict": 1024,  # Max tokens to generate
                            "num_thread": 8,  # CPU threads
                        }
                    },
                    timeout=120  # 2 minute timeout
                )
                
                if response.status_code == 200:
                    result_text = response.json()['response']
                    
                    # Clean up markdown fences
                    result_text = result_text.strip()
                    if result_text.startswith('```json'):
                        result_text = result_text[7:]
                    elif result_text.startswith('```'):
                        result_text = result_text[3:]
                    if result_text.endswith('```'):
                        result_text = result_text[:-3]
                    result_text = result_text.strip()
                    
                    if json_mode:
                        try:
                            parsed = json.loads(result_text)
                            print(f"[✓] LLM response parsed successfully")
                            return {
                                'content': parsed, 
                                'raw_text': result_text, 
                                'status': 'success'
                            }
                        except json.JSONDecodeError as e:
                            print(f"[✗] JSON parse error: {e}")
                            print(f"    Raw text: {result_text[:200]}...")
                            if attempt < max_retries - 1:
            None

def get_local_llm():
    """Get or create the global LLM instance"""
    global local_llm
    if local_llm is None:
        local_llm = LocalLLMWrapper()
    return local_llm

# Initialize on import
local_llm = LocalLLMWrapper()

# Quick test
def test_local_llm():
    print("\n" + "="*70)
    print("TESTING LOCAL LLM CONNECTION")
    print("="*70)
    
    if not local_llm.is_available:
        print("[✗] Test failed - Ollama not running")
        return False
    
    test_prompt = "Respond with exactly this JSON: {\"status\": \"working\", \"message\": \"LLM is operational\"}"
    result = local_llm.generate_content(test_prompt, json_mode=True, temperature=0.1)
    
    if result['status'] == 'success':
        print(f"[✓] Test passed!")
        print(f"    Response: {result['content']}")
        return True
    else:
        print(f"[✗] Test failed: {result.get('error', 'Unknown error')}")
        return False, 
                            'raw_text': result_text, 
                            'status': 'success'
                        }
                else:
                    print(f"[✗] HTTP {response.status_code}: {response.text}")
                
            except requests.exceptions.Timeout:
                print(f"[✗] Timeout on attempt {attempt + 1}")
            except Exception as e:
                print(f"[✗] Error: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return {
            'content': None, 
            'raw_text': '', 
            'status': 'failed',
            'error': 'All retry attempts exhausted'
        
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
