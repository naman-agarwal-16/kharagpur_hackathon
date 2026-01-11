# Test script for Gemini API
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.llm_api_fixed import api_wrapper

print("Testing Gemini API...")
print(f"API Key set: {bool(os.getenv('GEMINI_API_KEY'))}")

try:
    result = api_wrapper.generate_json('Return JSON with one key "status" set to "working"')
    print(f"✅ API Test Success!")
    print(f"Response: {result}")
except Exception as e:
    print(f"❌ API Test Failed: {e}")
