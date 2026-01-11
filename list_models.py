# List available Gemini models
import os
import google.generativeai as genai

api_key = os.getenv('GEMINI_API_KEY')
if api_key:
    genai.configure(api_key=api_key)
    
    print("Available Gemini models:")
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"  - {model.name}")
else:
    print("GEMINI_API_KEY not set")
