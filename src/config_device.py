# D:/kharagpur_hackathon/src/config_device.py
"""
Device-specific LLM configuration
- THIS device (no Ollama): USE_API = True
- Laptop (with Ollama): USE_API = False
"""

USE_API = True  # Set to False on laptop where Ollama is installed

# API Settings (for THIS device)
# Make sure GEMINI_API_KEY is set in environment
API_MODEL = "gemini-pro"
API_TEMPERATURE = 0.1
API_MAX_RETRIES = 3

# Local LLM Settings (for laptop)
LOCAL_MODEL = "llama3:8b-instruct-q4_0"
LOCAL_TEMPERATURE = 0.1
