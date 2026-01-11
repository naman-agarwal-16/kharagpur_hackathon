import os

# PATHS
BASE_DIR = "D:/kharagpur_hackathon"
DATA_DIR = os.path.join(BASE_DIR, "data")
NOVELS_DIR = os.path.join(DATA_DIR, "novels")
BACKSTORIES_DIR = os.path.join(DATA_DIR, "backstories")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# API SETTINGS
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-key-here")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
USE_LOCAL_LLM = False  # Disabled - using OpenRouter instead
LLM_PROVIDER = "openrouter"  # Options: "gemini", "openai", "openrouter", "local"

# LLM CONTROL - Set to True to skip LLM entirely and use fallback extraction
SKIP_LLM_USE_FALLBACK = False  # Disabled - OpenRouter API is working

# CLAIM DECOMPOSITION SETTINGS
MAX_CLAIMS_PER_BACKSTORY = 15
MIN_CLAIM_CONFIDENCE = 0.7
