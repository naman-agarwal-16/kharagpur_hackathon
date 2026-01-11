import os

# PATHS - Updated for dev container
BASE_DIR = "/workspaces/kharagpur_hackathon"
DATA_DIR = os.path.join(BASE_DIR, "data")
NOVELS_DIR = os.path.join(DATA_DIR, "novels")
BACKSTORIES_DIR = os.path.join(DATA_DIR, "backstories")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# API SETTINGS
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", None)
USE_LOCAL_LLM = False  # Disabled - using OpenRouter instead
LLM_PROVIDER = "openrouter"  # Options: "gemini", "openai", "openrouter", "local"

# LLM CONTROL - Keep False to always use LLM (avoid fallback to pattern matching)
SKIP_LLM_USE_FALLBACK = False  # Use LLM for claim extraction, not patterns

# CLAIM DECOMPOSITION SETTINGS
MAX_CLAIMS_PER_BACKSTORY = 15
MIN_CLAIM_CONFIDENCE = 0.7
