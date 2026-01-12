"""
Configuration for Narrative Consistency Checker
All settings and paths centralized here
"""
import os

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
NOVELS_DIR = os.path.join(DATA_DIR, "novels")
BACKSTORIES_DIR = os.path.join(DATA_DIR, "backstories")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

# Ensure directories exist
for directory in [RESULTS_DIR, LOGS_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# LLM PROVIDER CONFIGURATION
# ============================================================================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter")  # Options: "groq", "openrouter", "gemini"

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Provider-specific settings
LLM_CONFIG = {
    "groq": {
        "model": "llama-3.3-70b-versatile",
        "api_url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key": GROQ_API_KEY,
        "timeout": 60,
        "delay": 3  # Increased to 3s to avoid rate limits
    },
    "openrouter": {
        "model": "xiaomi/mimo-v2-flash:free",
        "api_url": "https://openrouter.ai/api/v1/chat/completions",
        "api_key": OPENROUTER_API_KEY,
        "timeout": 45,
        "delay": 1  # Free tier Xiaomi model
    },
    "gemini": {
        "model": "gemini-2.0-flash-exp",
        "api_key": GEMINI_API_KEY,
        "timeout": 30,
        "delay": 1
    }
}

# ============================================================================
# CLAIM EXTRACTION SETTINGS
# ============================================================================
MAX_CLAIMS_PER_BACKSTORY = 15
MIN_CLAIM_CONFIDENCE = 0.7
MAX_RETRY_ATTEMPTS = 3  # Retry on rate limits with exponential backoff

# ============================================================================
# EVIDENCE RETRIEVAL SETTINGS
# ============================================================================
TOP_K_EVIDENCE = 10  # Number of evidence chunks to retrieve per claim
EVIDENCE_SIMILARITY_THRESHOLD = 0.3
MAX_EVIDENCE_LENGTH = 500  # Characters per evidence snippet

# ============================================================================
# CACHING SETTINGS
# ============================================================================
ENABLE_CACHING = True
CACHE_DB_PATH = os.path.join(CACHE_DIR, "llm_cache.db")
NOVEL_CACHE_PATH = os.path.join(CACHE_DIR, "novel_cache.db")

# ============================================================================
# FALLBACK BEHAVIOR
# ============================================================================
USE_FALLBACK_ON_ERROR = True  # Use pattern matching when LLM rate limited

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
VERBOSE_OUTPUT = True  # Print detailed progress to console

# ============================================================================
# TESTING SETTINGS
# ============================================================================
TEST_BATCH_SIZE = 5  # Process this many examples before checking accuracy
AUTO_WAIT_ON_RATE_LIMIT = True
RATE_LIMIT_WAIT_HOURS = 12
