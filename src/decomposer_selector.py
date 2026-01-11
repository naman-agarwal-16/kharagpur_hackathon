# D:/kharagpur_hackathon/src/decomposer_selector.py
from config_device import USE_API

if USE_API:
    from claim_decomposer_api import APIClaimDecomposer
    ClaimDecomposer = APIClaimDecomposer
    print("✅ Using GEMINI API (no Ollama)")
else:
    from claim_decomposer_rules import RuleBasedClaimDecomposer
    ClaimDecomposer = RuleBasedClaimDecomposer
    print("✅ Using Rule-Based Decomposer (Ollama not available)")
