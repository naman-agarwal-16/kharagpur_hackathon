# Narrative Consistency Checker

**Clean, Production-Ready System** for verifying character backstory consistency against novel text using LLM-powered claim extraction and verification.

## 🎯 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Set your Groq API key (recommended for speed and reliability):

**Windows PowerShell:**
```powershell
$env:GROQ_API_KEY = "your-groq-api-key"
```

**Linux/Mac:**
```bash
export GROQ_API_KEY="your-groq-api-key"
```

**Alternative Providers:** OpenRouter, Gemini (configure in `src/config.py`)

### 3. Run Testing

Navigate to the src directory and run:

```bash
cd src
python run_afk_mode.py
```

The system will:
- ✅ Automatically test all training examples
- ✅ Save progress continuously (can resume anytime)
- ✅ Handle API rate limits automatically
- ✅ Generate `submission.csv` for test predictions when complete

## 📁 Clean Project Structure

```
kharagpur_hackathon/
├── src/
│   ├── config.py              # Unified configuration (LLM, paths, settings)
│   ├── data_loader.py         # Data loading utilities
│   ├── claim_decomposer.py    # LLM-based claim extraction
│   ├── consistency_checker.py # LLM-based claim verification
│   ├── evidence_retriever.py  # Evidence search from novel chunks
│   ├── novel_ingester.py      # Novel chunking and processing
│   ├── master_pipeline.py     # Main orchestration pipeline
│   ├── auto_test_loop.py      # Autonomous testing with resume
│   ├── run_afk_mode.py        # Runner script (START HERE)
│   ├── cache_manager.py       # SQLite caching for API responses
│   └── smart_fallback.py      # Pattern-based fallback extraction
│
├── data/
│   ├── train.csv              # Training backstories with labels
│   ├── test.csv               # Test backstories (predict these)
│   └── novels/                # Full novel texts
│
├── results/                   # Output predictions
├── logs/                      # Auto-test progress logs
├── cache/                     # SQLite caches (auto-created)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Features

### Core Capabilities
- **Multi-Provider LLM Support**: Groq, OpenRouter, Gemini
- **Smart Caching**: SQLite-based caching to avoid redundant API calls
- **Robust Error Handling**: Automatic retries, rate limit management
- **Resume Capability**: Saves progress continuously, can stop and restart anytime
- **Clean Code**: Refactored, readable, well-documented

### Pipeline Stages
1. **Claim Decomposition**: Extracts verifiable claims from backstory text
2. **Novel Ingestion**: Chunks novels into searchable segments
3. **Evidence Retrieval**: Finds relevant passages mentioning the character
4. **Consistency Verification**: LLM judges if claims match evidence
5. **Aggregation**: Combines multiple claim verdicts into final prediction

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
# LLM Provider
LLM_PROVIDER = "groq"  # Options: "groq", "openrouter", "gemini"

# Provider-specific settings (timeouts, delays, models)
LLM_CONFIG = {...}

# Claim extraction
MAX_CLAIMS_PER_BACKSTORY = 15
MIN_CLAIM_CONFIDENCE = 0.7

# Evidence retrieval
TOP_K_EVIDENCE = 10

# Testing
TEST_BATCH_SIZE = 5
AUTO_WAIT_ON_RATE_LIMIT = True
```

## 📊 Output

### Training Progress
Logged to `logs/auto_test_results.txt`:
```
Story 1: Pred=1, Actual=1.0, Conf=0.85, Rationale=2/3 claims supported...
Story 2: Pred=0, Actual=0.0, Conf=0.92, Rationale=4/4 claims contradicted...
```

### Test Predictions
Generated as `results/submission.csv`:
```csv
id,label
95,0
136,1
...
```

## 🛠️ Development

### Adding New LLM Providers

1. Add API configuration to `config.py`:
```python
LLM_CONFIG["new_provider"] = {
    "model": "model-name",
    "api_key": os.getenv("NEW_PROVIDER_KEY"),
    ...
}
```

2. Implement API call in `claim_decomposer.py` and `consistency_checker.py`

### Adjusting Verification Logic

Edit `master_pipeline.py` → `_aggregate_verifications()` to change how multiple claims are combined into final prediction.

## 🐛 Troubleshooting

### API Rate Limits
- System automatically waits 12 hours when hitting rate limits
- Reduce `TEST_BATCH_SIZE` in config.py for stricter limits
- Increase delays in `LLM_CONFIG` for each provider

### Low Accuracy
- Tune `MAX_CLAIMS_PER_BACKSTORY` (fewer = more focused)
- Adjust `TOP_K_EVIDENCE` (more evidence = better context)
- Modify aggregation weights in `_aggregate_verifications()`

### Memory Issues
- Novel chunks are cached - clear `cache/` folder if needed
- Reduce chunk size in `novel_ingester.py`

## 📈 Performance

- **Caching**: ~80% of claims reuse cached extractions on retry
- **Speed**: ~30-60 seconds per story (depending on LLM provider)
- **Accuracy**: Varies by model (70b models perform better)

## 🧹 Recently Cleaned

Removed unnecessary files:
- ❌ `test_ollama_setup.py` - Ollama is not used
- ❌ `test_api.py`, `test_quick.py` - Old test scripts
- ❌ `setup.py`, `run_predictions.py` - Replaced by clean pipeline
- ❌ `list_models.py` - Utility not needed
- ❌ `run.bat`, `run.sh` - Use Python directly

**Current Structure**: Clean, minimal, production-ready ✅

## 📝 License

MIT License - See LICENSE file for details
