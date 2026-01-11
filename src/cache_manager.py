# D:/kharagpur_hackathon/src/cache_manager.py
import sqlite3
import json
import hashlib
import time
from pathlib import Path
from functools import wraps

class CacheManager:
    """SQLite-based caching for API calls and processed data"""
    
    def __init__(self, cache_dir: str = "D:/kharagpur_hackathon/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Two caches: LLM responses and processed novels
        self.llm_db = sqlite3.connect(self.cache_dir / "llm_cache.db")
        self.novel_db = sqlite3.connect(self.cache_dir / "novel_cache.db")
        
        self._init_tables()
    
    def _init_tables(self):
        """Create cache tables if they don't exist"""
        
        # LLM response cache
        self.llm_db.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                prompt_hash TEXT PRIMARY KEY,
                response TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        
        # Novel processing cache
        self.novel_db.execute("""
            CREATE TABLE IF NOT EXISTS novel_cache (
                novel_id TEXT PRIMARY KEY,
                processed_data TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        
        self.llm_db.commit()
        self.novel_db.commit()
    
    def cache_llm_response(self, prompt: str, response: dict):
        """Cache an LLM response"""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        self.llm_db.execute(
            "INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?)",
            (prompt_hash, json.dumps(response), time.time())
        )
        self.llm_db.commit()
    
    def get_cached_llm_response(self, prompt: str) -> dict | None:
        """Retrieve cached LLM response"""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        cursor = self.llm_db.execute(
            "SELECT response FROM llm_cache WHERE prompt_hash = ?",
            (prompt_hash,)
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None
    
    def cache_processed_novel(self, novel_id: str, chunks: list):
        """Cache processed novel chunks"""
        self.novel_db.execute(
            "INSERT OR REPLACE INTO novel_cache VALUES (?, ?, ?)",
            (novel_id, json.dumps(chunks), time.time())
        )
        self.novel_db.commit()
    
    def get_cached_novel(self, novel_id: str) -> list | None:
        """Retrieve cached novel chunks"""
        cursor = self.novel_db.execute(
            "SELECT processed_data FROM novel_cache WHERE novel_id = ?",
            (novel_id,)
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None
    
    def clear_old_cache(self, max_age_hours: int = 24):
        """Clear cache older than X hours"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        self.llm_db.execute("DELETE FROM llm_cache WHERE timestamp < ?", (cutoff_time,))
        self.novel_db.execute("DELETE FROM novel_cache WHERE timestamp < ?", (cutoff_time,))
        
        self.llm_db.commit()
        self.novel_db.commit()
        
        print(f"[CACHE] Cleared entries older than {max_age_hours}h")

# Decorator for caching LLM calls
def cache_llm_response(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Only cache if it's an LLM call (has 'prompt' parameter)
        prompt = str(args[0]) if args else str(kwargs.get('prompt', ''))
        
        if prompt:
            cache_manager = getattr(self, 'cache_manager', None) or CacheManager()
            
            # Check cache
            cached = cache_manager.get_cached_llm_response(prompt)
            if cached is not None:
                print("[CACHE] Using cached LLM response")
                return cached
        
        # Call original function
        result = func(self, *args, **kwargs)
        
        # Cache the result
        if prompt and result:
            cache_manager.cache_llm_response(prompt, result)
        
        return result
    
    return wrapper
