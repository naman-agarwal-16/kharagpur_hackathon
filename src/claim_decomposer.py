"""
Clean Claim Decomposer - Extracts verifiable claims from character backstories
"""
import json
import re
import time
import requests
from typing import List, Dict, Any, Optional

from config import (
    LLM_PROVIDER, LLM_CONFIG, MAX_CLAIMS_PER_BACKSTORY, 
    MAX_RETRY_ATTEMPTS, USE_FALLBACK_ON_ERROR
)
from cache_manager import CacheManager
from smart_fallback import SmartFallback


class ClaimDecomposer:
    """Extracts structured, verifiable claims from character backstories using LLM"""
    
    def __init__(self):
        self.provider = LLM_PROVIDER
        self.config = LLM_CONFIG[self.provider]
        self.cache = CacheManager()
        self.fallback = SmartFallback()
        
        # Initialize provider-specific client
        if self.provider == "gemini":
            from google import genai
            self.client = genai.Client(api_key=self.config["api_key"])
    
    def decompose(self, backstory_text: str, character_name: str) -> List[Dict[str, Any]]:
        """
        Main entry point: decompose backstory into verifiable claims
        
        Args:
            backstory_text: Character's backstory paragraph
            character_name: Name of the character
            
        Returns:
            List of claim dictionaries with structure, vocabulary, and confidence
        """
        print(f"[CLAIM DECOMPOSER] Processing backstory for {character_name}...")
        
        # Check cache first
        cache_key = f"claims_{character_name}_{backstory_text[:100]}"
        cached = self.cache.get_cached_llm_response(cache_key)
        if cached:
            print("[CACHE] Using cached claim extraction")
            raw_claims = cached.get('claims', [])
        else:
            # Extract claims using LLM
            try:
                raw_claims = self._extract_with_llm(backstory_text, character_name)
                self.cache.cache_llm_response(cache_key, {'claims': raw_claims})
            except Exception as e:
                error_msg = str(e)
                print(f"[ERROR] Claim extraction failed: {error_msg}")
                if USE_FALLBACK_ON_ERROR:
                    print("[FALLBACK] Using pattern-based extraction (LLM unavailable)")
                    raw_claims = self.fallback.extract_claims_smart(backstory_text, character_name)
                else:
                    raise RuntimeError(f"Claim extraction failed: {error_msg}")
        
        print(f"[CLAIM DECOMPOSER] Extracted {len(raw_claims)} raw claims")
        
        # Enhance claims with vocabularies and patterns
        enhanced_claims = []
        for claim in raw_claims[:MAX_CLAIMS_PER_BACKSTORY]:
            enhanced = self._enhance_claim(claim, character_name)
            enhanced_claims.append(enhanced)
        
        print(f"[CLAIM DECOMPOSER] Final claims: {len(enhanced_claims)}")
        return enhanced_claims
    
    def _extract_with_llm(self, text: str, char_name: str) -> List[Dict]:
        """Extract claims using configured LLM provider"""
        prompt = self._build_extraction_prompt(text, char_name)
        
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                response_text = self._call_llm(prompt)
                claims = self._parse_llm_response(response_text)
                return claims
                
            except Exception as e:
                error_msg = str(e)
                print(f"[ERROR] LLM call failed (attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS}): {error_msg}")
                
                # Check if it's a rate limit error
                is_rate_limit = '429' in error_msg or 'rate limit' in error_msg.lower() or 'too many requests' in error_msg.lower()
                
                if attempt < MAX_RETRY_ATTEMPTS - 1:
                    # Exponential backoff: 5s, 15s, 45s
                    wait_time = 5 * (3 ** attempt)
                    print(f"[RETRY] Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    if is_rate_limit:
                        print(f"[WARN] Rate limit exceeded after {MAX_RETRY_ATTEMPTS} attempts")
                    raise RuntimeError(f"LLM extraction failed after {MAX_RETRY_ATTEMPTS} attempts: {error_msg}")
    
    def _call_llm(self, prompt: str) -> str:
        """Make API call to configured LLM provider"""
        if self.provider in ["groq", "openrouter"]:
            return self._call_openai_format(prompt)
        elif self.provider == "gemini":
            return self._call_gemini(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def _call_openai_format(self, prompt: str) -> str:
        """Call OpenAI-compatible APIs (Groq, OpenRouter)"""
        headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json"
        }
        
        # Add OpenRouter-specific headers if using OpenRouter
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com"
            headers["X-Title"] = "Narrative Consistency Checker"
        
        payload = {
            "model": self.config['model'],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
        
        response = requests.post(
            self.config['api_url'],
            headers=headers,
            json=payload,
            timeout=self.config['timeout']
        )
        response.raise_for_status()
        
        if not response.text:
            raise RuntimeError(f"Empty response from {self.provider}")
        
        try:
            data = response.json()
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON from {self.provider}: {response.text[:200]}")
        
        if 'choices' not in data or not data['choices']:
            raise RuntimeError(f"No choices in response: {data}")
        
        # Add delay to respect rate limits
        time.sleep(self.config['delay'])
        
        return data['choices'][0]['message']['content'].strip()
    
    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini API"""
        from google.genai import types
        
        response = self.client.models.generate_content(
            model=self.config['model'],
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1)
        )
        
        time.sleep(self.config['delay'])
        return response.text.strip()
    
    def _parse_llm_response(self, response_text: str) -> List[Dict]:
        """Extract JSON array from LLM response (handles markdown code blocks)"""
        # Try to find JSON in markdown code blocks first
        json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
        else:
            # Try to find raw JSON array
            json_match = re.search(r'(\[.*\])', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
        
        try:
            result = json.loads(response_text.strip())
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                return result.get('claims', [])
            return []
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse JSON: {e}\nResponse: {response_text[:500]}")
    
    def _build_extraction_prompt(self, text: str, char_name: str) -> str:
        """Build prompt for LLM claim extraction"""
        return f"""Extract testable claims from this character backstory. Return ONLY a JSON array.

Character: {char_name}
Backstory: {text}

Extract specific, factual, falsifiable claims. Each claim should be:
- SPECIFIC: Concrete events, traits, relationships, or skills
- VERIFIABLE: Can be proven true or false from novel text
- ATOMIC: One distinct fact per claim

Return JSON array with this structure:
[
    {{
        "claim_id": "descriptive_snake_case_id",
        "claim_text": "Full sentence describing the claim",
        "claim_type": "event|trait|relationship|skill|belief|motivation",
        "importance": "high|medium|low"
    }}
]

Focus on high-importance claims. Limit to {MAX_CLAIMS_PER_BACKSTORY} claims."""
    
    def _enhance_claim(self, claim: Dict, char_name: str) -> Dict[str, Any]:
        """Add search vocabularies and syntactic patterns to claim"""
        claim_type = claim.get('claim_type', 'event')
        importance = claim.get('importance', 'medium')
        
        # Generate search vocabulary based on claim type
        if claim_type == 'event':
            vocab = ['happened', 'occurred', 'when', 'after', 'before', 'during']
            anti_vocab = ['never happened', 'did not occur', 'impossible']
        elif claim_type == 'trait':
            vocab = ['was', 'is', 'known for', 'characterized by', 'nature']
            anti_vocab = ['not', 'never', 'opposite', 'contrary']
        elif claim_type == 'relationship':
            vocab = ['met', 'knew', 'friend', 'enemy', 'relation']
            anti_vocab = ['stranger', 'never met', 'unknown']
        elif claim_type == 'skill':
            vocab = ['skilled', 'expert', 'learned', 'trained', 'able to']
            anti_vocab = ['incapable', 'unable', 'never learned']
        else:
            vocab = ['mentioned', 'said', 'described']
            anti_vocab = ['denied', 'contradicted', 'opposite']
        
        # Syntactic patterns for evidence matching
        patterns = [
            f"{char_name} + (was|were|had)",
            f"{char_name} + (never|not|did not)",
            f"recall + {char_name}"
        ]
        
        # Confidence based on importance and type
        confidence_map = {'high': 0.8, 'medium': 0.6, 'low': 0.4}
        confidence = confidence_map.get(importance, 0.6)
        
        return {
            **claim,
            'search_vocabulary': vocab,
            'anti_vocabulary': anti_vocab,
            'syntactic_patterns': patterns,
            'confidence': confidence
        }
