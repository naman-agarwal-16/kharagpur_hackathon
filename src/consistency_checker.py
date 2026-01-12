"""
Clean Consistency Checker - Verifies claims against novel evidence using LLM
"""
import json
import re
import time
import requests
from typing import Dict, List, Any

from config import LLM_PROVIDER, LLM_CONFIG, MAX_RETRY_ATTEMPTS, USE_FALLBACK_ON_ERROR
from cache_manager import CacheManager


class ConsistencyChecker:
    """LLM-based verification of claims against novel evidence"""
    
    def __init__(self):
        self.provider = LLM_PROVIDER
        self.config = LLM_CONFIG[self.provider]
        self.cache = CacheManager()
        
        # Initialize provider-specific client
        if self.provider == "gemini":
            from google import genai
            self.client = genai.Client(api_key=self.config["api_key"])
    
    def verify_claim(self, claim: Dict, evidence_list: List[Dict]) -> Dict[str, Any]:
        """
        Verify a single claim against provided evidence
        
        Args:
            claim: Claim dictionary with claim_text and metadata
            evidence_list: List of evidence snippets with text and scores
            
        Returns:
            Dictionary with 'consistent' (0 or 1), 'confidence', 'rationale'
        """
        claim_text = claim.get('claim_text', '')
        
        # Check cache
        cache_key = f"verify_{claim_text[:100]}_{len(evidence_list)}"
        cached = self.cache.get_cached_llm_response(cache_key)
        if cached:
            print("[CACHE] Using cached verification result")
            return cached
        
        try:
            # Build verification prompt
            prompt = self._build_verification_prompt(claim_text, evidence_list)
            
            # Call LLM
            response_text = self._call_llm(prompt)
            
            # Parse response
            result = self._parse_verification_response(response_text)
            
            # Cache result
            self.cache.cache_llm_response(cache_key, result)
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Verification failed: {e}")
            if USE_FALLBACK_ON_ERROR:
                return self._fallback_verification(claim_text, evidence_list)
            else:
                raise RuntimeError(f"Verification failed: {e}")
    
    def _call_llm(self, prompt: str) -> str:
        """Make API call to configured LLM provider"""
        if self.provider in ["groq", "openrouter"]:
            return self._call_openai_format(prompt)
        elif self.provider == "gemini":
            return self._call_gemini(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _call_openai_format(self, prompt: str) -> str:
        """Call OpenAI-compatible APIs"""
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
            raise RuntimeError(f"Invalid JSON: {response.text[:200]}")
        
        if 'choices' not in data or not data['choices']:
            raise RuntimeError(f"No choices in response")
        
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
    
    def _parse_verification_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM verification response (handles JSON or plain text)"""
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
        else:
            # Try to find raw JSON object
            json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
        
        # Try parsing as JSON
        try:
            result = json.loads(response_text.strip())
            judgment = result.get('judgment', '').lower()
            confidence = result.get('confidence', 0.5)
            rationale = result.get('rationale', 'LLM response')
            
        except json.JSONDecodeError:
            # Fallback: extract from plain text
            response_lower = response_text.lower()
            
            if any(word in response_lower for word in ['contradicted', 'contradiction', 'inconsistent', 'false']):
                judgment = 'contradicted'
                confidence = 0.7
            elif any(word in response_lower for word in ['consistent', 'matches', 'supports', 'true']):
                judgment = 'consistent'
                confidence = 0.7
            else:
                judgment = 'consistent'  # Default to consistent if unclear
                confidence = 0.5
            
            rationale = response_text[:200]
        
        # Normalize to binary format
        return {
            'consistent': 1 if judgment == 'consistent' else 0,
            'confidence': float(confidence),
            'rationale': rationale
        }
    
    def _build_verification_prompt(self, claim_text: str, evidence_list: List[Dict]) -> str:
        """Build verification prompt with claim and evidence"""
        evidence_text = "\n\n".join([
            f"[Evidence {i+1}] {ev.get('text', '')[:500]}"
            for i, ev in enumerate(evidence_list[:5])  # Limit to top 5 pieces
        ])
        
        return f"""Verify if this claim is CONSISTENT with the novel evidence.

CLAIM: {claim_text}

EVIDENCE FROM NOVEL:
{evidence_text}

Respond with JSON:
{{
    "judgment": "consistent" or "contradicted",
    "confidence": 0.0 to 1.0,
    "rationale": "Brief explanation (1-2 sentences)"
}}

Rules:
- "consistent": Evidence supports or doesn't contradict the claim
- "contradicted": Evidence explicitly contradicts the claim
- If evidence is weak/missing, default to "consistent" with low confidence"""
    
    def _fallback_verification(self, claim_text: str, evidence_list: List[Dict]) -> Dict[str, Any]:
        """Simple keyword-based fallback verification"""
        # Count supporting vs contradicting keywords
        support_count = 0
        contradict_count = 0
        
        claim_lower = claim_text.lower()
        
        for evidence in evidence_list:
            evidence_text = evidence.get('text', '').lower()
            
            # Simple keyword matching
            if any(word in evidence_text for word in ['never', 'not', 'no', 'impossible']):
                contradict_count += 1
            else:
                support_count += 1
        
        # Decide based on counts
        if contradict_count > support_count:
            return {
                'consistent': 0,
                'confidence': 0.5,
                'rationale': 'Fallback: contradictory keywords found'
            }
        else:
            return {
                'consistent': 1,
                'confidence': 0.5,
                'rationale': 'Fallback: no clear contradiction'
            }
