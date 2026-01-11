import json
import re
from typing import List, Dict, Any
import requests
from config import GEMINI_API_KEY, OPENROUTER_API_KEY, USE_LOCAL_LLM, MAX_CLAIMS_PER_BACKSTORY, LLM_PROVIDER, SKIP_LLM_USE_FALLBACK
from cache_manager import CacheManager
from smart_fallback import SmartFallback

if USE_LOCAL_LLM:
    from llm_local import local_llm

class ClaimDecomposer:
    """
    Converts backstory prose into testable, falsifiable claims with search vocabularies
    """
    
    def __init__(self):
        self.llm_provider = LLM_PROVIDER
        if USE_LOCAL_LLM:
            self.local_llm = local_llm
            print("[CLAIM DECOMPOSER] Using local LLM")
        elif LLM_PROVIDER == "openrouter":
            self.api_key = OPENROUTER_API_KEY
            self.model_name = 'meta-llama/llama-3.2-3b-instruct:free'  # Smaller model, higher rate limits
            self.api_url = 'https://openrouter.ai/api/v1/chat/completions'
        elif LLM_PROVIDER == "gemini":
            from google import genai
            from google.genai import types
            self.client = genai.Client(api_key=GEMINI_API_KEY)
            self.model_name = 'gemini-2.5-flash'
        self.cache_manager = CacheManager()
        self.smart_fallback = SmartFallback()
        
        # Fix 3: Patterns for filtering vague claims
        self.vague_patterns = [
            'complex', 'complicated', 'difficult', 'hard', 'easy',
            'interesting', 'unique', 'special', 'different', 'normal',
            'good', 'bad', 'nice', 'strange', 'weird'
        ]
    
    def _filter_vague_claims(self, claims: List[Dict]) -> List[Dict]:
        """Remove overly vague claims that are hard to verify"""
        filtered = []
        for claim in claims:
            claim_text = claim.get('claim_text', '').lower()
            if not any(vague in claim_text for vague in self.vague_patterns):
                filtered.append(claim)
            else:
                print(f"[FILTER] Removed vague claim: {claim.get('claim_id', 'unknown')}")
        return filtered
    
    def decompose(self, backstory_text: str, character_name: str) -> List[Dict[str, Any]]:
        """
        Main entry point: Take backstory, return structured claims
        
        Args:
            backstory_text: The raw backstory text
            character_name: Name of the character this backstory is for
            
        Returns:
            List of claim dictionaries with search vocabularies
        """
        print(f"[CLAIM DECOMPOSER] Processing backstory for {character_name}...")
        
        # Step 1: Extract raw claims via LLM
        raw_claims = self._extract_claims_with_llm(backstory_text, character_name)
        print(f"[CLAIM DECOMPOSER] Extracted {len(raw_claims)} raw claims")
        
        # Step 2: Enrich each claim with search vocabulary
        enriched_claims = []
        for claim in raw_claims:
            enriched = self._enrich_claim(claim, character_name)
            if enriched['confidence'] >= 0.6:  # Filter low-confidence claims
                enriched_claims.append(enriched)
        
        # Step 3: Filter out vague claims that are hard to verify
        enriched_claims = self._filter_vague_claims(enriched_claims)
        
        print(f"[CLAIM DECOMPOSER] Final claims: {len(enriched_claims)}")
        return enriched_claims[:MAX_CLAIMS_PER_BACKSTORY]
    
    def _extract_claims_with_llm(self, text: str, char_name: str) -> List[Dict]:
        """Use LLM to extract structured claims with retry logic"""
        
        # Check cache first
        cache_key = f"claims_{char_name}_{text[:100]}"
        cached = self.cache_manager.get_cached_llm_response(cache_key)
        if cached:
            print("[CACHE] Using cached claim extraction")
            return cached.get('claims', [])
        
        prompt = f"""
        You are a forensic text analyst. Extract TESTABLE, FACTUAL claims from this character backstory.
        
        CHARACTER: {char_name}
        BACKSTORY: {text}
        
        INSTRUCTIONS:
        1. Extract claims about: personality traits, motivations, fears, beliefs, skills, past events, relationships
        2. Each claim must be VERIFIABLE from the novel text (either supported or contradicted)
        3. DO NOT extract vague claims like "he was complex" or "she had a hard life"
        4. Focus on SPECIFIC, BEHAVIORAL claims
        
        Return a JSON array of claims. Each claim must have:
        - claim_id: short snake_case identifier
        - claim_text: full sentence describing the claim
        - claim_type: one of [trait, motivation, fear, belief, skill, event, relationship]
        - importance: high/medium/low (how central to backstory)
        
        EXAMPLE OUTPUT:
        [
            {{
                "claim_id": "fear_of_abandonment",
                "claim_text": "{char_name} avoids close relationships due to fear of abandonment",
                "claim_type": "fear",
                "importance": "high"
            }},
            {{
                "claim_id": "orphaned_at_ten",
                "claim_text": "{char_name}'s parents died when he was ten years old",
                "claim_type": "event",
                "importance": "medium"
            }}
        ]
        """
        
        # Skip LLM entirely if configured or rate limited
        import time
        
        if SKIP_LLM_USE_FALLBACK:
            print("[INFO] SKIP_LLM enabled, using fallback extraction...")
            return self._fallback_extraction(text, char_name)
        
        # Use local LLM if enabled
        if USE_LOCAL_LLM:
            try:
                print("[INFO] Using local LLM for claim extraction...")
                result = self.local_llm.generate_content(prompt, json_mode=True, temperature=0.1)
                
                if result['status'] == 'success':
                    claims = result['content']
                    # Handle both list and dict responses
                    if isinstance(claims, list):
                        pass  # Already a list
                    elif isinstance(claims, dict):
                        claims = claims.get('claims', [])
                    else:
                        claims = []
                    
                    # Cache the successful result
                    cache_key = f"claims_{char_name}_{text[:100]}"
                    self.cache_manager.cache_llm_response(cache_key, {'claims': claims})
                    
                    return claims
                elif result['status'] == 'ollama_unavailable':
                    print(f"[WARN] Ollama not running, using fallback extraction")
                    return self._fallback_extraction(text, char_name)
                else:
                    print(f"[WARN] Local LLM failed: {result['status']}")
                    return self._fallback_extraction(text, char_name)
            except Exception as e:
                print(f"[ERROR] Local LLM error: {e}")
                return self._fallback_extraction(text, char_name)
        
        max_retries = 1  # Reduced from 3 - fail fast on rate limits
        
        for attempt in range(max_retries):
            try:
                # OpenRouter API call
                if self.llm_provider == "openrouter":
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    payload = {
                        "model": self.model_name,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1
                    }
                    
                    response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
                    response.raise_for_status()
                    
                    response_data = response.json()
                    response_text = response_data['choices'][0]['message']['content'].strip()
                    
                    # Add delay to avoid rate limits - increased to 5 seconds
                    time.sleep(10)
                
                # Gemini API call
                else:
                    from google.genai import types
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.1,
                        )
                    )
                    response_text = response.text.strip()
                
                # Extract JSON from response text (common for both)
                # Remove markdown code blocks if present
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.startswith('```'):
                    response_text = response_text[3:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                
                result = json.loads(response_text.strip())
                # Handle both list and dict responses
                if isinstance(result, list):
                    claims = result
                elif isinstance(result, dict):
                    claims = result.get('claims', [])
                else:
                    claims = []
                
                # Cache the successful result
                cache_key = f"claims_{char_name}_{text[:100]}"
                self.cache_manager.cache_llm_response(cache_key, {'claims': claims})
                
                return claims
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if rate limit error - skip retries and use fallback immediately
                if '429' in error_msg or 'quota' in error_msg.lower() or 'rate' in error_msg.lower():
                    print(f"[WARN] Rate limit detected: {error_msg[:100]}...")
                    print("[INFO] Skipping retries, using fallback extraction immediately")
                    return self._fallback_extraction(text, char_name)
                
                print(f"[ERROR] LLM call failed: {e}")
                
                if attempt == max_retries - 1:
                    # Final attempt failed, use fallback
                    print("[INFO] Using enhanced fallback claim extraction...")
                    return self._fallback_extraction(text, char_name)
    
    def _enrich_claim(self, claim: Dict, char_name: str) -> Dict:
        """Add search vocabulary and anti-trait vocabulary to a claim"""
        
        claim_text = claim['claim_text']
        
        # Try smart fallback vocabulary first
        try:
            vocab = self.smart_fallback.smart_vocabulary_generation(claim)
            claim['search_vocabulary'] = vocab['positive']
            claim['anti_vocabulary'] = vocab['negative']
            claim['confidence'] = vocab['confidence']
            claim['syntactic_patterns'] = []  # Smart fallback doesn't use patterns
            return claim
        except:
            pass
        
        # Generate search terms based on claim type
        if claim['claim_type'] == 'trait':
            vocab = self._generate_trait_vocab(claim_text, char_name)
        elif claim['claim_type'] == 'fear':
            vocab = self._generate_fear_vocab(claim_text, char_name)
        elif claim['claim_type'] == 'motivation':
            vocab = self._generate_motivation_vocab(claim_text, char_name)
        elif claim['claim_type'] == 'event':
            vocab = self._generate_event_vocab(claim_text, char_name)
        else:
            vocab = self._generate_generic_vocab(claim_text, char_name)
        
        claim['search_vocabulary'] = vocab['positive']
        claim['anti_vocabulary'] = vocab['negative']
        claim['syntactic_patterns'] = vocab['patterns']
        claim['confidence'] = vocab['confidence']
        
        return claim
    
    def _generate_trait_vocab(self, claim_text: str, char_name: str) -> Dict:
        """Generate search vocab for trait claims"""
        
        # Extract trait word(s)
        trait_match = re.search(r'(brave|cowardly|kind|cruel|smart|foolish|honest|deceitful|optimistic|pessimistic)', claim_text.lower())
        trait = trait_match.group(1) if trait_match else "trait"
        
        # Pre-computed trait lexicons (expand as needed)
        TRAIT_LEXICON = {
            'brave': {
                'positive': ['brave', 'courageous', 'heroic', 'fearless', 'stood his ground', 'faced danger', 'defended'],
                'negative': ['coward', 'cowardly', 'fled', 'ran away', 'hid', 'terrified', 'shaking with fear', 'too scared', 'panicked', 'abandoned', 'deserted'],
                'patterns': [
                    f"{char_name} + (was|is) + (brave|courageous)",
                    f"{char_name} + (faced|confronted|defied) + [threat]",
                    f"{char_name} + (ran from|hid from|avoided) + [threat]"
                ]
            },
            'honest': {
                'positive': ['honest', 'truthful', 'sincere', 'genuine', 'trustworthy', 'candid'],
                'negative': ['lied', 'deceived', 'tricked', 'false', 'fabricated', 'misled', 'dishonest'],
                'patterns': [f"{char_name} + (was|is) + honest"]
            },
            'loyal': {
                'positive': ['loyal', 'faithful', 'devoted', 'dedicated', 'steadfast', 'true'],
                'negative': ['betrayed', 'deserted', 'abandoned', 'defected', 'traitor', 'disloyal'],
                'patterns': [f"{char_name} + (was|is) + loyal"]
            },
            'peaceful': {
                'positive': ['peaceful', 'calm', 'gentle', 'kind', 'benevolent', 'compassionate'],
                'negative': ['violent', 'attacked', 'killed', 'fought', 'aggressive', 'brutal'],
                'patterns': [f"{char_name} + (was|is) + peaceful"]
            },
            'loving': {
                'positive': ['loving', 'caring', 'affectionate', 'tender', 'warm', 'nurturing'],
                'negative': ['hated', 'despised', 'resented', 'abused', 'neglected', 'cruel'],
                'patterns': [f"{char_name} + (was|is) + loving"]
            },
            'just': {
                'positive': ['just', 'fair', 'righteous', 'equitable', 'impartial', 'honorable'],
                'negative': ['revenge', 'vengeance', 'spite', 'retribution', 'malice', 'unjust'],
                'patterns': [f"{char_name} + (was|is) + just"]
            },
            'self_doubt': {
                'positive': ['unsure', 'hesitated', 'doubted himself', 'uncertain', 'second-guessed', 'lacked confidence'],
                'negative': ['confident', 'certain', 'decisive', 'never doubted', 'sure of himself', 'made decision instantly'],
                'patterns': [
                    f"{char_name} + (hesitated|paused|wavered)",
                    f"{char_name} + (asked|sought) + [advice]",
                    f"{char_name} + (decided|chose) + instantly"
                ]
            },
            'strict_upbringing': {
                'positive': ['strict', 'harsh', 'disciplined', 'rules', 'punishment', 'demanding', 'controlling', 'rigid', 'severe'],
                'negative': ['lenient', 'permissive', 'freedom', 'choice', 'flexible', 'relaxed', 'easy-going'],
                'patterns': [
                    f"(father|mother|parent) + was + strict",
                    f"(father|mother|parent) + was + demanding",
                    f"rules + were + strict"
                ]
            },
            'orphan': {
                'positive': ['orphan', 'parents died', 'lost parents', 'alone', 'no family', 'abandoned'],
                'negative': ['parents alive', 'family present', 'mother', 'father', 'siblings'],
                'patterns': [
                    f"{char_name} + (became|was) + orphan",
                    f"{char_name} + parents + died"
                ]
            }
        }
        
        # Default fallback
        default_vocab = {
            'positive': ['showed', 'demonstrated', 'was known for'],
            'negative': ['never showed', 'lacked', 'was not'],
            'patterns': [f"{char_name} + [verb] + [trait]"]
        }
        
        vocab = TRAIT_LEXICON.get(trait, default_vocab)
        vocab['confidence'] = 0.8 if trait in TRAIT_LEXICON else 0.5
        
        return vocab
    
    def _generate_fear_vocab(self, claim_text: str, char_name: str) -> Dict:
        """Generate vocabulary for fear-related claims"""
        
        # Extract fear object
        fear_match = re.search(r'fear of (\w+)', claim_text.lower())
        fear_object = fear_match.group(1) if fear_match else "unknown"
        
        vocab = {
            'positive': [
                'afraid', 'terrified', 'scared', 'frightened', 'panic', 'anxiety',
                f'fear of {fear_object}', f'afraid of {fear_object}',
                'avoided', 'shrank from', 'flinched', 'hesitated to approach'
            ],
            'negative': [
                'faced confidently', 'unafraid', 'calmly approached',
                f'no fear of {fear_object}', 'comfortable with',
                'embraced', 'sought out'
            ],
            'patterns': [
                f"{char_name} + (flinched|recoiled) + from + {fear_object}",
                f"{char_name} + (calmly|confidently) + [action] + {fear_object}",
                f"{char_name} + avoided + [anything related to {fear_object}]"
            ],
            'fear_object': fear_object
        }
        
        vocab['confidence'] = 0.85
        return vocab
    
    def _generate_motivation_vocab(self, claim_text: str, char_name: str) -> Dict:
        """Generate vocabulary for motivation claims"""
        
        vocab = {
            'positive': [
                'motivated by', 'driven by', 'desired', 'wanted', 'sought',
                'ambition', 'goal', 'purpose', 'reason', 'why'
            ],
            'negative': [
                'unmotivated', 'did not care about', 'ignored', 'rejected',
                'opposite of', 'against his nature'
            ],
            'patterns': [
                f"{char_name} + (wanted|desired|sought) + [goal]",
                f"{char_name} + (did not care|ignored) + [goal]",
                f"{char_name} + motivation + was + [goal]"
            ]
        }
        
        vocab['confidence'] = 0.75
        return vocab
    
    def _generate_event_vocab(self, claim_text: str, char_name: str) -> Dict:
        """Better event vocabulary - specific and actionable"""
        
        # Extract specific event details
        event_type = "general"
        details = []
        
        if any(word in claim_text.lower() for word in ['died', 'dead', 'death']):
            event_type = "death"
            details = ['died', 'death', 'passed away', 'killed', 'fatal', 'funeral', 'grave']
        elif any(word in claim_text.lower() for word in ['born', 'birth', 'child']):
            event_type = "birth"
            details = ['born', 'birth', 'childhood', 'grew up', 'young', 'as a child']
        elif any(word in claim_text.lower() for word in ['married', 'marriage', 'wedding']):
            event_type = "marriage"
            details = ['married', 'marriage', 'wedding', 'husband', 'wife', 'spouse']
        elif any(word in claim_text.lower() for word in ['orphan', 'parents died', 'alone']):
            event_type = "orphaned"
            details = ['orphan', 'parents died', 'lost parents', 'alone', 'no family']
        elif any(word in claim_text.lower() for word in ['accident', 'crash', 'tragedy']):
            event_type = "accident"
            details = ['accident', 'crash', 'tragedy', 'disaster', 'struck']
        else:
            # Generic but better than before
            details = ['happened', 'remember', 'recall', 'never forget', 'that day']
        
        # Extract age/date if present
        age_match = re.search(r'\b(age \d+|at \d+ years old|turned \d+)\b', claim_text, re.IGNORECASE)
        age_context = [age_match.group(0)] if age_match else []
        
        # Extract location if present
        location_match = re.search(r'\b(in \w+|at \w+|near \w+)\b', claim_text, re.IGNORECASE)
        location_context = [location_match.group(0)] if location_match else []
        
        vocab = {
            'positive': details + age_context + location_context,
            'negative': ['never happened', 'did not occur', 'contradicts', 'timeline mismatch'],
            'patterns': [
                f"{char_name} + (was|were) + [age]",
                f"{char_name} + (recalled|remembered) + {event_type}",
                f"{char_name} + (never|did not) + mention"
            ],
            'event_type': event_type
        }
        
        vocab['confidence'] = 0.85 if event_type != "general" else 0.6
        return vocab
    
    def _generate_generic_vocab(self, claim_text: str, char_name: str) -> Dict:
        """Fallback vocabulary generator"""
        
        vocab = {
            'positive': ['was', 'had', 'showed', 'demonstrated', 'known for'],
            'negative': ['was not', 'did not have', 'never showed', 'lacked'],
            'patterns': [
                f"{char_name} + [verb] + [claim keyword]",
                f"{char_name} + was + [adjective from claim]"
            ]
        }
        
        vocab['confidence'] = 0.5
        return vocab
    
    def _fallback_extraction(self, text: str, char_name: str) -> List[Dict]:
        """Enhanced fallback using SmartFallback system"""
        
        claims = self.smart_fallback.extract_claims_smart(text, char_name)
        print(f"[SMART FALLBACK] Extracted {len(claims)} claims")
        return claims


# Test function
def test_decomposer():
    """Quick test with sample backstory"""
    
    # Sample data - replace with actual backstory
    sample_backstory = """
    Sarah grew up in a strict household where her father's harsh criticism left her with deep self-doubt. 
    She learned to fear abandonment early on, and now avoids close relationships. Despite this, she's 
    incredibly brave when protecting others, though she doesn't see it in herself. She became an orphan 
    at age twelve when her parents died in a car crash, which she rarely talks about.
    """
    
    decomposer = ClaimDecomposer()
    claims = decomposer.decompose(sample_backstory, "Sarah")
    
    print("\n" + "="*60)
    print("EXTRACTED CLAIMS:")
    print("="*60)
    for claim in claims:
        print(f"\n[CLAIM: {claim['claim_id']}]")
        print(f"Text: {claim['claim_text']}")
        print(f"Type: {claim['claim_type']} | Importance: {claim['importance']}")
        print(f"Search Terms: {claim['search_vocabulary'][:5]}...")  # Show first 5
        print(f"Anti-Terms: {claim['anti_vocabulary'][:3]}...")  # Show first 3
        print(f"Confidence: {claim['confidence']:.2f}")


if __name__ == "__main__":
    test_decomposer()
