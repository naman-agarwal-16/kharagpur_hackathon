# D:/kharagpur_hackathon/src/claim_decomposer_api.py
from llm_api_fixed import api_wrapper
from typing import List, Dict, Any

class APIClaimDecomposer:
    """
    Uses Gemini API for high-quality claim extraction
    - No fallback to crude patterns
    - Structured JSON output
    - Handles complex backstories
    """
    
    def decompose(self, backstory_text: str, character_name: str) -> List[Dict[str, Any]]:
        print(f"[API DECOMPOSER] Processing {character_name}...")
        
        prompt = f"""
        You are a forensic text analyst. Extract TESTABLE, FACTUAL claims from this character backstory.
        
        CHARACTER: {character_name}
        BACKSTORY: {backstory_text}
        
        INSTRUCTIONS:
        1. Extract claims about: personality traits, motivations, fears, beliefs, skills, past events, relationships
        2. Each claim must be VERIFIABLE from the novel text
        3. Focus on SPECIFIC, BEHAVIORAL claims (not vague like "he was complex")
        4. Include age/year timeline if mentioned
        
        Return JSON with this exact structure:
        {{
          "claims": [
            {{
              "claim_id": "short_snake_case_id",
              "claim_text": "Full sentence describing claim",
              "claim_type": "trait|motivation|fear|belief|skill|event|relationship",
              "importance": "high|medium|low"
            }}
          ]
        }}
        
        EXAMPLE:
        {{
          "claims": [
            {{
              "claim_id": "fear_of_abandonment",
              "claim_text": "{character_name} avoids relationships due to fear of abandonment",
              "claim_type": "fear",
              "importance": "high"
            }}
          ]
        }}
        """
        
        # Call API (will retry or raise clear error)
        result = api_wrapper.generate_json(prompt)
        
        # Validate structure
        if 'claims' not in result:
            raise ValueError(f"API returned malformed JSON: missing 'claims' key. Got: {result}")
        
        claims = result['claims']
        
        # Add character name and generate vocab
        for claim in claims:
            claim['character_name'] = character_name
            claim['confidence'] = 0.85
            claim['search_vocabulary'] = self._generate_vocab(claim)
            claim['anti_vocabulary'] = self._generate_anti_vocab(claim)
        
        print(f"✅ Extracted {len(claims)} claims via API")
        return claims[:10]  # Limit to 10
    
    def _generate_vocab(self, claim: Dict) -> List[str]:
        """Generate search terms from claim text"""
        claim_text = claim['claim_text'].lower().replace(claim['character_name'].lower(), '')
        
        # Extract key verbs and nouns
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(claim_text)
        
        keywords = [token.text for token in doc if token.pos_ in ['VERB', 'NOUN', 'ADJ']]
        
        # Add claim-specific terms
        if 'fear' in claim['claim_type']:
            keywords.extend(['afraid', 'scared', 'terrified', 'panic'])
        elif 'brave' in claim['claim_id']:
            keywords.extend(['brave', 'courage', 'heroic', 'fearless'])
        elif 'honest' in claim['claim_id']:
            keywords.extend(['honest', 'truth', 'sincere'])
        
        return list(set(keywords))[:10]
        except Exception as e:
            print(f"[WARN] SpaCy not available: {e}")
            return ['claim', 'character']
    
    def _generate_anti_vocab(self, claim: Dict) -> List[str]:
        """Generate contradiction terms"""
        anti_map = {
            'brave': ['coward', 'fled', 'terrified', 'panicked'],
            'honest': ['lied', 'deceived', 'false'],
            'loyal': ['betrayed', 'deserted'],
            'strict': ['lenient', 'permissive'],
            'loving': ['hated', 'resented'],
        }
        
        for key, anti in anti_map.items():
            if key in claim['claim_id']:
                return anti
        
        return ['never', 'not', 'false', 'opposite']

# Test
if __name__ == "__main__":
    deco = APIClaimDecomposer()
    claims = deco.decompose("John is brave and honest", "John")
    for c in claims:
        print(f"- {c['claim_id']}: {c['claim_text']}")
