from typing import List, Dict, Any
import json
import re
import requests
from config import GEMINI_API_KEY, OPENROUTER_API_KEY, SKIP_LLM_USE_FALLBACK, LLM_PROVIDER, USE_LOCAL_LLM
from cache_manager import CacheManager

if USE_LOCAL_LLM:
    from llm_local import local_llm

class ConsistencyChecker:
    """
    Uses LLM to weigh evidence and make final consistency judgment
    """
    
    def __init__(self):
        self.llm_provider = LLM_PROVIDER
        if USE_LOCAL_LLM:
            self.local_llm = local_llm
            print("[CONSISTENCY CHECKER] Using local LLM")
        elif LLM_PROVIDER == "openrouter":
            self.api_key = OPENROUTER_API_KEY
            self.model_name = 'meta-llama/llama-3.2-3b-instruct:free'  # Smaller model, higher rate limits
            self.api_url = 'https://openrouter.ai/api/v1/chat/completions'
        else:
            from google import genai
            from google.genai import types
            self.client = genai.Client(api_key=GEMINI_API_KEY)
            self.model_name = 'gemini-2.5-flash'
        self.cache_manager = CacheManager()
    
    def verify_claim(self, claim: Dict, evidence: Dict) -> Dict[str, Any]:
        """
        Use LLM to evaluate if evidence supports or contradicts the claim
        
        Returns: {
            'judgment': 'consistent' | 'contradicted' | 'uncertain',
            'confidence': float,
            'rationale': str,
            'key_evidence': List[str]
        }
        """
        
        # Skip LLM if configured
        if SKIP_LLM_USE_FALLBACK:
            return self._fallback_verification(claim, evidence)
        
        # Handle empty evidence
        if not evidence.get('supporting') and not evidence.get('contradicting'):
            return {
                'consistent': 0,
                'confidence': 0.5,
                'rationale': 'No evidence found in novel for this claim',
                'key_passages': []
            }
        
        # Check cache
        cache_key = f"verify_{claim.get('claim_id', '')}_{len(evidence.get('supporting', []))}_{len(evidence.get('contradicting', []))}"
        cached = self.cache_manager.get_cached_llm_response(cache_key)
        if cached:
            print("[CACHE] Using cached verification result")
            return cached
        
        # Build evidence summary
        supporting_summary = self._summarize_evidence(evidence['supporting'][:3])
        contradicting_summary = self._summarize_evidence(evidence['contradicting'][:3])
        
        prompt = f"""
        You are evaluating whether a character's backstory is consistent with a novel.
        
        CLAIM: {claim['claim_text']}
        CLAIM TYPE: {claim['claim_type']}
        
        SUPPORTING EVIDENCE:
        {supporting_summary}
        
        CONTRADICTING EVIDENCE:
        {contradicting_summary}
        
        TASK:
        1. Analyze if the supporting evidence truly demonstrates the claim
        2. Analyze if the contradicting evidence truly refutes the claim
        3. Consider the type of evidence (actions > dialogue > thoughts)
        4. Check for narrative context (character growth, special circumstances)
        
        Provide your judgment in JSON format:
        {{
            "judgment": "consistent" or "contradicted" or "uncertain",
            "confidence": 0.0 to 1.0,
            "rationale": "Explain your reasoning in 2-3 sentences",
            "key_passages": ["list of 3 most important text excerpts"]
        }}
        """
        
        try:
            # Use local LLM if enabled
            if USE_LOCAL_LLM:
                result = self.local_llm.generate_content(prompt, json_mode=True, temperature=0.1)
                
                if result['status'] == 'success':
                    llm_result = result['content']
                    
                    # Convert to binary format
                    verification_result = {
                        'consistent': 1 if llm_result.get('judgment') == 'consistent' else 0,
                        'confidence': llm_result.get('confidence', 0.5),
                        'rationale': llm_result.get('rationale', 'Verification completed'),
                        'key_passages': llm_result.get('key_passages', [])
                    }
                    
                    # Cache the result
                    self.cache_manager.cache_llm_response(cache_key, verification_result)
                    return verification_result
                else:
                    print(f"[WARN] Local LLM failed: {result['status']}")
                    return self._fallback_verification(claim, evidence)
            
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
                import time
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
            
            # Extract JSON from response (common for both)
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            # Try to parse JSON first
            try:
                result = json.loads(response_text.strip())
                judgment = result.get('judgment', '').lower()
                confidence = result.get('confidence', 0.5)
                rationale = result.get('rationale', 'LLM response parsed')
            except json.JSONDecodeError:
                # Fallback: extract judgment from plain text
                response_lower = response_text.lower()
                if 'contradicted' in response_lower or 'contradiction' in response_lower or 'inconsistent' in response_lower:
                    judgment = 'contradicted'
                    confidence = 0.7
                elif 'consistent' in response_lower or 'matches' in response_lower or 'aligns' in response_lower:
                    judgment = 'consistent'
                    confidence = 0.7
                else:
                    # Default to consistent if unclear
                    judgment = 'consistent'
                    confidence = 0.5
                rationale = response_text[:200]
                print(f"[DEBUG] Extracted from text: judgment={judgment}")
            
            # Convert to binary format
            verification_result = {
                'consistent': 1 if judgment == 'consistent' else 0,
                'confidence': confidence,
                'rationale': rationale,
                'key_passages': []
            }
            
            # Cache the result
            cache_key = f"verify_{claim.get('claim_id', '')}_{len(evidence.get('supporting', []))}_{len(evidence.get('contradicting', []))}"
            self.cache_manager.cache_llm_response(cache_key, verification_result)
            
            return verification_result
            
        except Exception as e:
            print(f"[ERROR] LLM verification failed: {e}")
            # Fallback to simple scoring
            return self._fallback_verification(claim, evidence)
    
    def _summarize_evidence(self, evidence_list: List[Dict]) -> str:
        """Convert evidence passages to concise summary for LLM"""
        
        if not evidence_list:
            return "No evidence found."
        
        summary = ""
        for i, ev in enumerate(evidence_list[:3]):  # Top 3
            summary += f"{i+1}. Scene ({ev.get('scene_type', 'unknown')}): \"{ev['text'][:150]}...\"\n"
            summary += f"   Matched terms: {ev.get('matched_terms', [])[:3]}\n"
        
        return summary
    
    def _fallback_verification(self, claim: Dict, evidence: Dict) -> Dict:
        """Simple scoring fallback if LLM fails"""
        
        support_count = len(evidence.get('supporting', []))
        contra_count = len(evidence.get('contradicting', []))
        
        support_score = sum(e.get('adjusted_score', 0) for e in evidence.get('supporting', [])[:5])
        contra_score = sum(e.get('adjusted_score', 0) for e in evidence.get('contradicting', [])[:5])
        
        # Check if claim is specific or generic
        claim_id = claim.get('claim_id', '')
        # Claims that should default to consistent when no evidence found:
        # - event_death/birth: Generic life events
        # - backstory_summary: Full backstory as single claim
        # - relationship_*: Family details are often unverifiable
        # - event_education: Education details rarely in novels
        is_soft_claim = (
            claim_id.startswith('event_death') or 
            claim_id.startswith('event_birth') or
            claim_id.startswith('relationship_') or
            claim_id.startswith('event_education') or
            claim_id == 'backstory_summary'
        )
        
        # FIX 1: Handle "no evidence" case
        if support_count == 0 and contra_count == 0:
            if is_soft_claim:
                # Soft claims with no evidence - not conclusive, default consistent
                judgment = 1
                rationale = "Unverifiable claim with no contradicting evidence - defaulting to consistent"
                confidence = 0.5
            else:
                # Specific action claims with no evidence - suspicious (fabricated?)
                judgment = 0
                rationale = "No evidence found for specific claim - may be fabricated"
                confidence = 0.6
            return {
                'consistent': judgment,
                'confidence': confidence,
                'rationale': rationale,
                'key_passages': []
            }
        
        # Calculate net score - Fix 1: Increased contradiction weight from 1.2 to 2.5
        net_score = support_score - (contra_score * 2.5)
        
        # Check if claim is an action claim (more likely to be fabricated)
        is_action_claim = claim_id.startswith('action_')
        
        # Evidence-based decisions
        if support_count > 0 and contra_count == 0:
            # Only supporting evidence - consistent
            judgment = 1
            rationale = f"{support_count} supporting passages found, no contradictions"
            confidence = min(0.9, 0.5 + support_count * 0.1)
        elif contra_count > 0 and support_count == 0:
            # Only contradicting evidence - inconsistent
            judgment = 0
            rationale = f"{contra_count} contradicting passages found, no support"
            confidence = min(0.9, 0.5 + contra_count * 0.1)
        elif contra_count >= 3:
            # Multiple contradicting passages is a strong signal
            if contra_score > 0.3 * support_score:
                judgment = 0
                rationale = f"Multiple contradictions ({contra_count}) found despite supporting evidence"
                confidence = 0.7
            else:
                judgment = 1
                rationale = "Supporting evidence outweighs contradictions"
                confidence = 0.6
        elif net_score > 2:
            judgment = 1
            rationale = "Supporting evidence outweighs contradictions"
            confidence = min(abs(net_score) / 5.0, 0.9)
        elif net_score < -2:
            judgment = 0
            rationale = "Clear contradictions found in narrative"
            confidence = min(abs(net_score) / 5.0, 0.9)
        else:
            # Mixed/weak evidence - default to consistent (conservative)
            judgment = 1
            rationale = "Mixed evidence - slight lean toward consistent"
            confidence = 0.55
        
        return {
            'consistent': judgment,
            'confidence': confidence,
            'rationale': rationale,
            'key_passages': []
        }
    
    def check_temporal_consistency(self, claim: Dict, all_evidence: List[Dict]) -> Dict[str, Any]:
        """
        Enhanced temporal consistency checking with age/date alignment
        Example: Claim says "after trauma X, developed fear Y" - does Y appear after X?
        
        Returns: {
            'is_consistent': bool,
            'confidence': float,
            'issues': List[str]
        }
        """
        issues = []
        
        # Extract timeline markers from evidence
        timeline = []
        for ev in all_evidence:
            # Look for time markers in the evidence text
            text = ev['text']
            
            # Relative time markers
            rel_matches = re.findall(r'\b(after|before|when|while|since|until)\b.*?\b(\d+|child|young|grew up)\b', 
                                     text, re.IGNORECASE)
            if rel_matches:
                timeline.append({
                    'event': ev,
                    'relation': rel_matches[0][0],
                    'time_ref': rel_matches[0][1]
                })
        
        # Check age alignment
        claim_age = self._extract_age_from_claim(claim)
        if claim_age:
            # Find evidence with age mentions
            evidence_ages = []
            for ev in all_evidence:
                ev_age = re.search(r'\b(at age|was) (\d+)\b', ev['text'], re.IGNORECASE)
                if ev_age:
                    evidence_ages.append(int(ev_age.group(2)))
            
            if evidence_ages:
                # Check if ages roughly align
                age_diff = min(abs(claim_age - ea) for ea in evidence_ages)
                if age_diff > 10:
                    issues.append(f"Age mismatch: claim mentions age {claim_age}, evidence shows {evidence_ages}")
        
        # Simple check: if claim mentions early life, most evidence should be from early chapters
        if 'child' in claim['claim_text'].lower() or 'young' in claim['claim_text'].lower():
            early_evidence = sum(1 for ev in all_evidence if 'ch_1' in ev.get('chunk_id', '') or 'ch_2' in ev.get('chunk_id', ''))
            if early_evidence == 0 and len(all_evidence) > 0:
                issues.append("Claim about childhood but no evidence from early chapters")
        
        # Return consistency result
        is_consistent = len(issues) == 0
        confidence = 1.0 if is_consistent else max(0.3, 1.0 - len(issues) * 0.2)
        
        return {
            'is_consistent': is_consistent,
            'confidence': confidence,
            'issues': issues
        }
    
    def _extract_age_from_claim(self, claim: Dict) -> int:
        """Extract age mentioned in claim"""
        claim_text = claim.get('claim_text', '')
        age_match = re.search(r'\b(age |at |turned )(\d+)\b', claim_text, re.IGNORECASE)
        if age_match:
            return int(age_match.group(2))
        return None


# Test
def test_consistency_checker():
    """Test with example data"""
    
    checker = ConsistencyChecker()
    
    # Mock claim
    claim = {
        'claim_text': 'John is brave',
        'claim_type': 'trait'
    }
    
    # Mock evidence
    evidence = {
        'supporting': [
            {'text': 'John faced the dragon without fear', 'scene_type': 'action', 'adjusted_score': 2},
            {'text': 'John said "I am not afraid"', 'scene_type': 'dialogue', 'adjusted_score': 1}
        ],
        'contradicting': [
            {'text': 'John ran from the small dog', 'scene_type': 'action', 'adjusted_score': -2}
        ]
    }
    
    result = checker.verify_claim(claim, evidence)
    print(f"Judgment: {'consistent' if result['consistent'] else 'contradicted'}")
    print(f"Confidence: {result['confidence']}")
    print(f"Rationale: {result['rationale']}")

if __name__ == "__main__":
    test_consistency_checker()
