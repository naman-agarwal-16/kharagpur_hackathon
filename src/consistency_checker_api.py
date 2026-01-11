# D:/kharagpur_hackathon/src/consistency_checker_api.py
from llm_api_fixed import api_wrapper
from typing import List, Dict, Any

class APIConsistencyChecker:
    """
    Uses Gemini API for nuanced consistency judgment
    """
    
    def verify_claim(self, claim: Dict, evidence: Dict) -> Dict[str, Any]:
        print(f"[API CHECKER] Verifying claim: {claim['claim_id']}")
        
        # Format evidence
        supports = evidence['supporting'][:3]
        contradicts = evidence['contradicting'][:3]
        
        prompt = f"""
        You are evaluating a character's backstory against novel evidence.
        
        CLAIM: {claim['claim_text']}
        TYPE: {claim['claim_type']}
        
        SUPPORTING EVIDENCE (max 3):
        {self._format_evidence(supports)}
        
        CONTRADICTING EVIDENCE (max 3):
        {self._format_evidence(contradicts)}
        
        TASK:
        1. Does supporting evidence truly demonstrate the claim?
        2. Does contradicting evidence truly refute it?
        3. Consider scene type (actions > dialogue > thoughts)
        
        Return ONLY JSON:
        {{
          "judgment": "consistent" or "contradicted",
          "confidence": 0.0 to 1.0,
          "rationale": "Explain in 1-2 sentences",
          "key_passages": ["most important text excerpts"]
        }}
        """
        
        # Call API
        result = api_wrapper.generate_json(prompt)
        
        # Validate
        required = ['judgment', 'confidence', 'rationale', 'key_passages']
        for field in required:
            if field not in result:
                raise ValueError(f"API missing field '{field}' in response: {result}")
        
        return {
            'consistent': 1 if result['judgment'] == 'consistent' else 0,
            'confidence': result['confidence'],
            'rationale': result['rationale'],
            'key_passages': result['key_passages']
        }
    
    def _format_evidence(self, evidence_list: List[Dict]) -> str:
        """Format evidence for prompt"""
        if not evidence_list:
            return "None found."
        
        formatted = []
        for i, ev in enumerate(evidence_list, 1):
            text = ev.get('text', '')[:150]
            scene = ev.get('scene_type', 'unknown')
            formatted.append(f"{i}. [{scene}] \"{text}\"")
        
        return "\n".join(formatted)

# Test
if __name__ == "__main__":
    checker = APIConsistencyChecker()
    
    claim = {
        'claim_id': 'test_brave',
        'claim_text': 'John is brave',
        'claim_type': 'trait'
    }
    
    evidence = {
        'supporting': [
            {'text': 'John faced the dragon without fear', 'scene_type': 'action'}
        ],
        'contradicting': [
            {'text': 'John ran from a mouse', 'scene_type': 'action'}
        ]
    }
    
    result = checker.verify_claim(claim, evidence)
    print(f"Verdict: {'consistent' if result['consistent'] else 'contradicted'}")
    print(f"Confidence: {result['confidence']}")
