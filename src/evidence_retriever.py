"""
Clean Evidence Retriever - Finds relevant novel chunks for claim verification
"""
from typing import List, Dict, Any


class EvidenceRetriever:
    """
    Retrieves relevant evidence from novel chunks for claim verification
    """
    
    def __init__(self):
        pass
    
    def retrieve(self, claim: Dict, character_name: str, novel_chunks: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Retrieve top-k most relevant evidence chunks for a claim
        
        Args:
            claim: Claim dictionary with claim_text and search vocabulary
            character_name: Character name to search for
            novel_chunks: List of novel chunks with text and metadata
            top_k: Number of top evidence pieces to return
            
        Returns:
            List of evidence dictionaries with text, score, and type
        """
        claim_text = claim.get('claim_text', '')
        search_vocab = claim.get('search_vocabulary', [])
        anti_vocab = claim.get('anti_vocabulary', [])
        
        scored_chunks = []
        
        for chunk in novel_chunks:
            chunk_text = chunk.get('text', '')
            score = self._score_chunk(chunk_text, claim_text, character_name, search_vocab, anti_vocab)
            
            if score > 0:
                # Determine if supporting or contradicting based on anti-vocabulary
                is_contradicting = any(anti_word in chunk_text.lower() for anti_word in anti_vocab)
                
                scored_chunks.append({
                    'text': chunk_text[:500],  # Limit length
                    'score': score,
                    'type': 'contradicting' if is_contradicting else 'supporting',
                    'chunk_id': chunk.get('id', 'unknown')
                })
        
        # Sort by score and return top-k
        scored_chunks.sort(key=lambda x: x['score'], reverse=True)
        return scored_chunks[:top_k]
    
    def _score_chunk(self, chunk_text: str, claim_text: str, character_name: str, 
                    search_vocab: List[str], anti_vocab: List[str]) -> float:
        """
        Score a chunk's relevance to the claim
        """
        score = 0.0
        text_lower = chunk_text.lower()
        
        # Check if character is mentioned (required)
        char_lower = character_name.lower()
        char_parts = char_lower.split()
        
        char_mentioned = (
            char_lower in text_lower or
            any(part in text_lower for part in char_parts if len(part) > 3)
        )
        
        if not char_mentioned:
            return 0.0
        
        # Base score for mentioning character
        score += 1.0
        
        # Bonus for search vocabulary
        for word in search_vocab:
            if word.lower() in text_lower:
                score += 0.5
        
        # Check for anti-vocabulary (still relevant, just contradicting)
        for word in anti_vocab:
            if word.lower() in text_lower:
                score += 0.3  # Still relevant evidence
        
        # Bonus for direct quote or dialogue
        if '"' in chunk_text or "'" in chunk_text:
            score += 0.2
        
        return score
