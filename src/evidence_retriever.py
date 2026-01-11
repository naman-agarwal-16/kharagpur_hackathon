from typing import List, Dict, Any
import re
from collections import defaultdict

class EvidenceRetriever:
    """
    Sophisticated evidence retrieval that goes beyond keyword matching
    """
    
    def __init__(self, ingester):
        self.ingester = ingester
        self.scene_weights = {
            'action': 1.5,      # Actions > words
            'dialogue': 1.0,    # Dialogue is medium evidence
            'introspection': 1.2 # Internal thoughts are strong
        }
    
    def retrieve_evidence(self, character: str, claim: Dict) -> Dict[str, List[Dict]]:
        """
        Retrieve supporting and contradicting evidence for a claim
        
        Returns: {
            'supporting': [...],
            'contradicting': [...],
            'neutral': [...]
        }
        """
        
        # Get all matches from ingester
        all_matches = self.ingester.search_character(character, claim)
        
        # Classify each match
        evidence = {
            'supporting': [],
            'contradicting': [],
            'neutral': []
        }
        
        # Normalize character name for matching
        char_lower = character.lower()
        char_parts = char_lower.split()
        
        for match in all_matches:
            # FIX: Require character to be mentioned in the evidence passage
            text_lower = match['text'].lower()
            char_mentioned = (char_lower in text_lower or 
                              any(part in text_lower for part in char_parts if len(part) > 3))
            
            if not char_mentioned:
                # Evidence doesn't mention the character - skip it
                continue
            
            # Score adjustment based on scene type
            base_score = match['score']
            scene_type = match.get('scene_type', 'unknown')
            scene_multiplier = self.scene_weights.get(scene_type, 1.0)
            
            adjusted_score = base_score * scene_multiplier
            
            # Re-classify based on score and matched terms
            has_contradiction = any('CONTRADICTION' in term for term in match['matched_terms'])
            
            if adjusted_score > 0 and not has_contradiction:
                evidence['supporting'].append({
                    **match,
                    'adjusted_score': adjusted_score,
                    'weight': self._calculate_evidence_weight(match, claim)
                })
            elif adjusted_score < 0 or has_contradiction:
                evidence['contradicting'].append({
                    **match,
                    'adjusted_score': adjusted_score,
                    'weight': self._calculate_evidence_weight(match, claim)
                })
            else:
                evidence['neutral'].append(match)
        
        # Sort by weight
        for category in evidence:
            evidence[category].sort(key=lambda x: x.get('weight', 0), reverse=True)
        
        return evidence
    
    def _calculate_evidence_weight(self, match: Dict, claim: Dict) -> float:
        """
        Calculate evidence weight based on multiple factors
        """
        weight = 0
        
        # 1. Score magnitude
        weight += abs(match['adjusted_score']) * 0.5
        
        # 2. Scene type bonus
        scene_weights = {'action': 2.0, 'introspection': 1.5, 'dialogue': 1.0}
        weight += scene_weights.get(match.get('scene_type'), 0.5)
        
        # 3. Character presence (is the character active in scene?)
        if match.get('characters_in_scene') and claim.get('character_name'):
            if claim['character_name'] in match['characters_in_scene']:
                weight += 1.0  # Character is mentioned by name
        
        # 4. Temporal proximity to claim's time period
        # (If claim is about childhood, evidence from early chapters weights more)
        if 'early' in match.get('chunk_id', ''):
            weight += 0.5
        
        # 5. Negation detection (ALWAYS high weight)
        text_lower = match['text'].lower()
        negation_patterns = ['not', 'never', 'no longer', 'contrary to']
        if any(neg in text_lower for neg in negation_patterns):
            weight += 3.0  # Negations are powerful contradictions
        
        return weight
    
    def find_critical_moments(self, character: str, claim: Dict) -> List[Dict]:
        """
        Find scenes where claim-relevant behavior SHOULD appear if true
        This is the "negative evidence" detector
        
        Example: If claim is "fear of water", find all water scenes
        """
        critical_moments = []
        
        # Extract "trigger" concepts from claim
        # For fear, triggers are the feared object
        # For bravery, triggers are dangerous situations
        triggers = self._extract_triggers(claim)
        
        if not triggers:
            return critical_moments
        
        # Search for scenes containing triggers
        for chunk in self.ingester.chunks:
            text_lower = chunk['text'].lower()
            
            # Check if trigger is present
            trigger_present = any(trigger in text_lower for trigger in triggers)
            
            # Check if character is present
            char_present = character.lower() in text_lower
            
            if trigger_present and char_present:
                # Check if claim-specific behavior is ABSENT
                # This is evidence of contradiction
                has_positive_marker = any(term.lower() in text_lower for 
                                        term in claim.get('search_vocabulary', []))
                
                # If trigger is present but claim behavior is absent, this is a critical negative moment
                if not has_positive_marker:
                    critical_moments.append({
                        'chunk_id': chunk['id'],
                        'text': chunk['text'],
                        'trigger': triggers[0],
                        'absent_behavior': claim['claim_text'],
                        'weight': 2.5  # High weight for negative evidence
                    })
        
        return critical_moments
    
    def _extract_triggers(self, claim: Dict) -> List[str]:
        """
        Extract situation triggers from a claim
        """
        claim_text = claim['claim_text'].lower()
        
        trigger_map = {
            'fear': ['danger', 'threat', 'scared', 'frightened'],
            'brave': ['danger', 'risk', 'battle', 'fight', 'confront'],
            'water': ['water', 'river', 'sea', 'ocean', 'drown'],
            'height': ['height', 'cliff', 'mountain', 'high', 'fall'],
            'social': ['talk', 'speak', 'meet', 'people', 'crowd'],
        }
        
        # Check claim type first
        if claim.get('claim_type') == 'fear':
            # Extract object of fear
            fear_match = re.search(r'fear of (\w+)', claim_text)
            if fear_match:
                object_of_fear = fear_match.group(1)
                return [object_of_fear] + trigger_map.get(object_of_fear, [])
        
        # Check for keywords in claim text
        triggers = []
        for concept, keywords in trigger_map.items():
            if concept in claim_text:
                triggers.extend(keywords)
        
        return triggers[:5]  # Return max 5 triggers


# Test
def test_retriever():
    """Test with real example"""
    
    from data_loader import DataLoader
    from novel_ingester import NovelIngester
    
    # Load one example
    loader = DataLoader()
    example = loader.get_training_examples()[0]
    
    # Ingest novel
    ingester = NovelIngester(loader.load_novel(example['book_name']))
    ingester.ingest()
    
    # Create retriever
    retriever = EvidenceRetriever(ingester)
    
    # Decompose one claim
    from claim_decomposer import ClaimDecomposer
    decomposer = ClaimDecomposer()
    claims = decomposer.decompose(example['backstory'], example['character_name'])
    
    if claims:
        print(f"\nTesting claim: {claims[0]['claim_text']}")
        evidence = retriever.retrieve_evidence(example['character_name'], claims[0])
        
        print(f"\nSupporting evidence: {len(evidence['supporting'])} passages")
        print(f"Contradicting evidence: {len(evidence['contradicting'])} passages")
        
        # Show top contradiction if any
        if evidence['contradicting']:
            print(f"\nTop contradiction:")
            print(evidence['contradicting'][0]['text'][:200])
        
        # Check critical moments
        critical = retriever.find_critical_moments(example['character_name'], claims[0])
        print(f"\nCritical moments (negative evidence): {len(critical)}")

if __name__ == "__main__":
    test_retriever()
