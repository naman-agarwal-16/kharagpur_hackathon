"""
Improved Fallback System - Better pattern matching without LLM
Uses linguistic patterns, entity extraction, and heuristics
"""

import re
from typing import List, Dict, Any
from collections import Counter

class SmartFallback:
    """Enhanced pattern-based claim extraction and verification"""
    
    def __init__(self):
        # Expanded trait vocabulary
        self.traits = {
            'positive': ['brave', 'courageous', 'kind', 'generous', 'smart', 'intelligent', 
                        'honest', 'loyal', 'confident', 'optimistic', 'humble', 'patient'],
            'negative': ['cowardly', 'cruel', 'mean', 'selfish', 'foolish', 'stupid',
                        'deceitful', 'dishonest', 'arrogant', 'pessimistic', 'proud', 'impatient']
        }
        
        # Event indicators
        self.events = {
            'death': ['died', 'killed', 'murdered', 'passed away', 'death', 'funeral', 'grave'],
            'birth': ['born', 'birth', 'came into the world', 'entered the world'],
            'orphaned': ['orphan', 'lost parents', 'parents died', 'alone', 'abandoned'],
            'marriage': ['married', 'wedding', 'wed', 'spouse', 'husband', 'wife'],
            'arrest': ['arrested', 'imprisoned', 'jailed', 'captured', 'detained', 'sentenced'],
            'education': ['studied', 'learned', 'trained', 'taught', 'educated', 'school'],
            'injury': ['injured', 'wounded', 'hurt', 'accident', 'crash', 'struck'],
            'conflict': ['argued', 'fought', 'disagreed', 'conflict', 'dispute', 'quarrel']
        }
        
        # Relationship patterns
        self.relationships = ['father', 'mother', 'parent', 'son', 'daughter', 'brother',
                            'sister', 'family', 'relative', 'friend', 'mentor']
        
        # Temporal markers
        self.temporal = [r'\bat \d+\b', r'\bage \d+\b', r'\byoung\b', r'\bchild\b',
                        r'\bin \d{4}\b', r'\bwhen (?:he|she|they)\b']
    
    def extract_claims_smart(self, text: str, char_name: str) -> List[Dict]:
        """Smart claim extraction with better patterns"""
        
        claims = []
        sentences = re.split(r'[.!?;]', text)
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 15:
                continue
            
            sent_lower = sent.lower()
            
            # Check for traits
            trait_claim = self._extract_trait_claim(sent, sent_lower, char_name)
            if trait_claim:
                claims.append(trait_claim)
                continue
            
            # Check for events
            event_claim = self._extract_event_claim(sent, sent_lower, char_name)
            if event_claim:
                claims.append(event_claim)
                continue
            
            # Check for relationships
            relationship_claim = self._extract_relationship_claim(sent, sent_lower, char_name)
            if relationship_claim:
                claims.append(relationship_claim)
                continue
            
            # Default: Extract ANY sentence with action/state verbs as a generic claim
            # This is critical for catching fabricated backstories
            action_verbs = ['was', 'had', 'became', 'knew', 'felt', 'saw', 'met', 'found',
                           'made', 'took', 'gave', 'lost', 'won', 'joined', 'left', 'started',
                           'ended', 'began', 'finished', 'received', 'sent', 'arrived', 'departed',
                           'rescued', 'saved', 'helped', 'fought', 'discovered', 'learned', 'taught',
                           'created', 'built', 'destroyed', 'escaped', 'captured', 'freed']
            if any(word in sent_lower for word in action_verbs):
                claims.append({
                    'claim_id': f'action_{len(claims)}',
                    'claim_text': self._format_claim(sent, char_name),
                    'claim_type': 'event',  # Treat as event for better vocabulary
                    'event_type': 'action',
                    'importance': 'medium'
                })
        
        # If still no claims, create at least one from the full text
        if not claims and len(text) > 20:
            claims.append({
                'claim_id': 'backstory_summary',
                'claim_text': text[:200],  # First 200 chars as claim
                'claim_type': 'event',
                'event_type': 'general',
                'importance': 'medium'
            })
        
        return self._deduplicate(claims)[:12]
    
    def _extract_trait_claim(self, sent: str, sent_lower: str, char_name: str) -> Dict | None:
        """Extract trait-based claims"""
        
        # Check for explicit trait words
        for trait_list in [self.traits['positive'], self.traits['negative']]:
            for trait in trait_list:
                if re.search(rf'\b{trait}\b', sent_lower):
                    return {
                        'claim_id': f'trait_{trait}',
                        'claim_text': self._format_claim(sent, char_name),
                        'claim_type': 'trait',
                        'importance': 'high',
                        'detected_trait': trait
                    }
        
        # Check for emotional states
        emotions = ['fear', 'afraid', 'scared', 'worried', 'anxious', 'happy', 'sad', 
                   'angry', 'disappointed', 'hopeful', 'confident']
        for emotion in emotions:
            if re.search(rf'\b{emotion}\b', sent_lower):
                return {
                    'claim_id': f'emotion_{emotion}',
                    'claim_text': self._format_claim(sent, char_name),
                    'claim_type': 'fear' if emotion in ['fear', 'afraid', 'scared'] else 'trait',
                    'importance': 'high',
                    'detected_emotion': emotion
                }
        
        return None
    
    def _extract_event_claim(self, sent: str, sent_lower: str, char_name: str) -> Dict | None:
        """Extract event-based claims"""
        
        for event_type, keywords in self.events.items():
            for keyword in keywords:
                if re.search(rf'\b{keyword}\b', sent_lower):
                    # Extract temporal context if present
                    temporal_info = self._extract_temporal(sent)
                    
                    return {
                        'claim_id': f'event_{event_type}',
                        'claim_text': self._format_claim(sent, char_name),
                        'claim_type': 'event',
                        'importance': 'high',
                        'event_type': event_type,
                        'temporal': temporal_info
                    }
        
        return None
    
    def _extract_relationship_claim(self, sent: str, sent_lower: str, char_name: str) -> Dict | None:
        """Extract relationship-based claims"""
        
        for rel in self.relationships:
            if re.search(rf'\b{rel}\b', sent_lower):
                return {
                    'claim_id': f'relationship_{rel}',
                    'claim_text': self._format_claim(sent, char_name),
                    'claim_type': 'relationship',
                    'importance': 'medium',
                    'relationship_type': rel
                }
        
        return None
    
    def _extract_temporal(self, text: str) -> List[str]:
        """Extract temporal information from text"""
        
        temporal_info = []
        for pattern in self.temporal:
            matches = re.findall(pattern, text, re.IGNORECASE)
            temporal_info.extend(matches)
        
        return temporal_info
    
    def _format_claim(self, sent: str, char_name: str) -> str:
        """Format sentence as a proper claim"""
        
        sent = sent.strip()
        
        # If character name not in sentence, prepend it
        if char_name.lower() not in sent.lower():
            # Use possessive form
            if any(pronoun in sent.lower() for pronoun in ['his', 'her', 'their', 'he', 'she', 'they']):
                sent = f"{char_name}'s {sent}"
            else:
                sent = f"{char_name} {sent}"
        
        return sent
    
    def _deduplicate(self, claims: List[Dict]) -> List[Dict]:
        """Remove duplicate claims"""
        
        seen = set()
        unique = []
        
        for claim in claims:
            # Use claim_text as key
            key = claim['claim_text'].lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(claim)
        
        return unique
    
    def smart_vocabulary_generation(self, claim: Dict) -> Dict:
        """Generate better search vocabularies based on claim type"""
        
        claim_text = claim['claim_text'].lower()
        claim_type = claim['claim_type']
        
        positive_terms = []
        negative_terms = []
        
        if claim_type == 'trait' and 'detected_trait' in claim:
            trait = claim['detected_trait']
            
            # Use pre-defined synonyms
            if trait == 'brave':
                positive_terms = ['brave', 'courageous', 'heroic', 'fearless', 'bold', 'daring']
                negative_terms = ['coward', 'cowardly', 'afraid', 'scared', 'fled', 'hid']
            elif trait == 'cruel':
                positive_terms = ['cruel', 'harsh', 'brutal', 'merciless', 'ruthless']
                negative_terms = ['kind', 'gentle', 'merciful', 'compassionate']
            elif trait == 'intelligent':
                positive_terms = ['smart', 'intelligent', 'clever', 'wise', 'brilliant']
                negative_terms = ['foolish', 'stupid', 'dumb', 'ignorant']
            else:
                positive_terms = [trait]
                negative_terms = []
        
        elif claim_type == 'event' and 'event_type' in claim:
            event_type = claim['event_type']
            # Add generic event terms but prioritize claim-specific terms
            positive_terms = self.events.get(event_type, [])[:3]  # Only take 3 generic terms
            negative_terms = ['never happened', 'did not occur', 'false']
        
        elif claim_type == 'relationship':
            rel_type = claim.get('relationship_type', '')
            positive_terms = [rel_type, f'his {rel_type}', f'her {rel_type}']
            negative_terms = [f'no {rel_type}', f'without {rel_type}']
        
        # Extract key nouns, verbs, and proper nouns from claim_text
        # This is CRITICAL for specificity - use actual claim content
        stopwords = {
            'that', 'this', 'with', 'from', 'have', 'been', 'were', 'being',
            'when', 'where', 'what', 'which', 'while', 'there', 'their', 'they',
            'than', 'then', 'them', 'these', 'those', 'other', 'about', 'after',
            'before', 'would', 'could', 'should', 'might', 'must', 'shall',
            'very', 'just', 'only', 'even', 'also', 'some', 'such', 'like',
            'made', 'make', 'came', 'come', 'went', 'going', 'said', 'told',
            'himself', 'herself', 'itself', 'themselves', 'into', 'over', 'under'
        }
        
        # Extract words 4+ chars that aren't stopwords
        words = re.findall(r'\b\w{4,}\b', claim_text)
        specific_terms = [w for w in words if w.lower() not in stopwords]
        
        # Prioritize specific terms from claim_text OVER generic event terms
        positive_terms = specific_terms + positive_terms
        
        return {
            'positive': list(dict.fromkeys(positive_terms))[:15],  # Preserve order, dedupe
            'negative': list(set(negative_terms))[:10],
            'confidence': 0.8 if claim_type in ['trait', 'event'] else 0.6
        }


# Integration function
def enhance_fallback_in_decomposer():
    """Code to integrate SmartFallback into ClaimDecomposer"""
    return """
    # In ClaimDecomposer.__init__:
    from smart_fallback import SmartFallback
    self.smart_fallback = SmartFallback()
    
    # Replace _fallback_extraction method with:
    def _fallback_extraction(self, text: str, char_name: str) -> List[Dict]:
        claims = self.smart_fallback.extract_claims_smart(text, char_name)
        print(f"[SMART FALLBACK] Extracted {len(claims)} claims")
        return claims
    """
