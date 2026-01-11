import pathway as pw
import re
from typing import List, Dict, Any

class PathwayNarrativeGraph:
    """
    Uses Pathway to build a living graph of character-state-timeline
    
    Note: This is a simplified implementation. Pathway's actual streaming
    capabilities would allow real-time updates as new text is ingested.
    """
    
    def __init__(self, novel_text: str):
        self.text = novel_text
        self.events = []
        self.character_states = {}
        
    def build_graph(self):
        """Build character-event-state graph"""
        
        print("[PATHWAY] Extracting events from narrative...")
        
        # 1. Extract events and entities (lightweight NLP)
        self.events = self._extract_events()
        
        print(f"[PATHWAY] Extracted {len(self.events)} events")
        
        # 2. Build character state transitions
        for i, event in enumerate(self.events):
            for char in event['characters']:
                if char not in self.character_states:
                    self.character_states[char] = []
                
                self.character_states[char].append({
                    'event_id': i,
                    'timestamp': event['timestamp'],
                    'action': event['action'],
                    'context': event['text']
                })
        
        print(f"[PATHWAY] Tracked {len(self.character_states)} characters")
        return self.character_states
    
    def query_character_trajectory(self, character: str) -> List[Dict]:
        """Get character's state evolution over time"""
        
        if character not in self.character_states:
            return []
        
        trajectory = self.character_states[character]
        return sorted(trajectory, key=lambda x: x['timestamp'])
    
    def check_causal_consistency(self, cause_event: str, effect_state: str) -> bool:
        """
        Check if cause can plausibly lead to effect using temporal ordering
        """
        # Find cause event
        cause_events = [e for e in self.events 
                       if cause_event.lower() in e['text'].lower()]
        
        if not cause_events:
            return False
        
        # Check if effect appears downstream
        cause_pos = cause_events[0]['position']
        effect_pos = self.text.lower().find(effect_state.lower(), cause_pos)
        
        return effect_pos > cause_pos
    
    def find_contradictions(self, character: str, trait: str) -> List[Dict]:
        """
        Find events where character acts contrary to claimed trait
        """
        contradictions = []
        
        trajectory = self.query_character_trajectory(character)
        
        # Simple trait-action incompatibility checking
        incompatible_actions = {
            'brave': ['fled', 'hid', 'cowered', 'ran away'],
            'kind': ['hurt', 'killed', 'attacked', 'insulted'],
            'honest': ['lied', 'deceived', 'tricked', 'cheated'],
            'loyal': ['betrayed', 'abandoned', 'deserted']
        }
        
        if trait.lower() not in incompatible_actions:
            return contradictions
        
        incompatible = incompatible_actions[trait.lower()]
        
        for state in trajectory:
            action = state['action'].lower()
            if any(inc in action for inc in incompatible):
                contradictions.append({
                    'timestamp': state['timestamp'],
                    'action': action,
                    'context': state['context'],
                    'contradiction': f"Claimed {trait} but {action}"
                })
        
        return contradictions
    
    def _extract_events(self) -> List[Dict]:
        """Extract structured events from text (simplified)"""
        
        # Look for sentences with action verbs and character names
        sentences = re.split(r'[.!?]', self.text)
        events = []
        
        for i, sent in enumerate(sentences[:1000]):  # Limit for performance
            sent = sent.strip()
            if len(sent) < 50:
                continue
            
            # Find capitalized names (simple heuristic)
            names = re.findall(r'\b([A-Z][a-z]{2,})\b', sent)
            names = [n for n in names if n not in {'The', 'But', 'And', 'When', 'Where', 'How', 'That'}]
            
            # Find action verbs
            action_verbs = re.findall(
                r'\b(killed|saved|ran|fought|loved|hated|feared|attacked|defended|'
                r'helped|betrayed|fled|hid|protected|destroyed|built|created)\b', 
                sent, re.IGNORECASE
            )
            
            if names and action_verbs:
                events.append({
                    'text': sent,
                    'timestamp': i,  # Sentence index as timestamp
                    'characters': names[:3],  # Max 3 characters per event
                    'action': action_verbs[0] if action_verbs else 'unknown',
                    'position': self.text.find(sent)
                })
        
        return events
    
    def get_character_summary(self, character: str) -> Dict[str, Any]:
        """Get summary statistics for a character"""
        
        trajectory = self.query_character_trajectory(character)
        
        if not trajectory:
            return {}
        
        # Count action types
        action_counts = {}
        for state in trajectory:
            action = state['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            'character': character,
            'total_events': len(trajectory),
            'first_appearance': trajectory[0]['timestamp'],
            'last_appearance': trajectory[-1]['timestamp'],
            'action_distribution': action_counts,
            'most_common_action': max(action_counts.items(), key=lambda x: x[1])[0] if action_counts else 'none'
        }


# Test
def test_pathway_graph():
    """Test graph building"""
    
    sample_text = """
    John was a brave man. He fought the dragon and saved the princess.
    Later, John became king and ruled wisely. His bravery was legendary.
    Sarah joined John in his quest. She fought alongside him bravely.
    But later, Sarah fled from danger when the orcs attacked.
    """
    
    print("="*60)
    print("PATHWAY NARRATIVE GRAPH TEST")
    print("="*60)
    
    processor = PathwayNarrativeGraph(sample_text)
    graph = processor.build_graph()
    
    # Query trajectory
    print("\n[John's Trajectory]")
    trajectory = processor.query_character_trajectory('John')
    for state in trajectory:
        print(f"  {state['timestamp']}: {state['action']} - {state['context'][:50]}...")
    
    print("\n[Sarah's Trajectory]")
    trajectory = processor.query_character_trajectory('Sarah')
    for state in trajectory:
        print(f"  {state['timestamp']}: {state['action']} - {state['context'][:50]}...")
    
    # Check causality
    print("\n[Causal Consistency Check]")
    consistent = processor.check_causal_consistency('fought', 'king')
    print(f"  'fought' → 'king': {consistent}")
    
    # Find contradictions
    print("\n[Contradiction Detection]")
    contradictions = processor.find_contradictions('Sarah', 'brave')
    if contradictions:
        print(f"  Found {len(contradictions)} contradictions for Sarah being brave:")
        for c in contradictions:
            print(f"    - {c['contradiction']}: {c['context'][:60]}...")
    else:
        print("  No contradictions found")
    
    # Character summary
    print("\n[Character Summary - John]")
    summary = processor.get_character_summary('John')
    print(f"  Total events: {summary['total_events']}")
    print(f"  Most common action: {summary['most_common_action']}")
    print(f"  Action distribution: {summary['action_distribution']}")

if __name__ == "__main__":
    test_pathway_graph()
