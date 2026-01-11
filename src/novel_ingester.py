import re
import json
from pathlib import Path
from typing import List, Dict, Any
import nltk
from collections import defaultdict

# Download required NLTK data (run once)
# nltk.download('punkt')

class NovelIngester:
    """
    Ingests a full novel text and creates searchable chunks with metadata
    """
    
    def __init__(self, novel_input: str, is_text: bool = False):
        """
        Args:
            novel_input: Either a file path or the novel text itself
            is_text: If True, novel_input is treated as text content, not a path
        """
        if is_text:
            self.novel_path = None
            self.raw_text = novel_input
        else:
            self.novel_path = Path(novel_input)
            self.raw_text = ""
        self.chunks = []
        self.character_positions = defaultdict(list)
        self.timeline = []
        
    def ingest(self, chunk_method: str = "chapter") -> List[Dict[str, Any]]:
        """
        Main entry point: Load and chunk the novel
        
        Args:
            chunk_method: "chapter" (splits by Chapter X), "scene" (blank line breaks), or "fixed" (word count)
            
        Returns:
            List of chunks with metadata
        """
        # Load raw text if not already loaded
        if not self.raw_text:
            print(f"[INGESTER] Loading novel from {self.novel_path}...")
            with open(self.novel_path, 'r', encoding='utf-8') as f:
                self.raw_text = f.read()
        else:
            print(f"[INGESTER] Using pre-loaded novel text...")
        
        print(f"[INGESTER] Loaded {len(self.raw_text):,} characters")
        
        # Chunk based on method
        if chunk_method == "chapter":
            self.chunks = self._chunk_by_chapter()
        elif chunk_method == "scene":
            self.chunks = self._chunk_by_scene()
        else:
            self.chunks = self._chunk_fixed_size()
        
        print(f"[INGESTER] Created {len(self.chunks)} chunks")
        
        # Process each chunk
        for idx, chunk in enumerate(self.chunks):
            self._process_chunk(chunk, idx)
        
        print(f"[INGESTER] Processed all chunks with character positions")
        return self.chunks
    
    def _chunk_by_chapter(self) -> List[Dict]:
        """Split by Chapter/Section headers (most reliable)"""
        
        # Pattern: Chapter X, Book X, Part X, or all caps headers
        chapter_patterns = [
            r'\n\s*(Chapter|CHAPTER)\s+\d+',
            r'\n\s*(Book|BOOK)\s+\d+',
            r'\n\s*(Part|PART)\s+\d+',
            r'\n\s*[A-Z]{2,50}\s*\n',  # ALL CAPS titles
        ]
        
        combined_pattern = '|'.join(chapter_patterns)
        split_points = list(re.finditer(combined_pattern, self.raw_text))
        
        if len(split_points) < 5:  # Not enough chapters, fallback to scene
            print("[INGESTER] Few chapter markers found, using scene chunking")
            return self._chunk_by_scene()
        
        chunks = []
        start_pos = 0
        
        for i, match in enumerate(split_points):
            end_pos = match.start()
            chunk_text = self.raw_text[start_pos:end_pos].strip()
            
            if len(chunk_text) > 500:  # Only keep substantial chunks
                chunks.append({
                    'id': f"ch_{i}",
                    'text': chunk_text,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'type': 'chapter'
                })
            
            start_pos = match.start()
        
        # Add final chunk
        final_chunk = self.raw_text[start_pos:].strip()
        if len(final_chunk) > 500:
            chunks.append({
                'id': f"ch_{len(split_points)}",
                'text': final_chunk,
                'start_pos': start_pos,
                'end_pos': len(self.raw_text),
                'type': 'chapter'
            })
        
        return chunks
    
    def _chunk_by_scene(self) -> List[Dict]:
        """Split by blank lines (scene breaks)"""
        
        # Split by double newlines (scene breaks)
        raw_chunks = re.split(r'\n\s*\n', self.raw_text)
        
        chunks = []
        pos = 0
        
        for i, chunk_text in enumerate(raw_chunks):
            chunk_len = len(chunk_text)
            
            if len(chunk_text) > 300:  # Filter out tiny fragments
                chunks.append({
                    'id': f"sc_{i}",
                    'text': chunk_text.strip(),
                    'start_pos': pos,
                    'end_pos': pos + chunk_len,
                    'type': 'scene'
                })
            
            pos += chunk_len + 2  # +2 for the newlines
        
        return chunks
    
    def _chunk_fixed_size(self, target_size: int = 1500) -> List[Dict]:
        """Fallback: Fixed-size chunks by word count"""
        
        words = self.raw_text.split()
        chunks = []
        
        for i in range(0, len(words), target_size):
            chunk_words = words[i:i + target_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'id': f"fk_{i//target_size}",
                'text': chunk_text,
                'start_pos': i,
                'end_pos': i + len(chunk_words),
                'type': 'fixed'
            })
        
        return chunks
    
    def _process_chunk(self, chunk: Dict, index: int):
        """Extract metadata from a single chunk"""
        
        text = chunk['text']
        
        # 1. Extract character mentions (simplified NER)
        # Look for capitalized words that could be names
        potential_names = re.findall(r'\b([A-Z][a-z]{2,20})\b', text)
        
        # Filter common words and keep likely character names
        common_words = {'The', 'But', 'However', 'Nevertheless', 'When', 'Where', 'How'}
        likely_characters = [name for name in potential_names if name not in common_words]
        
        # Keep unique names and count frequency
        char_counts = defaultdict(int)
        for name in likely_characters:
            char_counts[name] += 1
        
        # Keep names that appear multiple times (more likely to be characters)
        main_characters = {name for name, count in char_counts.items() if count > 1}
        
        chunk['characters'] = list(main_characters)
        chunk['char_count'] = len(text.split())
        
        # 2. Extract timeline markers (dates, ages, relative time)
        timeline_markers = []
        
        # Age mentions
        age_matches = re.finditer(r'\b(at age|age|when (he|she) was|turned)\s+(\d+)', text, re.IGNORECASE)
        for match in age_matches:
            timeline_markers.append({
                'type': 'age',
                'value': int(match.group(3)),
                'context': text[max(0, match.start()):match.start()+100]
            })
        
        # Date mentions
        date_patterns = [
            r'\b(1[0-9]{3}|20[0-9]{2})\b',  # Years 1000-2999
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}',
        ]
        
        for pattern in date_patterns:
            dates = re.findall(pattern, text)
            if dates:
                timeline_markers.extend([{'type': 'date', 'value': d} for d in dates])
        
        chunk['timeline_markers'] = timeline_markers
        
        # 3. Identify scene type (dialogue-heavy, action, introspection)
        dialogue_lines = len(re.findall(r'["\'][^"\']+["\']\s*(said|asked|replied|shouted)', text))
        action_verbs = len(re.findall(r'\b(ran|jumped|fought|attacked|walked|moved)\b', text, re.IGNORECASE))
        
        if dialogue_lines > 5:
            chunk['scene_type'] = 'dialogue'
        elif action_verbs > 3:
            chunk['scene_type'] = 'action'
        else:
            chunk['scene_type'] = 'introspection'
        
        # 4. Store position for quick retrieval
        for char in main_characters:
            self.character_positions[char].append(index)
    
    def save_processed_novel(self, output_path: str):
        """Save processed chunks for fast reloading"""
        
        processed_data = {
            'chunks': self.chunks,
            'character_positions': dict(self.character_positions),
            'total_chunks': len(self.chunks),
            'total_characters': len(self.character_positions)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2)
        
        print(f"[INGESTER] Saved processed novel to {output_path}")
    
    def load_processed_novel(self, input_path: str) -> List[Dict]:
        """Fast reload from processed file"""
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.chunks = data['chunks']
        self.character_positions = defaultdict(list, data['character_positions'])
        
        print(f"[INGESTER] Loaded {len(self.chunks)} chunks for {len(self.character_positions)} characters")
        return self.chunks
    
    def search_character(self, character_name: str, claim_vocab: Dict) -> List[Dict]:
        """
        Search all chunks for a specific character and claim vocabulary
        
        Returns: List of matching passages with scores
        """
        
        # Get character aliases for better matching
        aliases = self._find_character_aliases(character_name)
        
        matches = []
        
        # Get chunk indices where any alias appears
        chunk_indices = set()
        for alias in aliases:
            chunk_indices.update(self.character_positions.get(alias, []))
        
        for idx in chunk_indices:
            chunk = self.chunks[idx]
            text = chunk['text']
            
            # Score based on vocabulary matches
            score = 0
            matched_terms = []
            
            # Check positive vocabulary
            for term in claim_vocab.get('positive', []):
                if term.lower() in text.lower():
                    score += 1
                    matched_terms.append(term)
            
            # Check anti-vocabulary (higher weight)
            for term in claim_vocab.get('negative', []):
                if term.lower() in text.lower():
                    score -= 2  # Contradictions weigh more
                    matched_terms.append(f"CONTRADICTION: {term}")
            
            # Check syntactic patterns
            for pattern in claim_vocab.get('patterns', []):
                # Simple pattern matching (can be improved with regex)
                pattern_words = pattern.replace(character_name, '').split()
                if all(word in text for word in pattern_words if word not in ['+', '[', ']', 'verb', 'adjective']):
                    score += 1.5
            
            # FIX: Require at least 2 matched terms for evidence to count
            # Single term matches are too weak/generic
            if score > 0 and len(matched_terms) < 2:
                score = 0.5  # Weak evidence, barely counts
            
            if score != 0:  # Only keep meaningful matches
                matches.append({
                    'chunk_id': chunk['id'],
                    'text': text,
                    'score': score,
                    'adjusted_score': score,  # Add adjusted_score for compatibility
                    'matched_terms': matched_terms,
                    'scene_type': chunk.get('scene_type', 'unknown'),
                    'characters_in_scene': chunk['characters']
                })
        
        # Sort by score (highest first)
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        return matches[:20]  # Return top 20 matches
    
    def _find_character_aliases(self, primary_name: str) -> List[str]:
        """Find if character is referred to by other names"""
        aliases = [primary_name]
        
        # First name only
        parts = primary_name.split()
        if len(parts) > 1:
            aliases.append(parts[0])  # First name
            aliases.append(parts[-1])  # Last name
        
        # Titles (common in classic novels)
        if len(parts) > 1:
            last_name = parts[-1]
            aliases.extend([
                f"Mr. {last_name}",
                f"Mrs. {last_name}",
                f"Miss {last_name}",
                f"Dr. {last_name}",
                f"Lord {last_name}",
                f"Lady {last_name}"
            ])
        else:
            aliases.extend([
                f"Mr. {primary_name}",
                f"Mrs. {primary_name}",
                f"Miss {primary_name}"
            ])
        
        return list(set(aliases))


# Test function
def test_ingester():
    """Test with a sample novel text"""
    
    # Create a tiny sample novel
    sample_novel = """
Chapter 1

Sarah stood at the cliff edge, her heart pounding. The wind howled around her, but she forced herself to stay calm. Below, the waves crashed against sharp rocks. She had always feared water since the accident, but now she had to be brave. 

"Are you sure about this?" asked Thomas, her older mentor.

Sarah hesitated, then nodded. "I can do this," she whispered, though her hands shook.

Chapter 2

Two years ago, at age twelve, Sarah's parents had died. The car crash left her alone in the world, and she had learned to be self-reliant. Now fourteen, she still woke up screaming from nightmares about the accident.

Thomas found her one morning, shaking. "It's okay to be afraid," he said gently.

"I'm not afraid," Sarah lied, pulling away. She didn't trust anyone since her parents left her.

Chapter 3

When the village was attacked, Sarah didn't think. She grabbed the child and ran through the flames, her fear forgotten. Later, villagers called her brave, but she only felt empty.

Thomas watched her. "You're stronger than you know."

Sarah avoided his eyes. "I'm not strong. I'm just... surviving."
"""
    
    # Save sample to file
    sample_path = Path("D:/kharagpur_hackathon/data/sample_novel.txt")
    sample_path.parent.mkdir(exist_ok=True)
    sample_path.write_text(sample_novel, encoding='utf-8')
    
    # Ingest the novel
    ingester = NovelIngester(sample_path)
    chunks = ingester.ingest(chunk_method="chapter")
    
    print(f"\n[TEST] Processed {len(chunks)} chapters")
    print(f"[TEST] Found characters: {list(ingester.character_positions.keys())}")
    
    # Save processed version
    ingester.save_processed_novel("D:/kharagpur_hackathon/data/sample_novel_processed.json")
    
    # Test search
    print("\n" + "="*60)
    print("SEARCH TEST: 'Sarah brave'")
    print("="*60)
    
    # Use the brave claim from our earlier test
    brave_claim = {
        'positive': ['brave', 'courageous', 'heroic', 'fearless', 'stood his ground', 'faced danger', 'defended'],
        'negative': ['coward', 'cowardly', 'fled', 'ran away', 'hid', 'terrified', 'shaking with fear', 'too scared'],
        'patterns': []
    }
    
    matches = ingester.search_character("Sarah", brave_claim)
    for match in matches:
        print(f"\n[CHUNK {match['chunk_id']}] Score: {match['score']}")
        print(f"Text preview: {match['text'][:150]}...")
        print(f"Matched: {match['matched_terms']}")
    
    return ingester, chunks


if __name__ == "__main__":
    test_ingester()
