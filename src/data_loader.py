import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Any

class DataLoader:
    """Handles loading and iterating over the Kharagpur dataset"""
    
    def __init__(self, data_dir: str = "D:/kharagpur_hackathon/data"):
        self.data_dir = Path(data_dir)
        self.novels_dir = self.data_dir / "novels"
        
        # Load metadata
        self.train_df = pd.read_csv(self.data_dir / "train.csv")
        self.test_df = pd.read_csv(self.data_dir / "test.csv")
        
        # Map story_id to novel file (use actual book_name from CSV)
        self.novel_files = {}
        for f in self.novels_dir.iterdir():
            if f.suffix == '.txt':
                # Normalize name for matching
                normalized = f.stem.lower().replace('_', ' ').replace('-', ' ')
                self.novel_files[normalized] = f
        
        # Convert labels to binary (handle both formats)
        self.train_df['label'] = self.train_df['label'].map({
            'consistent': 1, 
            'inconsistent': 0,
            'contradict': 0  # FIX: CSV uses "contradict" not "inconsistent"
        })
        
        print(f"[LOADER] Found {len(self.train_df)} training examples")
        print(f"[LOADER] Found {len(self.test_df)} test examples")
        print(f"[LOADER] Found {len(self.novel_files)} novels: {list(self.novel_files.keys())}")
    
    def load_backstories(self):
        """Load training and test dataframes"""
        return self.train_df, self.test_df
    
    def load_novels(self):
        """Load all novels into a dictionary"""
        novels = {}
        for key, path in self.novel_files.items():
            print(f"[LOADER] Loading novel: {path.name} ({path.stat().st_size // 1024} KB)")
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Clean Project Gutenberg header/footer
            text = self._clean_gutenberg_text(text)
            novels[key] = text
            print(f"[LOADER] Loaded {len(text):,} characters")
        
        return novels
    
    def _clean_gutenberg_text(self, text: str) -> str:
        """Remove Project Gutenberg header and footer"""
        # Remove header
        header_patterns = [
            r'\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*',
            r'START OF THIS PROJECT GUTENBERG.*?(?=\n\n)',
        ]
        for pattern in header_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                text = text[match.end():]
                print(f"[LOADER] Removed header using pattern: {pattern[:50]}...")
                break
        
        # Remove footer
        footer_patterns = [
            r'\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*',
            r'End of (the )?Project Gutenberg.*',
        ]
        for pattern in footer_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                text = text[:match.start()]
                print(f"[LOADER] Removed footer using pattern: {pattern[:50]}...")
                break
        
        return text.strip()
    
    def get_training_examples(self) -> List[Dict[str, Any]]:
        """Load all training examples"""
        examples = []
        for _, row in self.train_df.iterrows():
            examples.append({
                'story_id': row['id'],
                'book_name': row['book_name'],
                'character_name': row['char'],
                'backstory': row['content'],  # 'content' column has the backstory
                'label': row['label'],
                'novel_id': self._normalize_novel_name(row['book_name'])
            })
        return examples
    
    def get_test_examples(self) -> List[Dict[str, Any]]:
        """Load all test examples (no labels)"""
        examples = []
        for _, row in self.test_df.iterrows():
            examples.append({
                'story_id': row['id'],
                'book_name': row['book_name'],
                'character_name': row['char'],
                'backstory': row['content'],
                'novel_id': self._normalize_novel_name(row['book_name'])
            })
        return examples
    
    def _normalize_novel_name(self, book_name: str) -> str:
        """Convert book_name to match filename format"""
        normalized = book_name.lower().replace(' ', '_').replace('-', '_')
        return normalized
    
    def load_novel(self, book_name: str) -> str:
        """Load full text of a novel by book_name"""
        normalized = book_name.lower().replace(' ', '_').replace('-', '_')
        
        # Try direct match
        if normalized in self.novel_files:
            novel_path = self.novel_files[normalized]
        else:
            # Search for partial match
            matches = [k for k in self.novel_files.keys() if normalized.replace('_', '').replace(' ', '') in k.replace(' ', '').replace('_', '')]
            if matches:
                novel_path = self.novel_files[matches[0]]
            else:
                raise ValueError(f"Novel '{book_name}' not found. Available: {list(self.novel_files.keys())}")
        
        print(f"[LOADER] Loading novel: {novel_path.name} ({novel_path.stat().st_size / 1024:.0f} KB)")
        
        # Load and strip Gutenberg headers
        text = novel_path.read_text(encoding='utf-8', errors='ignore')
        return self._strip_gutenberg_headers(text)
    
    def _strip_gutenberg_headers(self, text: str) -> str:
        """Remove Project Gutenberg header/footer from text - handles multiple formats"""
        import re
        
        # Common start patterns (various formats)
        start_patterns = [
            r'\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*',
            r'\*\*\*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*',
            r'The Project Gutenberg eBook of.*?\n\n',
            r'Note: Project Gutenberg also has an HTML version.*?\n\n',
            r'This eBook is for the use of.*?\n\n',
            r'\*END\*THE SMALL PRINT.*?\n\n',  # Older format
        ]
        
        # Common end patterns
        end_patterns = [
            r'\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*',
            r'\*\*\*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*',
            r'End of the Project Gutenberg.*',
            r'End of Project Gutenberg.*',
        ]
        
        # Try to find and remove header
        header_removed = False
        for pattern in start_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                text = text[match.end():].strip()
                header_removed = True
                print(f"[LOADER] Removed header using pattern: {pattern[:50]}...")
                break
        
        # Try to find and remove footer
        for pattern in end_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                text = text[:match.start()].strip()
                print(f"[LOADER] Removed footer using pattern: {pattern[:50]}...")
                break
        
        # If no header found, try heuristic: look for first chapter marker
        if not header_removed:
            chapter_match = re.search(r'\n\s*(Chapter|CHAPTER|Book|BOOK)\s+\d+', text)
            if chapter_match:
                text = text[chapter_match.start():].strip()
                print("[LOADER] Used chapter marker as start point")
        
        return text


# Quick test
if __name__ == "__main__":
    loader = DataLoader()
    
    train_examples = loader.get_training_examples()
    print(f"\nFirst training example:")
    print(f"  ID: {train_examples[0]['story_id']}")
    print(f"  Book: {train_examples[0]['book_name']}")
    print(f"  Character: {train_examples[0]['character_name']}")
    print(f"  Backstory preview: {train_examples[0]['backstory'][:200]}...")
    print(f"  Label: {'Consistent' if train_examples[0]['label'] == 1 else 'Inconsistent'}")
    
    # Try loading a novel
    if train_examples:
        try:
            novel_text = loader.load_novel(train_examples[0]['book_name'])
            print(f"\nNovel text preview: {novel_text[:300]}...")
        except Exception as e:
            print(f"\nError loading novel: {e}")
