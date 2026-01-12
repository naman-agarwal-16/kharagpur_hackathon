"""
Clean Master Pipeline - Orchestrates the complete narrative consistency verification
"""
import pandas as pd
from typing import Dict, List, Any

from config import RESULTS_DIR
from data_loader import DataLoader
from claim_decomposer import ClaimDecomposer
from novel_ingester import NovelIngester
from evidence_retriever import EvidenceRetriever
from consistency_checker import ConsistencyChecker


class NarrativeConsistencyPipeline:
    """
    End-to-end pipeline for narrative consistency verification
    """
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.decomposer = ClaimDecomposer()
        self.retriever = EvidenceRetriever()
        self.checker = ConsistencyChecker()
        
        # Load data
        self.train_df, self.test_df = self.data_loader.load_backstories()
        self.novel_texts = self.data_loader.load_novels()
        self.novel_chunks_cache = {}  # Cache chunked novels
    
    def process_single_story(self, story_id: int, backstory: str, 
                            character: str, novel_name: str, 
                            actual_label: Any = None) -> Dict[str, Any]:
        """
        Process a single backstory through the complete pipeline
        
        Args:
            story_id: Unique identifier
            backstory: Character's backstory text
            character: Character name
            novel_name: Novel title
            actual_label: Ground truth label (optional, for training)
            
        Returns:
            Dictionary with prediction, confidence, and details
        """
        print(f"\n{'='*70}")
        print(f"PROCESSING STORY ID: {story_id}")
        print(f"{'='*70}")
        print(f"Character: {character}")
        print(f"Novel: {novel_name}")
        print(f"Backstory: {backstory[:100]}...")
        
        try:
            # Step 1: Decompose backstory into claims
            print(f"\n[1] DECOMPOSING CLAIMS...")
            claims = self.decomposer.decompose(backstory, character)
            
            if not claims:
                print("  ✗ No claims extracted")
                return self._default_prediction(story_id, actual_label, "No claims extracted")
            
            print(f"  ✓ Generated {len(claims)} claims")
            
            # Step 2: Ingest novel
            print(f"\n[2] INGESTING NOVEL...")
            novel_key = novel_name.lower()
            
            if novel_key not in self.novel_texts:
                print(f"  ✗ Novel not found: {novel_name}")
                return self._default_prediction(story_id, actual_label, "Novel not found")
            
            # Check cache first
            if novel_key in self.novel_chunks_cache:
                print(f"  ✓ Using cached: {novel_name}")
                chunks = self.novel_chunks_cache[novel_key]
            else:
                novel_text = self.novel_texts[novel_key]
                ingester = NovelIngester(novel_text, is_text=True)
                chunks = ingester.ingest(chunk_method="chapter")
                self.novel_chunks_cache[novel_key] = chunks  # Cache for reuse
            
            if not chunks:
                print(f"  ✗ Failed to chunk novel")
                return self._default_prediction(story_id, actual_label, "Novel chunking failed")
            
            print(f"  ✓ Ingested {len(chunks)} chunks")
            
            # Step 3: Retrieve evidence for each claim
            print(f"\n[3] RETRIEVING EVIDENCE...")
            claim_verifications = []
            
            for claim in claims:
                claim_id = claim.get('claim_id', 'unknown')
                claim_text = claim.get('claim_text', '')
                
                # Get evidence
                evidence = self.retriever.retrieve(
                    claim=claim,
                    character_name=character,
                    novel_chunks=chunks,
                    top_k=10
                )
                
                supporting = [e for e in evidence if e.get('type') == 'supporting']
                contradicting = [e for e in evidence if e.get('type') == 'contradicting']
                
                print(f"  Claim {len(claim_verifications)+1}: {claim_id}")
                print(f"    → {len(supporting)} supporting, {len(contradicting)} contradicting")
                
                # Verify claim
                if evidence:
                    verification = self.checker.verify_claim(claim, evidence)
                else:
                    # No evidence found - default to inconsistent with low confidence
                    verification = {
                        'consistent': 0,
                        'confidence': 0.3,
                        'rationale': 'No evidence found'
                    }
                
                claim_verifications.append({
                    'claim': claim,
                    'evidence': evidence,
                    'verification': verification
                })
            
            # Step 4: Aggregate results
            print(f"\n[4] AGGREGATING DECISION...")
            final_prediction = self._aggregate_verifications(claim_verifications)
            
            # Step 5: Display result
            print(f"\n[5] VERDICT: ", end="")
            
            if actual_label is not None:
                is_correct = (final_prediction['prediction'] == int(actual_label))
                print(f"{'✓ CORRECT' if is_correct else '✗ WRONG'}")
                print(f"    Predicted: {'consistent' if final_prediction['prediction'] == 1 else 'inconsistent'}")
                print(f"    Actual: {'consistent' if int(actual_label) == 1 else 'inconsistent'}")
            else:
                print(f"{'CONSISTENT' if final_prediction['prediction'] == 1 else 'INCONSISTENT'}")
            
            print(f"    Confidence: {final_prediction['confidence']:.2f}")
            print(f"    Key rationale: {final_prediction['rationale'][:100]}...")
            
            return {
                'id': story_id,
                'prediction': final_prediction['prediction'],
                'confidence': final_prediction['confidence'],
                'rationale': final_prediction['rationale'],
                'num_claims': len(claims),
                'actual_label': actual_label,
                'claim_verifications': claim_verifications
            }
            
        except Exception as e:
            print(f"\n[ERROR] Pipeline failed: {e}")
            return self._default_prediction(story_id, actual_label, f"Pipeline error: {str(e)}")
    
    def _aggregate_verifications(self, claim_verifications: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate multiple claim verifications into final prediction
        
        Strategy: Weighted voting based on confidence
        """
        if not claim_verifications:
            return {
                'prediction': 0,
                'confidence': 0.5,
                'rationale': 'No claims to verify'
            }
        
        # Calculate weighted consistency score
        total_weight = 0
        weighted_consistency = 0
        
        for cv in claim_verifications:
            verification = cv['verification']
            confidence = verification.get('confidence', 0.5)
            is_consistent = verification.get('consistent', 0)
            
            weighted_consistency += is_consistent * confidence
            total_weight += confidence
        
        # Average consistency score
        if total_weight > 0:
            avg_consistency = weighted_consistency / total_weight
        else:
            avg_consistency = 0.5
        
        # Make binary decision
        # If more than 50% consistent → predict consistent (1), else inconsistent (0)
        prediction = 1 if avg_consistency >= 0.5 else 0
        
        # Count contradicted claims for rationale
        contradicted_count = sum(1 for cv in claim_verifications 
                                if cv['verification'].get('consistent', 0) == 0)
        total_claims = len(claim_verifications)
        
        rationale = f"{contradicted_count}/{total_claims} claims contradicted (avg score: {avg_consistency:.2f})"
        
        return {
            'prediction': prediction,
            'confidence': abs(avg_consistency - 0.5) * 2,  # 0.5 maps to 0, 0 or 1 maps to 1
            'rationale': rationale
        }
    
    def _default_prediction(self, story_id: int, actual_label: Any, reason: str) -> Dict[str, Any]:
        """Return default prediction when pipeline fails"""
        return {
            'id': story_id,
            'prediction': 0,  # Default to inconsistent
            'confidence': 0.5,
            'rationale': reason,
            'num_claims': 0,
            'actual_label': actual_label
        }
    
    def generate_submission(self, output_path: str = None) -> pd.DataFrame:
        """
        Generate predictions for test set and save to CSV
        
        Args:
            output_path: Where to save results (defaults to results/submission.csv)
            
        Returns:
            DataFrame with id and label columns
        """
        if output_path is None:
            output_path = f"{RESULTS_DIR}/submission.csv"
        
        print("\n" + "="*70)
        print("GENERATING TEST SET PREDICTIONS (First 5 cases)")
        print("="*70)
        
        results = []
        
        for _, row in self.test_df.head(5).iterrows():
            result = self.process_single_story(
                story_id=row['id'],
                backstory=row['content'],
                character=row['char'],
                novel_name=row['book_name']
            )
            results.append({
                'id': row['id'],
                'label': result['prediction']
            })
        
        # Save to CSV
        submission_df = pd.DataFrame(results)
        submission_df.to_csv(output_path, index=False)
        
        print(f"\n✓ Submission saved to: {output_path}")
        return submission_df


if __name__ == "__main__":
    """
    Run the pipeline directly on test set
    """
    print("\n" + "="*70)
    print("STARTING NARRATIVE CONSISTENCY VERIFICATION PIPELINE")
    print("="*70)
    
    pipeline = NarrativeConsistencyPipeline()
    submission_df = pipeline.generate_submission()
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"Predictions: {len(submission_df)}")
    print(f"Output: {RESULTS_DIR / 'submission.csv'}")
