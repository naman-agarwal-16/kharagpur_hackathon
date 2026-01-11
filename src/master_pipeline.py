from data_loader import DataLoader
from claim_decomposer import ClaimDecomposer
from novel_ingester import NovelIngester
from evidence_retriever import EvidenceRetriever
from consistency_checker import ConsistencyChecker
from pathlib import Path
from typing import List, Dict, Any
import json

class MasterPipeline:
    """
    End-to-end pipeline: backstory → final judgment
    """
    
    def __init__(self):
        self.loader = DataLoader()
        self.decomposer = ClaimDecomposer()
        self.checker = ConsistencyChecker()
        
        # Cache ingested novels
        self.ingesters = {}
    
    def process_example(self, story_id: int, split: str = "train") -> Dict[str, Any]:
        """Process a single example end-to-end"""
        
        print(f"\n{'='*70}")
        print(f"PROCESSING STORY ID: {story_id}")
        print(f"{'='*70}")
        
        # 1. Load data
        if split == "train":
            row = self.loader.train_df[self.loader.train_df['id'] == story_id].iloc[0]
            actual_label = row['label']
        else:
            row = self.loader.test_df[self.loader.test_df['id'] == story_id].iloc[0]
            actual_label = None
        
        character_name = row['char']
        backstory = row['content']
        book_name = row['book_name']
        
        print(f"Character: {character_name}")
        print(f"Novel: {book_name}")
        print(f"Backstory: {backstory[:100]}...")
        
        # 2. Decompose claims
        print(f"\n[1] DECOMPOSING CLAIMS...")
        try:
            claims = self.decomposer.decompose(backstory, character_name)
        except Exception as e:
            print(f"  ✗ Claim decomposition failed: {e}")
            return {
                'story_id': story_id,
                'prediction': 0,
                'confidence': 0.0,
                'rationale': f'Claim decomposition error: {str(e)}',
                'actual_label': actual_label
            }
        
        if not claims:
            print("  ✗ No claims generated (likely LLM rate limit or API issue)")
            return {
                'story_id': story_id,
                'prediction': 0,
                'confidence': 0.0,
                'rationale': 'Failed to generate claims - LLM unavailable',
                'actual_label': actual_label
            }
        
        print(f"  ✓ Generated {len(claims)} claims")
        
        # 3. Ingest novel (cached)
        print(f"\n[2] INGESTING NOVEL...")
        if book_name not in self.ingesters:
            novel_text = self.loader.load_novel(book_name)
            ingester = NovelIngester(novel_text, is_text=True)
            ingester.ingest()
            self.ingesters[book_name] = ingester
            print(f"  ✓ Ingested and cached: {book_name}")
        else:
            ingester = self.ingesters[book_name]
            print(f"  ✓ Using cached: {book_name}")
        
        # 4. Retrieve evidence for each claim
        print(f"\n[3] RETRIEVING EVIDENCE...")
        retriever = EvidenceRetriever(ingester)
        
        claim_verifications = []
        for i, claim in enumerate(claims[:5]):  # Process top 5 claims
            print(f"  Claim {i+1}: {claim['claim_id']}")
            
            # Create vocab dict for evidence retrieval
            vocab_dict = {
                'positive': claim['search_vocabulary'],
                'negative': claim['anti_vocabulary'],
                'patterns': claim['syntactic_patterns']
            }
            
            evidence = retriever.retrieve_evidence(character_name, vocab_dict)
            critical_moments = retriever.find_critical_moments(character_name, claim)
            
            # Add critical moments as negative evidence
            evidence['contradicting'].extend(critical_moments)
            
            print(f"    → {len(evidence['supporting'])} supporting, {len(evidence['contradicting'])} contradicting")
            
            # 5. Verify claim with LLM
            try:
                verification = self.checker.verify_claim(claim, evidence)
            except Exception as e:
                print(f"    ⚠ Verification failed, using fallback: {e}")
                verification = self.checker._fallback_verification(claim, evidence)
            
            claim_verifications.append({
                'claim': claim,
                'evidence': evidence,
                'verification': verification
            })
        
        # 6. Aggregate final decision
        print(f"\n[4] AGGREGATING DECISION...")
        final_prediction = self._aggregate_decisions(claim_verifications)
        
        result = {
            'story_id': story_id,
            'character_name': character_name,
            'book_name': book_name,
            'prediction': final_prediction['consistent'],
            'confidence': final_prediction['confidence'],
            'rationale': final_prediction['rationale'],
            'claim_verifications': claim_verifications,
            'actual_label': actual_label
        }
        
        # 7. Compare with ground truth
        if actual_label is not None:
            import math
            # Handle NaN values in actual_label
            if isinstance(actual_label, float) and math.isnan(actual_label):
                result['correct'] = None
                print(f"\n[5] VERDICT: ? (No ground truth label)")
                print(f"    Predicted: {'consistent' if final_prediction['consistent'] else 'inconsistent'}")
                print(f"    Confidence: {final_prediction['confidence']:.2f}")
                print(f"    Key rationale: {final_prediction['rationale'][:100]}...")
            else:
                is_correct = int(final_prediction['consistent']) == int(actual_label)
                result['correct'] = is_correct
                print(f"\n[5] VERDICT: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
                print(f"    Predicted: {'consistent' if final_prediction['consistent'] else 'inconsistent'}")
                print(f"    Actual: {'consistent' if actual_label else 'inconsistent'}")
                print(f"    Confidence: {final_prediction['confidence']:.2f}")
                print(f"    Key rationale: {final_prediction['rationale'][:100]}...")
        
        return result
    
    def _aggregate_decisions(self, claim_verifications: List[Dict]) -> Dict:
        """
        Aggregate multiple claim verifications into final judgment
        """
        
        total_confidence = 0
        contradiction_count = 0
        support_count = 0
        
        for cv in claim_verifications:
            verification = cv['verification']
            total_confidence += verification['confidence']
            
            if verification['consistent']:
                support_count += 1
            else:
                contradiction_count += 1
        
        # Decision rule: Any strong contradiction = inconsistency
        # Otherwise, majority vote weighted by confidence
        
        # Find strongest contradiction
        contradictions = [cv for cv in claim_verifications if not cv['verification']['consistent']]
        strongest_contra = max(contradictions, 
                              key=lambda x: x['verification']['confidence']) if contradictions else None
        
        # Fix 2: Require at least 2 strong contradictions to override
        strong_contras = [cv for cv in claim_verifications 
                         if not cv['verification']['consistent'] and cv['verification']['confidence'] > 0.85]
        if len(strong_contras) >= 2:
            return {
                'consistent': 0,
                'confidence': max(cv['verification']['confidence'] for cv in strong_contras),
                'rationale': f"Multiple strong contradictions ({len(strong_contras)}) found in claims"
            }
        
        # Fix 1: Raised threshold from 0.7 to 0.9 for single contradiction override
        if strongest_contra and strongest_contra['verification']['confidence'] > 0.9:
            # Strong contradiction overrides everything
            return {
                'consistent': 0,
                'confidence': strongest_contra['verification']['confidence'],
                'rationale': f"Strong contradiction found: {strongest_contra['verification']['rationale'][:150]}"
            }
        
        # Fix 2: Require multiple corroborations - check for strong evidence
        claim_scores = [cv['verification'] for cv in claim_verifications]
        strong_evidence = [cs for cs in claim_scores if cs['confidence'] > 0.7]
        
        # Count claims with good supporting evidence (not just high confidence)
        claims_with_support = [cv for cv in claim_verifications if cv['verification']['consistent'] == 1]
        claims_with_contra = [cv for cv in claim_verifications if cv['verification']['consistent'] == 0]
        
        # NEW LOGIC: If majority of claims have support and no strong contradictions → consistent
        if len(claims_with_support) >= len(claims_with_contra) and len(strong_contras) == 0:
            total_claims = len(claim_verifications)
            avg_confidence = sum(cv['verification']['confidence'] for cv in claim_verifications) / total_claims if total_claims > 0 else 0.5
            return {
                'consistent': 1,
                'confidence': avg_confidence,
                'rationale': f"{len(claims_with_support)}/{total_claims} claims supported by evidence"
            }
        
        # If we have more contradictions than support → inconsistent
        if len(claims_with_contra) > len(claims_with_support):
            total_claims = len(claim_verifications)
            avg_confidence = sum(cv['verification']['confidence'] for cv in claim_verifications) / total_claims if total_claims > 0 else 0.5
            return {
                'consistent': 0,
                'confidence': avg_confidence,
                'rationale': f"{len(claims_with_contra)}/{total_claims} claims contradicted"
            }
        
        # Fallback: majority vote
        total_claims = len(claim_verifications)
        if support_count > contradiction_count:
            confidence = total_confidence / total_claims if total_claims > 0 else 0.5
            return {
                'consistent': 1,
                'confidence': confidence,
                'rationale': f"{support_count}/{total_claims} claims supported by evidence"
            }
        else:
            confidence = total_confidence / total_claims if total_claims > 0 else 0.5
            return {
                'consistent': 0,
                'confidence': confidence,
                'rationale': f"{contradiction_count}/{total_claims} claims contradicted"
            }
    
    def batch_process(self, story_ids: List[int], split: str = "train") -> List[Dict]:
        """Process multiple examples and return results"""
        
        results = []
        for story_id in story_ids:
            try:
                result = self.process_example(story_id, split)
                results.append(result)
            except Exception as e:
                print(f"\n✗ ERROR processing story {story_id}: {e}")
                results.append({
                    'story_id': story_id,
                    'prediction': 0,
                    'confidence': 0.0,
                    'rationale': f'Processing error: {str(e)}',
                    'actual_label': None,
                    'correct': False
                })
        
        return results
    
    def generate_submission(self, output_path: str):
        """Generate results.csv for test set submission"""
        
        print(f"\n{'='*70}")
        print("GENERATING SUBMISSION FILE")
        print(f"{'='*70}")
        
        test_examples = self.loader.get_test_examples()
        results = []
        
        for example in test_examples:
            print(f"Processing {example['story_id']}...")
            
            try:
                result = self.process_example(example['story_id'], split="test")
                
                results.append({
                    'Story ID': example['story_id'],
                    'Prediction': result['prediction'],
                    'Rationale': result['rationale'][:200]  # Truncate if needed
                })
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                results.append({
                    'Story ID': example['story_id'],
                    'Prediction': 0,  # Conservative default
                    'Rationale': 'Error during processing'
                })
        
        # Save to CSV
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        
        print(f"\n✓ Submission saved to {output_path}")
        print(f"  Total examples: {len(results)}")


# Run batch test
def test_pipeline():
    """Test on first 5 training examples"""
    
    pipeline = MasterPipeline()
    
    # Test on first 5 examples (using IDs from the dataset)
    train_ids = pipeline.loader.train_df['id'].tolist()[:5]
    print(f"Testing on story IDs: {train_ids}")
    
    results = pipeline.batch_process(train_ids, split="train")
    
    # Calculate accuracy (only for results with ground truth)
    results_with_labels = [r for r in results if r.get('correct') is not None]
    correct = sum(1 for r in results_with_labels if r.get('correct', False))
    total = len(results_with_labels)
    
    print(f"\n{'='*70}")
    if total > 0:
        print(f"BATCH ACCURACY: {correct}/{total} = {correct/total*100:.1f}%")
    else:
        print(f"BATCH ACCURACY: No labeled examples to evaluate")
    print(f"Total processed: {len(results)} (labeled: {total}, unlabeled: {len(results) - total})")
    print(f"{'='*70}")
    
    # Show failures (only for labeled examples)
    failures = [r for r in results_with_labels if not r.get('correct', False)]
    if failures:
        print("\nFAILURE CASES:")
        for r in failures:
            print(f"  Story {r['story_id']}: {r['rationale'][:100]}")
    elif total > 0:
        print("\n✓ All labeled predictions correct!")
    
    return results

if __name__ == "__main__":
    test_pipeline()
