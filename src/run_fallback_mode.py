"""
Run the pipeline WITHOUT using LLM (fallback mode only)
Use this when you've hit API rate limits
"""

import os

# Set environment variable to skip LLM
os.environ['SKIP_LLM'] = 'true'

# Now import and run
from master_pipeline import MasterPipeline

def run_fallback_mode():
    """Run pipeline in fallback mode (no LLM calls)"""
    
    print("="*70)
    print("RUNNING IN FALLBACK MODE (NO LLM)")
    print("Using pattern-based claim extraction only")
    print("="*70)
    
    pipeline = MasterPipeline()
    
    # Test on first 5 examples
    train_ids = pipeline.loader.train_df['id'].tolist()[:5]
    print(f"\nTesting on story IDs: {train_ids}\n")
    
    results = pipeline.batch_process(train_ids, split="train")
    
    # Calculate accuracy
    correct = sum(r.get('correct', False) for r in results)
    print(f"\n{'='*70}")
    print(f"BATCH ACCURACY (Fallback Mode): {correct}/{len(results)} = {correct/len(results)*100:.1f}%")
    print(f"{'='*70}")
    
    # Show details
    print("\nRESULTS:")
    for r in results:
        status = "✓" if r.get('correct', False) else "✗"
        print(f"  {status} Story {r['story_id']}: Predicted={r['prediction']}, Actual={r.get('actual_label', 'N/A')}")
        print(f"     Rationale: {r['rationale'][:80]}...")
    
    return results

if __name__ == "__main__":
    results = run_fallback_mode()
