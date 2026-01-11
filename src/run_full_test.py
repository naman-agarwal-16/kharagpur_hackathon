"""
Comprehensive test to verify all pipeline fixes work correctly
"""
from master_pipeline import MasterPipeline

def run_comprehensive_test(num_examples=40):
    """Test on more examples to validate fixes"""
    
    print("="*70)
    print("COMPREHENSIVE PIPELINE TEST")
    print("="*70)
    
    pipeline = MasterPipeline()
    
    # Get training IDs to test
    train_ids = pipeline.loader.train_df['id'].tolist()[:num_examples]
    print(f'\nTesting on {len(train_ids)} stories...\n')
    
    results = pipeline.batch_process(train_ids, split='train')
    
    # Calculate accuracy (only for results with ground truth)
    results_with_labels = [r for r in results if r.get('correct') is not None]
    correct = sum(1 for r in results_with_labels if r.get('correct', False))
    total = len(results_with_labels)
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    if total > 0:
        print(f"ACCURACY: {correct}/{total} = {correct/total*100:.1f}%")
    else:
        print("No labeled examples to evaluate")
    
    print(f"Total processed: {len(results)}")
    print(f"  - Labeled: {total}")
    print(f"  - Unlabeled: {len(results) - total}")
    
    # Show failures
    failures = [r for r in results_with_labels if not r.get('correct', False)]
    if failures:
        print(f"\nFAILURE CASES ({len(failures)}):")
        for r in failures:
            print(f"  Story {r['story_id']}: {r['rationale'][:80]}...")
    elif total > 0:
        print("\n✓ All labeled predictions correct!")
    
    # Show errors (processing failures)
    errors = [r for r in results if 'error' in r.get('rationale', '').lower()]
    if errors:
        print(f"\nPROCESSING ERRORS ({len(errors)}):")
        for r in errors:
            print(f"  Story {r['story_id']}: {r['rationale']}")
    
    print("="*70)
    return results

if __name__ == "__main__":
    run_comprehensive_test(5)  # Start small to test OpenRouter
