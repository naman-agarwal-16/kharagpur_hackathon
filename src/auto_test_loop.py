"""
Clean Auto-Test Loop - Autonomous continuous testing with resume capability
"""
import os
import time
from datetime import datetime
from typing import Set

from config import LOGS_DIR, TEST_BATCH_SIZE, AUTO_WAIT_ON_RATE_LIMIT, RATE_LIMIT_WAIT_HOURS
from master_pipeline import NarrativeConsistencyPipeline


class AutoTestLoop:
    """
    Autonomous testing system that:
    - Tests training examples one by one
    - Saves progress continuously
    - Can resume from where it left off
    - Handles API rate limits gracefully
    """
    
    def __init__(self):
        self.pipeline = NarrativeConsistencyPipeline()
        self.log_file = os.path.join(LOGS_DIR, "auto_test_results.txt")
        
        # Ensure log directory exists
        os.makedirs(LOGS_DIR, exist_ok=True)
    
    def run_forever(self):
        """
        Run continuous testing until all examples are processed
        Automatically waits and retries if API rate limits hit
        """
        print("\n" + "="*70)
        print("🤖 AUTOMATIC TESTING MODE STARTED")
        print("="*70)
        print("[AUTO] System will run continuously until all examples tested")
        print("[AUTO] Press Ctrl+C to stop")
        print("="*70 + "\n")
        
        while True:
            try:
                api_limit_hit = self.run_batch(batch_size=TEST_BATCH_SIZE)
                
                # Check if all examples are tested
                tested_ids = self._load_tested_ids()
                total_examples = len(self.pipeline.train_df)
                remaining = total_examples - len(tested_ids)
                
                if remaining == 0:
                    print("\n" + "="*70)
                    print("✓ ALL TRAINING EXAMPLES TESTED!")
                    print("="*70)
                    
                    # Generate submission for test set
                    print("\n[AUTO] Generating test set predictions...")
                    self.pipeline.generate_submission()
                    
                    print("\n[AUTO] ✓ Complete! Check results/ for submission.csv")
                    break
                
                # If API rate limit hit, wait and retry
                if api_limit_hit and AUTO_WAIT_ON_RATE_LIMIT:
                    self.wait_for_quota_reset(RATE_LIMIT_WAIT_HOURS)
                elif api_limit_hit:
                    print("\n[AUTO] API rate limit hit. Exiting. Re-run to resume.")
                    break
                    
            except KeyboardInterrupt:
                print("\n\n[AUTO] ⚠️ Stopped by user")
                print(f"[AUTO] Progress saved to {self.log_file}")
                break
            except Exception as e:
                print(f"\n[AUTO] ✗ Unexpected error: {e}")
                print("[AUTO] Waiting 60 seconds before retry...")
                time.sleep(60)
    
    def run_batch(self, batch_size: int = 5) -> bool:
        """
        Run a batch of test examples
        
        Returns:
            True if API rate limit was hit, False otherwise
        """
        # Load already tested IDs
        tested_ids = self._load_tested_ids()
        
        # Get untested examples
        untested_df = self.pipeline.train_df[~self.pipeline.train_df['id'].isin(tested_ids)]
        
        if len(untested_df) == 0:
            print("[AUTO] All examples already tested!")
            return False
        
        print("\n" + "="*70)
        print(f"[AUTO] Testing batch of {batch_size} examples...")
        print(f"[AUTO] Remaining: {len(untested_df)} / {len(self.pipeline.train_df)}")
        print("="*70 + "\n")
        
        # Process batch
        batch_df = untested_df.head(batch_size)
        results = []
        api_limit_hit = False
        
        for _, row in batch_df.iterrows():
            story_id = row['id']
            
            try:
                result = self.pipeline.process_single_story(
                    story_id=story_id,
                    backstory=row['content'],
                    character=row['char'],
                    novel_name=row['book_name'],
                    actual_label=row['label']
                )
                
                # Save result immediately
                self._save_result(result)
                results.append(result)
                
                print(f"[AUTO] ✓ Completed {len(tested_ids) + len(results)}/{len(self.pipeline.train_df)}")
                
            except Exception as e:
                error_msg = str(e)
                print(f"[AUTO] ✗ Error on story {story_id}: {e}")
                
                # Check if it's a rate limit error
                if '429' in error_msg or 'quota' in error_msg.lower() or 'rate limit' in error_msg.lower():
                    api_limit_hit = True
                    break
                
                # Save failed result
                self._save_result({
                    'id': story_id,
                    'prediction': 0,
                    'actual_label': row['label'],
                    'confidence': 0.5,
                    'rationale': f"Error: {str(e)[:100]}"
                })
                results.append({'id': story_id, 'prediction': 0, 'actual_label': row['label']})
        
        # Calculate and print batch accuracy
        if results:
            correct = sum(1 for r in results 
                         if r.get('actual_label') is not None 
                         and r['prediction'] == int(r['actual_label']))
            accuracy = correct / len(results) * 100 if results else 0
            print(f"\n[AUTO] Batch Accuracy: {correct}/{len(results)} = {accuracy:.1f}%")
        
        return api_limit_hit
    
    def wait_for_quota_reset(self, wait_hours: int = 12):
        """Wait for API quota to reset"""
        print("\n" + "="*70)
        print(f"⏰ API RATE LIMIT HIT - Waiting {wait_hours} hours for quota reset")
        print("="*70)
        
        wait_seconds = wait_hours * 3600
        end_time = time.time() + wait_seconds
        
        while time.time() < end_time:
            remaining = int(end_time - time.time())
            hours = remaining // 3600
            minutes = (remaining % 3600) // 60
            print(f"\r[AUTO] Time remaining: {hours}h {minutes}m", end="", flush=True)
            time.sleep(60)  # Update every minute
        
        print("\n[AUTO] ✓ Quota should be reset. Resuming testing...")
    
    def _load_tested_ids(self) -> Set[int]:
        """Load IDs of already tested examples from log file"""
        tested_ids = set()
        
        if not os.path.exists(self.log_file):
            return tested_ids
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith("Story"):
                        # Extract ID from line like "Story 1: Pred=0, Actual=1.0..."
                        parts = line.split(':')[0].split()
                        if len(parts) >= 2:
                            story_id = int(parts[1])
                            tested_ids.add(story_id)
        except Exception as e:
            print(f"[WARN] Error loading tested IDs: {e}")
        
        return tested_ids
    
    def _save_result(self, result: dict):
        """Append result to log file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_line = (
            f"Story {result['id']}: "
            f"Pred={result['prediction']}, "
            f"Actual={result.get('actual_label', 'N/A')}, "
            f"Conf={result.get('confidence', 0.5):.2f}, "
            f"Rationale={result.get('rationale', 'N/A')[:50]}, "
            f"Time={timestamp}\n"
        )
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_line)
