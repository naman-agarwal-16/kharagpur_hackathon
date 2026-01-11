"""
Automatic Testing Loop - Runs tests continuously, waiting for API quota resets
Logs all results and generates submission when ready
"""

import time
import os
from datetime import datetime
from pathlib import Path
from master_pipeline import MasterPipeline
from data_loader import DataLoader

class AutoTester:
    """Runs tests automatically, handles API limits, logs progress"""
    
    def __init__(self):
        self.log_dir = Path("D:/kharagpur_hackathon/logs")
        self.log_dir.mkdir(exist_ok=True)
        
        self.pipeline = MasterPipeline()
        self.loader = DataLoader()
        
        # Load all data once
        print("[AUTO] Loading all training data...")
        self.train_df = self.loader.train_df
        self.test_df = self.loader.test_df
        
        print(f"[AUTO] Loaded {len(self.train_df)} training examples")
        print(f"[AUTO] Loaded {len(self.test_df)} test examples")
        
        # Track progress
        self.results_file = self.log_dir / "auto_test_results.txt"
        self.tested_ids = set()
        self.load_progress()
    
    def load_progress(self):
        """Load previously tested IDs"""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                for line in f:
                    if line.startswith("Story"):
                        try:
                            story_id = int(line.split()[1].rstrip(':'))
                            self.tested_ids.add(story_id)
                        except:
                            pass
            print(f"[AUTO] Resumed - already tested {len(self.tested_ids)} examples")
    
    def log_result(self, story_id, predicted, actual, confidence, rationale):
        """Append result to log file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.results_file, 'a') as f:
            f.write(f"\n[{timestamp}] Story {story_id}: Pred={predicted}, Actual={actual}, Conf={confidence:.2f}\n")
            f.write(f"  Rationale: {rationale}\n")
    
    def run_batch(self, batch_size=5):
        """Run a batch of tests (reduced to avoid rate limits)"""
        
        # Get untested IDs
        all_ids = set(self.train_df['id'].values)
        remaining = list(all_ids - self.tested_ids)
        
        if not remaining:
            print("[AUTO] ✓ All training examples tested!")
            return True
        
        # Take next batch
        batch = remaining[:batch_size]
        
        print(f"\n{'='*70}")
        print(f"[AUTO] Testing batch of {len(batch)} examples...")
        print(f"[AUTO] Remaining: {len(remaining)} / {len(all_ids)}")
        print(f"{'='*70}\n")
        
        results = []
        api_limit_hit = False
        
        for story_id in batch:
            try:
                result = self.pipeline.process_example(story_id)
                
                # Check if we hit rate limit
                if 'rate_limited' in str(result.get('rationale', '')).lower():
                    print(f"[AUTO] ⚠️ Rate limit detected on story {story_id}")
                    api_limit_hit = True
                    break
                
                # Log result
                self.log_result(
                    story_id,
                    result['prediction'],
                    result.get('actual_label', 'unknown'),
                    result['confidence'],
                    result['rationale']
                )
                
                self.tested_ids.add(story_id)
                results.append(result)
                
                print(f"[AUTO] ✓ Completed {len(self.tested_ids)}/{len(all_ids)}")
                
                # Delay between tests to avoid rate limits
                time.sleep(5)
                
            except Exception as e:
                print(f"[AUTO] ✗ Error on story {story_id}: {e}")
                if '429' in str(e) or 'quota' in str(e).lower():
                    api_limit_hit = True
                    break
        
        # Calculate accuracy
        if results:
            correct = sum(1 for r in results 
                         if r.get('actual_label') is not None and 
                         r['prediction'] == int(r['actual_label']))
            accuracy = correct / len(results) * 100 if results else 0
            print(f"\n[AUTO] Batch Accuracy: {correct}/{len(results)} = {accuracy:.1f}%")
        
        return api_limit_hit
    
    def wait_for_quota_reset(self, wait_hours=12):
        """Wait for API quota to reset"""
        wait_seconds = wait_hours * 3600
        
        print(f"\n{'='*70}")
        print(f"[AUTO] 😴 API quota exhausted. Waiting {wait_hours} hours for reset...")
        print(f"[AUTO] Will resume at {datetime.now().replace(hour=(datetime.now().hour + wait_hours) % 24).strftime('%I:%M %p')}")
        print(f"{'='*70}\n")
        
        # Log to file
        with open(self.results_file, 'a') as f:
            f.write(f"\n[{datetime.now()}] Waiting {wait_hours}h for quota reset...\n")
        
        # Wait with progress updates
        intervals = 20
        for i in range(intervals):
            time.sleep(wait_seconds / intervals)
            elapsed = (i + 1) / intervals * 100
            remaining_hours = wait_hours * (1 - (i + 1) / intervals)
            print(f"[AUTO] Progress: {elapsed:.0f}% | Remaining: {remaining_hours:.1f}h", end='\r')
        
        print(f"\n[AUTO] ⏰ Quota should be reset now. Resuming tests...\n")
    
    def generate_submission(self):
        """Generate final submission file for test set"""
        print("\n" + "="*70)
        print("[AUTO] Generating test set predictions...")
        print("="*70 + "\n")
        
        submission_path = "D:/kharagpur_hackathon/results/submission.csv"
        self.pipeline.generate_submission(submission_path)
        
        print(f"[AUTO] ✓ Submission saved to {submission_path}")
    
    def run_forever(self):
        """Main loop - run until all training examples tested"""
        
        print("\n" + "="*70)
        print("🤖 AUTOMATIC TESTING MODE STARTED")
        print("="*70)
        print("[AUTO] System will run continuously until all examples tested")
        print("[AUTO] Press Ctrl+C to stop")
        print("="*70 + "\n")
        
        try:
            while True:
                # Run batch
                api_limit_hit = self.run_batch(batch_size=10)
                
                # Check if done
                all_ids = set(self.train_df['id'].values)
                if len(self.tested_ids) >= len(all_ids):
                    print("\n" + "="*70)
                    print("🎉 ALL TRAINING EXAMPLES TESTED!")
                    print("="*70 + "\n")
                    
                    # Generate submission
                    self.generate_submission()
                    break
                
                # If hit rate limit, wait
                if api_limit_hit:
                    self.wait_for_quota_reset(wait_hours=12)
                else:
                    # Small delay between batches
                    time.sleep(10)
        
        except KeyboardInterrupt:
            all_ids = set(self.train_df['id'].values)
            print("\n[AUTO] ⚠️ Stopped by user")
            print(f"[AUTO] Progress: {len(self.tested_ids)}/{len(all_ids)} examples tested")
            print(f"[AUTO] Resume anytime - progress saved to {self.results_file}")


if __name__ == "__main__":
    tester = AutoTester()
    tester.run_forever()
