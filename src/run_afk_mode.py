"""
Run AFK (Away From Keyboard) Mode
Continuously tests all training examples and generates submission
"""
import sys
import os

# Add src to path if needed
sys.path.insert(0, os.path.dirname(__file__))

from auto_test_loop import AutoTestLoop


def print_banner():
    print("\n╔════════════════════════════════════════════════════════════════╗")
    print("║                    AFK AUTO-TEST MODE                          ║")
    print("║                                                                 ║")
    print("║  🤖 System will run continuously until all examples tested     ║")
    print("║  ⏰ Automatically waits for API quota resets (12 hours)       ║")
    print("║  📊 Progress logged to logs/auto_test_results.txt             ║")
    print("║  💾 Can resume anytime - progress is saved                     ║")
    print("║  🎯 Generates submission.csv when training complete            ║")
    print("║                                                                 ║")
    print("║  Press Ctrl+C to stop                                          ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    print_banner()
    
    tester = AutoTestLoop()
    tester.run_forever()
