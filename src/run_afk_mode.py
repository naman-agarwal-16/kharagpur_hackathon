"""
AFK MODE LAUNCHER
Leave this running and it will automatically:
1. Test all training examples
2. Wait for API quota resets
3. Generate submission file when done
"""

print("""
╔════════════════════════════════════════════════════════════════╗
║                    AFK AUTO-TEST MODE                          ║
║                                                                 ║
║  🤖 System will run continuously until all examples tested     ║
║  ⏰ Automatically waits for API quota resets (12 hours)       ║
║  📊 Progress logged to logs/auto_test_results.txt             ║
║  💾 Can resume anytime - progress is saved                     ║
║  🎯 Generates submission.csv when training complete            ║
║                                                                 ║
║  Press Ctrl+C to stop                                          ║
╚════════════════════════════════════════════════════════════════╝
""")

from auto_test_loop import AutoTester

if __name__ == "__main__":
    tester = AutoTester()
    tester.run_forever()
