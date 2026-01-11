#!/usr/bin/env python3
"""
Test script to verify Ollama setup and integration
"""
import subprocess
import sys
import time
import os

def check_ollama_running():
    """Check if Ollama is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    print("="*70)
    print("OLLAMA SETUP CHECKER FOR BACKSTORY VERIFICATION SYSTEM")
    print("="*70)
    
    # Check if Ollama is installed
    print("\n[1/4] Checking Ollama installation...")
    try:
        result = subprocess.run(['which', 'ollama'], capture_output=True, text=True)
        if result.returncode != 0:
            print("[✗] Ollama not installed")
            print("    Install: curl -fsSL https://ollama.com/install.sh | sh")
            return False
        print("[✓] Ollama is installed")
    except Exception as e:
        print(f"[✗] Could not check Ollama installation: {e}")
        return False
    
    # Check if running
    print("\n[2/4] Checking if Ollama server is running...")
    if not check_ollama_running():
        print("[✗] Ollama not running")
        print("\n    To start Ollama, open a new terminal and run:")
        print("    $ ollama serve")
        print("\n    Then run this script again.")
        return False
    print("[✓] Ollama server is running")
    
    # Check if model is available
    print("\n[3/4] Checking for llama3.2:3b model...")
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if 'llama3.2:3b' in result.stdout or 'llama3.2' in result.stdout:
            print("[✓] Model llama3.2 is available")
        else:
            print("[!] Model llama3.2:3b not found")
            print("    Downloading model (this will take a few minutes)...")
            subprocess.run(['ollama', 'pull', 'llama3.2:3b'])
            print("[✓] Model downloaded")
    except Exception as e:
        print(f"[✗] Could not check models: {e}")
        return False
    
    # Test the LLM wrapper
    print("\n[4/4] Testing LLM integration...")
    print("="*70)
    sys.path.insert(0, '/workspaces/kharagpur_hackathon/src')
    
    try:
        from llm_local import test_local_llm
        success = test_local_llm()
        
        if not success:
            print("\n[✗] LLM integration test failed")
            return False
            
    except Exception as e:
        print(f"[✗] Error testing LLM: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # All checks passed
    print("\n" + "="*70)
    print("✓ SETUP COMPLETE - ALL CHECKS PASSED!")
    print("="*70)
    print("\nYour system is ready to verify backstories against novels.")
    print("\nNext steps:")
    print("  1. Test single example:")
    print("     $ cd /workspaces/kharagpur_hackathon")
    print("     $ python src/test_pipeline.py")
    print("\n  2. Run full test:")
    print("     $ python src/run_full_test.py")
    print("\n  3. Process test dataset:")
    print("     $ python src/master_pipeline.py")
    print("="*70)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
