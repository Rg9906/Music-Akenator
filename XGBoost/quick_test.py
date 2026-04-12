#!/usr/bin/env python3
"""
Quick Test - Skip data cleaning if already done
"""
import subprocess
import sys
import os
import pandas as pd

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                               capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"SUCCESS: {description}")
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
            return True
        else:
            print(f"FAILED: {description}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def check_files():
    """Check if required files exist"""
    print("Checking files...")
    
    required_files = [
        ("dataset_final.csv", "Cleaned dataset"),
        ("xgb_model.pkl", "Trained model"),
        ("model_columns.pkl", "Model columns")
    ]
    
    missing = []
    for file, desc in required_files:
        if os.path.exists(file):
            print(f"  FOUND: {desc} ({file})")
        else:
            print(f"  MISSING: {desc} ({file})")
            missing.append(file)
    
    return missing

def main():
    """Quick test workflow"""
    print("Quick Test - Music Akenator")
    print("=" * 40)
    
    # Check what files we have
    missing_files = check_files()
    
    # If we have the final dataset, skip cleaning
    if "dataset_final.csv" not in missing_files:
        print("\nDataset already exists - skipping data cleaning")
        
        # If model is missing, train it
        if "xgb_model.pkl" in missing_files:
            print("Model missing - training...")
            if not run_script("main.py", "Model Training"):
                print("Model training failed!")
                return
        else:
            print("Model already exists - skipping training")
    else:
        print("Dataset missing - need to run full pipeline")
        return
    
    # Test the engines
    print("\n" + "="*50)
    print("TESTING ENGINES")
    print("="*50)
    
    print("\n1. Testing Entropy Engine (pure logic):")
    run_script("entropy_engine.py", "Entropy Engine Test")
    
    print("\n2. Testing ML Engine (with probabilities):")
    run_script("ml_engine_final.py", "ML Engine Test")
    
    print("\n" + "="*50)
    print("QUICK TEST COMPLETE!")
    print("="*50)

if __name__ == "__main__":
    main()
