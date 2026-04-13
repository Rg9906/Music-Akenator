#!/usr/bin/env python3
"""
Complete Music Akenator Workflow Pipeline
1. Clean the dataset
2. Generate training data
3. Train ML model
4. Run inference
"""

import subprocess
import sys
import os

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*50}")
    print(f"🚀 {description}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                               capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            print(result.stdout)
        else:
            print(f"❌ {description} failed!")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False
    
    return True

def main():
    """Run the complete workflow pipeline"""
    print("🎵 Music Akenator - Complete Workflow Pipeline")
    print("=" * 60)
    
    # Step 1: Run unified system (handles all steps)
    if not run_script("music_akenator.py", "Music Akenator Unified System"):
        print("❌ Pipeline failed!")
        return
    
    print("\n" + "="*60)
    print("🎉 Music Akenator Unified System completed successfully!")
    print("📁 Generated files:")
    print("   - dataset_final.csv (all 21 features engineered)")
    print("   - training_data_clean_unique.csv (training data)")
    print("   - xgb_model.pkl (trained model)")
    print("   - model_columns.pkl (feature columns)")
    print("="*60)

if __name__ == "__main__":
    main()
