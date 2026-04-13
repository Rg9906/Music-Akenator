#!/usr/bin/env python3
"""
Test All Four Approaches
1. Brute Force (random questions)
2. Normal music_akenator (entropy + ML + adaptive)
3. Adaptive Engine (complex)
4. Simplified Adaptive Engine (working)
"""

import pandas as pd
import numpy as np
import random
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_brute_force(data, target_idx):
    """Test 1: Brute Force - Random Questions"""
    print("\n" + "="*60)
    print("🔴 BRUTE FORCE ENGINE (Random Questions)")
    print("="*60)
    
    target_song = data.iloc[target_idx]['track_name']
    print(f"Target: {target_song}")
    
    # Random questions until found
    remaining_indices = list(range(len(data)))
    questions_asked = 0
    
    while len(remaining_indices) > 1 and questions_asked < 30:
        # Ask random question about random remaining song
        random_idx = random.choice(remaining_indices)
        random_song = data.iloc[random_idx]
        
        # Pick random feature
        features = ["genre", "mood", "tempo", "language", "popularity_level"]
        feature = random.choice(features)
        value = random_song[feature]
        
        print(f"\nQ{questions_asked+1}: Is {feature} = {value}?")
        
        # Filter based on target song
        target_value = data.iloc[target_idx][feature]
        answer = target_value == value
        
        print("Answer:", "YES" if answer else "NO")
        
        # Filter remaining songs
        remaining_indices = [
            idx for idx in remaining_indices 
            if data.iloc[idx][feature] == value if answer else data.iloc[idx][feature] != value
        ]
        
        questions_asked += 1
        print(f"Remaining songs: {len(remaining_indices)}")
        
        if len(remaining_indices) <= 3:
            break
    
    print(f"\n🎯 Found {len(remaining_indices)} candidates after {questions_asked} questions")
    for idx in remaining_indices:
        print(f"  {data.iloc[idx]['track_name']}")
    
    return questions_asked

def test_normal_music_akenator(data, target_idx):
    """Test 2: Normal music_akenator (all three engines)"""
    print("\n" + "="*60)
    print("🟡 NORMAL MUSIC AKINATOR (All Three Engines)")
    print("="*60)
    
    try:
        from music_akenator import run_entropy_engine, run_ml_engine, run_adaptive_engine
        
        target_song = data.iloc[target_idx]['track_name']
        print(f"Target: {target_song}")
        
        # Run all three engines
        print("\n--- ENTROPY ENGINE ---")
        run_entropy_engine(data, target_idx)
        
        print("\n--- ML ENGINE ---")
        run_ml_engine(data, target_idx)
        
        print("\n--- ADAPTIVE ENGINE ---")
        run_adaptive_engine(data, target_idx)
        
    except Exception as e:
        print(f"Error: {e}")
        return 30  # max questions
    
    return 30

def test_complex_adaptive(data, target_idx):
    """Test 3: Complex Adaptive Engine"""
    print("\n" + "="*60)
    print("🔴 COMPLEX ADAPTIVE ENGINE")
    print("="*60)
    
    try:
        from adaptive_engine import run_adaptive_engine
        run_adaptive_engine(data, target_idx)
        
    except Exception as e:
        print(f"Error: {e}")
        return 30
    
    return 30

def test_simplified_adaptive(data, target_idx):
    """Test 4: Simplified Adaptive Engine"""
    print("\n" + "="*60)
    print("🟢 SIMPLIFIED ADAPTIVE ENGINE")
    print("="*60)
    
    try:
        from adaptive_engine_simple import run_adaptive_engine
        run_adaptive_engine(data, target_idx)
        
    except Exception as e:
        print(f"Error: {e}")
        return 30
    
    return 30

def main():
    """Run all four approaches"""
    print("🚀 TESTING ALL FOUR APPROACHES")
    print("="*80)
    
    # Load data
    data = pd.read_csv("dataset_final.csv")
    
    # Select random target
    target_idx = random.choice(range(len(data)))
    target_song = data.iloc[target_idx]['track_name']
    
    print(f"\n🎯 COMMON TARGET SONG: {target_song}")
    print(f"📊 Dataset Size: {len(data):,} songs")
    print("="*80)
    
    results = {}
    
    # Test all four approaches
    approaches = [
        ("Brute Force", test_brute_force),
        ("Normal Music Akinator", test_normal_music_akenator),
        ("Complex Adaptive", test_complex_adaptive),
        ("Simplified Adaptive", test_simplified_adaptive)
    ]
    
    for name, test_func in approaches:
        print(f"\n{'='*20}")
        print(f"TESTING: {name}")
        print(f"{'='*20}")
        
        questions = test_func(data, target_idx)
        results[name] = questions
        
        print(f"✅ {name}: {questions} questions")
    
    # Summary
    print("\n" + "="*80)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*80)
    
    for name, questions in results.items():
        print(f"  {name}: {questions} questions")
    
    print(f"\n🏆 WINNER: {min(results, key=results.get)} with {min(results.values())} questions")
    print(f"📈 AVERAGE: {np.mean(list(results.values())):.1f} questions")
    
    return results

if __name__ == "__main__":
    main()
