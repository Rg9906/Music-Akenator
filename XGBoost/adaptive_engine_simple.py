#!/usr/bin/env python3
"""
Simplified Adaptive Engine - WORKING VERSION
Main approach with unique stopping conditions
"""

import pandas as pd
import numpy as np
import random

def run_adaptive_engine(data, target_idx=None):
    """Simplified but effective adaptive engine"""
    print("\n=== SIMPLIFIED ADAPTIVE ENGINE ===")
    
    features = [
        "genre", "mood", "tempo", "language", "popularity_level",
        "duration_length", "danceability_level", "energy_level", "valence_level",
        "acoustic_level", "instrumental_level", "liveness_level", "speechiness_level",
        "loudness_level", "key_category", "mode_category", "time_signature_category",
        "content_rating", "artist_type", "release_type", "track_version"
    ]
    
    data_dict = {f: data[f].values for f in features}
    N = len(data)
    
    # Initialize probabilities
    popularity = data["popularity"].values + 1
    probs = popularity / np.sum(popularity)
    log_probs = np.log(probs)
    
    if target_idx is None:
        target_idx = random.randint(0, N-1)
    
    asked = set()
    feature_usage = {f: 0 for f in features}
    
    print(f"Target: {data.iloc[target_idx]['track_name']}")
    
    for step in range(30):
        # Normalize probabilities
        probs = np.exp(log_probs - np.max(log_probs))
        probs = probs / np.sum(probs)
        
        max_prob = np.max(probs)
        
        print(f"\nStep {step+1} | max_prob={max_prob:.4f}")
        
        # Simple but effective question selection
        best_score = -1
        best_q = None
        
        for f in features:
            values = np.unique(data_dict[f])
            
            for v in values:
                if (f, v) in asked:
                    continue
                
                mask = (data_dict[f] == v)
                prob_yes = np.sum(probs[mask])
                prob_no = 1 - prob_yes
                
                # Skip very imbalanced questions
                if min(prob_yes, prob_no) < 0.05:
                    continue
                
                # Adaptive scoring: balance + exploration bonus
                balance = 1 - abs(prob_yes - prob_no)
                
                # Exploration bonus for unused features
                exploration_bonus = 1.0 / (1 + 0.2 * feature_usage[f])
                
                # Information gain
                info_gain = -prob_yes * np.log2(prob_yes + 1e-9) - prob_no * np.log2(prob_no + 1e-9)
                
                # Combined score
                score = 0.5 * balance + 0.3 * info_gain + 0.2 * exploration_bonus
                
                if score > best_score:
                    best_score = score
                    best_q = (f, v)
        
        # Ask question
        f, v = best_q
        print(f"Q{step+1}: Is {f} = {v}?")
        
        answer = (data_dict[f][target_idx] == v)
        print("Answer:", "YES" if answer else "NO")
        
        # Update probabilities (fixed logic)
        p_correct = 0.75 + 0.2 * max_prob
        p_wrong = 1 - p_correct
        
        mask = (data_dict[f] == v)
        
        for i in range(N):
            match = mask[i]
            
            if answer:
                likelihood = p_correct if match else p_wrong
            else:
                likelihood = p_wrong if match else p_correct
            
            log_probs[i] += np.log(likelihood)
        
        asked.add((f, v))
        feature_usage[f] += 1
        
        # Show top candidates
        top_idx = np.argsort(probs)[-5:][::-1]
        sorted_probs = probs[top_idx]
        
        print("Top candidates:")
        for idx in top_idx:
            print(f"  {data.iloc[idx]['track_name']} -> {probs[idx]:.4f}")
        
        # UNIQUE STOPPING CONDITIONS (different from ML engine)
        if max_prob > 0.5:  # Higher confidence than ML engine
            print("\nAdaptive confidence threshold (0.5) reached")
            break
        
        elif len(sorted_probs) >= 2 and (sorted_probs[0] - sorted_probs[1]) > 0.15:  # 15% absolute gap
            print(f"\nAdaptive gap: {sorted_probs[0]:.3f} vs {sorted_probs[1]:.3f}")
            break

def main():
    data = pd.read_csv("dataset_final.csv")
    run_adaptive_engine(data)

if __name__ == "__main__":
    main()
