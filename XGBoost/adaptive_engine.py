import pandas as pd
import numpy as np
import random
import joblib
from math import log2

# ================================
# ADAPTIVE QUESTIONING ENGINE
# Based on ML Engine with Adaptive Question Selection
# ================================

def run_adaptive_engine(data, target_idx=None):
    """Run adaptive questioning engine with dynamic exploration-exploitation"""
    print("\n=== ADAPTIVE ENGINE TEST (Dynamic Exploration-Exploitation) ===")
    
    # Skip ML model loading - use pure heuristic approach
    print("Using pure heuristic adaptive scoring (no ML model dependency)")
    
    features = [
        "genre", "mood", "tempo", "language", "popularity_level",
        "duration_length", "danceability_level", "energy_level", "valence_level",
        "acoustic_level", "instrumental_level", "liveness_level", "speechiness_level",
        "loudness_level", "key_category", "mode_category", "time_signature_category",
        "content_rating", "artist_type", "release_type", "track_version"
    ]
    
    data_dict = {col: data[col].values for col in features}
    data_len = len(data)
    
    # ================================
    # PRIOR INITIALIZATION (same as ML engine)
    # ================================
    
    popularity = data["popularity"].values + 1
    probs = popularity / popularity.sum()
    log_probs = np.log(probs)
    
    asked_questions = set()
    asked_categories = {f: 0 for f in features}
    
    # Use provided target_idx or random if not provided
    if target_idx is None:
        target_idx = random.choice(range(data_len))
    
    print(f"Target song: {data.iloc[target_idx]['track_name']}")
    
    # No need for build_input function - using pure heuristic approach
    
    def compute_entropy(probs):
        """Compute entropy of probability distribution"""
        return -np.sum(probs * np.log2(probs + 1e-9))
    
    # ================================
    # MAIN LOOP (same structure as ML engine)
    # ================================
    
    for step in range(30):
        
        # Convert log probs to probs for display and calculations
        probs = np.exp(log_probs - np.max(log_probs))
        probs = probs / np.sum(probs)
        
        max_prob = np.max(probs)
        
        # ENHANCED STOP CONDITIONS
        top_idx = np.argsort(probs)[-5:][::-1]
        sorted_probs = probs[top_idx]
        
        # Check for early stopping conditions
        early_stop = False
        stop_reason = ""
        
        # Condition 1: Confidence threshold
        if max_prob > 0.4:
            early_stop = True
            stop_reason = "Confidence threshold (0.4) reached"
        
        # Condition 2: Big gap between first and second
        elif len(sorted_probs) >= 2 and sorted_probs[0] > 2 * sorted_probs[1]:
            early_stop = True
            stop_reason = f"Big gap: {sorted_probs[0]:.3f} vs {sorted_probs[1]:.3f}"
        
        # Condition 3: Maximum questions
        elif step >= 29:
            early_stop = True
            stop_reason = "Maximum questions (30) reached"
        
        if early_stop:
            print(f"\nADAPTIVE ENGINE FINAL RESULT (Step {step+1}):")
            print(f"Stop reason: {stop_reason}")
            print(f"Max Probability: {max_prob:.4f}")
            for idx in top_idx:
                print(f"{data.iloc[idx]['track_name']} -> {probs[idx]:.4f}")
            break
        
        remaining = np.sum(probs > 0.01)
        entropy = compute_entropy(probs)
        
        # Determine current phase
        if max_prob < 0.05:
            phase = "AGGRESSIVE EXPLORATION"
        elif max_prob <= 0.3:
            phase = "BALANCED NARROWING"
        else:
            phase = "FINE-GRAINED EXPLOITATION"
        
        print(f"\nStep {step+1} - Phase: {phase} (max_prob={max_prob:.4f})")
        
        best_score = -1
        best_f = None
        best_v = None
        
        # ================================
        # ADAPTIVE QUESTION SELECTION (MODIFIED PART)
        # ================================
        
        for f in features:
            for v in set(data_dict[f]):
                
                if (f, v) in asked_questions:
                    continue
                
                mask = data_dict[f] == v
                prob_yes = np.sum(probs[mask])
                prob_no = np.sum(probs[~mask])
                
                balance = 1 - abs(prob_yes - prob_no)
                
                # ================================
                # PURE HEURISTIC ADAPTIVE SCORING
                # ================================
                
                # Dynamic weights based on current confidence
                exploration_weight = 1 - max_prob  # High when uncertain
                exploitation_weight = max_prob      # High when confident
                
                # Information gain (for exploration)
                info_gain = -prob_yes * np.log2(prob_yes + 1e-9) - prob_no * np.log2(prob_no + 1e-9)
                
                # Heuristic ML score based on question characteristics
                reduction_potential = min(prob_yes, prob_no)
                ml_score = reduction_potential * balance
                
                # Adaptive scoring function
                score = (
                    0.5 * ml_score +                              # Heuristic effectiveness
                    0.3 * exploration_weight * info_gain +        # Exploration: high-impact questions
                    0.2 * exploitation_weight * balance           # Exploitation: balanced splits
                )
                
                score *= (1 / (1 + asked_categories[f]))
                
                if score > best_score:
                    best_score = score
                    best_f = f
                    best_v = v
        
        print(f"\nQ{step+1}: Is {best_f} = {best_v}?")
        
        # Simulated answer (same as ML engine)
        true_answer = data_dict[best_f][target_idx] == best_v
        answer = true_answer
        
        print(f"Answer: {'YES' if answer else 'NO'}")
        
        # ================================
        # PROBABILITY UPDATE (same as ML engine)
        # ================================
        
        # Smooth + noise-aware
        p_correct = 0.75 + 0.2 * max_prob
        p_wrong = 1 - p_correct
        
        for i in range(data_len):
            match = data_dict[best_f][i] == best_v
            
            if answer:
                likelihood = p_correct if match else p_wrong
            else:
                likelihood = p_wrong if match else p_correct
            
            log_probs[i] += np.log(likelihood)
        
        asked_questions.add((best_f, best_v))
        asked_categories[best_f] += 1
        
        # ================================
        # SHOW TOP CANDIDATES (same as ML engine)
        # ================================
        
        top_idx = np.argsort(probs)[-5:][::-1]
        print("Top candidates:")
        for idx in top_idx:
            print(f"  {data.iloc[idx]['track_name']}: {probs[idx]:.4f}")

def main(target_idx=None):
    """Main function to run adaptive engine"""
    data = pd.read_csv("dataset_final.csv")
    run_adaptive_engine(data, target_idx)

if __name__ == "__main__":
    main()
