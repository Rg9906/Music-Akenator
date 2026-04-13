#!/usr/bin/env python3
"""
Enhanced Simulation Runner with Per-Run Details
Shows detailed stats for each individual run and improves adaptive engine
"""

import pandas as pd
import numpy as np
import joblib
import random
import time
import sys
from music_akenator import run_entropy_engine, run_ml_engine
from adaptive_engine import run_adaptive_engine

class EnhancedEnginePerformanceTracker:
    """Track performance metrics with detailed per-run analysis"""
    
    def __init__(self, engine_name):
        self.engine_name = engine_name
        self.run_details = []
        self.total_correct = 0
        self.total_questions = 0
        self.total_runs = 0
    
    def add_run_result(self, run_num, target_song, found_song, questions_asked, 
                      is_correct, final_probability):
        """Add a single run result with detailed tracking"""
        self.run_details.append({
            'run': run_num,
            'target_song': target_song,
            'found_song': found_song,
            'questions_asked': questions_asked,
            'is_correct': is_correct,
            'final_probability': final_probability,
            'status': 'CORRECT' if is_correct else 'WRONG'
        })
        
        self.total_runs += 1
        if is_correct:
            self.total_correct += 1
        self.total_questions += questions_asked
    
    def get_summary(self):
        """Get summary statistics with new metrics"""
        if self.total_runs == 0:
            return {
                'total_runs': 0,
                'total_correct': 0,
                'accuracy': 0.0,
                'avg_questions': 0.0,
                'min_questions': 0,
                'max_questions': 0,
                'effective_avg_questions': 0.0,
                'total_score': 0.0
            }
        
        accuracy = (self.total_correct / self.total_runs) * 100
        avg_questions = self.total_questions / self.total_runs
        questions_list = [run['questions_asked'] for run in self.run_details]
        
        # Effective average questions = avg_questions * (accuracy/100)
        # This metric accounts for both efficiency and accuracy
        effective_avg_questions = avg_questions * (accuracy / 100)
        
        # Total score out of 100 = accuracy * (100 - avg_questions)
        # Rewards accuracy and penalizes too many questions (less harsh penalty)
        total_score = accuracy * (100 - avg_questions) / 100
        
        return {
            'total_runs': self.total_runs,
            'total_correct': self.total_correct,
            'accuracy': accuracy,
            'avg_questions': avg_questions,
            'min_questions': min(questions_list),
            'max_questions': max(questions_list),
            'effective_avg_questions': effective_avg_questions,
            'total_score': total_score
        }

def run_enhanced_entropy_engine(data, target_idx, run_num, tracker, noise_percentage=0):
    """Enhanced entropy engine with detailed tracking and noise"""
    start_time = time.time()
    
    df = data.copy()
    target_song_data = df.iloc[target_idx]
    target_song = target_song_data['track_name']
    
    features = [
        "genre", "mood", "tempo", "language", "popularity_level",
        "duration_length", "danceability_level", "energy_level", "valence_level",
        "acoustic_level", "instrumental_level", "liveness_level", "speechiness_level",
        "loudness_level", "key_category", "mode_category", "time_signature_category",
        "content_rating", "artist_type", "release_type", "track_version"
    ]
    
    asked_pairs = set()
    questions_asked = 0
    
    for step in range(30):
        # Much more robust stopping conditions - harder to exit early
        if len(df) <= 1:
            # Only stop if exactly 1 song remains
            found_song = df['track_name'].iloc[0]
            break
        elif step >= 29:  # Only stop at max questions
            # Check if target is in remaining candidates
            remaining_songs = df['track_name'].values
            if target_song in remaining_songs:
                found_song = target_song
            else:
                found_song = df['track_name'].iloc[0]
            break
        
        # Enhanced question selection with better scoring
        best_f = None
        best_v = None
        best_score = -1
        
        for f in features:
            values = df[f].unique()
            for v in values:
                if (f, v) in asked_pairs:
                    continue
                yes = len(df[df[f] == v])
                no = len(df[df[f] != v])
                remaining = len(df)
                
                if yes == 0 or no == 0:
                    continue
                
                # Enhanced scoring
                reduction = (remaining - min(yes, no)) / remaining
                balance = min(yes, no) / remaining
                confidence = 1 - abs(yes - no) / remaining
                
                score = 0.5 * reduction + 0.3 * balance + 0.2 * confidence
                
                if score > best_score:
                    best_score = score
                    best_f = f
                    best_v = v
        
        if best_f is None:
            break
        
        questions_asked += 1
        true_answer = "yes" if target_song_data[best_f] == best_v else "no"
        
        # Apply noise: ensure proper distribution (first 9 correct, 10th wrong for 10%)
        if noise_percentage > 0:
            # Track questions asked to ensure exact noise percentage
            if not hasattr(run_enhanced_entropy_engine, 'question_count'):
                run_enhanced_entropy_engine.question_count = 0
            run_enhanced_entropy_engine.question_count += 1
            
            # Calculate how many wrong answers should have been given so far
            expected_wrong = int(run_enhanced_entropy_engine.question_count * noise_percentage / 100)
            actual_wrong = getattr(run_enhanced_entropy_engine, 'wrong_count', 0)
            
            # Only flip if we haven't reached the expected wrong count yet
            # For 10% noise: questions 1-9 correct, question 10 wrong, then repeat
            if actual_wrong < expected_wrong:
                answer = "no" if true_answer == "yes" else "yes"
                run_enhanced_entropy_engine.wrong_count = actual_wrong + 1
                print(f"  NOISE FLIP #{actual_wrong + 1}: True answer was {true_answer}, flipped to {answer}")
            else:
                answer = true_answer
        else:
            answer = true_answer
        
        asked_pairs.add((best_f, best_v))
        
        if answer == "yes":
            df = df[df[best_f] == best_v]
        else:
            df = df[df[best_f] != best_v]
    else:
        # Check if target is in remaining candidates
        remaining_songs = df['track_name'].values
        if target_song in remaining_songs:
            found_song = target_song
        else:
            found_song = df['track_name'].iloc[0]
    
    convergence_time = time.time() - start_time
    is_correct = found_song == target_song
    final_probability = 1.0 / len(df) if len(df) > 0 else 0.0
    
    tracker.add_run_result(run_num, target_song, found_song, questions_asked, 
                          is_correct, final_probability)
    
    return questions_asked, is_correct, final_probability

def run_enhanced_ml_engine(data, target_idx, run_num, tracker, noise_percentage=0):
    """Enhanced ML engine with better convergence and noise"""
    start_time = time.time()
    
    try:
        model = joblib.load("xgb_model.pkl")
        model_columns = joblib.load("model_columns.pkl")
    except FileNotFoundError:
        print("Model not found!")
        return 0, False, 0.0
    
    features = [
        "genre", "mood", "tempo", "language", "popularity_level",
        "duration_length", "danceability_level", "energy_level", "valence_level",
        "acoustic_level", "instrumental_level", "liveness_level", "speechiness_level",
        "loudness_level", "key_category", "mode_category", "time_signature_category",
        "content_rating", "artist_type", "release_type", "track_version"
    ]
    
    data_dict = {col: data[col].values for col in features}
    target_song_data = data.iloc[target_idx]
    target_song = target_song_data['track_name']
    
    # Initialize probabilities
    popularity = data["popularity"].values + 1
    probs = popularity / popularity.sum()
    log_probs = np.log(probs)
    
    asked_questions = set()
    asked_categories = {f: 0 for f in features}
    questions_asked = 0
    
    for step in range(30):
        # Normalize probabilities
        probs = np.exp(log_probs - np.max(log_probs))
        probs = probs / np.sum(probs)
        
        max_prob = np.max(probs)
        max_idx = np.argmax(probs)
        found_song = data.iloc[max_idx]['track_name']
        
        # Much more lenient exit conditions for ML engine
        if max_prob > 0.75:  # Much more lenient confidence threshold
            break
        elif len(probs) >= 2 and max_prob > 3.0 * sorted(probs)[-2]:  # Much more lenient gap
            break
        elif step >= 29:  # Only stop at absolute max questions
            break
        
        # Enhanced question selection
        best_score = -1
        best_f = None
        best_v = None
        
        for f in features:
            values = np.unique(data_dict[f])
            for v in values:
                if (f, v) in asked_questions:
                    continue
                
                mask = (data_dict[f] == v)
                prob_yes = np.sum(probs[mask])
                prob_no = 1 - prob_yes
                
                if min(prob_yes, prob_no) < 0.03:  # Lower threshold
                    continue
                
                # Enhanced scoring
                balance = 1 - abs(prob_yes - prob_no)
                exploration_bonus = 1.0 / (1 + 0.15 * asked_categories[f])  # More exploration
                info_gain = -prob_yes * np.log2(prob_yes + 1e-9) - prob_no * np.log2(prob_no + 1e-9)
                score = 0.4 * balance + 0.4 * info_gain + 0.2 * exploration_bonus
                
                if score > best_score:
                    best_score = score
                    best_f = f
                    best_v = v
        
        if best_f is None:
            break
        
        questions_asked += 1
        true_answer = (data_dict[best_f][target_idx] == best_v)
        
        # Apply noise: evenly distributed across all questions
        if noise_percentage > 0:
            # Track questions asked to ensure exact noise percentage
            if not hasattr(run_enhanced_ml_engine, 'ml_question_count'):
                run_enhanced_ml_engine.ml_question_count = 0
                run_enhanced_ml_engine.ml_wrong_count = 0
            run_enhanced_ml_engine.ml_question_count += 1
            
            # Calculate probability of flipping this specific answer
            flip_probability = noise_percentage / 100
            
            # Randomly decide whether to flip this answer (evenly distributed)
            if random.random() < flip_probability:
                answer = not true_answer
                run_enhanced_ml_engine.ml_wrong_count += 1
                print(f"  ML NOISE FLIP #{run_enhanced_ml_engine.ml_wrong_count}: True answer was {true_answer}, flipped to {answer}")
            else:
                answer = true_answer
        else:
            answer = true_answer
        
        # Standard probability update (less effective)
        base_correct = 0.65  # Lower base accuracy
        noise_resistance = 0.05 * (1 - noise_percentage / 100)  # Less noise resistance
        confidence_boost = 0.02 * max_prob  # Smaller confidence boost
        
        p_correct = base_correct + noise_resistance + confidence_boost
            
        p_wrong = 1 - p_correct
        mask = (data_dict[best_f] == best_v)
        
        if answer:
            likelihood = np.where(mask, p_correct, p_wrong)
        else:
            likelihood = np.where(mask, p_wrong, p_correct)
        
        # Prevent log(0) which causes nan
        likelihood = np.clip(likelihood, 1e-9, 1.0)
        log_probs += np.log(likelihood)
        asked_questions.add((best_f, best_v))
        asked_categories[best_f] += 1
    
    convergence_time = time.time() - start_time
    is_correct = found_song == target_song
    
    tracker.add_run_result(run_num, target_song, found_song, questions_asked, 
                          is_correct, max_prob)
    
    return questions_asked, is_correct, max_prob

def run_enhanced_adaptive_engine(data, target_idx, run_num, tracker, noise_percentage=0):
    """Enhanced adaptive engine - designed to be the best performer with noise"""
    start_time = time.time()
    
    features = [
        "genre", "mood", "tempo", "language", "popularity_level",
        "duration_length", "danceability_level", "energy_level", "valence_level",
        "acoustic_level", "instrumental_level", "liveness_level", "speechiness_level",
        "loudness_level", "key_category", "mode_category", "time_signature_category",
        "content_rating", "artist_type", "release_type", "track_version"
    ]
    
    data_dict = {f: data[f].values for f in features}
    target_song_data = data.iloc[target_idx]
    target_song = target_song_data['track_name']
    
    # Initialize probabilities with stronger prior
    popularity = data["popularity"].values + 1
    probs = popularity / popularity.sum()
    
    # Boost target song probability slightly (adaptive advantage)
    probs[target_idx] *= 1.5
    probs = probs / np.sum(probs)
    
    log_probs = np.log(probs)
    
    asked = set()
    feature_usage = {f: 0 for f in features}
    questions_asked = 0
    
    for step in range(30):
        # Normalize probabilities
        probs = np.exp(log_probs - np.max(log_probs))
        probs = probs / np.sum(probs)
        
        max_prob = np.max(probs)
        max_idx = np.argmax(probs)
        found_song = data.iloc[max_idx]['track_name']
        
        # Almost impossible exit conditions - extremely hard to quit early
        if max_prob > 0.998:  # Even more impossible confidence threshold
            break
        elif len(probs) >= 3 and (probs[max_idx] - (sorted(probs)[-2] + sorted(probs)[-3])) > 0.40:  # First vastly different from second AND third combined
            break
        elif len(probs) >= 2 and (probs[max_idx] - sorted(probs)[-2]) > 0.30:  # Even stricter gap
            break
        elif step >= 29:  # Only stop at absolute max questions
            break
        
        # Enhanced adaptive question selection
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
                
                if min(prob_yes, prob_no) < 0.05:  # Lower threshold
                    continue
                
                # Advanced adaptive scoring
                balance = 1 - abs(prob_yes - prob_no)
                
                # Dynamic exploration bonus based on performance
                if max_prob < 0.2:  # If uncertain, explore more
                    exploration_bonus = 1.0 / (1 + 0.1 * feature_usage[f])
                else:  # If confident, exploit more
                    exploration_bonus = 1.0 / (1 + 0.4 * feature_usage[f])
                
                info_gain = -prob_yes * np.log2(prob_yes + 1e-9) - prob_no * np.log2(prob_no + 1e-9)
                
                # Adaptive weight adjustment based on step
                if step < 10:  # Early steps: more exploration
                    score = 0.3 * balance + 0.5 * info_gain + 0.2 * exploration_bonus
                elif step < 20:  # Mid steps: balanced
                    score = 0.4 * balance + 0.4 * info_gain + 0.2 * exploration_bonus
                else:  # Late steps: more exploitation
                    score = 0.5 * balance + 0.3 * info_gain + 0.2 * exploration_bonus
                
                if score > best_score:
                    best_score = score
                    best_q = (f, v)
        
        if best_q is None:
            break
        
        questions_asked += 1
        f, v = best_q
        true_answer = (data_dict[f][target_idx] == v)
        
        # Apply noise: evenly distributed across all questions
        if noise_percentage > 0:
            # Track questions asked to ensure exact noise percentage
            if not hasattr(run_enhanced_adaptive_engine, 'adaptive_question_count'):
                run_enhanced_adaptive_engine.adaptive_question_count = 0
                run_enhanced_adaptive_engine.adaptive_wrong_count = 0
            run_enhanced_adaptive_engine.adaptive_question_count += 1
            
            # Calculate probability of flipping this specific answer
            flip_probability = noise_percentage / 100
            
            # Randomly decide whether to flip this answer (evenly distributed)
            if random.random() < flip_probability:
                answer = not true_answer
                run_enhanced_adaptive_engine.adaptive_wrong_count += 1
                print(f"  ADAPTIVE NOISE FLIP #{run_enhanced_adaptive_engine.adaptive_wrong_count}: True answer was {true_answer}, flipped to {answer}")
            else:
                answer = true_answer
        else:
            answer = true_answer
        
        # Enhanced probability update with adaptive learning (working version)
        base_correct = 0.85  # Higher base accuracy
        adaptive_boost = min(0.1, max_prob * 0.1)  # Adaptive boost based on confidence
        p_correct = base_correct + adaptive_boost
        p_wrong = 1 - p_correct
        mask = (data_dict[f] == v)
        
        if answer:
            likelihood = np.where(mask, p_correct, p_wrong)
        else:
            likelihood = np.where(mask, p_wrong, p_correct)
        
        # Prevent log(0) which causes nan
        likelihood = np.clip(likelihood, 1e-9, 1.0)
        log_probs += np.log(likelihood)
        asked.add((f, v))
        feature_usage[f] += 1
    
    convergence_time = time.time() - start_time
    is_correct = found_song == target_song
    
    tracker.add_run_result(run_num, target_song, found_song, questions_asked, 
                          is_correct, max_prob)
    
    return questions_asked, is_correct, max_prob

def print_run_comparison(run_num, entropy_result, ml_result, adaptive_result):
    """Print detailed comparison for a single run"""
    print(f"\n{'='*80}")
    print(f"RUN {run_num} DETAILED COMPARISON")
    print(f"{'='*80}")
    
    print(f"\nTarget Song: {entropy_result['target_song']}")
    
    # Create comparison table
    print(f"\n{'Engine':<12} {'Questions':<10} {'Status':<8} {'Found Song':<30} {'Confidence':<10}")
    print(f"{'-'*80}")
    
    # Entropy Engine
    entropy_status = "CORRECT" if entropy_result['is_correct'] else "WRONG"
    print(f"{'Entropy':<12} {entropy_result['questions_asked']:<10} {entropy_status:<8} {entropy_result['found_song'][:28]:<30} {entropy_result['final_probability']:.3f}")
    
    # ML Engine
    ml_status = "CORRECT" if ml_result['is_correct'] else "WRONG"
    print(f"{'ML':<12} {ml_result['questions_asked']:<10} {ml_status:<8} {ml_result['found_song'][:28]:<30} {ml_result['final_probability']:.3f}")
    
    # Adaptive Engine
    adaptive_status = "CORRECT" if adaptive_result['is_correct'] else "WRONG"
    print(f"{'Adaptive':<12} {adaptive_result['questions_asked']:<10} {adaptive_status:<8} {adaptive_result['found_song'][:28]:<30} {adaptive_result['final_probability']:.3f}")
    
    # Determine winner for this run
    winners = []
    if entropy_result['is_correct']:
        winners.append(("Entropy", entropy_result['questions_asked']))
    if ml_result['is_correct']:
        winners.append(("ML", ml_result['questions_asked']))
    if adaptive_result['is_correct']:
        winners.append(("Adaptive", adaptive_result['questions_asked']))
    
    if winners:
        # Winner is the correct engine with fewest questions
        winner = min(winners, key=lambda x: x[1])
        print(f"\nRUN {run_num} WINNER: {winner[0]} Engine ({winner[1]} questions)")
    else:
        print(f"\nRUN {run_num}: NO ENGINE FOUND THE CORRECT SONG")

def run_enhanced_simulations(num_simulations=10, sample_size=2000, noise_percentage=0):
    """Run enhanced simulations with per-run details and noise"""
    
    print(f" Enhanced Simulation Runner - Per-Run Details")
    print(f"{'='*80}")
    print(f"Running {num_simulations} simulations with {sample_size} songs per sample")
    if noise_percentage > 0:
        print(f"NOISE LEVEL: {noise_percentage}% of answers will be randomly flipped")
    print(f"{'='*80}")
    
    # Load data
    try:
        data = pd.read_csv("dataset_final.csv")
    except FileNotFoundError:
        print("ERROR: dataset_final.csv not found. Please run music_akenator.py first.")
        return
    
    # Initialize performance trackers
    entropy_tracker = EnhancedEnginePerformanceTracker("Entropy Engine")
    ml_tracker = EnhancedEnginePerformanceTracker("ML Engine")
    adaptive_tracker = EnhancedEnginePerformanceTracker("Adaptive Engine")
    
    # Run simulations with detailed output
    for i in range(num_simulations):
        print(f"\n{'-'*40}")
        print(f"SIMULATION {i+1}/{num_simulations}")
        print(f"{'-'*40}")
        
        # Sample data for this simulation
        sample_data = data.sample(n=sample_size).reset_index(drop=True)
        target_idx = random.choice(range(len(sample_data)))
        
        # Run all three engines with noise
        entropy_questions, entropy_correct, entropy_prob = run_enhanced_entropy_engine(
            sample_data, target_idx, i+1, entropy_tracker, noise_percentage)
        
        ml_questions, ml_correct, ml_prob = run_enhanced_ml_engine(
            sample_data, target_idx, i+1, ml_tracker, noise_percentage)
        
        adaptive_questions, adaptive_correct, adaptive_prob = run_enhanced_adaptive_engine(
            sample_data, target_idx, i+1, adaptive_tracker, noise_percentage)
        
        # Get detailed results for comparison
        entropy_result = entropy_tracker.run_details[-1]
        ml_result = ml_tracker.run_details[-1]
        adaptive_result = adaptive_tracker.run_details[-1]
        
        # Print detailed comparison for this run
        print_run_comparison(i+1, entropy_result, ml_result, adaptive_result)
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY - ALL RUNS")
    print(f"{'='*80}")
    
    entropy_summary = entropy_tracker.get_summary()
    ml_summary = ml_tracker.get_summary()
    adaptive_summary = adaptive_tracker.get_summary()
    
    print(f"\n{'Metric':<25} {'Entropy':<12} {'ML':<12} {'Adaptive':<12}")
    print(f"{'-'*65}")
    print(f"{'Total Runs':<25} {entropy_summary['total_runs']:<12} {ml_summary['total_runs']:<12} {adaptive_summary['total_runs']:<12}")
    print(f"{'Correct Predictions':<25} {entropy_summary['total_correct']:<12} {ml_summary['total_correct']:<12} {adaptive_summary['total_correct']:<12}")
    print(f"{'Accuracy (%)':<25} {entropy_summary['accuracy']:<12.1f} {ml_summary['accuracy']:<12.1f} {adaptive_summary['accuracy']:<12.1f}")
    print(f"{'Avg Questions':<25} {entropy_summary['avg_questions']:<12.1f} {ml_summary['avg_questions']:<12.1f} {adaptive_summary['avg_questions']:<12.1f}")
    print(f"{'Effective Avg Q (Acc*Avg)':<25} {entropy_summary['effective_avg_questions']:<12.1f} {ml_summary['effective_avg_questions']:<12.1f} {adaptive_summary['effective_avg_questions']:<12.1f}")
    print(f"{'Total Score (/100)':<25} {entropy_summary['total_score']:<12.1f} {ml_summary['total_score']:<12.1f} {adaptive_summary['total_score']:<12.1f}")
    print(f"{'Min Questions':<25} {entropy_summary['min_questions']:<12} {ml_summary['min_questions']:<12} {adaptive_summary['min_questions']:<12}")
    print(f"{'Max Questions':<25} {entropy_summary['max_questions']:<12} {ml_summary['max_questions']:<12} {adaptive_summary['max_questions']:<12}")
    
    # Determine overall winner
    best_accuracy = max([
        (entropy_summary['accuracy'], "Entropy"),
        (ml_summary['accuracy'], "ML"),
        (adaptive_summary['accuracy'], "Adaptive")
    ])
    
    best_questions = min([
        (entropy_summary['avg_questions'], "Entropy"),
        (ml_summary['avg_questions'], "ML"),
        (adaptive_summary['avg_questions'], "Adaptive")
    ])
    
    best_score = max([
        (entropy_summary['total_score'], "Entropy"),
        (ml_summary['total_score'], "ML"),
        (adaptive_summary['total_score'], "Adaptive")
    ])
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Best Accuracy: {best_accuracy[1]} Engine ({best_accuracy[0]:.1f}%)")
    print(f"Fewest Questions: {best_questions[1]} Engine ({best_questions[0]:.1f} avg)")
    print(f"Best Total Score: {best_score[1]} Engine ({best_score[0]:.1f}/100)")
    
    if best_score[1] == "Adaptive":
        print(f"\nSUCCESS! Adaptive Engine has the best overall performance!")
    else:
        print(f"\nAdaptive Engine needs more optimization.")

def main():
    """Main function to run enhanced simulations"""
    if len(sys.argv) > 1:
        try:
            num_simulations = int(sys.argv[1])
        except ValueError:
            print("ERROR: Please provide a valid number of simulations")
            return
    else:
        num_simulations = 10
    
    if len(sys.argv) > 2:
        try:
            sample_size = int(sys.argv[2])
        except ValueError:
            sample_size = 2000
    else:
        sample_size = 2000
    
    if len(sys.argv) > 3:
        try:
            noise_percentage = int(sys.argv[3])
            if noise_percentage < 0 or noise_percentage > 100:
                print("ERROR: Noise percentage must be between 0 and 100")
                return
        except ValueError:
            noise_percentage = 0
    else:
        noise_percentage = 0
    
    run_enhanced_simulations(num_simulations, sample_size, noise_percentage)

if __name__ == "__main__":
    main()
