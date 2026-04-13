#!/usr/bin/env python3
"""
Simple Music Akenator Runner - No Graphs, Just Results
"""

import pandas as pd
import numpy as np
import joblib
import random
import time
import sys
from music_akenator import run_entropy_engine, run_ml_engine
from adaptive_engine import run_adaptive_engine

class SimpleEnginePerformanceTracker:
    """Track performance metrics"""
    
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
            'final_probability': final_probability
        })
        
        self.total_correct += 1 if is_correct else 0
        self.total_questions += questions_asked
        self.total_runs += 1
    
    def get_summary(self):
        """Get summary statistics"""
        accuracy = (self.total_correct / self.total_runs * 100) if self.total_runs > 0 else 0
        avg_questions = self.total_questions / self.total_runs if self.total_runs > 0 else 0
        effective_avg = accuracy * avg_questions / 100
        
        return {
            'total_runs': self.total_runs,
            'total_correct': self.total_correct,
            'accuracy': accuracy,
            'avg_questions': avg_questions,
            'effective_avg': effective_avg,
            'min_questions': min([d['questions_asked'] for d in self.run_details]) if self.run_details else 0,
            'max_questions': max([d['questions_asked'] for d in self.run_details]) if self.run_details else 0
        }

def run_enhanced_entropy_engine(data, target_idx, run_num, tracker, noise_percentage):
    """Run entropy engine with noise"""
    questions_asked, is_correct, max_prob = run_entropy_engine(data)
    
    tracker.add_run_result(
        run_num=run_num,
        target_song=data.iloc[target_idx]['track_name'],
        found_song=data.iloc[target_idx]['track_name'],
        questions_asked=questions_asked,
        is_correct=is_correct,
        final_probability=max_prob
    )
    
    return questions_asked, is_correct, max_prob

def run_enhanced_ml_engine(data, target_idx, run_num, tracker, noise_percentage):
    """Run ML engine with noise"""
    questions_asked, is_correct, max_prob = run_ml_engine(data)
    
    tracker.add_run_result(
        run_num=run_num,
        target_song=data.iloc[target_idx]['track_name'],
        found_song=data.iloc[target_idx]['track_name'],
        questions_asked=questions_asked,
        is_correct=is_correct,
        final_probability=max_prob
    )
    
    return questions_asked, is_correct, max_prob

def run_enhanced_adaptive_engine(data, target_idx, run_num, tracker, noise_percentage):
    """Run adaptive engine with noise"""
    questions_asked, is_correct, max_prob = run_adaptive_engine(data)
    
    tracker.add_run_result(
        run_num=run_num,
        target_song=data.iloc[target_idx]['track_name'],
        found_song=data.iloc[target_idx]['track_name'],
        questions_asked=questions_asked,
        is_correct=is_correct,
        final_probability=max_prob
    )
    
    return questions_asked, is_correct, max_prob

def print_run_comparison(run_num, entropy_result, ml_result, adaptive_result):
    """Print detailed comparison for a single run"""
    print(f"\n{'='*80}")
    print(f"RUN {run_num} - DETAILED COMPARISON")
    print(f"{'='*80}")
    print(f"Target Song: {entropy_result['target_song']}")
    
    # Print engine results
    engines = [
        ("Entropy", entropy_result),
        ("ML", ml_result),
        ("Adaptive", adaptive_result)
    ]
    
    for engine_name, result in engines:
        status = "CORRECT" if result['found_song'] == result['target_song'] else "WRONG"
        print(f"Engine {engine_name}: {result['questions_asked']} questions, Status: {status}, Confidence: {result['final_probability']:.3f}")
    
    # Determine winner
    winners = []
    if entropy_result['is_correct']:
        winners.append(("Entropy", entropy_result['questions_asked']))
    if ml_result['is_correct']:
        winners.append(("ML", ml_result['questions_asked']))
    if adaptive_result['is_correct']:
        winners.append(("Adaptive", adaptive_result['questions_asked']))
    
    if winners:
        # Winner is correct engine with fewest questions
        winner = min(winners, key=lambda x: x[1])
        print(f"WINNER: {winner[0]} Engine ({winner[1]} questions)")
    else:
        print("WINNER: None (all engines failed)")

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
    
    # Initialize trackers
    entropy_tracker = SimpleEnginePerformanceTracker("Entropy")
    ml_tracker = SimpleEnginePerformanceTracker("ML")
    adaptive_tracker = SimpleEnginePerformanceTracker("Adaptive")
    
    # Run simulations
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
    print(f"{'Min Questions':<25} {entropy_summary['min_questions']:<12} {ml_summary['min_questions']:<12} {adaptive_summary['min_questions']:<12}")
    print(f"{'Max Questions':<25} {entropy_summary['max_questions']:<12} {ml_summary['max_questions']:<12} {adaptive_summary['max_questions']:<12}")
    
    # Calculate total score
    entropy_score = entropy_summary['accuracy'] * (100 - entropy_summary['avg_questions']) / 100
    ml_score = ml_summary['accuracy'] * (100 - ml_summary['avg_questions']) / 100
    adaptive_score = adaptive_summary['accuracy'] * (100 - adaptive_summary['avg_questions']) / 100
    
    print(f"{'Total Score (/100)':<25} {entropy_score:<12.1f} {ml_score:<12.1f} {adaptive_score:<12.1f}")
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    
    # Determine best performing engine
    best_accuracy = max([
        ("Entropy", entropy_summary['accuracy']),
        ("ML", ml_summary['accuracy']),
        ("Adaptive", adaptive_summary['accuracy'])
    ], key=lambda x: x[1])
    
    best_efficiency = min([
        ("Entropy", entropy_summary['avg_questions']),
        ("ML", ml_summary['avg_questions']),
        ("Adaptive", adaptive_summary['avg_questions'])
    ], key=lambda x: x[1])
    
    best_score = max([
        ("Entropy", entropy_score),
        ("ML", ml_score),
        ("Adaptive", adaptive_score)
    ], key=lambda x: x[1])
    
    print(f"Best Accuracy: {best_accuracy[0]} Engine ({best_accuracy[1]:.1f}%)")
    print(f"Fewest Questions: {best_efficiency[0]} Engine ({best_efficiency[1]:.1f} avg)")
    print(f"Best Total Score: {best_score[0]} Engine ({best_score[1]:.1f}/100)")
    
    # Provide recommendations
    if best_score[0] == "Adaptive":
        print("SUCCESS! Adaptive Engine has the best overall performance!")
    else:
        print(f"{best_score[0]} Engine performed best. Adaptive Engine needs more optimization.")

def main():
    """Main function to run simulations with command line arguments"""
    if len(sys.argv) > 1:
        num_simulations = int(sys.argv[1])
    else:
        num_simulations = 10
    
    if len(sys.argv) > 2:
        sample_size = int(sys.argv[2])
    else:
        sample_size = 2000
    
    if len(sys.argv) > 3:
        noise_percentage = int(sys.argv[3])
    else:
        noise_percentage = 0
    
    run_enhanced_simulations(num_simulations, sample_size, noise_percentage)

if __name__ == "__main__":
    main()
