import numpy as np

def create_terminal_visualization(entropy_tracker, ml_tracker, adaptive_tracker, noise_percentage):
    """Create colored line plot visualization in terminal"""
    print(f"\n{'='*80}")
    print(f"🎵 MODEL PERFORMANCE VISUALIZATION (Noise: {noise_percentage}%)")
    print(f"{'='*80}")
    
    # Get data for visualization
    entropy_data = entropy_tracker.run_details
    ml_data = ml_tracker.run_details
    adaptive_data = adaptive_tracker.run_details
    
    # Create visualization data
    runs = list(range(1, len(entropy_data) + 1))
    entropy_questions = [d['questions_asked'] for d in entropy_data]
    ml_questions = [d['questions_asked'] for d in ml_data]
    adaptive_questions = [d['questions_asked'] for d in adaptive_data]
    
    # Create terminal plot
    max_questions = max(max(entropy_questions), max(ml_questions), max(adaptive_questions))
    plot_height = 15
    plot_width = 60
    
    print("\n📊 Questions per Simulation:")
    print("┌" + "─" * plot_width + "┐")
    
    for i in range(plot_height, 0, -1):
        line = "│"
        for j in range(plot_width):
            # Calculate position
            run_idx = int((j / plot_width) * len(runs))
            if run_idx >= len(runs):
                line += " "
                continue
            
            # Calculate question threshold for this line
            threshold = (max_questions / plot_height) * i
            
            # Add points for each engine
            if run_idx < len(entropy_questions):
                if abs(entropy_questions[run_idx] - threshold) < 0.5:
                    line += "🔴"  # Red for Entropy
                elif entropy_questions[run_idx] < threshold:
                    line += " "
            
            if run_idx < len(ml_questions):
                if abs(ml_questions[run_idx] - threshold) < 0.5:
                    line += "🔵"  # Blue for ML
                elif ml_questions[run_idx] < threshold:
                    line += " "
            
            if run_idx < len(adaptive_questions):
                if abs(adaptive_questions[run_idx] - threshold) < 0.5:
                    line += "🟢"  # Green for Adaptive
                elif adaptive_questions[run_idx] < threshold:
                    line += " "
            
            if len(line) - 1 <= j:
                line += " "
        
        line += "│"
        print(line)
    
    print("└" + "─" * plot_width + "┘")
    
    # Legend
    print("\n🔴 = Entropy Engine")
    print("🔵 = ML Engine") 
    print("🟢 = Adaptive Engine")
    
    # Statistics
    print(f"\n📈 Average Questions:")
    print(f"   Entropy: {np.mean(entropy_questions):.1f}")
    print(f"   ML: {np.mean(ml_questions):.1f}")
    print(f"   Adaptive: {np.mean(adaptive_questions):.1f}")
    
    # Accuracy
    entropy_correct = sum(1 for d in entropy_data if d['found_song'] == d['target_song'])
    ml_correct = sum(1 for d in ml_data if d['found_song'] == d['target_song'])
    adaptive_correct = sum(1 for d in adaptive_data if d['found_song'] == d['target_song'])
    
    total_runs = len(entropy_data)
    print(f"\n🎯 Accuracy:")
    print(f"   Entropy: {(entropy_correct/total_runs)*100:.1f}%")
    print(f"   ML: {(ml_correct/total_runs)*100:.1f}%")
    print(f"   Adaptive: {(adaptive_correct/total_runs)*100:.1f}%")
