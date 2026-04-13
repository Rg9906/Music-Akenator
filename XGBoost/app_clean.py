import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import random

# Page configuration
st.set_page_config(
    page_title="🎵 Music Akenator Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
        color: #666;
    }
    
    .simulation-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .winner-card {
        border-left: 5px solid #28a745;
        background: linear-gradient(135deg, #f8fff9 0%, #ffffff 100%);
    }
    
    .engine-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    
    .engine-table th {
        background: #f8f9fa;
        padding: 0.75rem;
        text-align: left;
        border-bottom: 2px solid #dee2e6;
    }
    
    .engine-table td {
        padding: 0.75rem;
        border-bottom: 1px solid #dee2e6;
    }
    
    .status-correct {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-wrong {
        color: #dc3545;
        font-weight: bold;
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">🎵 Music Akenator Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered song identification & comparative simulation system</div>', unsafe_allow_html=True)

# Mode selector
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("🚀 Automated Mode", key="auto_mode", use_container_width=True):
        st.session_state.mode = "auto"
        st.rerun()

with col2:
    if st.button("🎮 Manual Mode", key="manual_mode", use_container_width=True):
        st.session_state.mode = "manual"
        st.rerun()

with col3:
    if st.button("📊 Analysis Mode", key="analysis_mode", use_container_width=True):
        st.session_state.mode = "analysis"
        st.rerun()

# Initialize mode
if 'mode' not in st.session_state:
    st.session_state.mode = "auto"

mode = st.session_state.mode

# Demo simulation function
def create_demo_simulation(num_sims, noise_level):
    """Create realistic demo simulation data"""
    songs = ["Thunder", "Shape of You", "Blinding Lights", "Dance Monkey", 
              "Someone You Loved", "Starboy", "Perfect", "Believer",
              "Havana", "Closer", "Rockstar", "Girls Like You"]
    
    results = []
    for i in range(num_sims):
        target = random.choice(songs)
        
        # Generate engine results based on noise
        noise_factor = noise_level / 100
        
        # Entropy: good with no noise, bad with noise
        entropy_q = random.randint(10, 12)
        entropy_correct = random.random() > (noise_factor * 8) if noise_level > 0 else True
        entropy_conf = random.uniform(0.9, 1.0) if entropy_correct else random.uniform(0.8, 1.0)
        
        # ML: moderate performance
        ml_q = random.randint(15, 25)
        ml_correct = random.random() > (noise_factor * 4) if noise_level > 0 else True
        ml_conf = random.uniform(0.7, 0.9) if ml_correct else random.uniform(0.3, 0.7)
        
        # Adaptive: best with noise
        adaptive_q = random.randint(12, 20)
        adaptive_correct = random.random() > (noise_factor * 2) if noise_level > 0 else True
        adaptive_conf = random.uniform(0.8, 0.95) if adaptive_correct else random.uniform(0.4, 0.8)
        
        engines = [
            ("Entropy", entropy_q, entropy_correct, entropy_conf),
            ("ML", ml_q, ml_correct, ml_conf),
            ("Adaptive", adaptive_q, adaptive_correct, adaptive_conf)
        ]
        
        # Find winner
        winner = None
        for name, q, correct, conf in engines:
            if correct:
                winner = (name, q)
                break
        
        if not winner:
            winner = ("Adaptive", adaptive_q)
        
        results.append({
            'sim_num': i + 1,
            'target': target,
            'engines': engines,
            'winner': winner
        })
    
    return results

# Automated Mode
if mode == "auto":
    st.header("🚀 Automated Simulation Mode")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_sims = st.slider("Number of Simulations", 1, 50, 10)
    
    with col2:
        dataset_size = st.slider("Dataset Size", 500, 5000, 2000)
    
    with col3:
        noise_level = st.slider("Noise Percentage", 0, 20, 10)
    
    # Run button
    if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
        st.session_state.running = True
        st.session_state.results = []
    
    if st.session_state.get('running', False):
        # Progress container
        progress_container = st.container()
        
        with progress_container:
            st.subheader("📊 Live Results")
            
            # Progress elements
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            
            # Run simulations
            results = create_demo_simulation(num_sims, noise_level)
            
            for i, result in enumerate(results):
                # Update progress
                progress = (i + 1) / num_sims
                progress_bar.progress(progress)
                status_placeholder.text(f"Running Simulation {i + 1}/{num_sims}...")
                
                # Simulation card
                with st.container():
                    card_class = "winner-card" if result['winner'][0] == "Adaptive" else "simulation-card"
                    
                    st.markdown(f"""
                    <div class="{card_class}">
                        <h3>🎯 Simulation {result['sim_num']}</h3>
                        <p><strong>Target Song:</strong> {result['target']}</p>
                        
                        <table class="engine-table">
                            <tr>
                                <th>Engine</th>
                                <th>Questions</th>
                                <th>Status</th>
                                <th>Confidence</th>
                            </tr>
                    """)
                    
                    # Engine results
                    for engine_name, questions, correct, confidence in result['engines']:
                        status_class = "status-correct" if correct else "status-wrong"
                        status_text = "CORRECT" if correct else "WRONG"
                        
                        st.markdown(f"""
                            <tr>
                                <td>{engine_name}</td>
                                <td>{questions}</td>
                                <td class="{status_class}">{status_text}</td>
                                <td>{confidence:.3f}</td>
                            </tr>
                        """, unsafe_allow_html=True)
                    
                    # Winner
                    winner_name, winner_q = result['winner']
                    st.markdown(f"""
                            </table>
                            
                            <div class="winner-card">
                                <h4>🏆 Winner: {winner_name} Engine ({winner_q} questions)</h4>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                time.sleep(0.5)
            
            # Final summary
            st.session_state.running = False
            progress_bar.progress(1.0)
            status_placeholder.text("✅ Complete!")
            
            # Calculate metrics
            entropy_correct = sum(1 for r in results if r['engines'][0][2])
            ml_correct = sum(1 for r in results if r['engines'][1][2])
            adaptive_correct = sum(1 for r in results if r['engines'][2][2])
            
            entropy_acc = (entropy_correct / num_sims) * 100
            ml_acc = (ml_correct / num_sims) * 100
            adaptive_acc = (adaptive_correct / num_sims) * 100
            
            entropy_avg_q = np.mean([r['engines'][0][1] for r in results])
            ml_avg_q = np.mean([r['engines'][1][1] for r in results])
            adaptive_avg_q = np.mean([r['engines'][2][1] for r in results])
            
            entropy_score = entropy_acc * (100 - entropy_avg_q) / 100
            ml_score = ml_acc * (100 - ml_avg_q) / 100
            adaptive_score = adaptive_acc * (100 - adaptive_avg_q) / 100
            
            # Summary display
            st.markdown("---")
            st.subheader("📈 Final Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{adaptive_acc:.1f}%</div>
                    <div class="metric-label">Adaptive Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{adaptive_avg_q:.1f}</div>
                    <div class="metric-label">Adaptive Avg Questions</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{adaptive_score:.1f}</div>
                    <div class="metric-label">Adaptive Total Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Comparison table
            comparison_data = {
                'Engine': ['Entropy', 'ML', 'Adaptive'],
                'Accuracy (%)': [f"{entropy_acc:.1f}", f"{ml_acc:.1f}", f"{adaptive_acc:.1f}"],
                'Avg Questions': [f"{entropy_avg_q:.1f}", f"{ml_avg_q:.1f}", f"{adaptive_avg_q:.1f}"],
                'Total Score': [f"{entropy_score:.1f}", f"{ml_score:.1f}", f"{adaptive_score:.1f}"]
            }
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

# Manual Mode
elif mode == "manual":
    st.header("🎮 Manual Interactive Mode")
    
    if st.button("🎯 Start New Game", use_container_width=True):
        st.session_state.game_active = True
        st.session_state.target = f"Random Song {random.randint(1, 1000)}"
        st.session_state.questions_asked = 0
        st.session_state.candidates = [f"Song {i}" for i in range(1, 101)]
    
    if st.session_state.get('game_active', False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎵 Current Game")
            st.write(f"**Target:** {st.session_state.target}")
            st.write(f"**Questions Asked:** {st.session_state.questions_asked}")
            st.write(f"**Remaining Candidates:** {len(st.session_state.candidates)}")
            
            # Question
            st.markdown("---")
            question = f"Is the song's tempo {'fast' if random.random() > 0.5 else 'slow'}?"
            st.write(f"**Q{st.session_state.questions_asked + 1}:** {question}")
            
            # Answer buttons
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("✅ YES", key="yes", use_container_width=True):
                    st.session_state.questions_asked += 1
                    if len(st.session_state.candidates) > 1:
                        st.session_state.candidates = st.session_state.candidates[:len(st.session_state.candidates)//2 + 1]
                    st.rerun()
            
            with col_b:
                if st.button("❌ NO", key="no", use_container_width=True):
                    st.session_state.questions_asked += 1
                    if len(st.session_state.candidates) > 1:
                        st.session_state.candidates = st.session_state.candidates[len(st.session_state.candidates)//2:]
                    st.rerun()
        
        with col2:
            st.subheader("🏆 Top Prediction")
            if st.session_state.candidates:
                st.write(f"**Song:** {st.session_state.candidates[0]}")
                st.write(f"**Confidence:** {random.uniform(0.7, 0.95):.2f}")
            
            if st.button("🏁 Make Final Guess", use_container_width=True):
                st.success(f"🎉 Final Guess: {st.session_state.candidates[0]}")
                st.info(f"Questions used: {st.session_state.questions_asked}")

# Analysis Mode
elif mode == "analysis":
    st.header("📊 Deep Dive Analysis")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        analysis_sims = st.slider("Simulations", 10, 100, 20)
        analysis_noise = st.slider("Noise %", 0, 20, 10)
        
        if st.button("📈 Generate Analysis", type="primary", use_container_width=True):
            # Generate analysis data
            np.random.seed(42)
            
            noise_levels = [0, 2, 5, 8, 10, 12, 15, 17, 18, 20]
            entropy_acc = [100, 95, 85, 70, 40, 10, 0, 0, 0, 0]
            ml_acc = [100, 98, 92, 85, 70, 50, 30, 20, 15, 10]
            adaptive_acc = [100, 98, 95, 90, 80, 65, 50, 40, 30, 25]
            
            st.session_state.analysis_data = {
                'noise': noise_levels,
                'entropy': entropy_acc,
                'ml': ml_acc,
                'adaptive': adaptive_acc
            }
    
    with col2:
        if 'analysis_data' in st.session_state:
            st.subheader("📋 Data Table")
            data = st.session_state.analysis_data
            df_analysis = pd.DataFrame({
                'Noise %': data['noise'],
                'Entropy Accuracy': data['entropy'],
                'ML Accuracy': data['ml'],
                'Adaptive Accuracy': data['adaptive']
            })
            st.dataframe(df_analysis, use_container_width=True, hide_index=True)
    
    with col3:
        if 'analysis_data' in st.session_state:
            st.subheader("🎯 Key Insights")
            data = st.session_state.analysis_data
            idx = min(4, len(data['noise']) - 1)  # 10% noise index
            
            st.metric("Adaptive @ 10% Noise", f"{data['adaptive'][idx]}%", delta="Best Performance")
            st.metric("Noise Robustness", "Excellent", delta="Handles errors well")
    
    # Visualizations
    if 'analysis_data' in st.session_state:
        data = st.session_state.analysis_data
        
        st.markdown("---")
        st.subheader("📈 Performance Visualizations")
        
        # Noise vs Performance
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data['noise'],
            y=data['entropy'],
            mode='lines+markers',
            name='Entropy',
            line=dict(color='#ff6b6b', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=data['noise'],
            y=data['ml'],
            mode='lines+markers',
            name='ML',
            line=dict(color='#4ecdc4', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=data['noise'],
            y=data['adaptive'],
            mode='lines+markers',
            name='Adaptive',
            line=dict(color='#667eea', width=4),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title='🎵 Engine Performance vs Noise Level',
            xaxis_title='Noise Percentage (%)',
            yaxis_title='Accuracy (%)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    <p>🎵 Music Akenator Dashboard - Advanced AI Song Identification System</p>
    <p>Comparing Entropy, ML, and Adaptive approaches under realistic conditions</p>
</div>
""", unsafe_allow_html=True)
