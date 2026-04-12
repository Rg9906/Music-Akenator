#!/usr/bin/env python3
"""
Music Akenator - Complete Unified System
Single file solution for music guessing game
"""

import pandas as pd
import numpy as np
import joblib
import random
import re
from math import log2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# ================================
# FULL FEATURE ENGINEERING (ALL 21 ATTRIBUTES)
# ================================

def extract_all_features(df):
    """Extract meaningful features from ALL 21 dataset attributes"""
    df = df.copy()
    
    # 1. GENRE (direct mapping)
    df['genre'] = df['track_genre']
    
    # 2. MOOD (valence + energy combination)
    def get_mood(valence, energy):
        if valence < 0.33 and energy < 0.4:
            return 'sad'
        elif valence > 0.66 and energy > 0.6:
            return 'happy'
        elif valence < 0.4 and energy > 0.7:
            return 'energetic'
        elif valence > 0.6 and energy < 0.4:
            return 'calm'
        else:
            return 'neutral'
    
    df['mood'] = df.apply(lambda row: get_mood(row['valence'], row['energy']), axis=1)
    
    # 3. TEMPO (BPM categories)
    def get_tempo_category(tempo):
        if tempo < 60:
            return 'very_slow'
        elif tempo < 90:
            return 'slow'
        elif tempo < 120:
            return 'medium'
        elif tempo < 140:
            return 'fast'
        else:
            return 'very_fast'
    
    df['tempo'] = df['tempo'].apply(get_tempo_category)
    
    # 4. LANGUAGE (from artist names)
    def infer_language(artist_name):
        if pd.isna(artist_name):
            return 'unknown'
        
        artist_name = str(artist_name).lower()
        
        non_english_patterns = [
            r'ñ', r'é', r'á', r'í', r'ó', r'ú',
            r'ç', r'ã', r'õ', r'â', r'ê', r'î', r'ô', r'û',
            r'ü', r'ö', r'ä', r'ß',
            r'ø', r'æ', r'å',
        ]
        
        for pattern in non_english_patterns:
            if re.search(pattern, artist_name):
                return 'non_english'
        
        non_english_words = ['y', 'el', 'la', 'los', 'las', 'de', 'del', 'da', 'das', 'von', 'van', 'le', 'du']
        words = artist_name.split()
        if any(word in non_english_words for word in words):
            return 'non_english'
        
        return 'english'
    
    df['language'] = df['artists'].apply(infer_language)
    
    # 5. POPULARITY (popularity buckets)
    def get_popularity_bucket(popularity):
        if popularity < 20:
            return 'very_low'
        elif popularity < 40:
            return 'low'
        elif popularity < 60:
            return 'medium'
        elif popularity < 80:
            return 'high'
        else:
            return 'very_high'
    
    df['popularity_level'] = df['popularity'].apply(get_popularity_bucket)
    
    # 6. DURATION (song length categories)
    def get_duration_category(duration_ms):
        duration_min = duration_ms / 60000
        if duration_min < 2:
            return 'very_short'
        elif duration_min < 3:
            return 'short'
        elif duration_min < 4:
            return 'medium'
        elif duration_min < 5:
            return 'long'
        else:
            return 'very_long'
    
    df['duration_length'] = df['duration_ms'].apply(get_duration_category)
    
    # 7. DANCEABILITY (how danceable)
    def get_danceability_level(danceability):
        if danceability < 0.2:
            return 'not_danceable'
        elif danceability < 0.4:
            return 'low_danceability'
        elif danceability < 0.6:
            return 'moderate_danceability'
        elif danceability < 0.8:
            return 'high_danceability'
        else:
            return 'very_danceable'
    
    df['danceability_level'] = df['danceability'].apply(get_danceability_level)
    
    # 8. ENERGY (energy levels)
    def get_energy_level(energy):
        if energy < 0.2:
            return 'very_calm'
        elif energy < 0.4:
            return 'calm'
        elif energy < 0.6:
            return 'moderate_energy'
        elif energy < 0.8:
            return 'energetic'
        else:
            return 'very_energetic'
    
    df['energy_level'] = df['energy'].apply(get_energy_level)
    
    # 9. VALENCE (emotional positivity)
    def get_valence_level(valence):
        if valence < 0.2:
            return 'very_negative'
        elif valence < 0.4:
            return 'negative'
        elif valence < 0.6:
            return 'neutral_valence'
        elif valence < 0.8:
            return 'positive'
        else:
            return 'very_positive'
    
    df['valence_level'] = df['valence'].apply(get_valence_level)
    
    # 10. ACOUSTICNESS (how acoustic)
    def get_acoustic_level(acousticness):
        if acousticness < 0.2:
            return 'electronic'
        elif acousticness < 0.4:
            return 'mostly_electronic'
        elif acousticness < 0.6:
            return 'balanced'
        elif acousticness < 0.8:
            return 'mostly_acoustic'
        else:
            return 'very_acoustic'
    
    df['acoustic_level'] = df['acousticness'].apply(get_acoustic_level)
    
    # 11. INSTRUMENTALNESS (vocal vs instrumental)
    def get_instrumental_level(instrumentalness):
        if instrumentalness < 0.1:
            return 'vocal'
        elif instrumentalness < 0.3:
            return 'mostly_vocal'
        elif instrumentalness < 0.7:
            return 'mixed'
        elif instrumentalness < 0.9:
            return 'mostly_instrumental'
        else:
            return 'pure_instrumental'
    
    df['instrumental_level'] = df['instrumentalness'].apply(get_instrumental_level)
    
    # 12. LIVENESS (live vs studio)
    def get_liveness_level(liveness):
        if liveness < 0.2:
            return 'studio'
        elif liveness < 0.4:
            return 'mostly_studio'
        elif liveness < 0.6:
            return 'possibly_live'
        elif liveness < 0.8:
            return 'likely_live'
        else:
            return 'definitely_live'
    
    df['liveness_level'] = df['liveness'].apply(get_liveness_level)
    
    # 13. SPEECHINESS (spoken vs sung)
    def get_speechiness_level(speechiness):
        if speechiness < 0.33:
            return 'music'
        elif speechiness < 0.66:
            return 'mixed_speech'
        else:
            return 'speech'
    
    df['speechiness_level'] = df['speechiness'].apply(get_speechiness_level)
    
    # 14. LOUDNESS (volume categories)
    def get_loudness_level(loudness):
        if loudness < -20:
            return 'very_quiet'
        elif loudness < -10:
            return 'quiet'
        elif loudness < -5:
            return 'moderate'
        elif loudness < 0:
            return 'loud'
        else:
            return 'very_loud'
    
    df['loudness_level'] = df['loudness'].apply(get_loudness_level)
    
    # 15. KEY (musical key categories)
    def get_key_category(key):
        if pd.isna(key):
            return 'no_key'
        key = int(key)
        if key in [0, 5, 7]:  # C, F, G
            return 'natural_key'
        elif key in [2, 9, 10]:  # D, A, Bb
            return 'sharp_key'
        elif key in [4, 11, 6]:  # E, B, F#
            return 'flat_key'
        else:
            return 'other_key'
    
    df['key_category'] = df['key'].apply(get_key_category)
    
    # 16. MODE (major vs minor)
    def get_mode_category(mode):
        if mode == 1:
            return 'major'
        elif mode == 0:
            return 'minor'
        else:
            return 'unknown_mode'
    
    df['mode_category'] = df['mode'].apply(get_mode_category)
    
    # 17. TIME SIGNATURE (beat pattern)
    def get_time_signature(time_sig):
        if time_sig == 4:
            return '4_4_time'
        elif time_sig == 3:
            return '3_4_time'
        elif time_sig == 6:
            return '6_8_time'
        else:
            return 'other_time'
    
    df['time_signature_category'] = df['time_signature'].apply(get_time_signature)
    
    # 18. EXPLICIT (content rating)
    def get_explicit_level(explicit):
        if explicit == 1:
            return 'explicit_content'
        else:
            return 'clean_content'
    
    df['content_rating'] = df['explicit'].apply(get_explicit_level)
    
    # 19. ARTIST COUNT (solo vs collaboration)
    def get_artist_count(artists):
        if pd.isna(artists):
            return 'unknown_artist_count'
        artist_str = str(artists)
        separators = [',', ';', '&', 'feat.', 'ft.']
        count = 1
        for sep in separators:
            count = max(count, len(artist_str.split(sep)))
        if count == 1:
            return 'solo_artist'
        elif count == 2:
            return 'duo'
        elif count <= 4:
            return 'small_group'
        else:
            return 'large_collaboration'
    
    df['artist_type'] = df['artists'].apply(get_artist_count)
    
    # 20. ALBUM NAME (presence)
    def get_album_status(album_name):
        if pd.isna(album_name) or album_name == '':
            return 'single'
        else:
            return 'album_track'
    
    df['release_type'] = df['album_name'].apply(get_album_status)
    
    # 21. TRACK NAME (characteristics)
    def get_track_characteristics(track_name):
        if pd.isna(track_name):
            return 'unknown_track'
        track_name = str(track_name).lower()
        
        if any(word in track_name for word in ['remix', 'remaster', 'edit', 'mix']):
            return 'remix_version'
        elif any(word in track_name for word in ['live', 'concert', 'performance']):
            return 'live_version'
        elif any(word in track_name for word in ['radio', 'edit']):
            return 'radio_version'
        elif any(word in track_name for word in ['demo', 'rough']):
            return 'demo_version'
        else:
            return 'original_version'
    
    df['track_version'] = df['track_name'].apply(get_track_characteristics)
    
    return df

# ================================
# DATA CLEANING
# ================================

def clean_dataset(df):
    """Clean the Spotify dataset"""
    print("Cleaning dataset...")
    original_shape = df.shape
    
    # Remove rows with null values in critical columns
    critical_cols = ['track_id', 'artists', 'track_name', 'danceability', 'energy', 'valence', 'tempo']
    df = df.dropna(subset=critical_cols)
    
    # Remove extremely long/short songs
    df = df[df['duration_ms'].between(30000, 600000)]  # 30s to 10min
    
    # Remove songs with invalid audio features
    valid_audio = (
        (df['danceability'].between(0, 1)) &
        (df['energy'].between(0, 1)) &
        (df['valence'].between(0, 1)) &
        (df['acousticness'].between(0, 1)) &
        (df['instrumentalness'].between(0, 1)) &
        (df['liveness'].between(0, 1)) &
        (df['speechiness'].between(0, 1))
    )
    df = df[valid_audio]
    
    # Remove duplicate tracks
    df = df.drop_duplicates(subset=['track_id'], keep='first')
    
    # Remove zero popularity songs
    df = df[df['popularity'] >= 1]
    
    print(f"Cleaned: {original_shape[0]} -> {len(df)} songs")
    return df

# ================================
# TRAINING DATA GENERATION
# ================================

def generate_training_data(data, num_simulations=1000):
    """Generate training data from songs"""
    print("Generating training data...")
    
    features = [
        "genre", "mood", "tempo", "language", "popularity_level",
        "duration_length", "danceability_level", "energy_level", "valence_level",
        "acoustic_level", "instrumental_level", "liveness_level", "speechiness_level",
        "loudness_level", "key_category", "mode_category", "time_signature_category",
        "content_rating", "artist_type", "release_type", "track_version"
    ]
    data_dict = {col: data[col].values for col in features}
    data_len = len(data)
    
    def compute_entropy(n):
        if n == 0:
            return 0
        probs = np.ones(n) / n
        return -sum(p * log2(p) for p in probs)
    
    rows = []
    
    for sim in range(num_simulations):
        if sim % 100 == 0:
            print(f"Simulation {sim}")
        
        indices = list(range(data_len))
        target_idx = random.choice(indices)
        asked_categories = {f: 0 for f in features}
        
        for step in range(10):
            if len(indices) <= 2:
                break
            
            remaining = len(indices)
            entropy_before = compute_entropy(remaining)
            
            # Evaluate all possible questions
            for f in features:
                values = set(data_dict[f][i] for i in indices)
                
                for v in values:
                    yes_indices = [i for i in indices if data_dict[f][i] == v]
                    no_indices = [i for i in indices if data_dict[f][i] != v]
                    
                    if len(yes_indices) == 0 or len(no_indices) == 0:
                        continue
                    
                    if data_dict[f][target_idx] == v:
                        after = len(yes_indices)
                    else:
                        after = len(no_indices)
                    
                    reduction = (remaining - after) / remaining
                    
                    rows.append({
                        "feature": f,
                        "value": v,
                        "remaining": remaining,
                        "entropy": entropy_before,
                        "step": step,
                        "category_count": asked_categories[f],
                        "target_score": reduction
                    })
            
            # Select best question for simulation
            best_f = None
            best_v = None
            best_score = -1
            
            for f in features:
                values = set(data_dict[f][i] for i in indices)
                
                for v in values:
                    yes_indices = [i for i in indices if data_dict[f][i] == v]
                    no_indices = [i for i in indices if data_dict[f][i] != v]
                    
                    if len(yes_indices) == 0 or len(no_indices) == 0:
                        continue
                    
                    yes = len(yes_indices)
                    no = len(no_indices)
                    remaining = len(indices)
                    after = len(yes_indices) if data_dict[f][target_idx] == v else len(no_indices)
                    
                    reduction = (remaining - after) / remaining
                    balance = min(yes, no) / remaining
                    confidence = abs(yes - no) / remaining
                    
                    target_score = (
                        0.5 * reduction +
                        0.3 * balance +
                        0.2 * confidence
                    )
                    
                    penalty = 1 / (1 + asked_categories[f])
                    score = target_score * penalty
                    
                    if score > best_score:
                        best_score = score
                        best_f = f
                        best_v = v
                        best_yes = yes_indices
                        best_no = no_indices
            
            if best_f is None or best_score <= 0:
                break
            
            asked_categories[best_f] += 1
            
            if data_dict[best_f][target_idx] == best_v:
                indices = best_yes
            else:
                indices = best_no
    
    return pd.DataFrame(rows)

# ================================
# MODEL TRAINING
# ================================

def train_model(training_data):
    """Train XGBoost model"""
    print("Training model...")
    
    # Encode features
    training_data["feature_value"] = training_data["feature"] + "_" + training_data["value"]
    training_data = pd.get_dummies(training_data, columns=["feature_value"])
    training_data = training_data.drop(["feature", "value"], axis=1)
    
    # Split data
    X = training_data.drop("target_score", axis=1)
    y = training_data["target_score"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model MSE: {mse}")
    
    # Save model and columns
    joblib.dump(model, "xgb_model.pkl")
    joblib.dump(X.columns, "model_columns.pkl")
    
    return model, X.columns

# ================================
# GAME ENGINES
# ================================

def run_entropy_engine(data, target_idx=None):
    """Run entropy-based questioning engine"""
    print("\n=== ENTROPY ENGINE TEST ===")
    
    features = [
        "genre", "mood", "tempo", "language", "popularity_level",
        "duration_length", "danceability_level", "energy_level", "valence_level",
        "acoustic_level", "instrumental_level", "liveness_level", "speechiness_level",
        "loudness_level", "key_category", "mode_category", "time_signature_category",
        "content_rating", "artist_type", "release_type", "track_version"
    ]
    
    def compute_entropy(df):
        n = len(df)
        if n == 0:
            return 0
        probs = np.ones(n) / n
        return -sum(p * log2(p) for p in probs)
    
    def value_info_gain(df, feature, value):
        total_entropy = compute_entropy(df)
        yes_subset = df[df[feature] == value]
        no_subset = df[df[feature] != value]
        if len(yes_subset) == 0 or len(no_subset) == 0:
            return -1
        p_yes = len(yes_subset) / len(df)
        p_no = len(no_subset) / len(df)
        new_entropy = p_yes * compute_entropy(yes_subset) + p_no * compute_entropy(no_subset)
        return total_entropy - new_entropy
    
    def select_best_question(df, asked_categories, asked_pairs):
        best_score = -1
        best_q = None
        for f in features:
            values = df[f].unique()
            for v in values:
                if (f, v) in asked_pairs:
                    continue
                yes = len(df[df[f] == v])
                no = len(df[df[f] != v])
                remaining = len(df)
                reduction = (remaining - min(yes, no)) / remaining
                balance = min(yes, no) / remaining
                confidence = 1 - abs(yes - no) / remaining  # Fixed: higher confidence = more balanced split
                target_score = 0.5 * reduction + 0.3 * balance + 0.2 * confidence
                penalty = 1 / (1 + asked_categories[f])
                score = target_score * penalty
                if score > best_score:
                    best_score = score
                    best_q = (f, v)
        if best_q is None:
            best_q = (features[0], df[features[0]].iloc[0])
        return best_q
    
    df = data.copy()
    asked_categories = {f: 0 for f in features}
    asked_pairs = set()
    
    if target_idx is None:
        target_song = df.sample(1).iloc[0]
    else:
        target_song = df.iloc[target_idx]
    
    print("🎯 Target song (hidden):", target_song["track_name"])
    
    for step in range(30):
        if len(df) <= 3:
            print("\n🎯 Final candidates:")
            print(df["track_name"].values)
            break
        
        q = select_best_question(df, asked_categories, asked_pairs)
        
        f, v = q
        print(f"\nQ{step+1}: Is {f} = {v}?")
        
        answer = "yes" if target_song[f] == v else "no"
        print("Answer:", answer)
        
        asked_pairs.add((f, v))
        asked_categories[f] += 1
        
        if answer == "yes":
            df = df[df[f] == v]
        else:
            df = df[df[f] != v]
        
        print("Remaining songs:", len(df))

def run_ml_engine(data, target_idx=None):
    """Run ML-based questioning engine (FIXED VERSION - NO COLLAPSE)"""
    print("\n=== ML ENGINE TEST (FIXED) ===")
    
    # Load model
    try:
        model = joblib.load("xgb_model.pkl")
        model_columns = joblib.load("model_columns.pkl")
    except FileNotFoundError:
        print("Model not found! Run training first.")
        return
    
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
    # 🔥 PRIOR INITIALIZATION (NEW)
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
    
    def build_input(feature, value, remaining, entropy, step, category_count):
        row = {col: 0 for col in model_columns}
        
        row["remaining"] = remaining
        row["entropy"] = entropy
        row["step"] = step
        row["category_count"] = category_count
        
        key = f"feature_value_{feature}_{value}"
        if key in row:
            row[key] = 1
        
        return pd.DataFrame([row])
    
    # ================================
    # MAIN LOOP
    # ================================
    
    for step in range(30):
        
        # 🔥 STABLE NORMALIZATION
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
            print(f"\n🎯 FINAL RESULT (Step {step+1}):")
            print(f"Stop reason: {stop_reason}")
            print(f"Max Probability: {max_prob:.4f}")
            for idx in top_idx:
                print(f"{data.iloc[idx]['track_name']} → {probs[idx]:.4f}")
            break
        
        remaining = np.sum(probs > 0.01)
        entropy = -np.sum(probs * np.log2(probs + 1e-9))
        
        best_score = -1
        best_f = None
        best_v = None
        
        # ================================
        # QUESTION SELECTION
        # ================================
        
        for f in features:
            for v in set(data_dict[f]):
                
                if (f, v) in asked_questions:
                    continue
                
                mask = np.array([data_dict[f][i] == v for i in range(data_len)])
                
                prob_yes = np.sum(probs[mask])
                prob_no = np.sum(probs[~mask])
                
                if prob_yes == 0 or prob_no == 0:
                    continue
                
                balance = 1 - abs(prob_yes - prob_no)
                
                inp = build_input(f, v, remaining, entropy, step, asked_categories[f])
                ml_score = model.predict(inp)[0]
                
                score = (0.6 * ml_score) + (0.4 * balance)
                score *= (1 / (1 + asked_categories[f]))
                
                if score > best_score:
                    best_score = score
                    best_f = f
                    best_v = v
        
        print(f"\nQ{step+1}: Is {best_f} = {best_v}?")
        
        # Simulated answer
        true_answer = data_dict[best_f][target_idx] == best_v
        answer = true_answer
        
        print(f"Answer: {'YES' if answer else 'NO'}")
        
        # ================================
        # 🔥 FIXED PROBABILITY UPDATE
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
        # SHOW TOP CANDIDATES
        # ================================
        
        top_idx = np.argsort(probs)[-5:][::-1]
        print("Top candidates:")
        for idx in top_idx:
            print(f"  {data.iloc[idx]['track_name']}: {probs[idx]:.4f}")
# ================================
# MAIN PIPELINE
# ================================

def main():
    """Main unified pipeline"""
    print("Music Akenator - Unified System")
    print("=" * 50)
    
    # Only regenerate if dataset_final.csv doesn't exist or has old features
    need_regenerate = False
    
    if not os.path.exists("dataset_final.csv"):
        print("dataset_final.csv not found - creating with all 21 features...")
        need_regenerate = True
    else:
        # Check if existing dataset has all 21 features
        try:
            existing_data = pd.read_csv("dataset_final.csv")
            required_features = [
                'popularity_level', 'duration_length', 'danceability_level', 'energy_level', 
                'valence_level', 'acoustic_level', 'instrumental_level', 'liveness_level', 
                'speechiness_level', 'loudness_level', 'key_category', 'mode_category', 
                'time_signature_category', 'content_rating', 'artist_type', 'release_type', 'track_version'
            ]
            missing_features = [f for f in required_features if f not in existing_data.columns]
            
            if missing_features:
                print(f"dataset_final.csv missing {len(missing_features)} features - regenerating...")
                need_regenerate = True
            else:
                print("dataset_final.csv already has all 21 features - loading...")
                data = existing_data
        except Exception as e:
            print(f"Error checking dataset_final.csv - regenerating: {e}")
            need_regenerate = True
    
    if need_regenerate:
        # Load and clean data
        try:
            raw_data = pd.read_csv("dataset.csv")
            print(f"Loaded raw dataset: {raw_data.shape}")
        except FileNotFoundError:
            print("ERROR: dataset.csv not found!")
            return
        
        # Clean data
        clean_data = clean_dataset(raw_data)
        
        # Extract ALL features (21 attributes)
        processed_data = extract_all_features(clean_data)
        
        # Save processed data
        processed_data.to_csv("dataset_final.csv", index=False)
        print(f"Saved processed dataset: {processed_data.shape}")
        
        data = processed_data
    
    # Check if model exists
    if not os.path.exists("xgb_model.pkl"):
        print("Training model...")
        
        # Generate training data
        training_data = generate_training_data(data, num_simulations=1000)
        training_data = training_data.drop_duplicates()
        print(f"Generated training data: {training_data.shape}")
        
        # Train model
        model, columns = train_model(training_data)
        print("Model trained and saved!")
    else:
        print("Model already exists.")
    
    # Run tests
    print("\n" + "=" * 50)
    print("RUNNING ALL THREE APPROACHES - SAME TARGET SONG")
    print("=" * 50)
    
    # Select same target song for all three engines
    target_idx = random.choice(range(len(data)))
    target_song = data.iloc[target_idx]
    print(f"🎯 Common Target Song: {target_song['track_name']}")
    print("=" * 50)
    
    # Test entropy engine (Baseline)
    run_entropy_engine(data, target_idx)
    
    # Test ML engine (Probabilistic)
    try:
        run_ml_engine(data, target_idx)
    except Exception as e:
        print(f"ML engine error: {e}")
        print("Skipping ML engine test...")
    
    # Test adaptive engine (Dynamic Exploration-Exploitation)
    try:
        from adaptive_engine import main as adaptive_main
        adaptive_main(target_idx)
    except Exception as e:
        print(f"Adaptive engine error: {e}")
        print("Skipping adaptive engine test...")
    
    print("\n" + "=" * 50)
    print("ALL THREE APPROACHES TEST COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    import os
    main()
