import pandas as pd
import numpy as np
import re

def extract_all_features(df):
    """
    Extract meaningful features from ALL 21 dataset attributes
    Convert raw audio features into human-understandable questions
    """
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
        # Count artists separated by commas, semicolons, etc.
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
        
        # Check for common patterns
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

def test_full_features(df):
    """Test the quality of all engineered features"""
    print("🔍 FULL FEATURE ANALYSIS:")
    print(f"Total songs: {len(df)}")
    
    # All engineered features
    all_features = [
        'genre', 'mood', 'tempo', 'language', 'popularity_level',
        'duration_length', 'danceability_level', 'energy_level', 'valence_level',
        'acoustic_level', 'instrumental_level', 'liveness_level', 'speechiness_level',
        'loudness_level', 'key_category', 'mode_category', 'time_signature_category',
        'content_rating', 'artist_type', 'release_type', 'track_version'
    ]
    
    print(f"\n📊 FEATURE BREAKDOWN:")
    for feature in all_features:
        if feature in df.columns:
            unique_values = df[feature].nunique()
            value_counts = df[feature].value_counts()
            most_common = value_counts.index[0]
            count = value_counts.iloc[0]
            print(f"  {feature}: {unique_values} unique values (most: {most_common} - {count} songs)")
    
    print(f"\n✅ Total features used: {len(all_features)}")
    
    return df

if __name__ == "__main__":
    # Test the full feature engineering
    df = pd.read_csv("dataset.csv")
    df_full = extract_all_features(df)
    
    print("✅ Full feature engineering applied!")
    df_full = test_full_features(df_full)
    
    # Save full dataset
    df_full.to_csv("dataset_full_features.csv", index=False)
    print(f"\n📁 Saved full feature dataset: dataset_full_features.csv ({df_full.shape})")
