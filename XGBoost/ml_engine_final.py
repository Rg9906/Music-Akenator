import pandas as pd
import numpy as np
import joblib
import random

# -------------------------
# LOAD FINAL DATA
# -------------------------
data = pd.read_csv("dataset_final.csv")

# -------------------------
# FEATURE ENGINEERING (FINAL)
# -------------------------
def extract_features(df):
    """Extract game features from final dataset"""
    df = df.copy()
    
    # All features already engineered in dataset_final.csv
    # Just ensure required features exist
    required_features = ["genre", "mood", "tempo", "language"]
    
    for feature in required_features:
        if feature not in df.columns:
            raise ValueError(f"Missing required feature: {feature}")
    
    return df

# Process data
data = extract_features(data)

model = joblib.load("xgb_model.pkl")
model_columns = joblib.load("model_columns.pkl")

features = ["genre", "mood", "tempo", "language"]

data_dict = {col: data[col].values for col in features}
data_len = len(data)

# -------------------------
# INITIAL BELIEF
# -------------------------
probs = np.ones(data_len) / data_len

asked_questions = set()
asked_categories = {f: 0 for f in features}

target_idx = random.choice(range(data_len))

print("🎯 Target:", data.iloc[target_idx]["title"])

# -------------------------
# BUILD INPUT
# -------------------------
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

# -------------------------
# MAIN LOOP
# -------------------------
for step in range(12):

    # 🔥 GENTLE SHARPENING
    probs = probs ** 1.1
    probs = probs / np.sum(probs)

    max_prob = np.max(probs)

    # -------------------------
    # STOP CONDITION
    # -------------------------
    if max_prob > 0.75:
        print("\n🎯 CONFIDENT!")
        print("Prediction:", data.iloc[np.argmax(probs)]["title"])
        break

    remaining = np.sum(probs > 0.01)
    entropy = -np.sum(probs * np.log2(probs + 1e-9))

    best_score = -1
    best_f = None
    best_v = None

    # -------------------------
    # QUESTION SELECTION
    # -------------------------
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

            inp = build_input(
                f, v,
                remaining,
                entropy,
                step,
                asked_categories[f]
            )

            ml_score = model.predict(inp)[0]

            score = (0.6 * ml_score) + (0.4 * balance)
            score *= (1 / (1 + asked_categories[f]))

            if score > best_score:
                best_score = score
                best_f = f
                best_v = v

    print(f"\nQ{step+1}: Is {best_f} = {best_v}?")

    # -------------------------
    # SIMULATED ANSWER
    # -------------------------
    true_answer = data_dict[best_f][target_idx] == best_v

    if random.random() < 0.1:
        answer = not true_answer
        print("Answer (noisy):", "YES" if answer else "NO")
    else:
        answer = true_answer
        print("Answer:", "YES" if answer else "NO")

    # -------------------------
    # 🔥 STABLE UPDATE
    # -------------------------
    for i in range(data_len):

        match = data_dict[best_f][i] == best_v

        # dynamic strength
        strength = 1.2 + 0.6 * max_prob   # grows with confidence

        if answer:
            probs[i] *= strength if match else (2 - strength)
        else:
            probs[i] *= (2 - strength) if match else strength

    probs = probs / np.sum(probs)

    asked_questions.add((best_f, best_v))
    asked_categories[best_f] += 1

    # -------------------------
    # DEBUG
    # -------------------------
    top_idx = np.argsort(probs)[-5:][::-1]

    print("\nTop candidates:")
    for idx in top_idx:
        print(data.iloc[idx]["title"], "→", round(probs[idx], 3))