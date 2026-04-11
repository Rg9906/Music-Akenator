import pandas as pd
import numpy as np
import joblib
import random

# -------------------------
# LOAD DATA + MODEL
# -------------------------
data = pd.read_csv("data.csv")

model = joblib.load("xgb_model.pkl")
model_columns = joblib.load("model_columns.pkl")

features = ["genre", "mood", "tempo", "language", "era"]

data_dict = {col: data[col].values for col in features}
data_len = len(data)

# -------------------------
# INITIAL PROBABILITIES
# -------------------------
probs = np.ones(data_len) / data_len

asked_categories = {f: 0 for f in features}
asked_questions = set()

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
for step in range(10):

    # 🔥 SHARPEN DISTRIBUTION
    probs = probs ** 1.3
    probs = probs / np.sum(probs)

    remaining = np.sum(probs > 0.01)

    # -------------------------
    # STOP CONDITION
    # -------------------------
    if np.max(probs) > 0.65:
        print("\n🎯 High confidence!")
        best_idx = np.argmax(probs)
        print("Prediction:", data.iloc[best_idx]["title"])
        break

    best_score = -1
    best_f = None
    best_v = None

    entropy = -np.sum(probs * np.log2(probs + 1e-9))

    # -------------------------
    # SELECT BEST QUESTION
    # -------------------------
    for f in features:
        values = set(data_dict[f])

        for v in values:

            if (f, v) in asked_questions:
                continue  # 🔥 avoid repetition

            mask = np.array([data_dict[f][i] == v for i in range(data_len)])

            prob_yes = np.sum(probs[mask])
            prob_no = np.sum(probs[~mask])

            if prob_yes == 0 or prob_no == 0:
                continue

            # 🔥 PERFECT BALANCE SCORE
            balance = 1 - abs(prob_yes - prob_no)

            # ML score
            inp = build_input(
                f, v,
                remaining,
                entropy,
                step,
                asked_categories[f]
            )

            ml_score = model.predict(inp)[0]

            # 🔥 COMBINED SCORE
            score = (0.5 * ml_score) + (0.5 * balance)

            # category penalty
            score *= (1 / (1 + asked_categories[f]))

            if score > best_score:
                best_score = score
                best_f = f
                best_v = v

    print(f"\nQ{step+1}: Is {best_f} = {best_v}?")

    # -------------------------
    # SIMULATED ANSWER (WITH NOISE)
    # -------------------------
    true_answer = data_dict[best_f][target_idx] == best_v

    if random.random() < 0.1:
        answer = not true_answer
        print("Answer (noisy):", "YES" if answer else "NO")
    else:
        answer = true_answer
        print("Answer:", "YES" if answer else "NO")

    # -------------------------
    # 🔥 STRONGER UPDATE RULE
    # -------------------------
    for i in range(data_len):

        match = data_dict[best_f][i] == best_v

        if answer:
            probs[i] *= 2.0 if match else 0.3
        else:
            probs[i] *= 0.3 if match else 2.0

    probs = probs / np.sum(probs)

    asked_categories[best_f] += 1
    asked_questions.add((best_f, best_v))

    # -------------------------
    # DEBUG OUTPUT
    # -------------------------
    top_idx = np.argsort(probs)[-5:][::-1]

    print("\nTop candidates:")
    for idx in top_idx:
        print(data.iloc[idx]["title"], "→", round(probs[idx], 3))