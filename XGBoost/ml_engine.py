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

# -------------------------
# PREPROCESS
# -------------------------
data_dict = {col: data[col].values for col in features}
data_len = len(data)

# -------------------------
# BUILD INPUT
# -------------------------
def build_input(feature, value, remaining, entropy, step, category_count):

    # initialize all columns
    row = {col: 0 for col in model_columns}

    # state features
    row["remaining"] = remaining
    row["entropy"] = entropy
    row["step"] = step
    row["category_count"] = category_count

    # feature encoding
    key = f"feature_value_{feature}_{value}"
    if key in row:
        row[key] = 1

    return pd.DataFrame([row])

# -------------------------
# SIMULATION
# -------------------------
indices = list(range(data_len))
target_idx = random.choice(indices)

asked_categories = {f: 0 for f in features}

print("🎯 Target:", data.iloc[target_idx]["title"])

for step in range(10):

    if len(indices) <= 2:
        print("\n🎯 Final candidates:")
        print(data.iloc[indices]["title"].values)
        break

    best_score = -1
    best_f = None
    best_v = None

    remaining = len(indices)
    entropy = np.log2(remaining)

    # -------------------------
    # TRY ALL QUESTIONS
    # -------------------------
    for f in features:
        values = set(data_dict[f][i] for i in indices)

        for v in values:

            yes_indices = [i for i in indices if data_dict[f][i] == v]
            no_indices = [i for i in indices if data_dict[f][i] != v]

            if len(yes_indices) == 0 or len(no_indices) == 0:
                continue

            # -------------------------
            # 🔥 LATE-GAME SWITCH
            # -------------------------
            if remaining <= 6:
                # pure elimination strategy
                score = min(len(yes_indices), len(no_indices)) / len(indices)

            else:
                # build model input
                inp = build_input(
                    f,
                    v,
                    remaining,
                    entropy,
                    step,
                    asked_categories[f]
                )

                # -------------------------
                # BASE MODEL SCORE
                # -------------------------
                score = model.predict(inp)[0]

                # -------------------------
                # CATEGORY PENALTY
                # -------------------------
                penalty = 1 / (1 + asked_categories[f])
                score *= penalty

                # -------------------------
                # BALANCE FACTOR
                # -------------------------
                balance = min(len(yes_indices), len(no_indices)) / len(indices)
                score *= balance

                # -------------------------
                # EXPLORATION
                # -------------------------
                score += random.uniform(0, 0.01)

            # -------------------------
            # SELECT BEST
            # -------------------------
            if score > best_score:
                best_score = score
                best_f = f
                best_v = v
                best_yes = yes_indices
                best_no = no_indices

    # -------------------------
    # SAFETY CHECK
    # -------------------------
    if best_f is None:
        break

    print(f"\nQ{step+1}: Is {best_f} = {best_v}?")

    # simulate answer
    if data_dict[best_f][target_idx] == best_v:
        print("Answer: YES")
        indices = best_yes
    else:
        print("Answer: NO")
        indices = best_no

    asked_categories[best_f] += 1

    print("Remaining:", len(indices))