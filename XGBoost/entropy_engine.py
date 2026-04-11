import pandas as pd
import numpy as np
from math import log2

# -------------------------
# LOAD DATA
# -------------------------
data = pd.read_csv("data.csv")

features = ["genre", "mood", "tempo", "language", "era"]

# -------------------------
# ENTROPY
# -------------------------
def compute_entropy(df):
    n = len(df)
    if n == 0:
        return 0
    probs = np.ones(n) / n
    return -sum(p * log2(p) for p in probs)

# -------------------------
# VALUE-BASED INFORMATION GAIN
# -------------------------
def value_info_gain(df, feature, value):
    total_entropy = compute_entropy(df)

    yes_subset = df[df[feature] == value]
    no_subset = df[df[feature] != value]

    if len(yes_subset) == 0 or len(no_subset) == 0:
        return -1  # useless split

    p_yes = len(yes_subset) / len(df)
    p_no = len(no_subset) / len(df)

    new_entropy = (
        p_yes * compute_entropy(yes_subset) +
        p_no * compute_entropy(no_subset)
    )

    return total_entropy - new_entropy

# -------------------------
# SELECT BEST QUESTION
# -------------------------
def select_best_question(df, asked_categories, asked_pairs):
    best_score = -1
    best_q = None

    for f in features:
        values = df[f].unique()

        for v in values:
            if (f, v) in asked_pairs:
                continue

            gain = value_info_gain(df, f, v)

            if gain <= 0:
                continue

            # CATEGORY PENALTY
            penalty = 1 / (1 + asked_categories[f])

            score = gain * penalty

            if score > best_score:
                best_score = score
                best_q = (f, v)

    return best_q

# -------------------------
# SIMULATION
# -------------------------

df = data.copy()

asked_categories = {f: 0 for f in features}
asked_pairs = set()

# pick a hidden target (simulate user)
target_song = df.sample(1).iloc[0]

print("🎯 Target song (hidden):", target_song["title"])

for step in range(10):

    if len(df) <= 3:
        print("\n🎯 Final candidates:")
        print(df["title"].values)
        break

    q = select_best_question(df, asked_categories, asked_pairs)

    if q is None:
        print("\nNo more useful questions.")
        break

    f, v = q

    print(f"\nQ{step+1}: Is {f} = {v}?")

    # simulate real answer based on hidden target
    answer = "yes" if target_song[f] == v else "no"

    print("Answer:", answer)

    asked_pairs.add((f, v))
    asked_categories[f] += 1

    # FILTER based on answer
    if answer == "yes":
        df = df[df[f] == v]
    else:
        df = df[df[f] != v]

    print("Remaining songs:", len(df))