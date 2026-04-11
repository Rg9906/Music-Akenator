import pandas as pd
import numpy as np
from math import log2

data = pd.read_csv("data.csv")

features = ["genre", "mood", "tempo", "language", "era"]

def compute_entropy(df):
    n = len(df)
    if n == 0:
        return 0
    probs = np.ones(n) / n
    return -sum(p * log2(p) for p in probs)

def info_gain(df, feature):
    total_entropy = compute_entropy(df)
    values = df[feature].unique()

    weighted_entropy = 0
    for v in values:
        subset = df[df[feature] == v]
        prob = len(subset) / len(df)
        weighted_entropy += prob * compute_entropy(subset)

    return total_entropy - weighted_entropy

def select_best_question(df, asked_categories, asked_pairs):
    best_score = -1
    best_q = None

    for f in features:
        values = df[f].unique()

        for v in values:
            if (f, v) in asked_pairs:
                continue

            subset = df[df[f] == v]
            if len(subset) == 0:
                continue

            gain = info_gain(df, f)

            # 🔥 CATEGORY PENALTY
            penalty = 1 / (1 + asked_categories[f])

            score = gain * penalty

            if score > best_score:
                best_score = score
                best_q = (f, v)

    return best_q


# -------------------------
# SIMULATION LOOP
# -------------------------

df = data.copy()

asked_categories = {f: 0 for f in features}
asked_pairs = set()

for step in range(5):
    q = select_best_question(df, asked_categories, asked_pairs)

    if q is None:
        break

    f, v = q

    print(f"\nQ{step+1}: Is {f} = {v}?")

    # simulate answer
    answer = "yes" if df[f].iloc[0] == v else "no"

    print("Answer:", answer)

    asked_pairs.add((f, v))
    asked_categories[f] += 1

    # 🔥 FILTER DATA
    if answer == "yes":
        df = df[df[f] == v]
    else:
        df = df[df[f] != v]

    print("Remaining songs:", len(df))

    if len(df) <= 1:
        break