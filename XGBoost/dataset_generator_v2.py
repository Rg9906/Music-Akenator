import pandas as pd
import numpy as np
from math import log2
import random

# -------------------------
# LOAD DATA
# -------------------------
data = pd.read_csv("data.csv")

features = ["genre", "mood", "tempo", "language", "era"]

# -------------------------
# ENTROPY
# -------------------------
def compute_entropy(n):
    if n == 0:
        return 0
    probs = np.ones(n) / n
    return -sum(p * log2(p) for p in probs)

# -------------------------
# PREPROCESS (SPEED BOOST)
# -------------------------
data_dict = {col: data[col].values for col in features}
data_len = len(data)

# -------------------------
# DATASET GENERATOR
# -------------------------
def generate_dataset(num_simulations=3000):
    rows = []

    for sim in range(num_simulations):

        # 🔥 progress print (important for long runs)
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

            # -------------------------
            # EVALUATE ALL QUESTIONS
            # -------------------------
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

            # -------------------------
            # SELECT BEST QUESTION
            # -------------------------
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

                    reduction = (len(indices) - min(len(yes_indices), len(no_indices))) / len(indices)

                    penalty = 1 / (1 + asked_categories[f])

                    score = reduction * penalty

                    if score > best_score:
                        best_score = score
                        best_f = f
                        best_v = v
                        best_yes = yes_indices
                        best_no = no_indices

            # -------------------------
            # SAFETY CHECK
            # -------------------------
            if best_f is None or best_score <= 0:
                break

            # -------------------------
            # UPDATE STATE
            # -------------------------
            asked_categories[best_f] += 1

            if data_dict[best_f][target_idx] == best_v:
                indices = best_yes
            else:
                indices = best_no

    return pd.DataFrame(rows)

# -------------------------
# RUN GENERATION
# -------------------------
dataset = generate_dataset(num_simulations=3000)

dataset.to_csv("training_data_v2.csv", index=False)

print("Dataset created:", dataset.shape)