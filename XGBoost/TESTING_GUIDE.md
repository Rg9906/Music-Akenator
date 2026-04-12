# Music Akenator - Testing Guide

## Quick Start Commands

### 1. Generate Training Data
```bash
python dataset_generator_clean.py
```
**Expected Output:**
- Shows simulation progress (Simulation 0, 100, 200...)
- Final message: "Clean training dataset created: (rows, columns)"
- Creates: `training_data_clean_unique.csv`

### 2. Train ML Model
```bash
python main.py
```
**Expected Output:**
- Dataset shape: (X, Y)
- Feature shape: (X, Z)
- "Training model..."
- MSE score
- "Model saved!"
- Top 15 important features

### 3. Test Entropy Engine (Pure Logic)
```bash
python entropy_engine.py
```
**Expected Output:**
- Target song name
- Questions like "Q1: Is genre = pop?"
- Answers (yes/no)
- Remaining songs count
- Final candidates or prediction

### 4. Test ML Engine (With Probability Updates)
```bash
python ml_engine_final.py
```
**Expected Output:**
- Target song name
- Questions with ML scores
- Top candidates with probabilities
- "CONFIDENT!" message when prediction is ready

---

## Full Pipeline Test

### Option A: Run Everything Automatically
```bash
python workflow_pipeline.py
```
**Expected Output:**
- Complete pipeline with all 4 steps
- Progress indicators for each stage
- Final success message

### Option B: Step-by-Step Manual Test
```bash
# Step 1: Generate training data
python dataset_generator_clean.py

# Step 2: Train model
python main.py

# Step 3: Test entropy engine
python entropy_engine.py

# Step 4: Test ML engine
python ml_engine_final.py
```

---

## What to Look For

### Successful Training Data Generation:
- No error messages
- Simulation progress completes
- Training CSV file created

### Successful Model Training:
- MSE score < 0.1 (good)
- Model files created: `xgb_model.pkl`, `model_columns.pkl`

### Successful Engine Tests:
- Questions are generated
- Answers are processed
- Candidate list shrinks over time
- Final prediction is made

---

## Troubleshooting

### If you get "FileNotFoundError":
- Make sure you run steps in order
- Check that `dataset_final.csv` exists

### If you get "Model not found":
- Run `python main.py` first to train the model
- Check that `xgb_model.pkl` exists

### If questions are repetitive:
- This is normal - the system learns optimal questioning
- Different runs will produce different questions

### If no final prediction:
- Try running with more songs or different target
- Check the confidence threshold (0.75 in ml_engine_final.py)

---

## Quick Test Command

For a fast test of just the ML engine:
```bash
python ml_engine_final.py
```

This will show the complete game loop with probability updates and should give you a good idea if everything is working!
