import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sympy
from transformers import T5Tokenizer, FlaxT5ForConditionalGeneration
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report


# ----------------------
# Output dir for predictions
# ----------------------
os.makedirs("../predictions", exist_ok=True)

# ----------------------
# Load small model for local test
# ----------------------
model_name = "google/t5-v1_1-large"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = FlaxT5ForConditionalGeneration.from_pretrained(model_name)

# ----------------------
# Load Q&A pairs from a single file
# ----------------------
def load_pairs(file_path):
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return [(lines[i], lines[i+1]) for i in range(0, len(lines), 2)]

# ----------------------
# Batch process for interpolate & extrapolate data
# ----------------------
interpolate_files = glob.glob("../math/interpolate/*.txt")
extrapolate_files = glob.glob("../math/extrapolate/*.txt")

interpolate_pairs = []
for file in interpolate_files:
    interpolate_pairs.extend(load_pairs(file))

extrapolate_pairs = []
for file in extrapolate_files:
    extrapolate_pairs.extend(load_pairs(file))

# ----------------------
# Prepare questions & answers
# ----------------------
"""
Uncomment for full test and/or validation set. 
Comment out the smaller test and/or validation sample sets
"""
#test_questions = [q for q, a in extrapolate_pairs]
#test_answers = [a for q, a in extrapolate_pairs]
test_questions = [q for q, a in interpolate_pairs][:10]  # first 10
test_answers = [a for q, a in interpolate_pairs][:10]



# ----------------------
# Evaluate (inference only, no fine-tune here)
# ----------------------
predictions = []

for question in tqdm(test_questions, desc="Evaluating"):
    prompt = f"solve: {question}"
    input_ids = tokenizer(prompt, return_tensors="jax").input_ids
    output_ids = model.generate(input_ids, max_length=50).sequences
    pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    predictions.append(pred)

# ----------------------
# Build DataFrame
# ----------------------
df = pd.DataFrame({
    "question": test_questions,
    "true_answer": test_answers,
    "predicted": predictions
})

output_file = os.path.join("../predictions", "predictions3.csv")
df.to_csv(output_file, index=False)
print(f" Predictions saved to: {output_file}")

# ----------------------
# Model Performance Metrics
# ----------------------
# Text-based correctness
df["text_match"] = df["true_answer"] == df["predicted"]

# Calculate metrics
accuracy_text = accuracy_score(df["true_answer"], df["predicted"])
f1_text = f1_score(df["true_answer"], df["predicted"], average="macro", zero_division=0)
precision_text = precision_score(df["true_answer"], df["predicted"], average="macro", zero_division=0)
recall_text = recall_score(df["true_answer"], df["predicted"], average="macro", zero_division=0)

print("\nText Match Metrics (Exact string):")
print(f" Accuracy:  {accuracy_text:.2f}")
print(f" Precision: {precision_text:.2f}")
print(f" Recall:    {recall_text:.2f}")
print(f" F1 Score:  {f1_text:.2f}")

print("\nClassification Report (Text Match):")
print(classification_report(df["true_answer"], df["predicted"], zero_division=0))

# ----------------------
# Symbolic equivalence check
# ----------------------
def is_equivalent(a1, a2):
    try:
        e1 = sympy.simplify(sympy.sympify(a1))
        e2 = sympy.simplify(sympy.sympify(a2))
        return e1 == e2
    except:
        return False

df["equivalent"] = [is_equivalent(t, p) for t, p in zip(df["true_answer"], df["predicted"])]
accuracy = df["equivalent"].mean()
print(f"\n Symbolic equivalence accuracy: {accuracy:.2f}")

# ----------------------
# Confusion matrix plot
# ----------------------
labels = list(set(df["true_answer"]).union(set(df["predicted"])))
cm = confusion_matrix(df["true_answer"], df["predicted"], labels=labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

plt.figure(figsize=(6, 1))
sns.heatmap(df[["equivalent"]].T, cmap="Greens", cbar=False, annot=True)
plt.title("Correctness Heatmap")
plt.yticks(rotation=0)
plt.show()

