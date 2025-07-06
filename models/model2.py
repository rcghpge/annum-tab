import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, GridSearchCV

np.random.seed(578)

# --- Helper functions ---
def load_docs_and_labels(paths):
    docs = []
    labels = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            content = f.read().strip()
            docs.append(content)
        filename = os.path.basename(path)
        label = os.path.splitext(filename)[0]
        labels.append(label)
    return docs, labels

def preview_counter(counter_obj, n=10):
    return dict(counter_obj.most_common(n))

def extract_coarse_label(label):
    return label.split("__")[0]

def safe_transform(labels, le):
    known_classes = set(le.classes_)
    new_labels = []
    unknown_count = 0
    for label in labels:
        if label in known_classes:
            new_labels.append(le.transform([label])[0])
        else:
            new_labels.append(-1)
            unknown_count += 1
    return np.array(new_labels), unknown_count

# --- File paths ---
train_paths = glob.glob("math/train-medium/*.txt")
inter_paths = glob.glob("math/interpolate/*.txt")
extra_paths = glob.glob("math/extrapolate/*.txt")

train_docs, train_labels = load_docs_and_labels(train_paths)
inter_docs, inter_labels = load_docs_and_labels(inter_paths)
extra_docs, extra_labels = load_docs_and_labels(extra_paths)

# --- Map categorical features to main math subject areas ---
train_labels_coarse = [extract_coarse_label(lab) for lab in train_labels]
inter_labels_coarse = [extract_coarse_label(lab) for lab in inter_labels]
extra_labels_coarse = [extract_coarse_label(lab) for lab in extra_labels]

print("Train label distribution preview (coarse):", preview_counter(collections.Counter(train_labels_coarse)))
print("Interpolate label distribution preview (coarse):", preview_counter(collections.Counter(inter_labels_coarse)))
print("Extrapolate label distribution preview (coarse):", preview_counter(collections.Counter(extra_labels_coarse)))

# --- Encode labels ---
le = LabelEncoder()
le.fit(train_labels_coarse)
y_train = le.transform(train_labels_coarse)

y_inter, unknown_count_inter = safe_transform(inter_labels_coarse, le)
print(f"Interpolate unknown classes count: {unknown_count_inter}")

y_extra, unknown_count_extra = safe_transform(extra_labels_coarse, le)
print(f"Extrapolate unknown classes count: {unknown_count_extra}")

# --- Define pipeline ---
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000)),
    ('select', SelectKBest(score_func=chi2, k=5000)),
    ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=578))
])

# === Main fixed hyperparameter build ===
pipeline.fit(train_docs, y_train)

# --- Evaluate interpolate set ---
mask_inter = y_inter != -1
if np.any(mask_inter):
    y_inter_known = y_inter[mask_inter]
    inter_docs_known = np.array(inter_docs)[mask_inter]
    y_inter_pred = pipeline.predict(inter_docs_known)

    present_labels = np.unique(y_inter_known)
    present_class_names = le.inverse_transform(present_labels)

    print("\nClassification report on interpolate set (known classes only):")
    print(classification_report(
        y_inter_known, 
        y_inter_pred, 
        labels=present_labels,
        target_names=present_class_names,
        zero_division=0
    ))
    ConfusionMatrixDisplay.from_predictions(
        y_inter_known, 
        y_inter_pred, 
        display_labels=present_class_names,
        xticks_rotation="vertical"
    )
    plt.title("Confusion Matrix — Interpolate")
    plt.show()
else:
    print("No known classes in interpolate set to evaluate.")

# --- Evaluate extrapolate set ---
mask_extra = y_extra != -1
if np.any(mask_extra):
    y_extra_known = y_extra[mask_extra]
    extra_docs_known = np.array(extra_docs)[mask_extra]
    y_extra_pred = pipeline.predict(extra_docs_known)

    present_labels = np.unique(y_extra_known)
    present_class_names = le.inverse_transform(present_labels)

    print("\nClassification report on extrapolate set (known classes only):")
    print(classification_report(
        y_extra_known, 
        y_extra_pred, 
        labels=present_labels,
        target_names=present_class_names,
        zero_division=0
    ))
    ConfusionMatrixDisplay.from_predictions(
        y_extra_known, 
        y_extra_pred, 
        display_labels=present_class_names,
        xticks_rotation="vertical"
    )
    plt.title("Confusion Matrix — Extrapolate")
    plt.show()
else:
    print("No known classes in extrapolate set to evaluate.")


# === Optional tuning block for separate evaluation (uncomment for test passes) ===
"""
param_grid = {
    'select__k': [3000, 5000, 7000],
    'clf__C': [0.1, 1.0, 10]
}

cv = KFold(n_splits=3, shuffle=True, random_state=578)
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)

grid.fit(train_docs, y_train)

print("\nBest parameters from tuning:", grid.best_params_)
print("Best CV score from tuning:", grid.best_score_)

# Evaluate tuned model
pipeline_tuned = grid.best_estimator_

# Evaluate interpolate set for tuned
if np.any(mask_inter):
    y_inter_pred_tuned = pipeline_tuned.predict(inter_docs_known)

    present_labels = np.unique(y_inter_known)
    present_class_names = le.inverse_transform(present_labels)

    print("\n[Tuned] Classification report on interpolate set (known classes only):")
    print(classification_report(
        y_inter_known, 
        y_inter_pred_tuned, 
        labels=present_labels,
        target_names=present_class_names,
        zero_division=0
    ))
    ConfusionMatrixDisplay.from_predictions(
        y_inter_known, 
        y_inter_pred_tuned, 
        display_labels=present_class_names,
        xticks_rotation="vertical"
    )
    plt.title("[Tuned] Confusion Matrix — Interpolate")
    plt.show()

# Evaluate extrapolate set for tuned
if np.any(mask_extra):
    y_extra_pred_tuned = pipeline_tuned.predict(extra_docs_known)

    present_labels = np.unique(y_extra_known)
    present_class_names = le.inverse_transform(present_labels)

    print("\n[Tuned] Classification report on extrapolate set (known classes only):")
    print(classification_report(
        y_extra_known, 
        y_extra_pred_tuned, 
        labels=present_labels,
        target_names=present_class_names,
        zero_division=0
    ))
    ConfusionMatrixDisplay.from_predictions(
        y_extra_known, 
        y_extra_pred_tuned, 
        display_labels=present_class_names,
        xticks_rotation="vertical"
    )
    plt.title("[Tuned] Confusion Matrix — Extrapolate")
    plt.show()

print("\nAll done!")
"""

