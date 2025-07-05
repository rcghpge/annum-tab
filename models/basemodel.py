#!/usr/bin/env python
# coding: utf-8

# # Tabular Prototype

# # prototype annum

# ![annum project](../src/projects/annum/prototypesubmission.png) 

# # prototype version

# ![version project](../src/projects/version/prototypesubmission.png) 

# In[8]:


import joblib

sa_vec = joblib.load("../builds/sa_vectorizer.joblib")
esa_vec = joblib.load("../builds/esa_vectorizer.joblib")

print("sa vectorizer vocab size:", len(sa_vec.get_feature_names_out()))
print("esa vectorizer vocab size:", len(esa_vec.get_feature_names_out()))


# In[11]:


def load_txt_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

train_medium = load_txt_lines("../math/train-medium/arithmetic__div.txt")
# Similarly load train-medium, train-hard, test, etc.


# In[12]:


from sklearn.feature_extraction.text import TfidfVectorizer

# For standalone TF‑IDF transformation:
X_train_sa = sa_vec.transform(train_medium)
X_train_esa = esa_vec.transform(train_medium)

print("SA matrix shape:", X_train_sa.shape)
print("ESA matrix shape:", X_train_esa.shape)


# In[13]:


def load_lines(path):
    return [line.strip() for line in open(path, encoding="utf-8") if line.strip()]

train = load_lines("../math/train-medium/polynomials__collect.txt")  # adjust paths
test = load_lines("../math/interpolate/calculus__differentiate.txt")

print("Train count:", len(train), "Test count:", len(test))


# In[15]:


from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# Reduce ESA features to 2D
svd = TruncatedSVD(n_components=2)
coords = svd.fit_transform(X_train_esa)

plt.figure(figsize=(6, 6))
plt.scatter(coords[:, 0], coords[:, 1], s=5, alpha=0.5)
plt.title("ESA Feature PCA Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


# In[17]:


# Test for variance
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

svd = TruncatedSVD(n_components=2)
coords = svd.fit_transform(X_train_esa)
ratios = svd.explained_variance_ratio_

plt.figure(figsize=(6,4))
plt.plot(range(1, len(ratios)+1), ratios, 'o-')
plt.title("Scree Plot: Explained Variance by Component")
plt.xlabel("Component Number")
plt.ylabel("Explained Variance Ratio")
plt.grid(True)
plt.show()


# In[16]:


import numpy as np

components = svd.components_
feature_names = esa_vec.get_feature_names_out()

for idx, pc in enumerate(components):
    top_idx = np.argsort(np.abs(pc))[::-1][:10]
    print(f"Top features for PC{idx + 1}:")
    for i in top_idx:
        print(f"  {feature_names[i]} (loading {pc[i]:.3f})")
    print()


# In[18]:


import pandas as pd
import seaborn as sns

loadings = svd.components_.T  # shape: (n_features × 2)
feature_names = esa_vec.get_feature_names_out()  # your concept names

df_load = pd.DataFrame(loadings, index=feature_names, columns=["PC1", "PC2"])
top = df_load.abs().sum(axis=1).sort_values(ascending=False).head(20)
df_top = df_load.loc[top.index]

plt.figure(figsize=(8,10))
sns.heatmap(df_top, cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Top 20 Feature Loadings on PC1 & PC2")
plt.xlabel("Principal Components")
plt.ylabel("Features")
plt.show()


# In[19]:


import glob
import os

label_paths = glob.glob("../math/interpolate/*.txt")
labels = []

for path in label_paths:
    label = os.path.basename(os.path.dirname(path))
    labels.append(label)

print(f"Found {len(labels)} labels from interpolate/")


# In[21]:


import glob, os
from sklearn.feature_extraction.text import TfidfVectorizer

# Check working directory
print("Current working directory:", os.getcwd())

# List concept files
paths = glob.glob("../math/train-medium/*.txt")
print("Found concept files:", paths)

# Load into docs list
docs = []
for path in paths:
    with open(path, encoding='utf-8') as f:
        docs.append(f.read())

print("\nLoaded docs count:", len(docs))

# Preview first document
if docs:
    print("\nPreview:", docs[0][:200], "…")

# Build TF-IDF model
concept_vec = TfidfVectorizer(stop_words='english', max_features=10000)
X_concepts = concept_vec.fit_transform(docs)
print("\nConcept matrix shape:", X_concepts.shape)


# In[23]:


import matplotlib.pyplot as plt

# Select top K features by magnitude
K = 10
scores = coords
loadings = svd.components_.T
mags = np.linalg.norm(loadings, axis=1)
top_features = np.argsort(mags)[::-1][:K]

plt.figure(figsize=(7, 7))
plt.scatter(scores[:, 0], scores[:, 1], s=5, alpha=0.4)

for i in top_features:
    x, y = loadings[i] * np.abs(scores).max(axis=0)
    plt.arrow(0, 0, x, y, color='green', alpha=0.6, width=0.005)
    plt.text(x * 1.1, y * 1.1, feature_names[i], color='orange', fontsize=9)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("ESA Biplot with Top Concept Loadings")
plt.grid(alpha=0.3)
plt.show()


# In[33]:


import glob
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def load_docs_and_labels(base_path):
    paths = glob.glob(os.path.join(base_path, "*.txt"))
    docs = []
    labels = []

    for path in paths:
        with open(path, encoding='utf-8') as f:
            docs.append(f.read())

        # Extract label from file name prefix before '__'
        filename = os.path.basename(path)
        label = filename.split("__")[0]
        labels.append(label)

    print(f"Loaded {len(docs)} docs from {base_path}")
    return docs, labels

# === Train set ===
train_docs, train_labels = load_docs_and_labels("../math/train-medium")

# Fit TF-IDF on train
concept_vec = TfidfVectorizer(stop_words='english', max_features=10000)
X_train = concept_vec.fit_transform(train_docs)

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(train_labels)

print("\nTrain matrix shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("Label classes:", le.classes_)

# === Interpolate set ===
interpolate_docs, interpolate_labels = load_docs_and_labels("../math/interpolate")

X_interpolate = concept_vec.transform(interpolate_docs)
y_interpolate = le.transform(interpolate_labels)

print("\nInterpolate matrix shape:", X_interpolate.shape)
print("y_interpolate shape:", y_interpolate.shape)

# === Extrapolate set ===
extrapolate_docs, extrapolate_labels = load_docs_and_labels("../math/extrapolate")

X_extrapolate = concept_vec.transform(extrapolate_docs)
y_extrapolate = le.transform(extrapolate_labels)

print("\nExtrapolate matrix shape:", X_extrapolate.shape)
print("y_extrapolate shape:", y_extrapolate.shape)

# === Final shape summary ===
print("\nFinal shape summary:")
print("Train X:", X_train.shape, "Train y:", y_train.shape)
print("Interpolate X:", X_interpolate.shape, "Interpolate y:", y_interpolate.shape)
print("Extrapolate X:", X_extrapolate.shape, "Extrapolate y:", y_extrapolate.shape)


# In[37]:


import numpy as np
import tensorflow as tf
import random

# Set seeds
np.random.seed(578)
tf.random.set_seed(578)
random.seed(578)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

basemodel = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(len(np.unique(y_train)), activation='softmax')
])

basemodel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

basemodel.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


# In[25]:


print("Train shape:", X_train.shape)
print("Test shape:", X_interpolate.shape)


# In[55]:


# Evaluate on interpolate
interpolate_loss, interpolate_acc = basemodel.evaluate(X_interpolate, y_interpolate, verbose=1)
print(f"\nInterpolate set accuracy: {interpolate_acc:.4f}")

# Evaluate on extrapolate
extrapolate_loss, extrapolate_acc = basemodel.evaluate(X_extrapolate, y_extrapolate, verbose=1)
print(f"\nExtrapolate set accuracy: {extrapolate_acc:.4f}")


# # --------------------------------------------------------------------------

# In[56]:


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred_interpolate = basemodel.predict(X_interpolate)
y_pred_interpolate_classes = np.argmax(y_pred_interpolate, axis=1)

print("\nInterpolate classification report:")
print(classification_report(y_interpolate, y_pred_interpolate_classes, target_names=le.classes_))


# In[57]:


loss, acc = basemodel.evaluate(X_interpolate, y_interpolate)
print(f"\nInterpolate set accuracy: {acc:.4f}")


# In[58]:


y_pred_probs = basemodel.predict(X_interpolate)
y_pred_classes = np.argmax(y_pred_probs, axis=1)


# In[47]:


cm = confusion_matrix(y_interpolate, y_pred_classes, labels=range(len(le.classes_)))

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix - Interpolate Set")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()


# # ------------------------------------------------------------------------

# In[48]:


loss, acc = basemodel.evaluate(X_extrapolate, y_extrapolate)
print(f"\nExtrapolate set accuracy: {acc:.4f}")


# In[61]:


y_pred_probs = basemodel.predict(X_extrapolate)
y_pred_classes = np.argmax(y_pred_probs, axis=1)


# In[62]:


print("\nExtrapolate classification report:")
print(classification_report(
    y_extrapolate, y_pred_classes,
    labels=range(len(le.classes_)),
    target_names=le.classes_,
    zero_division=0
))


# In[51]:


cm = confusion_matrix(y_extrapolate, y_pred_classes, labels=range(len(le.classes_)))

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix - Extrapolate Set")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()


# # --------------------------------------------------------------------------------

# In[32]:


import os
from glob import glob

print("Working directory:", os.getcwd())
print("Top-level items:", os.listdir("."))

matches = glob("**/*", recursive=True)
print("All files and folders:")
for item in matches:
    print("  ", item)


# In[ ]:


from transformers import TFT5ForConditionalGeneration


# Open file for writing
with open("model_summary.txt", "w") as f:
    # Define custom print function to redirect to file
    def print_to_file(*args, **kwargs):
        print(*args, **kwargs, file=f)

    # Call summary and pass custom print function
    basemodel.summary(print_fn=print_to_file)

print(" Model summary saved to model_summary.txt")

# end of base model demo

