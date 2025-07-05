import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Load data
data = pd.DataFrame({
    "problem": ["Integrate x^2 dx", "Solve for x: x + 3 = 5"],
    "label": [1, 0]  # Example labels, e.g., solvable or difficulty level
})

# Generate NLP feature extractor
vectorizer = TfidfVectorizer()

# Run test run on tabular + NLP model
model = make_pipeline(vectorizer, LogisticRegression())

# Train
X = data["problem"]
y = data["label"]
model.fit(X, y)

# Run Inference
pred = model.predict(["Integrate x^3 dx"])
print(f"Predicted label: {pred[0]}")
