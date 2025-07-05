# annum-tab

#[![CodeQL Advanced](https://github.com/rcghpge/annum-tab/actions/workflows/codeql.yml/badge.svg)](https://github.com/rcghpge/annum-tab/actions/workflows/codeql.yml)
#[![Bandit](https://github.com/rcghpge/annum-tab/actions/workflows/bandit.yml/badge.svg)](https://github.com/rcghpge/annum-tab/actions/workflows/bandit.yml)

A research-focused deep learning repository dedicated to **tabular and NLP machine learning workflows**, symbolic reasoning, and advanced vectorization methods.

Built on the foundation of Saxton et al., 2019, this project explores ESA/SA vectorizers, T5-based inference, symbolic analysis, and hybrid modeling. It extends `annum-sdk`, a custom software development kit designed for data science, tabular ML, NLP, and broader AI workflows.

**Reference Paper:** [Analysing Mathematical Reasoning Abilities of Neural Models](https://openreview.net/pdf?id=H1gR5iR5FX)

**Key Datasets:**
- [Google DeepMind Mathematics Dataset](https://github.com/google-deepmind/mathematics_dataset)
- [Hugging Face Math Dataset](https://huggingface.co/datasets/deepmind/math_dataset)

---

## 📁 Project Structure

```
annum-tab/
├── LICENSE
├── README.md
├── __init__.py
├── data/
├── docs/
├── models/
├── notebooks/
├── predictions/
├── pyproject.toml
├── requirements.txt
├── src/
└── uv.lock
```

---

## ⚡ Setup

```bash
# Install uv if not already installed
pip install uv

# Create and activate a uv virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

---

## 🚀 Quick Start

```bash
# Run the base model
python models/basemodel.py

# Launch Jupyter Lab
jupyter lab
```

---

## 📊 Running & Viewing Results

1️⃣ Place your data in `data/` or corresponding tabular/NLP subdirectories.  

2️⃣ Run:

```bash
python models/basemodel.py
```

- Generates predictions
- Outputs metrics (accuracy, F1 score, symbolic correctness)
- Saves results to `predictions/`

3️⃣ Review summaries in `models/basemodel_summary.txt`.

---

## 🧪 Notebooks

Start Jupyter Lab:

```bash
jupyter lab
```

Then open `basemodel.ipynb` or `prototype.ipynb` inside `notebooks/` for exploratory analysis and interactive workflows.

---

## 🔗 Useful Links

- [Google DeepMind Mathematics Dataset](https://github.com/google-deepmind/mathematics_dataset)
- [Hugging Face Math Dataset](https://huggingface.co/datasets/deepmind/math_dataset)

---

### 💡 FreeBSD Tip

If you'd like to visualize the folder structure (like `tree`), install:

```bash
pkg install tree
```

Then run:

```bash
tree -L 2
```

---

## 🛡️ License

This project is licensed under the terms of the BSD-2 Clause License. See the [LICENSE](LICENSE) file for details.

