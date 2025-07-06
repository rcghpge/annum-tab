# annum-tab

[![CodeQL Advanced](https://github.com/rcghpge/annum-tab/actions/workflows/codeql.yml/badge.svg)](https://github.com/rcghpge/annum-tab/actions/workflows/codeql.yml)
[![Bandit](https://github.com/rcghpge/annum-tab/actions/workflows/bandit.yml/badge.svg)](https://github.com/rcghpge/annum-tab/actions/workflows/bandit.yml)

annum-tab is a research-driven machine learning repository for tabular data + NLP architectures with an implementation test focus on mathematical problem solving, symbolic reasoning, math-based vectorization, large language model (LLM) development, and Python R$D for FreeBSD and BSD systems.

Based on the original work of Saxton et al., 2019, annum-tab is an extension of `annum-sdk`, a BSD-native software development kit designed for data science, Python development for the Python ecosystem on FreeBSD + BSD systems, and other domains.

---

## 📄 **Research Base**

**Original Paper:** [Analysing Mathematical Reasoning Abilities of Neural Models](https://openreview.net/pdf?id=H1gR5iR5FX)

**Datasets:**
- [Google DeepMind Mathematics Dataset](https://github.com/google-deepmind/mathematics_dataset)
- [Hugging Face Math Dataset](https://huggingface.co/datasets/deepmind/math_dataset)

---

## 📁 **Project Structure**

```
annum-tab/
├── src/
├── data/               
├── docs/             
├── models/            
├── notebooks/         
├── requirements.txt    
├── pyproject.toml      
├── LICENSE
├── README.md
├── __init__.py
└── uv.lock
```

---

## ⚙️ **Setup (FreeBSD 14.3 & uv)**

### Install `uv` package manager

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

### Install Python (if not already installed)

```bash
pkg install python
```

This project is tested on **Python 3.13.2** with FreeBSD **14.3**.

---

## 📦 **Install dependencies using uv**

```bash
uv pip install -r requirements.txt
```

---

## 🐚 **Shell Data Scripts (in `data/`)**

Shell scripts inside `data/` automate pulling and preparing math datasets.

### Example: Fetch math data

```bash
cd data/
sh fetch_math_data.sh
```

**Direct download link to bash script:**  
[data.sh](./data/data.sh)

---

## 🚀 **Quick Start**

### Run baseline model

```bash
python models/basemodel.py
```

### Start Jupyter Lab

```bash
jupyter lab
```

Open any notebook in `notebooks/` to explore symbolic math reasoning workflows.

---

## 🧪 **Notebooks**

Recommended entry points:

- `notebooks/basemodel.ipynb`
- `notebooks/model2.ipynb`
- `notebooks/testbsd.ipynb`

Start:

```bash
jupyter lab
```

---

## 📊 **Viewing Results**

1️⃣ Place or pull math dataset files into `data/` (or run `data.sh` and `extract.sh`).  
2️⃣ Run model builds in `models/` to generate model(s) inference test use case focus and the model's metrics.  
3️⃣ Outputs include accuracy, F1 score, symbolic correctness metrics, and other relevant inference to example implementation.

---

## 🛰 **FreeBSD Python Ecosystem Notes**

- Built and tested on FreeBSD **14.3** only.
- Uses `uv` with pip integration dependency resolution.
- Shell scripts are POSIX-compliant for broad compatibility on BSD systems.

---

## 🔗 **Links**

- [Google DeepMind Mathematics Dataset](https://github.com/google-deepmind/mathematics_dataset)
- [Hugging Face Math Dataset](https://huggingface.co/datasets/deepmind/math_dataset)

---

## 💬 **License**

BSD 3-Clause License and MIT.  

---
