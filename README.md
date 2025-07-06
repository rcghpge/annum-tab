# annum-tab

[![Dependabot Updates](https://github.com/rcghpge/annum-tab/actions/workflows/dependabot/dependabot-updates/badge.svg)](https://github.com/rcghpge/annum-tab/actions/workflows/dependabot/dependabot-updates)
[![CodeQL Advanced](https://github.com/rcghpge/annum-tab/actions/workflows/codeql.yml/badge.svg)](https://github.com/rcghpge/annum-tab/actions/workflows/codeql.yml)
[![Bandit](https://github.com/rcghpge/annum-tab/actions/workflows/bandit.yml/badge.svg)](https://github.com/rcghpge/annum-tab/actions/workflows/bandit.yml)
[![pages-build-deployment](https://github.com/rcghpge/annum-tab/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/rcghpge/annum-tab/actions/workflows/pages/pages-build-deployment)

annum-tab is a research-driven machine learning repository for tabular data + NLP architectures with an implementation test focus on mathematical problem solving, symbolic reasoning, math-based vectorization, large language model (LLM) development, and Python R&D for FreeBSD and BSD systems.

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

This project is tested on **Python version 3.13.2** running on FreeBSD version **14.3**.

---

## 📦 **Install dependencies using uv**

```bash
uv pip install -r requirements.txt
```

---

## 📊 **Data**

Shell scripts inside `data/` automate pulling and preparing math datasets.

### Example: Fetch math data

```bash
cd data/
sh download.sh
sh extract.sh
```

**Direct download link to bash script:**  
[download.sh](./data/download.sh)

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

1️⃣ Place or pull math dataset files into `data/` (or run `download.sh` and `extract.sh`).  
2️⃣ Run model builds in `models/` to generate model(s) inference on the example domain of the model's expertise, performance, and accuracy metrics.  
3️⃣ Outputs include accuracy, F1 score, symbolic correctness metrics, and other relevant results to the example test focus.

---

## 🛰 **FreeBSD + Python Ecosystem**

- Built and tested on FreeBSD **14.3** only due to time constraints.
- This is research-driven repository. This is not a completed build.
- Developed with `uv` and pip integration dependency resolution.
- Shell scripts are POSIX-compliant for broad compatibility on BSD systems.

---

## 🔗 **Links**

- [Google DeepMind Mathematics Dataset](https://github.com/google-deepmind/mathematics_dataset)
- [Hugging Face Math Dataset](https://huggingface.co/datasets/deepmind/math_dataset)

---

## 💬 **License**

BSD 3-Clause License and MIT.  

---
