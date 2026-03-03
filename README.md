# 🔬 ML From Scratch

> No sklearn. No shortcuts. Just Python, NumPy, and first principles.

This repo implements classical Machine Learning algorithms and Deep Learning architectures completely from scratch — understanding every equation, every gradient, every design decision before ever touching a library.

> **Analogy:** Most ML tutorials teach you to drive by handing you the keys. This repo teaches you how the engine works first.

---

## 🗂️ What's Inside

### 📐 Classical ML

| # | Notebook | Key Concepts | Blog |
|---|---|---|---|
| 01 | [Linear Regression](https://github.com/Sonal12061/ml-from-scratch/blob/main/LinearRegression.ipynb) | MSE loss, gradient descent, normal equation | — |
| 02 | [Logistic Regression](https://github.com/Sonal12061/ml-from-scratch/blob/main/LogisticRegression.ipynb) | Sigmoid, BCE loss, log-odds, decision boundary | [Intuitive](https://medium.com/@sonal.mishra1297/logistic-regression-explained-intuitively-from-probabilities-to-log-odds-25d32c39de4e) · [Mathematical](https://medium.com/@sonal.mishra1297/logistic-regression-explained-mathematically-from-loss-function-to-gradients-and-training-6d39a696c948) |
| 03 | [Decision Tree](https://github.com/Sonal12061/ml-from-scratch/blob/main/Decision_Tree.ipynb) | Gini impurity, information gain, recursive splitting | — |
| 04 | [Naive Bayes](https://github.com/Sonal12061/ml-from-scratch/blob/main/Naive_Bayes.ipynb) | Bayes theorem, conditional independence, MLE | — |
| 05 | [K-Means Clustering](https://github.com/Sonal12061/ml-from-scratch/blob/main/K_Means_Clustering.ipynb) | Centroid initialisation, inertia, convergence | — |

### 🧠 Deep Learning

| # | Notebook | Key Concepts | Blog |
|---|---|---|---|
| 01 | [RNN From Scratch](https://github.com/Sonal12061/ml-from-scratch/blob/main/RNN_from_scratch.ipynb) | Hidden state, BPTT, vanishing gradient | — |
| 02 | [LSTM From Scratch](https://github.com/Sonal12061/ml-from-scratch/blob/main/LSTM_from_scratch.ipynb) | Forget/input/output gates, cell state | — |
| 03 | [Transformer From Scratch](https://github.com/Sonal12061/ml-from-scratch/blob/main/Transformer_from_scratch.ipynb) | Self-attention, positional encoding, multi-head attention,PyTorch |  |
| 04 | [miniGPT From Scratch](https://github.com/Sonal12061/ml-from-scratch/blob/main/miniGPT_from_scratch.ipynb) | Causal attention, token generation, training on real text,PyTorch |  |

---

## 🧭 The Philosophy

Most people learn ML top-down — they start with `model.fit()` and work backwards when something breaks. This repo goes bottom-up:

```
Mathematical intuition
        ↓
Implement in pure NumPy / Python
        ↓
Validate against known datasets
        ↓
Understand what libraries are doing under the hood
```

Each notebook follows the same structure:
1. **The Problem** — what are we trying to solve and why?
2. **The Math** — derive the key equations from scratch
3. **The Implementation** — code it up with no shortcuts
4. **The Sanity Check** — verify against sklearn or a known result
5. **The Intuition** — what does this actually mean?

---

## 📐 Classical ML — What "From Scratch" Means

No `sklearn`, no `statsmodels`. Just `numpy` and `matplotlib`.

**Example — Logistic Regression:**
```python
# What sklearn hides:
model = LogisticRegression().fit(X, y)

# What this repo shows:
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def gradient_descent(X, y, lr=0.01, epochs=1000):
    w = np.zeros(X.shape[1])
    b = 0
    for _ in range(epochs):
        y_pred = sigmoid(X @ w + b)
        dw = X.T @ (y_pred - y) / len(y)
        db = np.mean(y_pred - y)
        w -= lr * dw
        b -= lr * db
    return w, b
```

---

## 🧠 Deep Learning — What "From Scratch" Means

No `torch.nn`, no `model.compile()`. Pure `PyTorch` tensors and autograd only.

**Example — Self-Attention:**
```python
# What HuggingFace hides:
output = BertModel(input_ids)

# What this repo shows:
def self_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)  # scaled dot product
    weights = F.softmax(scores, dim=-1)                   # attention distribution
    return weights @ V                                     # weighted sum of values
```

---

## 🔁 How This Repo Connects to the Rest of My Work

This repo is the **foundation layer** of a larger portfolio:

```
ml-from-scratch          ← understand the internals (this repo)
        ↓
distilbert-sentiment     ← apply pre-trained transformers (full fine-tuning)
        ↓
lora-tinyllama-medical   ← efficient adaptation of LLMs (PEFT/LoRA)
        ↓
LangGraph_Implementations ← orchestrate multiple LLM agents
```

Building from scratch isn't just academic — it's what lets you debug production systems, explain model behaviour to stakeholders, and make principled architecture decisions.

---

## 🛠️ Setup

All notebooks run in **Google Colab** or locally.

```bash
git clone https://github.com/Sonal12061/ml-from-scratch
cd ml-from-scratch
pip install numpy matplotlib torch jupyter
```

No GPU needed for classical ML notebooks. Deep learning notebooks run on Colab T4.

---

## ✍️ Paired Reading

Each implementation is paired with a blog that explains the *why*, not just the *what*:

- [Logistic Regression Explained Intuitively](https://medium.com/@sonal.mishra1297/logistic-regression-explained-intuitively-from-probabilities-to-log-odds-25d32c39de4e)
- [Logistic Regression Explained Mathematically](https://medium.com/@sonal.mishra1297/logistic-regression-explained-mathematically-from-loss-function-to-gradients-and-training-6d39a696c948)

More blogs coming as new notebooks are added.

---

*Built with 🔬 curiosity | No shortcuts taken*
