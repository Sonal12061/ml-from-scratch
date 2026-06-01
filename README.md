# ML from Scratch

> No shortcuts. Every algorithm implemented three ways — **pure math**, **NumPy**, and **scikit-learn** — so you understand what the library is doing before you use it.

A progressive, interview-ready collection of classical ML algorithms on real datasets. Every notebook runs top-to-bottom, saves a model file to `Models/`, and feeds into a single `model_comparison.ipynb` that benchmarks all methods without re-training.

---

## Project Structure

```
ml-from-scratch/
│
├── Models/                          # Saved model files (auto-created on first run)
│   ├── lr_scratch.json / lr_numpy.npz / lr_sklearn.joblib
│   ├── ridge_numpy.npz / ridge_sklearn.joblib
│   ├── lasso_numpy.npz / lasso_sklearn.joblib
│   ├── logreg_scratch.json / logreg_numpy.npz / logreg_sklearn.joblib
│   ├── naive_bayes_scratch.json / naive_bayes_numpy.npz / naive_bayes_sklearn.joblib
│   ├── dt_scratch.json / dt_numpy.json / dt_sklearn.joblib
│   ├── knn_scratch.json / knn_numpy.npz / knn_sklearn.joblib
│   ├── kmeans_scratch.json / kmeans_numpy.npz / kmeans_sklearn.joblib
│   ├── pca_scratch.json / pca_numpy.npz / pca_sklearn.joblib
│   ├── svm_scratch.npz / svm_numpy.npz / svm_sklearn.joblib
│   ├── nn_scratch.npz / nn_numpy.npz
│   └── nn_pytorch_weights.pth / nn_pytorch_config.json
│
├── Linear_Regression/
│   ├── linear_regression_scratch.ipynb
│   ├── linear_regression_numpy.ipynb
│   ├── linear_regression_sklearn.ipynb
│   └── linear_regression_assumptions.ipynb
│
├── Logistic_Regression/
│   ├── logistic_regression_scratch.ipynb
│   ├── logistic_regression_numpy.ipynb
│   └── logistic_regression_sklearn.ipynb
│
├── Naive Bayes/
│   ├── naive_bayes_scratch.ipynb
│   ├── naive_bayes_numpy.ipynb
│   └── naive_bayes_sklearn.ipynb
│
├── Decision Trees/
│   ├── decision_tree_scratch.ipynb
│   ├── decision_tree_numpy.ipynb
│   └── decision_tree_sklearn.ipynb
│
├── KNN/
│   ├── knn_scratch.ipynb
│   ├── knn_numpy.ipynb
│   └── knn_sklearn.ipynb
│
├── K_Means/
│   ├── kmeans_scratch.ipynb
│   ├── kmeans_numpy.ipynb
│   └── kmeans_sklearn.ipynb
│
├── PCA/
│   ├── pca_scratch.ipynb
│   ├── pca_numpy.ipynb
│   └── pca_sklearn.ipynb
│
├── Ridge lasso/
│   ├── ridge_lasso_numpy.ipynb
│   └── ridge_lasso_sklearn.ipynb
│
├── SVM/
│   ├── svm_scratch.ipynb
│   ├── svm_numpy.ipynb
│   └── svm_sklearn.ipynb
│
├── Neural Network/
│   ├── nn_scratch.ipynb
│   ├── nn_numpy.ipynb
│   └── nn_pytorch.ipynb
│
├── Data_Preprocessing/
│   └── data_preprocessing.ipynb
│
└── model_comparison.ipynb
```

---

## Algorithms

### Linear Regression
**Dataset:** California Housing · **Split:** `train_test_split(random_state=42)`

| Notebook | What's implemented |
|---|---|
| `linear_regression_scratch` | OLS via Gauss-Jordan elimination + Batch GD (min-max scaling) — pure pandas |
| `linear_regression_numpy` | OLS via `np.linalg.inv` + GD (z-score scaling) — numpy only |
| `linear_regression_sklearn` | `LinearRegression` with coefficient table |
| `linear_regression_assumptions` | All 7 OLS assumptions: linearity, independence (Durbin-Watson), homoscedasticity, normality (Q-Q + Shapiro-Wilk), VIF, Cook's Distance, ACF |

**Metrics:** MSE · RMSE · R²

---

### Ridge & Lasso Regression
**Dataset:** California Housing · **Split:** `train_test_split(random_state=42)`

| Notebook | What's implemented |
|---|---|
| `ridge_lasso_numpy` | Ridge closed-form `(XᵀX + αI)⁻¹Xᵀy` + Lasso coordinate descent with soft thresholding — numpy only |
| `ridge_lasso_sklearn` | `RidgeCV`, `LassoCV`, `ElasticNetCV` + coefficient paths + LassoCV MSE path |

**Key concepts:** L1 vs L2 penalty · soft thresholding · feature selection · regularisation paths · ElasticNet

---

### Logistic Regression
**Dataset:** NLTK Twitter Samples (5k positive + 5k negative tweets) · **Split:** shuffle `seed=42` → 80/20

| Notebook | What's implemented |
|---|---|
| `logistic_regression_scratch` | TF-IDF from scratch (pure Python) + sigmoid BCE gradient descent — pandas |
| `logistic_regression_numpy` | Vectorised TF-IDF + `sigmoid(X @ beta)` GD — numpy |
| `logistic_regression_sklearn` | `TfidfVectorizer(ngram_range=(1,2))` + `LogisticRegression` + ROC curve + C sweep |

**Metrics:** Accuracy · Precision · Recall · F1 · ROC-AUC

---

### Naive Bayes
**Dataset:** NLTK Twitter Samples · **Split:** shuffle `seed=42` → 80/20

| Notebook | What's implemented |
|---|---|
| `naive_bayes_scratch` | Multinomial NB with Laplace smoothing via `Counter` + `math.log` — pure Python |
| `naive_bayes_numpy` | BoW matrix + vectorised log-posterior `X @ log_likelihood.T` + manual ROC |
| `naive_bayes_sklearn` | `Pipeline(TfidfVectorizer + MultinomialNB)` + α sweep |

**Metrics:** Accuracy · Precision · Recall · F1 · ROC-AUC

---

### Decision Tree
**Dataset:** Iris (4 features, 3 classes) · **Split:** `train_test_split(stratify=y, random_state=42)`

| Notebook | What's implemented |
|---|---|
| `decision_tree_scratch` | CART with Gini impurity — pure Python `Node` class, text tree printer, split-count importance |
| `decision_tree_numpy` | Vectorised Gini + boolean masking + Gini-weighted impurity decrease importance |
| `decision_tree_sklearn` | `DecisionTreeClassifier` + `plot_tree` + cost-complexity pruning + decision boundary |

**Metrics:** Accuracy · Precision · Recall · F1 · ROC-AUC (OVR)

---

### K-Nearest Neighbours
**Dataset:** Iris · **Split:** `train_test_split(stratify=y, random_state=42)`

| Notebook | What's implemented |
|---|---|
| `knn_scratch` | Euclidean + Manhattan distance — pure Python, k sweep, distance metric comparison |
| `knn_numpy` | Vectorised `(n_test, n_train, p)` broadcasting + inverse-distance weighting + manual 5-fold CV |
| `knn_sklearn` | `KNeighborsClassifier` + `GridSearchCV` over k × metric × weights + decision boundary |

**Metrics:** Accuracy · Precision · Recall · F1 · ROC-AUC (OVR)

---

### K-Means Clustering
**Dataset:** `make_blobs(n=1500, centers=4, random_state=42)` · **Unsupervised**

| Notebook | What's implemented |
|---|---|
| `kmeans_scratch` | Lloyd's algorithm, random init, `math.sqrt` distances — pure Python + pandas |
| `kmeans_numpy` | K-Means++ init (D² sampling) + vectorised `(n, k, d)` broadcasting + init comparison boxplot |
| `kmeans_sklearn` | `Pipeline(StandardScaler + KMeans(n_init=10))` + elbow + silhouette plot |

**Metrics:** Inertia (WCSS) · Silhouette · Davies-Bouldin · Calinski-Harabasz

---

### PCA
**Dataset:** California Housing (8 features) · **Unsupervised**

| Notebook | What's implemented |
|---|---|
| `pca_scratch` | Covariance matrix (pandas) → `np.linalg.eigh` → project with `dot` + biplot |
| `pca_numpy` | Cov+Eigh AND SVD side-by-side — proves numerical equivalence; explains why SVD is more stable |
| `pca_sklearn` | `Pipeline(StandardScaler + PCA)` + loadings heatmap + PCA→LR cross-val benchmark |

**Outputs:** Scree plot · Cumulative variance · 2D scatter · Biplot · Reconstruction MSE vs k

---

### SVM (Support Vector Machine)
**Dataset:** Breast Cancer (30 features, binary) · **Split:** `train_test_split(stratify=y, random_state=42)`

| Notebook | What's implemented |
|---|---|
| `svm_scratch` | Soft-margin primal via subgradient descent — hinge loss, support vector identification, decision boundary (PCA 2D), C sweep |
| `svm_numpy` | `LinearSVM` class (mini-batch, 3 LR schedules) + **Random Fourier Features** for kernel approximation without O(n²) Gram matrix |
| `svm_sklearn` | `SVC` with linear/RBF/poly/sigmoid kernels + `GridSearchCV` + decision boundary + ROC curves |

**Key concepts:** Hinge loss · margin maximisation · support vectors · kernel trick · RFF approximation

---

### Neural Network (MLP + Backpropagation)
**Dataset:** Breast Cancer (30 features, binary) · **Split:** `train_test_split(stratify=y, random_state=42)`

| Notebook | What's implemented |
|---|---|
| `nn_scratch` | Manual forward pass, full backprop derivation, gradient check (numerical vs analytical) — pure numpy |
| `nn_numpy` | OOP `Dense` + `MLP` classes, mini-batch Adam, SGD vs Momentum vs Adam comparison, L2 sweep |
| `nn_pytorch` | `nn.Module`, `BCEWithLogitsLoss`, Adam + `ReduceLROnPlateau`, early stopping, architecture comparison |

**Key concepts:** He initialisation · chain rule · soft thresholding · Adam bias correction · `BCEWithLogitsLoss` vs `BCE(sigmoid(z))`

---

### Data Preprocessing
**Dataset:** Synthetic (numerical) + NLTK Twitter Samples (text)

| Section | What's covered |
|---|---|
| **Numerical — Missing Values** | Mean/Median/Mode, KNN, MICE (IterativeImputer) — distribution comparison |
| **Numerical — Outliers** | IQR, Z-score, Isolation Forest — box plots, winsorizing, log transform |
| **Numerical — Scaling** | StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler — side-by-side grid |
| **Numerical — Encoding** | Label, Ordinal, One-Hot, Target, Frequency encoding |
| **Numerical — Transformations** | Log, sqrt, Box-Cox, Yeo-Johnson, Polynomial features |
| **Numerical — Feature Selection** | VarianceThreshold, correlation heatmap, SelectKBest (F + MI), RFE, Lasso |
| **Numerical — Imbalanced Data** | SMOTE (from scratch), random over/under-sampling (from scratch, no imblearn) |
| **Numerical — Full Pipeline** | `ColumnTransformer` + `Pipeline` + 5-fold CV in one block |
| **Text — Cleaning** | URLs, HTML, mentions, hashtags, contractions, punctuation — configurable `clean_text()` |
| **Text — Tokenisation** | Word tokenise, sentence tokenise, whitespace baseline |
| **Text — Stop Words** | Removal + token frequency visualisation before/after |
| **Text — Stemming/Lemmatisation** | Porter, Snowball, WordNet (POS-aware) — comparison table |
| **Text — Vectorisation** | BoW, binary BoW, TF-IDF heatmap, unigrams/bigrams |
| **Text — Feature Engineering** | 12 features: char/word/sent count, lexical diversity, uppercase ratio |
| **Text — End-to-End Pipeline** | `full_text_preprocess()` → TF-IDF → MultinomialNB → predict new tweets |

---

## Model Comparison

`model_comparison.ipynb` loads every saved model from `Models/` and evaluates all methods on the same test set — **no re-training, no duplicated code**.

| Task | Dataset | Methods compared | Primary metric |
|---|---|---|---|
| Linear Regression | California Housing | OLS Pandas · GD Pandas · OLS NumPy · GD NumPy · sklearn · Ridge · Lasso | R² |
| Tweet Classification | Twitter Samples | LogReg GD Pandas/NumPy/sklearn · NB Scratch/NumPy/sklearn | F1 / ROC-AUC |
| K-Means | make_blobs | Scratch · NumPy · sklearn | Silhouette / Inertia |
| Iris Classification | Iris | DT Scratch/NumPy/sklearn · KNN Scratch/NumPy/sklearn | F1 / ROC-AUC |

> **Run order:** individual notebooks (any order) → `model_comparison.ipynb` last.

---

## Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install core dependencies
pip install jupyter numpy pandas scikit-learn matplotlib seaborn statsmodels scipy nltk joblib

# 3. Install PyTorch (for Neural Network notebook)
pip install torch

# 4. Download NLTK corpora
python -c "
import nltk
nltk.download('twitter_samples')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
"

# 5. Launch Jupyter
jupyter notebook
```

---

## Key Concepts by Notebook

| Concept | Where to look |
|---|---|
| OLS normal equations derivation | `Linear_Regression/linear_regression_scratch` |
| Why GD needs feature scaling | `Linear_Regression/linear_regression_numpy` |
| All 7 OLS assumptions with tests | `Linear_Regression/linear_regression_assumptions` |
| Soft thresholding — why Lasso zeroes coefficients | `Ridge lasso/ridge_lasso_numpy` |
| Coordinate descent for Lasso | `Ridge lasso/ridge_lasso_numpy` |
| TF-IDF math from first principles | `Logistic_Regression/logistic_regression_scratch` |
| Why sigmoid + BCE gradient simplifies to `(ŷ - y)/n` | `Neural Network/nn_scratch` |
| Laplace smoothing — why it's needed | `Naive Bayes/naive_bayes_scratch` |
| K-Means++ vs random init (boxplot) | `K_Means/kmeans_numpy` |
| Cov+Eigh vs SVD — same result, different stability | `PCA/pca_numpy` |
| Hinge loss subgradient derivation | `SVM/svm_scratch` |
| Random Fourier Features — kernel without O(n²) | `SVM/svm_numpy` |
| Backpropagation — full chain rule derivation | `Neural Network/nn_scratch` |
| Gradient check (numerical vs analytical) | `Neural Network/nn_scratch` |
| Adam — bias-corrected moments | `Neural Network/nn_numpy` |
| SMOTE from scratch (no imblearn) | `Data_Preprocessing/data_preprocessing` |
| When to use which imputation strategy | `Data_Preprocessing/data_preprocessing` |
| POS-aware lemmatisation vs stemming | `Data_Preprocessing/data_preprocessing` |

---

## Philosophy

**Scratch notebooks** — implement the math, not the library. Every step is explicit and traceable to the equation.

**NumPy notebooks** — vectorise. Replace Python loops with broadcasting. Understand what the library computes internally.

**sklearn notebooks** — best practices: pipelines, stratified splits, full metric suites, cross-validation, hyperparameter sweeps.

**Comparison notebook** — no re-training. Train once, save, load, compare. This is how real evaluation works.

**Preprocessing notebook** — every technique explained with visuals and comparison plots, not just `fit_transform`.

---

## Related Work

This repo is the foundation layer of a larger ML portfolio:

```
ml-from-scratch           ← understand the internals (this repo)
        ↓
distilbert-sentiment      ← apply pre-trained transformers
        ↓
lora-tinyllama-medical    ← efficient LLM adaptation (PEFT/LoRA)
        ↓
LangGraph_Implementations ← multi-agent LLM orchestration
```

---

*Built with curiosity. No shortcuts taken.*
