# ⚙️📚 Optimizing Knowledge Retrieval in Retrieval-Augmented Generation  
![Optimization](https://img.shields.io/badge/Optimization-MIO%20%7C%20Robust-blue)
![RAG](https://img.shields.io/badge/RAG-Retrieval--Augmented%20Generation-green)
![Gurobi](https://img.shields.io/badge/Gurobi-Mixed--Integer--Programs-red)
![Embeddings](https://img.shields.io/badge/Embeddings-BERT%20%7C%20E5%20%7C%20MPNet-yellow)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)

> **Authors**  
> Kaibo Zhang, Annie Liu, and Lauren Zhang  
> **Affiliation**: MIT, 15.C57 Machine Learning & Optimization  
---

## 🌐 Overview

This project rethinks retrieval in **Retrieval-Augmented Generation (RAG)** by treating it not as a heuristic ranking task but as a **formal optimization problem**. Most RAG systems simply select the **Top-k** documents by cosine similarity, but this overlooks issues such as:

- Redundancy among retrieved documents  
- Encoder-specific variability in embedding spaces  
- Sensitivity to noise in long or uneven chunks  
- Lack of robustness to adversarial or structural embedding perturbations  

We propose a suite of optimization-based retrieval methods that incorporate:

- **Diversity constraints** to reduce redundancy  
- **Sparsity penalties** that adaptively control the number of retrieved documents  
- **Robust optimization** (norm-based and polyhedral) to model embedding uncertainty  

Experiments reveal consistent improvements in semantic answer quality and remarkable stability across embedding models.

---

## 📂 Dataset

We use the **Mini-Wikipedia RAG dataset** [HuggingFace, 2024], containing:

- **3,200** short Wikipedia-style passages  
- **918** factual question–answer pairs  

Each passage and question is embedded using **four encoders**:

| Encoder | Dimensionality |
|--------|----------------|
| BERT-base-uncased | 768 |
| multi-qa-mpnet-base-dot-v1 | 768 |
| hkunlp/instructor-large | 1024 |
| intfloat/e5-small-v2 | 384 |

Embeddings are **L2-normalized** and used throughout retrieval and robustness experiments.

---

## 🔍 Methodology

### 1. Top-k Retrieval as an Optimization Problem  
We show that choosing the top-k documents by similarity is equivalent to solving a tiny **binary knapsack** with identical weights:

\[
\max \sum_i s_i x_i \quad\text{s.t.}\quad \sum_i x_i = k, \; x_i \in \{0,1\}
\]

This casts the classical heuristic in a principled optimization framework.  
(Section 3.1, p. 2–3 of the report)

---

### 2. Embedding Model Variability

Different encoders induce different geometric structures.  
Empirically, the **mean Jaccard distance** between Top-k sets across embedding models is **0.8935**, meaning retrieval varies drastically.  
Best models (MPNet, Instructor) show higher semantic alignment.  
(Table 1, p. 7)

---

### 3. Optimization-Enhanced Retrieval  
We extend Top-k retrieval in three directions:

#### **a. Diversity Constraints**  
Prevent selecting documents that are too similar.  
Implemented using McCormick linearization for pairwise cosine constraints.

#### **b. Sparsity Penalties**  
Remove the hard-coded value of k.  
Model automatically chooses how many documents to keep.

#### **c. Robust Retrieval**  
Solve:

\[
\max_x \min_{\tilde\mu \in \mathcal{U}} \left( \sum_i s_i(\tilde\mu_i) x_i - \lambda \sum_i x_i \right)
\]

Uncertainty sets:

- ℓ₁-ball  
- ℓ₂-ball  
- ℓ∞-ball  
- **k-sparse polyhedral** (with optional coordinate-protection)

ℓ∞ is stable but insensitive to hyperparameters; k-sparse reveals low-dimensional relevance structure.

---

## 🧪 Experiments

### Retrieval → Llama-3 Answering Pipeline

Retrieved documents are fed into a strict QA prompt (Table 2, p. 7).

---

### ⭐ Key Findings

#### **1. Robustness dramatically improves stability and semantic accuracy**

All robust variants outperform baseline Top-k on cosine similarity, Manhattan distance, and all three BERTScore metrics.  
(Tables 3–7, p. 7–9)

#### **2. ℓ∞ uncertainty produces extremely consistent retrieval sets**

Across runs and embeddings, ℓ∞ yields the lowest Jaccard instability.  
(Section 4.3, p. 5–6)

#### **3. Retrieval often depends on only a small number of informative coordinates**

A striking toy example shows:

| Doc | Cosine (Original) | Cosine (After Erasing 20 Noisy Dimensions) |
|------|---------------------|--------------------------------------------|
| Correct document | 0.724 | **0.751** |
| Distractor | 0.747 | 0.750 |

(Table 8, p. 9)

This demonstrates that **chunk length and embedding noise can flip rankings**, and simple coordinate erasure restores correct ordering.

---

## 📈 Representative Results

### Heuristic Top-k Performance (k = 2–10)
From Table 3:

| k | Cosine | BERT-F1 |
|---|--------|----------|
| 2 | 0.8352 | 0.8436 |
| 10 | **0.9246** | **0.8680** |

---

### Example Robust ℓ₁ Model Retrieval  
(Table 9, p. 10)

For the question **“Are beetles endopterygotes?”**:

| λ | Retrieved Set |
|----|----------------|
| 0.50 | Many documents including 2384, 2385 |
| 0.65 | **Only 2394** |
| 0.72 | **Only 2394** |

Doc 2394 explicitly states that beetles are endopterygotes—robust retrieval isolates it.

---

## 📘 Conclusion

This study demonstrates that:

- Viewing retrieval as an **optimization problem** yields more interpretable and effective systems.  
- Diversity and sparsity help avoid redundancy and oversampling noisy documents.  
- Robust optimization reveals that embedding noise can dramatically distort cosine similarities.  
- A **small number of embedding dimensions** often carry the true semantic signal.  

These insights open paths toward:

- Learned sparse masking of embedding dimensions  
- Hybrid robust-retrieval + embedding-training frameworks  
- More stable and interpretable RAG systems  

---

## 📚 References  
All references correspond to citations in the original report.  
See full PDF for tables, figures, and mathematical formulations.  
:contentReference[oaicite:1]{index=1}
