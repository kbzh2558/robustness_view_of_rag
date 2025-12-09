# ⚙️📚 Optimizing Knowledge Retrieval in Retrieval-Augmented Generation  
![Optimization](https://img.shields.io/badge/Optimization-MIO%20%7C%20Robust-blue)
![RAG](https://img.shields.io/badge/RAG-Retrieval--Augmented%20Generation-green)
![Gurobi](https://img.shields.io/badge/Gurobi-Mixed--Integer--Programs-red)
![Embeddings](https://img.shields.io/badge/Embeddings-BERT%20%7C%20E5%20%7C%20MPNet-yellow)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)

> **Title**  
> Optimizing Knowledge Retrieval in Retrieval-Augmented Generation  
>
> **Authors**  
> Kaibo Zhang, Annie Liu, and Lauren Zhang  
>
> **Course**  
> 15.C57 – Optimization Methods, MIT  

---

## Overview  

Most Retrieval-Augmented Generation (RAG) systems use a simple heuristic: retrieve the top-k documents with the highest cosine similarity to the query. This works reasonably well, but it:

- Often returns highly **redundant** passages  
- Is **sensitive** to which embedding model is used  
- Can be **unstable** under small changes in embeddings (e.g., long chunks, noise)  

This project reframes retrieval as an **optimization problem** instead of a ranking heuristic. We:

- Show that classical top-k retrieval is equivalent to a tiny **binary knapsack** problem  
- Add **diversity constraints** to avoid redundant documents  
- Use **sparsity penalties** so the model can decide how many documents to keep instead of fixing k  
- Introduce **robust optimization** to handle uncertainty in embedding vectors  

We then compare these approaches on a Mini-Wikipedia RAG dataset using multiple embedding models and Llama-3 as the answer generator.

---

## Dataset  

We use the **Mini-Wikipedia RAG dataset** [1], which contains:

- **3,200** short Wikipedia-style passages  
- **918** factual question–answer pairs  

Each passage and question is embedded using four pretrained encoders:

| Encoder                         | Dim |
|---------------------------------|-----|
| `bert-base-uncased`             | 768 |
| `multi-qa-mpnet-base-dot-v1`    | 768 |
| `hkunlp/instructor-large`       | 1024 |
| `intfloat/e5-small-v2`          | 384 |

Embeddings are:

- Computed in batches  
- Mean-pooled where needed  
- L2-normalized for cosine similarity search  

Questions include factual verification (e.g., “Was Abraham Lincoln the sixteenth President?”) and entity-centric queries (e.g., “Did his mother die of pneumonia?”).

---

## Methodology  

### 1. Top-k Retrieval as an Optimization Baseline  

We interpret top-k similarity search as solving a simple 0–1 knapsack:

- Each document i has a **value** equal to its similarity score `s_i`
- Each document has the same **cost** (1)
- Total “budget” is k documents
- Decision variable `x_i` is 1 if document i is selected, 0 otherwise  

Because all costs are identical, this reduces to a deterministic **sorting problem**: rank by `s_i` and take the top k. This gives us a clean optimization interpretation of the standard heuristic and serves as our baseline.

---

### 2. Variation Across Embedding Models  

Different encoders define different vector spaces, so they may disagree on which documents are “similar”:

- We compute the Top-k set for each encoder and compare overlap  
- The **mean Jaccard distance** between Top-k sets across encoders is **0.8935**, meaning they largely pick different documents for the same query  
- Models like `multi-qa-mpnet-base-dot-v1` and `hkunlp/instructor-large` achieve better F1, cosine similarity, and BERTScore than `bert-base-uncased`  

All subsequent experiments use the best-performing encoder to reduce noise and computational cost (see Table 1 in the report).

---

### 3. Robust and Diversity-Aware Retrieval  

We extend the baseline in three ways:

1. **Diversity Constraint**  
   - Add a constraint on the average pairwise cosine similarity between selected documents  
   - Implemented using McCormick linearization to handle the product terms between selection variables  

2. **Sparsity Penalty**  
   - Instead of fixing k, we penalize the number of selected documents  
   - A sparsity parameter `lambda` controls how aggressively the model prunes the set  

3. **Robustness to Embedding Uncertainty**  
   - Model retrieval as a max–min problem:  
     - The outer problem selects a document subset (relevance, diversity, sparsity)  
     - The inner problem perturbs embeddings inside a chosen **uncertainty set** to represent noise or encoder variability  
   - We experiment with several uncertainty sets:
     - `l1` norm ball  
     - `l2` norm ball  
     - `l∞` norm ball  
     - **k-sparse polyhedral set**, where only a limited number of coordinates can be perturbed  

To solve the robust models, we formulate mixed-integer optimization (MIO) problems and use **Gurobi**.

---

## Experiments  

### Pipeline  

1. Embed questions and passages  
2. Solve retrieval models (baseline top-k, diversity, robust variants)  
3. Feed retrieved documents into **Llama-3** with a strict QA prompt (direct short answer only, no explanations)  
4. Evaluate answer quality using:
   - Cosine similarity between ground truth and predicted answer embedding  
   - Manhattan distance  
   - BERTScore (precision, recall, F1)  

The RAG prompt template and exact evaluation metrics are listed in Tables 2–7 of the report.

---

## Results Summary  

### 1. Heuristic Top-k vs Robust Models  

- Varying k in the baseline reveals a trade-off between **coverage** and **redundancy**  
- Robust models with `l1`, `l2`, and `l∞` uncertainty sets achieve **similar or better** answer quality compared to the best heuristic k values  
- Robust variants consistently improve semantic metrics such as BERTScore F1  

(See Tables 3–7 in the report for detailed numbers.)

---

### 2. Behavior of Different Uncertainty Sets  

**Norm-based sets (`l1`, `l2`, `l∞`):**

- `l1` and `l2` behave intuitively:  
  - As the sparsity penalty `lambda` increases, fewer documents survive, but the ones that remain are more robustly relevant  
- `l∞` quickly **saturates**:  
  - Once the perturbation radius is large enough to alter the direction of an embedding, increasing it further barely changes the worst-case similarity  
  - This leads to almost flat performance across a wide range of hyperparameters  

**k-sparse polyhedral set:**

- Allows an adversary to change only a small number of coordinates  
- In early experiments, even small k could **destroy** relevance by targeting the most informative dimensions  
- Adding **protection constraints** (coordinates that cannot be attacked) restores meaningful behavior and reveals that retrieval often depends on a **small subset of “core” coordinates**  

A toy example in the report shows:

- The correct document initially has a lower cosine score than a distractor  
- After adversarially erasing just 20 noisy coordinates, the correct document becomes the top-ranked one  

This suggests that many embedding dimensions behave as noise (especially for longer chunks), while a small subset encodes the true semantic match.

---

### 3. Consistency of Robust Retrieval  

Using the best hyperparameters for each model, we measure the **Jaccard distance** between retrieval sets across embedding models and runs:

- `l∞` uncertainty achieves the **strongest cross-embedding consistency**  
- Some distances are mechanically inflated because different runs may produce sets of different sizes, but the **qualitative content** of the sets remains stable  
- Case study: For the question *“Are beetles endopterygotes?”*, robust `l1` retrieval consistently centers on a single key document that explicitly answers the question, after increasing the sparsity penalty  

(See Table 9 and Figure 1 in the report.)

---

## Insights and Future Work  

**Key takeaways:**

- Viewing retrieval as an optimization problem provides a **unified framework** that subsumes and improves upon heuristic top-k search  
- Diversity constraints and sparsity penalties reduce redundancy and adaptively control the number of retrieved documents  
- Robust optimization exposes **failure modes** of standard embeddings (e.g., long noisy chunks) and improves stability across encoders  
- Empirical evidence suggests that document relevance is often **low-dimensional**: only a small subset of embedding coordinates is truly informative  

**Future directions:**

- Learn a **sparse masking vector** over embedding dimensions to systematically down-weight noisy coordinates  
- Integrate robust retrieval objectives into **embedding training** itself, not just inference-time selection  
- Combine robust optimization with **diversity-aware clustering** for large-scale RAG systems  

---

## Citation  

If you use or build upon this work, please cite:

> Zhang, K., Liu, A., & Zhang, L. (2025). *Optimizing Knowledge Retrieval in Retrieval-Augmented Generation*. Final report for 15.C57 Optimization Methods, MIT.

### References  

[1] Hugging Face Community and Contributors. RAG datasets: A collection of small-scale and domain-specific question-answer-passage corpora. 2024.  

[2] Wang, Z., Bi, B., Luo, Y., Asur, S., & Cheng, C. N. (2025). *Diversity improves RAG: Ranking, clustering, and coverage for retrieval-augmented generation*. arXiv preprint arXiv:2502.09017.  
