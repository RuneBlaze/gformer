# **Refined Model Architecture Overview**

## **1. Introduction**

This model is designed for converting a set of “gene trees” (in Newick format) into a single “species tree” (also in Newick format). Each gene tree is an unrooted phylogenetic tree; the species tree is typically rooted. The process can be seen as an **encoder-decoder** transformation:

- **Encoder** consumes the set of gene trees (treated as a set, thus order-invariant).
- **Decoder** then predicts the tokens of the species tree, one at a time, using an autoregressive approach.

The goal is to capture high-level phylogenetic signals from multiple gene trees and produce a single species tree that explains these signals.

---

## **2. Model Inputs and Representation**

### **2.1 Gene Trees**

1. **Input Format**  
   We assume \( k \) unrooted gene trees, each given in Newick notation.  
   - Example gene tree:  
     ```
     (0, (1, 2))
     ```
   - Each gene tree has up to 256 possible taxa labels (0 through 255).

2. **Distance Matrix**  
   For each gene tree, we create an **all-pairs topological distance matrix**.  
   - **Topological distance** here is the number of edges on the path between two taxa (i.e., ignoring branch lengths).  
   - If a tree has \( N \) leaves, this yields an \( N \times N \) matrix.  
   - We often fix \( N = 16 \) (or a known maximum) to simplify zero-padding.

3. **Binary Encoding**  
   We flatten the upper-triangular part of the \( N \times N \) matrix (excluding the diagonal) and store each distance in **binary** using 8 bits. This yields a 3D tensor of shape:  
   \[
   (\text{num\_upper\_triangular\_entries}) \times 8
   \]
   or in code terms, something like \(\tfrac{N(N-1)}{2} \times 8\).  
   - Each bit is extracted, then transformed by a small activation (SiLU).  
   - This flattening results in a consistent-length representation for each gene tree.

### **2.2 Species Tree (Output)**

1. **Tokenizer**  
   We have a simple integer-based tokenizer for Newick strings:
   - **Taxa labels**: integers \([0..255]\).  
   - **Special tokens**:
     - 256 = “(” (LEFT_PAREN)
     - 257 = “)” (RIGHT_PAREN)
     - 258 = End-of-input (EOI)
     - 259 = End-of-sequence (EOS)
   - The final vocabulary size is 260.

2. **Autoregressive Generation**  
   The decoder produces the species tree tokens one at a time. At step \( t \), it can only see the tokens \(\leq t-1\) that it has already produced (or teacher-forced).

---

## **3. High-Level Architecture**

### **3.1 Overview: Encoder–Decoder**

The model follows a **Transformer** design:

1. **Encoder**  
   Processes all \( k \) gene trees, each turned into an embedding vector.  
   - **Set Input**: We treat the \( k \) gene tree embeddings as a set (hence no positional encoding).  
   - The encoder uses standard multi-head self-attention across these \( k \) embeddings.

2. **Decoder**  
   Predicts the species tree token-by-token.  
   - **Positional Encoding** is applied to the partial species tree tokens.  
   - **Causal Masking** ensures the decoder can’t see future tokens.  
   - The decoder also attends to the encoder’s outputs (cross-attention) to incorporate the gene tree information.

### **3.2 Detailed Components**

Below is a diagram of the major stages:

```
Gene Trees --> DistanceMatrixMLP --> TransformerEncoder --> *encoded memory*
                                                   \ 
                                                    -> TransformerDecoder -> Output Tokens
Species Tokens + Positional Encoding ------------/
```

1. **DistanceMatrixMLP**  
   - Input dimension:  
     \[
     \underbrace{\left(\frac{N(N-1)}{2}\right)}_{\text{upper-triangular pairs}} \times \underbrace{8}_{\text{bits}}
     \]
   - Simple feedforward of shape \(\text{[in_dim, hidden_dim, out_dim]}\), with SiLU activation in the hidden layer.  
   - Outputs a **single embedding vector** per gene tree.

2. **Transformer Encoder**  
   - Takes a batch of gene-tree embeddings (\( k \)-by-\(\text{embedding\_dim}\)) and processes them with multi-head attention.  
   - We do not apply any position-based ordering, because the set of gene trees is permutation-invariant.  
   - Standard pre-layer normalization is used for training stability.  
   - This yields a memory representation of shape \(\text{[batch, k, d\_model]}\).

3. **Transformer Decoder**  
   - Consumes partial species-tree tokens, each embedded to dimension \( d_{\text{model}} \) via `nn.Embedding`.  
   - A **sinusoidal positional encoding** is added to these embeddings (since sequences, not sets, must reflect order).  
   - Self-attention with a **causal mask** ensures token \( t \) can’t see beyond \( t-1 \).  
   - **Cross-attention** keys/values come from the encoder memory.  
   - Produces a hidden state for each species-tree token.

4. **Final Projection**  
   - A linear projection maps each decoder hidden state to \(\text{vocab\_size} = 260\) logits.  
   - Softmax over these logits yields the next token distribution.

---

## **4. Training**

### **4.1 Setup**

1. **Teacher Forcing**  
   During training, the decoder sees the ground-truth partial sequence at each timestep. We shift the target tokens by one, so the model predicts the next token from the current partial sequence.

2. **Loss Function**  
   - We use cross-entropy loss over the species tree tokens.  
   - Special or padding tokens can be ignored using PyTorch’s built-in ability to exclude specific indices (e.g., the EOI token).

3. **Hyperparameters**  
   - **Batch size**: e.g., 32 for the large model or 1 for a minimal test.  
   - **Learning rate**: typically in the \(1\mathrm{e}{-4}\) range (configurable).  
   - **Warmup steps**: e.g., 1000, with a cosine decay schedule afterward.  
   - **Gradient checkpointing**: can optionally be enabled to reduce memory usage at the cost of some recomputation.

### **4.2 Optimizers**  
Common optimizers are supported, including **Adam, AdamW, SGD**, and **Adafactor**. We often rely on memory-efficient “fused” variants if available.

### **4.3 Validation**  
For validation, we measure:
1. **Validation Loss** using cross-entropy on a held-out subset.  
2. **Token Accuracy** by comparing predicted tokens vs. reference.  

Because tree topologies can have multiple correct forms, a direct token-level comparison is only a rough approximation of correctness. More advanced tree-specific metrics could be substituted if desired.

---

## **5. Implementation Notes**

### **5.1 Parallel Preprocessing**  
- A custom `TreeDataset` uses parallel workers to:
  - Parse and tokenize gene trees.
  - Build and cache distance-matrix encodings.
  - Construct token sequences for the species tree.

### **5.2 Handling Variable Gene Tree Counts**  
- We zero-pad the dimension “number of gene trees” to a fixed maximum (e.g., `MAX_GTREES = 300`), so the encoder can process them in a single batch.  
- If a particular input has fewer than `MAX_GTREES`, we pad with zeros; if more, we can truncate.

### **5.3 Limitations / Assumptions**  
1. **Fixed Taxa Limit**: The code expects a fixed \( N \leq 256 \). Large expansions might require memory or architectural changes.  
2. **No explicit root modeling** in the gene trees. We rely on topological distance.  
3. **Set-based gene trees**: We do not impose any order or weighting among the gene trees.

---

## **6. Example Workflow**

1. **Data**: Provide a dataset of pairs \(\{\text{list of gene trees}, \text{species tree}\}\).  
2. **Preprocessing**:
   - Convert each gene tree into its binary-encoded distance matrix flattening.  
   - Tokenize the species tree into integer IDs.  
3. **Training**:
   - The encoder processes the gene-tree embeddings.  
   - The decoder is fed partial species-tree tokens plus a causal mask.  
   - The model outputs distributions over the next token until the entire species tree is generated.  
4. **Inference**:
   - Given new gene trees, do the same distance-matrix encoding.  
   - Feed them to the encoder, then **autoregressively** decode the species tree tokens.  
   - Stop when an EOS token is generated.

---

## **7. Configurations**

We supply two example YAML configuration files:

1. **`model_large.yaml`**  
   - ~200M parameters, with GPT2-small-like dimensions (embedding_dim=768, 12 heads, 12 layers, MLP=3072).  
   - Good for production usage.

2. **`model_small.yaml`**  
   - ~28M parameters, smaller test version (embedding_dim=256, only 2 layers, etc.).  
   - Typically used for debugging or quick experiments.

---

## **8. Summary**

This architecture leverages a **set-based Transformer encoder** on binary-encoded distance matrices from multiple gene trees, followed by a **causal Transformer decoder** to produce the final species tree tokens. The design is flexible, relatively straightforward to train, and can handle large numbers of taxa and gene trees (subject to memory constraints). It captures phylogenetic signals by converting each gene tree to a topological embedding, then synthesizes them in an autoregressive manner to predict the species tree.

Overall, the model aims to unify signals from multiple gene trees in a robust, end-to-end trainable neural framework, producing consistent and meaningful species-tree predictions.

---

**End of Document**