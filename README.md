# Architecture Overview

## Input Processing
- Input: k unrooted gene trees in Newick format
- Per tree conversion:
  1. Compute all-pairs distance matrix d(x,y) where d(x,y) = number of edges between taxa x and y
  2. Flatten upper triangular portion of distance matrix
  3. Project through MLP to fixed-dimension embedding

## Encoder-Decoder Architecture
### Encoder
- Input: Set of k gene tree embeddings
- All gene trees processed in parallel through transformer encoder
- No positional encoding for gene trees (set is permutation-invariant)
- Output: Encoded memory of all gene tree information

### Decoder
- Input: Previously generated tokens of species tree
- Uses both:
  1. Self-attention with causal masking (can't peek at future tokens)
  2. Cross-attention to encoded gene trees
- Includes sinusoidal positional encoding for output sequence
- Generates species tree tokens one at a time

## Vocabulary
1. Taxa labels [0-255]
2. Left parenthesis
3. Right parenthesis 
4. End-of-input token
5. End-of-output token

## Output Format
Standard Newick format for rooted species tree:
```
<(> <3> <(> <4> <)> <)> <end_of_output>
```

## Training
- Teacher forcing: Use ground truth tokens as decoder input
- Shifted right by one position (predict next token)
- Loss computed only on output sequence predictions
- All gene trees assumed to have same number of taxa (for initial implementation)
- Uses cross-entropy loss with padding token ignored

## Model Details
- Based on standard Transformer architecture
- Embedding dimension: 768 (GPT2-small)
- Number of heads: 12
- Number of layers: 12 (both encoder and decoder)
- MLP hidden dimension: 3072
- Pre-layer normalization for better training stability
