# Architecture Overview

## Input Processing
- Input: k unrooted gene trees in Newick format
- Per tree conversion:
  1. Compute all-pairs distance matrix d(x,y) where d(x,y) = number of edges between taxa x and y
  2. Flatten upper triangular portion of distance matrix
  3. Project through MLP to fixed-dimension embedding -> becomes `<gtree_tok>`

## Transformer Input Sequence
```
<gtree_tok0> <gtree_tok1> ... <gtree_tokk> <end_of_user_input>
```
- All `<gtree_tok>` assigned position 0 (permutation-invariant)
- Position encodings only used for output sequence tokens + <end_of_output> + <end_of_user_input>. Use sinusoidal positional encoding.

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
- Next token prediction with causal masking
- Loss computed only on output sequence tokens
- All gene trees assumed to have same number of taxa (for initial implementation)
