Input: (a) Memory of gene trees, (b) boolean vector of 16 dimension.
Output: A probability distribution over the 3 possible quartets.


Given a boolean vector of 16 dimension.

We have:

(1) Encoder, {0, 1}^16 -> float of dimension d_model. This is the "quartet query token"

(2) Cross-attention, between the quartet query token and the gene tree encodings.

(3) A MLP, which takes the output of the cross-attention and the gene tree encodings, and outputs a probability distribution over the the 3 possible quartets. AB|CD, AC|BD, AD|BC.



