# Understanding Transformers, the MLE Way

Post: [mlwhiz.com/p/transformers](https://www.mlwhiz.com/p/transformers)
Date: 2026-05-29

*GenAI Series Part 1: What even are transformers?*

## Summary

This is the foundation post for the series. It walks through the full transformer from the "Attention Is All You Need" paper using one running example: translating an English sentence into German. The post is written as a Q&A with the reader, with the math kept light.

LSTMs and GRUs process words one at a time, so runtime grows with sentence length. The transformer drops recurrence and relies on attention alone, which makes it fast.

The encoder is a stack of 6 layers. Each has multi-head self-attention and a position-wise feed-forward network, with inputs and outputs of shape S x D (D = 512). Self-attention builds Q, K and V matrices (d = 64), computes softmax(QKᵀ / √d), and uses the result to take a weighted average of the value vectors. Eight heads run in parallel (h x d = D) and get concatenated and projected back with Wo. The post also covers sine/cosine positional encodings and the residual "Add & Norm" connections.

The decoder has 6 layers too: masked self-attention, attention over the encoder output ("memory"), and a feed-forward network. Masking adds -inf above the diagonal before the softmax, so the model can train on the whole shifted target sentence at once without peeking ahead.

Then come the output head, training (cross-entropy, or KL divergence with label smoothing at 0.1, plus the paper's learning rate schedule), and prediction. Greedy search is explained as beam search with k = 1. A worked beam search example uses k = 2, log-probabilities, and a length penalty (the paper uses α = 0.6 and a beam of 4). The post closes with BPE, shared weights and checkpoint averaging.

## Key takeaways

- Attention is a soft dictionary lookup: each word's output is a weighted blend of the value vectors of all words.
- Multi-head attention runs 8 smaller attentions in parallel (64 dims each) and concatenates them back to 512.
- Positional encodings are needed because there's no recurrence, so order has to be injected.
- The decoder mask lets the model train on a whole sentence in parallel. At prediction time it still generates one token at a time.
- Greedy search is beam search with k = 1. Beam search scores whole sequences with summed log-probabilities and a length penalty.

## Code

This post is conceptual, no code. The implementation is in the next post: [2. Transformer from Scratch](../2.%20Transformer%20from%20Scratch/).
