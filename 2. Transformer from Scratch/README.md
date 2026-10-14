# The Transformer, Demystified — Let's Actually Build One

Post: [mlwhiz.com/p/the-transformer-demystified-lets](https://www.mlwhiz.com/p/the-transformer-demystified-lets)
Date: 2026-06-05

*GenAI Series Part 2: Implementing a Transformer*

## Summary

The previous post explained how transformers work. This one builds an English-to-German translator in PyTorch with a transformer at its core.

The data is the English-German subset of OPUS-100 from HuggingFace datasets. The post filters to sentences of at most 20 tokens (MAX_LEN = 20) and caps training at 200k pairs (OPUS-100 has about 1M). Tokenization uses spaCy. A small Vocab class registers three special tokens first: `<blank>` = 0, `<s>` = 1, `</s>` = 2, and drops words seen fewer than 2 times.

Source sentences are padded with `<blank>`. Target sentences get a start token, an end token, and padding. A collate function builds the batches (batch size 64).

The model is a `MyTransformer` class built on PyTorch's `nn.TransformerEncoder` and `nn.TransformerDecoder` (6 layers each, d_model = 512, 8 heads). PyTorch's own Transformer module doesn't include the embedding or positional encoding layers, so the post adds those. Training uses plain Adam (lr 0.0001) instead of the paper's schedule, and cross-entropy instead of KL divergence. Each step builds a source padding mask, a target padding mask and a causal mask.

Training runs for 35 epochs. Epoch 1 gives train loss 43.785 and val loss 39.631, and the debug sentence comes out as gibberish. By epoch 35 train loss is 15.168 and val loss is 31.852, and the translation is a plausible German sentence. A greedy decoding function generates translations one token at a time. The post lists what it left out (full data, BPE, LR schedule, label smoothing, beam search, checkpoint ensembling).

## Key takeaways

- The model needs two inputs: the padded source matrix and the shifted target matrix (start token, end token, padding).
- PyTorch gives you the encoder and decoder blocks. You still add embeddings, positional encoding and the output layer yourself.
- Three masks do the work in training: source padding, target padding and the causal mask.
- Train loss falls to 15.168 while val loss only reaches 31.852, so the gap is large. Output is still readable German after a short run.
- Greedy decoding generates one token at a time until the end token or a max length.

## Code

- Notebook: [translation-using-transformers.ipynb](translation-using-transformers.ipynb)
- Also on [Kaggle](https://www.kaggle.com/code/mlwhiz/translation-using-transformers)
