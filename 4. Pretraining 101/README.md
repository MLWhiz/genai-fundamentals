# Pretraining 101: Data, Scale, and the Loss Function

Post: [mlwhiz.com/p/pretraining-101-data-scale-and-the](https://www.mlwhiz.com/p/pretraining-101-data-scale-and-the)
Date: 2026-06-26

*GenAI Fundamentals Series Part 4: How a pile of random weights becomes a base model. Next-token cross-entropy, trillion-token data pipelines, and compute budgets*

## Summary

Pretraining turns random weights into a base model with one objective: predict the next token. It's self-supervised, because any text is its own answer key. "The cat sat on the mat." gives five training examples, one per position.

The post separates pretraining, post-training and fine-tuning. Pretraining is where most raw knowledge comes from and where almost all the cost sits.

The loss section walks through cross-entropy, -log(p of the true token), with a toy five-token example (average loss 0.54). It covers perplexity (a fresh model with a 128K vocabulary starts near 128,000 and a loss of about 11.7) and KL divergence. Cross-entropy equals H(p) plus KL(p||q), so the loss can never reach zero, because language has irreducible entropy. It also covers hard versus soft targets, label smoothing and distillation.

The data section describes five stages: collect (Common Crawl plus curated sources), filter, deduplicate, tokenize and pack. Llama 3 was pretrained on over 15 trillion tokens.

On compute, the post uses C ≈ 6ND and the Chinchilla rule of about 20 tokens per parameter. GPT-3 (175B parameters, about 300B tokens) was undertrained by that measure. Chinchilla (70B, 1.4T tokens) beat larger models. Llama 3 8B was trained on 15T tokens, about 1,875 tokens per parameter, because inference cost favors smaller models. Llama 3 405B used about 3.8×10²⁵ FLOPs.

It then covers warmup plus cosine decay, loss spikes and checkpointing, and a tiny GPT (about 163K parameters, 3 blocks, 64 dims, 4 heads) pretrained on Alice's Adventures in Wonderland. Loss and perplexity are shown falling over 3,000 steps. It ends with why the result is a base model, not a chatbot.

## Key takeaways

- Pretraining is one loss, next-token cross-entropy, applied to trillions of self-labeled tokens.
- Perplexity is exp(loss): roughly how many tokens the model is effectively choosing between.
- Filtering and deduplication of data matter more than adding raw tokens.
- Compute is roughly 6 x parameters x tokens. Chinchilla says a smaller model with more data beats a bigger, undertrained one.
- A base model completes text. It doesn't follow instructions until post-training.

## Code

- Notebook: [pretraining-from-scratch.ipynb](pretraining-from-scratch.ipynb) (pretrains a tiny GPT on a Project Gutenberg book, runs on a laptop CPU in a few minutes)
- Also on [Kaggle](https://www.kaggle.com/code/mlwhiz/pretraining-from-scratch)
