# What is an LLM? Tokens, Embeddings, and the Big Picture

Post: [mlwhiz.com/p/what-is-an-llm-tokens-embeddings](https://www.mlwhiz.com/p/what-is-an-llm-tokens-embeddings)
Date: 2026-06-14

*GenAI Fundamentals Series Part 3: From raw text to next-token prediction. The complete mental model*

## Summary

The post starts from the LinkedIn "gotcha" screenshots (strawberry, lollipop, 9.11 vs 9.9) and argues these failures are predictable once you know the pipeline. Its definition: an LLM is a function that takes a list of integers and outputs a probability distribution over which integer comes next.

Tokenization comes first. Characters make sequences too long, whole words make the vocabulary huge, so modern LLMs use subwords (32K to 256K vocabularies). The post explains Unicode, UTF-8 bytes, and the BPE algorithm (start with 256 byte units, merge the most frequent pair, repeat). Byte-level BPE means there's no unknown token. SentencePiece and tiktoken are both libraries that run BPE. The post gives tiktoken's encodings (cl100k_base at 100,277 tokens, o200k_base at about 200K, gpt2 at 50,257) and compares tokenizers across model families.

Next is the embedding table, a (vocab_size, d_model) lookup. For a 128,256 x 4,096 table that's 525 million parameters. Then the transformer block, and the LM head (a linear layer to vocabulary-sized logits, then softmax). Weight tying is covered with examples: GPT-2, Gemma 2, Qwen2.5-0.5B and Llama 3.2 1B tie, while Llama 3 8B, Qwen2.5 7B and DeepSeek-V3 don't. Temperature, top-k and top-p follow.

The post traces "The cat sat on" through the full pipeline, then covers autoregressive generation, the KV cache and the EOS token. It ends with practical consequences: why LLMs can't count letters, why Hindi needs about 3x the tokens of English, and why you can't swap a tokenizer after training.

## Key takeaways

- An LLM is one function: token IDs in, next-token probabilities out. Generation loops that function one token at a time.
- "strawberry" becomes "str" + "aw" + "berry", so the model never sees individual letters.
- Byte-level BPE starts from 256 byte values, so any text can be represented and nothing is "unknown".
- The embedding table is a lookup and the LM head is a linear projection. Small models often tie them, large ones often don't.
- Tokenizer choice affects cost, context length and multilingual quality, and it's fixed once the model is trained.

## Code

- Notebook: [what-is-an-llm-really.ipynb](what-is-an-llm-really.ipynb)
- Also on [Kaggle](https://www.kaggle.com/code/mlwhiz/what-is-a-llm-really)
