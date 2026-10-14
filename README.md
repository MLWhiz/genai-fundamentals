# GenAI Fundamentals

Code for the [GenAI Fundamentals series](https://www.mlwhiz.com) on mlwhiz.com. Each post picks one building block of modern LLMs and explains it from first principles, with runnable code.

## Repository Structure

```
genai-fundamentals/
├── 3. What is an LLM/
│   └── what-is-an-llm-really.ipynb
└── 4. Pretraining 101/
    └── pretraining-from-scratch.ipynb
```

## Notebooks

### 3. What is an LLM?

**Notebook:** `3. What is an LLM/what-is-an-llm-really.ipynb`

**Blog post:** [What is an LLM? Tokens, Embeddings, and the Big Picture](https://www.mlwhiz.com/p/what-is-an-llm-tokens-embeddings)

Walks through the LLM pipeline step by step: tokenization (BPE, SentencePiece, tiktoken), the embedding table, weight tying, and next-token prediction.

---

### 4. Pretraining 101

**Notebook:** `4. Pretraining 101/pretraining-from-scratch.ipynb`

**Blog post:** [Pretraining 101: Data, Scale, and the Loss Function](https://www.mlwhiz.com/p/pretraining-101-data-scale-and-the)

Pretrains a tiny GPT from scratch on a public-domain book from Project Gutenberg. It runs on a laptop CPU in a few minutes. Also on [Kaggle](https://www.kaggle.com/code/mlwhiz/pretraining-from-scratch).

---

## Blog Series

| # | Post | Code |
|---|------|------|
| 1 | [Understanding Transformers, the MLE Way](https://www.mlwhiz.com/p/transformers) | Concept post, no notebook |
| 2 | [The Transformer, Demystified — Let's Actually Build One](https://www.mlwhiz.com/p/the-transformer-demystified-lets) | Not in this repo yet |
| 3 | [What is an LLM? Tokens, Embeddings, and the Big Picture](https://www.mlwhiz.com/p/what-is-an-llm-tokens-embeddings) | `3. What is an LLM/` |
| 4 | [Pretraining 101: Data, Scale, and the Loss Function](https://www.mlwhiz.com/p/pretraining-101-data-scale-and-the) | `4. Pretraining 101/` |
| 5 | [Post-Training 101: From Base Model to Assistant](https://www.mlwhiz.com/p/post-training-101-from-base-model) | Concept post, code companion coming |

## Related

- [recsysSeries](https://github.com/MLWhiz/recsysSeries): notebooks for the RecSys series
- [genai](https://github.com/MLWhiz/genai): code and papers for the earlier Generative AI series
