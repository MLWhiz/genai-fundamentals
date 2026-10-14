# GenAI Fundamentals

Code for the [GenAI Fundamentals series](https://www.mlwhiz.com) on mlwhiz.com. Each post picks one building block of modern LLMs and explains it from first principles, with runnable code.

## Repository Structure

```
genai-fundamentals/
├── 1. Understanding Transformers/
│   └── README.md
├── 2. Transformer from Scratch/
│   ├── README.md
│   └── translation-using-transformers.ipynb
├── 3. What is an LLM/
│   ├── README.md
│   └── what-is-an-llm-really.ipynb
├── 4. Pretraining 101/
│   ├── README.md
│   └── pretraining-from-scratch.ipynb
├── 5. Post-Training 101/
│   ├── README.md
│   └── code-snippets.md
└── 6. Post-Training in Code/
    ├── README.md
    └── post-training-sft-dpo.ipynb
```

Each folder has a README with a summary of the post.

## Notebooks

### 2. The Transformer, Demystified

**Notebook:** `2. Transformer from Scratch/translation-using-transformers.ipynb`

**Blog post:** [The Transformer, Demystified — Let's Actually Build One](https://www.mlwhiz.com/p/the-transformer-demystified-lets)

Builds a transformer from scratch and trains it on a translation task. Also on [Kaggle](https://www.kaggle.com/code/mlwhiz/translation-using-transformers).

---

### 3. What is an LLM?

**Notebook:** `3. What is an LLM/what-is-an-llm-really.ipynb`

**Blog post:** [What is an LLM? Tokens, Embeddings, and the Big Picture](https://www.mlwhiz.com/p/what-is-an-llm-tokens-embeddings)

Also on [Kaggle](https://www.kaggle.com/code/mlwhiz/what-is-a-llm-really). Walks through the LLM pipeline step by step: tokenization (BPE, SentencePiece, tiktoken), the embedding table, weight tying, and next-token prediction.

---

### 4. Pretraining 101

**Notebook:** `4. Pretraining 101/pretraining-from-scratch.ipynb`

**Blog post:** [Pretraining 101: Data, Scale, and the Loss Function](https://www.mlwhiz.com/p/pretraining-101-data-scale-and-the)

Pretrains a tiny GPT from scratch on a public-domain book from Project Gutenberg. It runs on a laptop CPU in a few minutes. Also on [Kaggle](https://www.kaggle.com/code/mlwhiz/pretraining-from-scratch).

---

### 6. Post-Training in Code

**Notebook:** `6. Post-Training in Code/post-training-sft-dpo.ipynb`

**Blog post:** [Post-Training in Code: Fine-Tune Your Own Assistant with SFT and DPO](https://www.mlwhiz.com/p/post-training-in-code-sft-dpo)

The hands-on companion to Part 5. Takes Qwen3.5-0.8B-Base through SFT (twice, full fine-tuning and LoRA) and then DPO, on one GPU, with the real generations and loss curves at every stage.

---

## Blog Series

| # | Post | Code |
|---|------|------|
| 1 | [Understanding Transformers, the MLE Way](https://www.mlwhiz.com/p/transformers) | [`1. Understanding Transformers/`](1.%20Understanding%20Transformers/) |
| 2 | [The Transformer, Demystified — Let's Actually Build One](https://www.mlwhiz.com/p/the-transformer-demystified-lets) | `2. Transformer from Scratch/` |
| 3 | [What is an LLM? Tokens, Embeddings, and the Big Picture](https://www.mlwhiz.com/p/what-is-an-llm-tokens-embeddings) | `3. What is an LLM/` |
| 4 | [Pretraining 101: Data, Scale, and the Loss Function](https://www.mlwhiz.com/p/pretraining-101-data-scale-and-the) | `4. Pretraining 101/` |
| 5 | [Post-Training 101: From Base Model to Assistant](https://www.mlwhiz.com/p/post-training-101-from-base-model) | [`5. Post-Training 101/`](5.%20Post-Training%20101/) (code snippets) |
| 6 | [Post-Training in Code: Fine-Tune Your Own Assistant with SFT and DPO](https://www.mlwhiz.com/p/post-training-in-code-sft-dpo) | [`6. Post-Training in Code/`](6.%20Post-Training%20in%20Code/) |

## Related

- [recsysSeries](https://github.com/MLWhiz/recsysSeries): notebooks for the RecSys series
- [genai](https://github.com/MLWhiz/genai): code and papers for the earlier Generative AI series
