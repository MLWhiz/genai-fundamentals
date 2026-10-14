# Post-Training in Code: Fine-Tune Your Own Assistant with SFT and DPO

Post: [mlwhiz.com/p/post-training-in-code-sft-dpo](https://www.mlwhiz.com/p/post-training-in-code-sft-dpo)
Date: 2026-10-15

*genAI Fundamentals Part 6, the first hands-on lab: take a small base model, instruction-tune it (SFT), then align it (DPO), with real before/after generations you can reproduce.*

## Summary

Part 5 explained post-training. This one runs it, end to end, in a single notebook on one GPU.

The base model is Qwen3.5-0.8B-Base, small enough to train twice. Before any training it answers a hash-map question correctly in one sentence and then invents its own follow-up questions, because nothing has ever taught it that a reply ends. The knowledge is there; the behaviour is not.

Stage 1 is SFT on `yahma/alpaca-cleaned`, run two ways on identical data and identical steps: once by updating all 0.8B weights, and once with LoRA adapters on a frozen base. Both learn to answer and stop. LoRA came out clearly ahead, and the loss curves explain why: at the shared `learning_rate=8e-5`, the full fine-tune destabilised (1.44 to over 2.2 within fifty steps, finishing near 1.7) while LoRA settled around 1.2. 8e-5 is a reasonable LoRA learning rate and an aggressive one for moving every weight, so the comparison handed both methods a setting that suits only one of them. The honest finding isn't that LoRA is better, it's that LoRA is far more forgiving of the settings you guess wrong.

Stage 2 is DPO on `trl-lib/ultrafeedback_binarized`, starting from the merged LoRA model with the adapter itself acting as the reference. The shift is real and small: a warmer answer that volunteers a next step, while the repetition a 0.8B model shows inside a list stays exactly where it was.

Along the way the notebook covers the gotchas that cost real time: two different stop tokens (`<|endoftext|>` is the document separator the tokenizer points at, `<|im_end|>` is the turn ender SFT teaches), generating while the model is still in training mode with LoRA dropout active, and why a base checkpoint should not get a chat template at inference.

## Key takeaways

- LoRA was far more forgiving than full fine-tuning at a shared learning rate. A better-conditioned problem tolerates settings that destabilise the full run.
- Stopping is a separate skill from answering, and `generate` has to be told about it. SFT teaches `<|im_end|>`, but it never updates the tokenizer's `eos_token`.
- DPO loss starts at exactly 0.693 (ln 2), because at step zero the policy is the reference. If your run starts anywhere else, your reference model isn't what you think it is.
- DPO's effect is subtle. Judge it by reading answers, not by watching the loss.
- TRL handles the `-100` prompt mask for you with `assistant_only_loss=True`. Don't apply the chat template by hand as well.

## Code

[`post-training-sft-dpo.ipynb`](post-training-sft-dpo.ipynb) runs top to bottom on a single GPU, with outputs from the run that appears in the post, including the ones where the model gets it wrong.

All the knobs live in one config cell at the top (`MODEL_NAME`, `DATASET_NAME`, `PREFERENCE_DATASET_NAME`, learning rates, token caps), so you can point it at your own data without hunting through cells. The post's closing section has a template for the two row shapes TRL expects: a `messages` row for SFT, and a `prompt`/`chosen`/`rejected` row for DPO.

Built on HuggingFace `transformers`, `trl`, `peft` and `datasets`.
