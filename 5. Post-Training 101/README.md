# Post-Training 101: From Base Model to Assistant

Post: [mlwhiz.com/p/post-training-101-from-base-model](https://www.mlwhiz.com/p/post-training-101-from-base-model)
Date: 2026-07-07

*genAI Fundamentals Part 5: instruction tuning (SFT), RLHF vs DPO, and the RLVR/GRPO reasoning era, how a base model learns to follow instructions and which answers people actually prefer.*

## Summary

A base model has the knowledge but not the behavior of an assistant. Post-training has two jobs: behavior (answer instead of continuing the text) and preference (which answer is better). It's also much cheaper than pretraining, using thousands to low millions of curated examples.

SFT (instruction tuning) is the same next-token training on instruction and response pairs. It teaches format, not facts. The post notes SFT is a poor way to add new knowledge and may increase hallucination. The prompt tokens are masked from the loss with -100.

SFT can only imitate, so the post moves to preferences. RLHF has three steps. First, SFT. Second, train a reward model on human comparisons with the Bradley-Terry loss. Third, run PPO against a KL penalty to a frozen reference. A value network (critic) supplies the baseline for the advantage. A full PPO run juggles four models, and the KL term guards against reward hacking.

DPO derives that the policy contains an implicit reward, so the reward model, critic and RL loop can be dropped. The loss is a Bradley-Terry loss over log-probability ratios, with β = 0.1 in the example. The tradeoff is that it's offline. KTO handles unpaired good/bad labels and borrows loss aversion from prospect theory.

RLVR uses a program (math checker, unit tests) as the reward. GRPO drops the critic by standardizing rewards within a group of sampled answers. The post works through "What is 17 × 24?" with four samples. If all answers agree, the advantage is zero. DeepSeek-R1-Zero learned to reason from pure RLVR with no SFT.

It ends with RLAIF and Constitutional AI, and a decision guide for which method to use when.

## Key takeaways

- Post-training teaches behavior and preferences. The facts come from pretraining.
- SFT always comes first. Then DPO or RLHF for subjective alignment, and GRPO/RLVR where the answer is checkable.
- DPO removes the reward model and RL loop but only learns from a fixed preference dataset.
- GRPO uses the group mean as the baseline instead of a critic, and learns only when answers in a group disagree.
- The KL penalty keeps the model anchored to its reference, in PPO explicitly and in DPO inside the loss.

## Code

This post has no notebook. The code blocks from the post are collected in [code-snippets.md](code-snippets.md). They're fragments, not a runnable notebook.

The post says the next one in the series is a hands-on lab building SFT and DPO from scratch.
