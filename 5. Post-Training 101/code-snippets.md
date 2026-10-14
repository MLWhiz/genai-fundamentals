# Code blocks from: Post-Training 101: From Base Model to Assistant

Code blocks copied from the post, in order. They are fragments, not a runnable notebook.

Source: https://www.mlwhiz.com/p/post-training-101-from-base-model

## 1. 2. Instruction Tuning (SFT): Teaching Format, Not Facts

```
# one training example, in a chat format
prompt   = "<|user|>\nWhat is the capital of France?\n<|assistant|>\n"
response = "The capital of France is Paris."

ids    = tokenize(prompt + response)
labels = ids.copy()

# ignore the prompt tokens in the loss (-100 = "don't train on this")
labels[:len(tokenize(prompt))] = [-100] * len(tokenize(prompt))

# from here it is the SAME next-token cross-entropy as pretraining,
# just scored only on the assistant's reply
loss = cross_entropy(model(ids)[:-1], labels[1:])
```

## 2. The recipe now →

```
prompt: "Explain quantum entanglement simply."
chosen (y_w): "Quantum entanglement is when two particles..."
rejected (y_l): "It's basically magic where particles..."
```

## 3. The recipe now →

```python
class RewardModel(nn.Module):
    def __init__(self, sft_model):
        super().__init__()
        self.backbone = sft_model.model      # transformer body, no LM head
        self.reward_head = nn.Linear(sft_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids, attention_mask=attention_mask).last_hidden_state
        last_token = hidden[torch.arange(hidden.size(0)), attention_mask.sum(1) - 1]
        return self.reward_head(last_token).squeeze(-1)   # one number per (x, y)
```

## 4. The recipe now →

```python
def bradley_terry_loss(r_chosen, r_rejected):
    return -F.logsigmoid(r_chosen - r_rejected).mean()
```

## 5. The recipe now →

```
for batch in dataloader:
    r_w = model(batch["chosen_ids"], batch["chosen_mask"])
    r_l = model(batch["rejected_ids"], batch["rejected_mask"])
    loss = bradley_terry_loss(r_w, r_l)
    loss.backward()
    optimizer.step()
```

## 6. 5. Skipping the Loop: DPO

```python
import torch.nn.functional as F

beta = 0.1  # how hard to push toward preferences (also controls the KL leash)

# implicit reward = how much more the policy likes a response than the reference does
#   this is exactly  beta * log( pi_theta(y|x) / pi_ref(y|x) ),  with beta applied below
chosen   = policy_logp_chosen   - ref_logp_chosen      # implicit reward, chosen
rejected = policy_logp_rejected - ref_logp_rejected    # implicit reward, rejected

# push the chosen implicit reward above the rejected one — the Bradley-Terry loss again
loss = -F.logsigmoid(beta * (chosen - rejected)).mean()
```

## 7. 7. Rewards You Can Check: RLVR and GRPO

```python
# sample K answers to the same prompt, then score each with a verifier
rewards = torch.tensor([verify(a) for a in answers])   # e.g. 1.0 if correct else 0.0

# advantage = how far above/below the group's average (no critic network needed)
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
```
