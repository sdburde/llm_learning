# GRPO (Group Relative Policy Optimization) — From First Principles

> GRPO is the core RL algorithm behind DeepSeek R1. It's a simpler, cheaper alternative to PPO.
> Now adopted by other LLM providers like Qwen.

**Papers:**
- GRPO (DeepSeekMath): https://arxiv.org/pdf/2402.03300
- DeepSeek-R1: https://arxiv.org/pdf/2501.12948
- PPO: https://arxiv.org/pdf/1707.06347

---

## Where GRPO Fits in the LLM Pipeline

```
Pre-training          → base model, lots of world knowledge, just completes sentences
Instruction Fine-Tuning → conversational, follows instructions
         │
         ├── Preference Fine-Tuning (RLHF / DPO)
         │     Signal: human preference between two responses
         │     Algorithm: PPO (RLHF) or DPO
         │
         └── Reasoning Fine-Tuning (RLVR)
               Signal: binary correctness on math/code/logic
               Algorithm: PPO (expensive) or GRPO (cheaper)
```

GRPO sits at the reasoning fine-tuning stage. It updates model weights based on whether the answer was correct — no human labels needed, just a verifier.

---

## RL Fundamentals (Quick Recap)

| RL Term | Physical World (Roomba) | LLM World |
|---|---|---|
| Agent | The robot | The LLM |
| Environment | Your living room | Humans, tools, Python interpreter |
| State Sₜ | XY position | Prompt + tokens generated so far |
| Action aₜ | Move one foot | Predict the next token |
| Reward rₜ | Covered more floor | Answer was correct / helpful |
| Trajectory τ | All moves in one session | All tokens in one response |
| Episode | Full cleaning session | Full response generation |

### Math Problem Walkthrough

```
Prompt: "How many books does John have if he starts with 3 and buys 5 more?"

S₀ = [prompt]
a₀ = "To"     →  S₁ = [prompt, "To"]
a₁ = "find"   →  S₂ = [prompt, "To", "find"]
a₂ = "out"    →  S₃ = [prompt, "To", "find", "out"]
...
aₙ = <end>    →  Episode ends

Environment checks: answer == 8 (ground truth) → reward = 1
All intermediate rewards = 0
```

The hard problem: one number (reward = 1 or 0) must backpropagate through hundreds of tokens and billions of parameters. Making this stable is the core challenge.

---

## Step 1 — Policy Gradient Methods & REINFORCE

### Terminology

The LLM is called the **policy**, written as πθ (pi theta), where θ are the model weights. At each state Sₜ, the policy outputs a probability distribution over all tokens:

```
πθ(Sₜ) → probability over vocab → sample token → action aₜ
```

### Goal of One Training Step

If an action aₜ led to a positive reward → increase the probability of that action.
If it led to a negative reward → decrease it.

**Policy Gradient Update:**
```
θ ← θ + α × Σₜ ∇θ log πθ(aₜ|Sₜ) × Gₜ
```

Where:
- `log πθ(aₜ|Sₜ)` — log probability of the token the model chose
- `Gₜ` — the return (sum of future rewards from step t)
- `α` — learning rate
- The log is for mathematical convenience but pulls in the same direction

### The Return Gₜ

If we have intermediate rewards (from a reward model scoring each sentence):
```
Gₜ = rₜ + γ·rₜ₊₁ + γ²·rₜ₊₂ + ...
```

γ (gamma) is the **discount factor** (0 to 1). Rewards further in the future contribute less — the model takes less credit for things that happen much later.

For our math task with only a final reward: Gₜ = R (the single final reward) for all tokens.

This update rule — policy gradient with discounted returns — is called **REINFORCE**, the very first policy gradient algorithm (1992). GRPO and PPO are its descendants.

---

## Step 2 — Reward Baselines & The Problem They Solve

### Why a Baseline?

Imagine the agent's goal is to become the richest person on Earth and it succeeds. Was the last action (aₜ₋₁) responsible for this?

- If the agent **starts as Jeff Bezos**: success is expected, the action deserves little credit
- If the agent **starts as an average person**: the action must have been extraordinary

**The baseline = where you started from.** The same reward means very different things depending on the starting point. Without a baseline, all positive rewards look equally good — even trivial ones.

So instead of the raw return Gₜ, we want:

```
Advantage Aₜ = Gₜ - baseline
```

"How much better than average was this action?"

### Actor-Critic Methods (What PPO Does)

A whole family of algorithms estimates the baseline using a **state value function Vφ**:

```
Vφ(Sₜ) = expected total future reward from state Sₜ
```

This gives:
```
Advantage Aₜ = Gₜ - Vφ(Sₜ)
```

The architecture is called **Actor-Critic**:
- **Actor** = the LLM πθ — takes actions
- **Critic** = the value model Vφ — judges how good each state is

**The cost:** The critic is typically initialized from a large LLM, so it:
- Doubles memory requirements
- Doubles compute
- Adds cognitive complexity — harder to understand and tune

---

## Step 3 — GRPO: Ditch the Critic

### The Core Idea

GRPO asks: in the language domain, do we really need per-step advantages computed by a sophisticated critic model?

In video games, you get constant feedback (gold medals, damage, points). In LLMs with math/reasoning tasks, you usually only get a reward at the very end. Intermediate estimates from a reward model are approximate and often wrong anyway.

**GRPO's answer:** Just use a simple group-average baseline instead.

### How GRPO Computes the Advantage

For each instruction, instead of one response, GRPO samples **G responses** (typically 4–8):

```
Instruction: "What is 3 + 5 × 2?"

Response 1: "The answer is 13."  → reward = 1  ✅
Response 2: "The answer is 16."  → reward = 0  ❌
Response 3: "The answer is 13."  → reward = 1  ✅
Response 4: "The answer is 11."  → reward = 0  ❌
```

Baseline = mean reward of the group = (1 + 0 + 1 + 0) / 4 = 0.5

Advantage for each response:
```
Aᵢ = (rᵢ - mean(rewards)) / std(rewards)
```

Response 1: A = (1 - 0.5) / 0.5 = +1.0  → reinforce
Response 2: A = (0 - 0.5) / 0.5 = -1.0  → discourage
Response 3: A = (1 - 0.5) / 0.5 = +1.0  → reinforce
Response 4: A = (0 - 0.5) / 0.5 = -1.0  → discourage

This is why it's called **Group Relative** Policy Optimization — advantages are relative to the group.

No critic model. No value function. No second large model to train.

### On Intermediate Rewards (Process Supervision)

DeepSeek did experiment with intermediate rewards — scoring each reasoning step (sentence) separately using a reward model. They call this "process supervision."

Result: small gains on GSM8K math benchmark vs just using a final reward. The overhead of training and running a step-level reward model was not worth it. The conclusion from their research: **keep it simple, final reward is enough**.

---

## Step 4 — The GRPO Loss

### From Policy Gradient to PPO

The simple policy gradient loss (REINFORCE) has high **variance**: noisy rewards or outlier updates can push the model into a bad state it can't recover from (e.g., probability of a token goes to 1, model can never predict anything else).

TRPO introduced "trust regions" — mathematically constrain how far the new parameters can move from the old ones per update. But it required expensive second-order derivatives (the Hessian). Impractical at LLM scale.

**PPO's fix: clipping the ratio**

Define:
```
rₜ(θ) = πθ(aₜ|Sₜ) / πθ_old(aₜ|Sₜ)
```

This ratio measures how much the model changed between updates:
- rₜ = 1 → no change
- rₜ > 1 → model now assigns higher probability to this action
- rₜ < 1 → model now assigns lower probability

PPO clips this ratio to `[1-ε, 1+ε]` (ε typically 0.2):

```
L_PPO = min( rₜ × Aₜ,  clip(rₜ, 1-ε, 1+ε) × Aₜ )
```

Taking the minimum:
- When advantage and ratio **agree** (good action → model got more likely, or bad action → model got less likely): clip to prevent overzealous update
- When advantage and ratio **disagree** (gradient going wrong direction due to noise): allow the unclipped loss to penalize fully

### GRPO Loss = PPO Loss + KL Penalty

GRPO uses the PPO clipped loss directly and adds one extra term:

```
L_GRPO = L_PPO  -  β × KL(πθ || πref)
```

Where:
- `L_PPO` — clipped policy gradient loss using group-relative advantages
- `KL(πθ || πref)` — KL divergence between current policy and **reference policy**
- `πref` — the model **before reasoning fine-tuning started** (e.g., the DPO checkpoint)
- `β` — hyperparameter controlling how much to regularize

**Two levels of constraint:**
1. **Clipping** — keeps updates small between consecutive training steps
2. **KL penalty** — keeps the model from drifting too far from where it started overall

The KL penalty compensates for GRPO removing the value function. Without a critic, variance is higher — so GRPO anchors the model to a stable reference point to prevent catastrophic drift.

---

## PPO vs GRPO — Side by Side

| Aspect | PPO | GRPO |
|---|---|---|
| Baseline | State value function Vφ (critic model) | Mean reward of group |
| Extra models | Actor + Critic (2× memory) | Actor only |
| Advantage | Per-token, per-state | Same value for all tokens in response |
| Intermediate rewards | Supported via critic | Usually just final reward |
| Variance control | Clipping + value function | Clipping + KL penalty to reference |
| Complexity | Higher | Lower |
| Cost | Higher | ~Half |

### One Training Step Comparison

**PPO:**
```
State Sₜ → πθ → action aₜ
         → Vφ → value of Sₜ
Advantage = Q(Sₜ, aₜ) - Vφ(Sₜ)
Loss = PPO clipped loss
Update both πθ and Vφ
```

**GRPO:**
```
Instruction → sample G responses → get G rewards
Baseline = mean(rewards)
Advantage = (reward - mean) / std
Loss = PPO clipped loss + KL penalty to πref
Update only πθ
```

---

## Full Code: GRPO Training Loop

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from dataclasses import dataclass
from typing import List
import re

# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class GRPOConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    learning_rate: float = 1e-6
    group_size: int = 8           # G — number of responses per instruction
    clip_eps: float = 0.2         # PPO clipping epsilon
    kl_beta: float = 0.01         # KL penalty coefficient
    ppo_epochs: int = 1           # how many times to update on same batch
    max_new_tokens: int = 512
    temperature: float = 1.0

cfg = GRPOConfig()

# ── Reward Function ────────────────────────────────────────────────────────────

def compute_reward(response: str, ground_truth: str) -> float:
    """Verifiable reward: 1.0 if correct, 0.0 if wrong."""
    numbers = re.findall(r'\b\d+\.?\d*\b', response)
    predicted = numbers[-1] if numbers else ""
    return 1.0 if predicted.strip() == ground_truth.strip() else 0.0

# ── Group-Relative Advantage ──────────────────────────────────────────────────

def compute_group_advantages(rewards: List[float]) -> torch.Tensor:
    """
    GRPO advantage: normalize rewards within the group.
    Aᵢ = (rᵢ - mean(r)) / std(r)
    """
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    mean = rewards_t.mean()
    std  = rewards_t.std() + 1e-8    # avoid div by zero
    advantages = (rewards_t - mean) / std
    return advantages

# ── KL Divergence ─────────────────────────────────────────────────────────────

def compute_kl_divergence(current_logits: torch.Tensor,
                           ref_logits: torch.Tensor) -> torch.Tensor:
    """
    Token-level KL divergence between current policy and reference policy.
    KL(πθ || πref) = Σ πθ(a) × log(πθ(a) / πref(a))
    """
    current_log_probs = F.log_softmax(current_logits, dim=-1)
    ref_log_probs     = F.log_softmax(ref_logits,     dim=-1)
    current_probs     = current_log_probs.exp()

    kl = (current_probs * (current_log_probs - ref_log_probs)).sum(dim=-1)
    return kl.mean()

# ── GRPO Loss ─────────────────────────────────────────────────────────────────

def grpo_loss(new_log_probs: torch.Tensor,
              old_log_probs: torch.Tensor,
              advantage: float,
              current_logits: torch.Tensor,
              ref_logits: torch.Tensor,
              clip_eps: float,
              kl_beta: float) -> torch.Tensor:
    """
    GRPO loss = PPO clipped surrogate loss + KL penalty

    new_log_probs: log πθ(tokens)       — current policy
    old_log_probs: log πθ_old(tokens)   — policy that generated the response
    advantage:     scalar Aᵢ for this response
    """
    # Importance sampling ratio: πθ / πθ_old
    ratio = torch.exp(new_log_probs - old_log_probs)

    # PPO clipped surrogate loss
    adv_tensor = torch.tensor(advantage)
    unclipped = ratio * adv_tensor
    clipped   = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_tensor
    policy_loss = -torch.min(unclipped, clipped).mean()

    # KL divergence penalty (keeps model close to reference)
    kl = compute_kl_divergence(current_logits, ref_logits)

    total_loss = policy_loss + kl_beta * kl
    return total_loss, policy_loss.item(), kl.item()

# ── GRPO Training Step ────────────────────────────────────────────────────────

def grpo_step(model, ref_model, tokenizer, optimizer, instruction, ground_truth, cfg):
    """
    One GRPO training step for a single instruction.
    1. Sample G responses
    2. Compute rewards
    3. Compute group-relative advantages
    4. PPO update with KL penalty
    """
    device = next(model.parameters()).device
    model.train()
    ref_model.eval()

    input_ids = tokenizer(
        instruction, return_tensors="pt", padding=True
    ).input_ids.to(device)
    prompt_len = input_ids.shape[1]

    # ── 1. Sample G responses ─────────────────────────────────────────────────
    responses, rewards = [], []

    with torch.no_grad():
        for _ in range(cfg.group_size):
            generated = model.generate(
                input_ids,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=cfg.temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
            response_ids  = generated[0][prompt_len:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
            reward = compute_reward(response_text, ground_truth)
            responses.append(generated[0])
            rewards.append(reward)

    print(f"  Rewards: {rewards} | Mean: {sum(rewards)/len(rewards):.2f}")

    # ── 2. Group-relative advantages ─────────────────────────────────────────
    advantages = compute_group_advantages(rewards)  # shape: [G]

    # ── 3. Get old log probs (reference for importance sampling) ──────────────
    old_log_probs_list = []
    with torch.no_grad():
        for gen_ids in responses:
            full_ids = gen_ids.unsqueeze(0)
            logits   = model(full_ids).logits[0]   # [T, vocab]
            log_probs = F.log_softmax(logits, dim=-1)
            # log probs for generated tokens only
            resp_log_probs = log_probs[prompt_len - 1:-1].gather(
                1, gen_ids[prompt_len:].unsqueeze(1)
            ).squeeze(1)
            old_log_probs_list.append(resp_log_probs)

    # ── 4. PPO update with KL penalty ─────────────────────────────────────────
    total_loss_val = 0.0
    optimizer.zero_grad()

    for i, (gen_ids, advantage) in enumerate(zip(responses, advantages)):
        full_ids = gen_ids.unsqueeze(0)
        resp_ids = gen_ids[prompt_len:]

        # Current policy logits
        outputs = model(full_ids)
        cur_logits  = outputs.logits[0]                      # [T, vocab]
        log_probs   = F.log_softmax(cur_logits, dim=-1)
        new_log_probs = log_probs[prompt_len - 1:-1].gather(
            1, resp_ids.unsqueeze(1)
        ).squeeze(1)

        # Reference policy logits (frozen)
        with torch.no_grad():
            ref_logits = ref_model(full_ids).logits[0]

        loss, p_loss, kl_val = grpo_loss(
            new_log_probs,
            old_log_probs_list[i].detach(),
            advantage.item(),
            cur_logits[prompt_len - 1:-1],     # current logits for KL
            ref_logits[prompt_len - 1:-1],      # ref logits for KL
            cfg.clip_eps,
            cfg.kl_beta,
        )

        # Accumulate gradients across group
        (loss / cfg.group_size).backward()
        total_loss_val += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "mean_reward":  sum(rewards) / len(rewards),
        "total_loss":   total_loss_val / cfg.group_size,
    }

# ── Main Training Loop ────────────────────────────────────────────────────────

def train_grpo():
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Policy model (gets updated)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Reference model (frozen — this is πref, the pre-RLVR checkpoint)
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)

    # Sample reasoning dataset (replace with GSM8K or similar)
    dataset = [
        ("Solve step by step: John has 3 apples and buys 5 more. How many?", "8"),
        ("Solve step by step: A train goes 60 mph for 2 hours. How far?",    "120"),
        ("Solve step by step: 15 students, 1/3 go home early. How many stay?", "10"),
    ]

    for step in range(50):
        instruction, ground_truth = dataset[step % len(dataset)]
        metrics = grpo_step(
            model, ref_model, tokenizer, optimizer,
            instruction, ground_truth, cfg
        )
        if step % 5 == 0:
            print(f"Step {step:3d} | Reward: {metrics['mean_reward']:.3f} | "
                  f"Loss: {metrics['total_loss']:.4f}")

    model.save_pretrained("./grpo-reasoning-model")
    tokenizer.save_pretrained("./grpo-reasoning-model")

if __name__ == "__main__":
    train_grpo()
```

---

## Summary — The Full Derivation Chain

```
REINFORCE (1992)
  Update = ∇ log π(aₜ|Sₜ) × Gₜ
  Problem: high variance from noisy rewards
      ↓
Add Baseline
  Advantage Aₜ = Gₜ - baseline
  Problem: what's a good baseline?
      ↓
Actor-Critic (PPO's approach)
  Baseline = Vφ(Sₜ) (state value function, a second model)
  Problem: doubles memory and compute
      ↓
TRPO
  Trust regions: constrain parameter updates mathematically
  Problem: needs second-order derivatives, too expensive
      ↓
PPO
  Clips ratio rₜ = πθ/πθ_old to [1-ε, 1+ε]
  Cheap and effective variance control
  Still needs critic model though
      ↓
GRPO (DeepSeek's approach)
  Drops the critic entirely
  Baseline = mean reward of a group of G responses
  Advantage = (rᵢ - mean) / std
  Loss = PPO clipped loss + KL penalty to reference model
  Result: ~half the memory, comparable performance
```

### Key Hyperparameters

| Parameter | Typical Value | Controls |
|---|---|---|
| G (group_size) | 4–8 | Number of responses sampled per question |
| ε (clip_eps) | 0.2 | Max allowed policy change per step |
| β (kl_beta) | 0.01 | How much to penalize drift from reference model |
| γ (gamma) | 1.0 | Discount factor (usually 1 for LLMs, single final reward) |

### The One Key Insight

> GRPO's bet: in the language domain with sparse end-of-episode rewards, per-step value estimates from a critic are wrong anyway. A simple group average is just as good a baseline — and it's free.

The savings are real. No second 671B parameter critic model to train and store.

---

> "Research papers are a bit like Instagram — Greek symbols are one way to flex.
> Once you see past that, the core ideas are often quite simple." — Julia Turc
