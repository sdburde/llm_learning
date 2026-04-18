# PPO (Proximal Policy Optimization) — From First Principles

> PPO is the RL algorithm behind RLHF (aligning LLMs with human preferences) and RLVR (giving LLMs reasoning abilities).
> Used to train OpenAI o1/o3, Claude 3.7, Grok 3, and more.

**Papers:**
- PPO: https://arxiv.org/pdf/1707.06347
- GAE: https://arxiv.org/pdf/1506.02438
- TRPO: https://arxiv.org/pdf/1502.05477

---

## The Big Picture — Why PPO Exists

In standard supervised learning, you show the model the correct answer and it learns to copy it. Simple.

In RL for LLMs, you don't have a correct answer token-by-token. You only have a reward at the end — a single number saying "this response was good or bad." Somehow you have to backpropagate that single number through hundreds of generated tokens and billions of model parameters to update weights. This is brutally hard to do stably.

PPO is the result of decades of research accumulated to make this work in practice.

---

## Step 1 — RL Basics (in the LLM Context)

### The Framework

| RL Term | Physical World | LLM World |
|---|---|---|
| Agent | A Roomba robot | The LLM (e.g. Llama) |
| Environment | Your living room | Humans, tools, data, Python interpreter |
| Episode | One full cleaning session | One full response generation |
| State (Sₜ) | XY position of robot | Prompt + all tokens generated so far |
| Action (aₜ) | Move one foot right | Predict the next token |
| Reward (rₜ) | Covered 5% more floor | Response was correct / helpful |
| Trajectory (τ) | All states + actions in one session | All states + tokens in one response |

### Concrete Example: Math Reasoning Task

Dataset: GSM8K (middle school math questions)

```
Instruction: "John has 3 apples and buys 5 more. How many does he have?"
Model generates: "To find... the answer is 8"
Environment checks: 8 == ground truth (8) → reward = 1
```

The RL loop for one token at a time:
```
State S₀ = [prompt]
Action a₀ = "To"       → State S₁ = [prompt, "To"]
Action a₁ = "find"     → State S₂ = [prompt, "To", "find"]
Action a₂ = "out"      → State S₃ = [prompt, "To", "find", "out"]
...
Action aₙ = <end>      → Episode ends → reward computed
```

**Key point:** All intermediate rewards are 0. Only the final token gets a reward. This makes training hard — how do you know which of the 200 tokens in your response caused the success or failure?

---

## Step 2 — Policy Gradient (The Core Loss)

### Terminology Shift
In RL, the LLM is called the **policy**, written as:
```
πθ  (pi theta)
```
where θ are the model weights. It's just a new name for the LLM.

At each state Sₜ, the policy outputs a probability distribution over all possible tokens:
```
πθ(Sₜ) → probability distribution over vocab
```
The model samples a token from this distribution and that becomes action aₜ.

### Supervised vs RL Training Signal

**In supervised learning:**
- You know the correct token at every step
- Label = probability 1 for correct token, 0 for everything else
- Loss = cross-entropy between model output and label
- Effect: correct token becomes slightly more likely, others slightly less

**In RL:**
- No correct token is known
- You only have a reward at the end
- You need to figure out a "scaling factor" that plays the same role as the supervised label

That scaling factor is called the **Advantage**, written as **Aₜ**.

### The Policy Gradient Loss

Goal: if action aₜ was good (high advantage), increase its probability. If bad (negative advantage), decrease it.

```
Loss = -Σₜ log πθ(aₜ | Sₜ) × Aₜ
```

Breaking it down:
- `log πθ(aₜ | Sₜ)` — log probability of the action the model took
- `Aₜ` — how good was this action compared to average
- The negative sign — because we minimize loss, so maximizing reward means minimizing negative reward
- Summed over all time steps in the trajectory

The log is added for mathematical convenience — it still pulls the model in the right direction but is easier to work with in practice.

**The remaining question:** How do you compute Aₜ? This requires the value function.

---

## Step 3 — The Value Function

### What is the Advantage?

PPO's philosophy:
> "Since we don't know the correct action, compare our action against the *average* action."

Advantage = how much better than average was the action we took?
- Advantage > 0 → this action was better than average → reinforce it
- Advantage < 0 → this action was worse than average → discourage it

To compute "average expected outcome from a state," we need the **value function**.

### The Value Function V(Sₜ)

The value function answers: **"How good is the current state, regardless of what we do next?"**

Formally:
```
V(Sₜ) = E[sum of all future rewards from step t onwards]
```

It's the expected total discounted return from state Sₜ, averaged over all possible continuations.

Examples for our math problem:
```
State: "beep boop bop"          → V ≈ very low  (dead end, hard to recover)
State: "To find out"            → V ≈ medium    (some good, some bad completions possible)
State: "The answer is 72"       → V ≈ very high (already correct)
```

### Why Not Compute V Directly?

Computing V exactly requires expanding a tree with branching factor = vocabulary size (50,000+ tokens) at every step. Impossible.

Instead: train a **separate model Vφ** to estimate it.
- φ are the weights of the value model
- It's trained separately from the policy

### Training the Value Function

Use mean squared error between the model's value estimate and the actual total discounted return:

```
V_loss = MSE(Vφ(Sₜ), Gₜ)
```

Where Gₜ is the total discounted return:
```
Gₜ = rₜ + γ·rₜ₊₁ + γ²·rₜ₊₂ + ... + γᵀ⁻ᵗ·rₜ
```

γ (gamma) is a discount factor between 0 and 1. Rewards further in the future contribute less — because they're less certain and less attributable to the current action.

### Actor-Critic Architecture

The agent now has two models:
- **Actor** = the policy πθ (the LLM) — decides what to do
- **Critic** = the value function Vφ — judges how good the current state is

This is called an **Actor-Critic model**.

```
Actor:  πθ  →  generates tokens
Critic: Vφ  →  scores each state
```

Both are trained simultaneously, each with their own loss.

### Computing the Advantage from V

Now that we have V, we can compute the Advantage.

We need Q(Sₜ, aₜ) — the quality of taking action aₜ in state Sₜ.

Then:
```
Advantage Aₜ = Q(Sₜ, aₜ) - V(Sₜ)
```

"How much better was this specific action vs. the average expected outcome from this state?"

---

## Step 4 — Generalized Advantage Estimate (GAE)

### Two Ways to Estimate Q

**Option 1 — Monte Carlo:**
Wait for the entire episode to finish. Sum up all actual rewards received.
```
Q(Sₜ, aₜ) = rₜ + γ·rₜ₊₁ + γ²·rₜ₊₂ + ...  (actual rewards)
```
- ✅ Low bias — uses real rewards, no assumptions
- ❌ High variance — noisy rewards (especially from humans) cause huge swings
- ❌ Must wait for full episode before any update

**Option 2 — Bootstrapping:**
Don't wait for the full episode. Use the value function estimate for the next state.
```
Q(Sₜ, aₜ) ≈ rₜ + γ·Vφ(Sₜ₊₁)
```
This gives the **temporal difference residual** (TD residual), also called δₜ:
```
δₜ = rₜ + γ·Vφ(Sₜ₊₁) - Vφ(Sₜ)
```
- ✅ Low variance — smooth because it relies on V, not raw rewards
- ❌ High bias — Vφ is an approximation, so errors compound

### The Bias-Variance Trade-off

| Method | Bias | Variance |
|---|---|---|
| Monte Carlo (full episode) | Low | High |
| Single-step TD bootstrapping | High | Low |
| Somewhere in between | Medium | Medium |

You can express Q using 1 reward, 2 rewards, 3 rewards... all the way to T rewards (Monte Carlo). Each gives a different trade-off.

### GAE: Best of All Worlds

GAE takes an **exponentially weighted average** of all these estimates:

```
Aₜ^GAE = Σₖ (γλ)ᵏ · δₜ₊ₖ
```

Where:
- δₜ = temporal difference residual at step t
- λ (lambda) — controls the bias-variance trade-off
  - λ = 0 → pure single-step bootstrapping (low variance, high bias)
  - λ = 1 → pure Monte Carlo (low bias, high variance)
  - λ ≈ 0.95 in practice — balance between both

GAE is used in virtually all modern RL for LLMs because it's empirically the most stable.

---

## Step 5 — End-to-End Training Algorithm

At this point we have:
- **Policy gradient loss** — trains the actor using advantages
- **Value function loss** — trains the critic using discounted returns

The basic loop:

```
Initialize: θ₀ (LLM weights), φ₀ (value model weights)

For each training step:
  1. Sample a mini-batch of prompts
  2. Generate complete responses using current policy πθ
     → This gives a set of trajectories τ
  3. For each token in each trajectory:
     a. Compute discounted return Gₜ → used for V_loss
     b. Compute GAE advantage Aₜ → used for policy gradient loss
  4. Compute gradients of both losses
  5. Update θ (actor) and φ (critic) via backprop
  6. Discard trajectories, start next step
```

**Problem:** Step 6 throws away all the generated responses after just one update. That's expensive — generating responses is slow. In supervised learning we'd reuse the data for multiple epochs.

---

## Step 6 — Importance Sampling (Reusing Trajectories)

### The Problem with Multiple Updates

If we try to update the model on the same trajectories multiple times, we hit a mismatch: the trajectories were generated using old parameters θ_old, but we're computing the loss using updated parameters θ_new.

After a few updates, the model has drifted far from θ_old. The gradient computed from old trajectories no longer points in the right direction → training breaks.

### The Fix: Importance Sampling

Importance sampling is a mathematical trick that lets you estimate an expectation under one distribution using samples from another distribution, by reweighting with a ratio:

```
E_P[f(x)] ≈ E_Q[f(x) · P(x)/Q(x)]
```

Applied to PPO: multiply the loss by the ratio of new policy probability to old policy probability:

```
ratio rₜ = πθ(aₜ | Sₜ) / πθ_old(aₜ | Sₜ)
```

New surrogate loss:
```
L_surrogate = rₜ(θ) × Aₜ
```

When the model hasn't changed much, rₜ ≈ 1 and the loss is approximately what it was. This lets us safely do multiple gradient updates on the same batch of trajectories.

This is called a **surrogate loss** — it's not exactly the original loss, but it points in the same direction and is easier to optimize.

---

## Step 7 — PPO Clipping (The Novel Contribution)

Everything up to here existed before PPO. The actual new idea in PPO is **clipping**.

### The Problem

Even with importance sampling, if the model updates are too large, training becomes unstable. A previous algorithm called TRPO addressed this by mathematically constraining how far the new weights could be from the old weights — but it required computing second-order derivatives (the Hessian), which is enormously expensive at LLM scale.

### The PPO Solution: Clip the Ratio

The ratio `rₜ = πθ/πθ_old` naturally measures how much the model has changed:
- rₜ = 1 → model hasn't changed
- rₜ > 1 → model now assigns higher probability to this action
- rₜ < 1 → model now assigns lower probability to this action

PPO clips this ratio to stay within `[1-ε, 1+ε]` where ε is typically 0.2:

```
L_clip = min(rₜ × Aₜ,  clip(rₜ, 1-ε, 1+ε) × Aₜ)
```

Taking the minimum means:
- If the ratio is within bounds → use the unclipped loss (normal update)
- If the ratio goes outside bounds → clip it (limit the update size)

This prevents any single mini-batch from updating the model too drastically. Cheap to compute (just a clip operation), no second-order derivatives needed.

### Final PPO Loss

```
Total Loss = L_clip  +  c₁ × V_loss  +  c₂ × entropy_bonus
```

- `L_clip` — policy gradient loss with clipped ratio
- `V_loss` — value function MSE loss
- `entropy_bonus` — encourages exploration (often dropped in practice)
- c₁, c₂ — weighting coefficients

---

## Full Code: PPO for LLM Reasoning

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from dataclasses import dataclass
from typing import List

# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    learning_rate: float = 1e-6
    gamma: float = 0.99          # discount factor
    lam: float = 0.95            # GAE lambda
    clip_eps: float = 0.2        # PPO clipping epsilon
    vf_coef: float = 0.5         # value function loss weight
    ppo_epochs: int = 4          # how many times to reuse each batch
    mini_batch_size: int = 4
    max_new_tokens: int = 512

cfg = PPOConfig()

# ── Actor-Critic Model ─────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Wraps a causal LLM with an added value head.
    Actor  = the LLM (produces token probabilities)
    Critic = a linear layer on top of hidden states (produces state values)
    """
    def __init__(self, model_name: str):
        super().__init__()
        self.actor = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        hidden_size = self.actor.config.hidden_size
        # Value head: maps last hidden state → scalar value
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.actor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = outputs.logits                        # shape: [B, T, vocab]
        hidden = outputs.hidden_states[-1]             # last layer: [B, T, H]
        values = self.value_head(hidden).squeeze(-1)   # shape: [B, T]
        return logits, values

    def generate(self, input_ids, **kwargs):
        return self.actor.generate(input_ids, **kwargs)

# ── Reward Function ────────────────────────────────────────────────────────────

def compute_reward(response: str, ground_truth: str) -> float:
    """
    Verifiable reward: 1.0 if correct, 0.0 if wrong.
    In practice: parse the boxed answer from response and compare.
    """
    import re
    # Extract number from response (simplified)
    numbers = re.findall(r'\b\d+\.?\d*\b', response)
    predicted = numbers[-1] if numbers else ""
    return 1.0 if predicted.strip() == ground_truth.strip() else 0.0

# ── GAE (Generalized Advantage Estimate) ─────────────────────────────────────

def compute_gae(rewards: List[float], values: List[float],
                gamma: float, lam: float):
    """
    rewards: [r₀, r₁, ..., rₙ]  (intermediate=0, final=1 or 0)
    values:  [V(S₀), V(S₁), ..., V(Sₙ)]
    Returns: advantages and discounted returns
    """
    T = len(rewards)
    advantages = [0.0] * T
    returns = [0.0] * T

    gae = 0.0
    next_value = 0.0  # value after episode ends = 0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]   # TD residual δₜ
        gae = delta + gamma * lam * gae                        # GAE accumulation
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
        next_value = values[t]

    return torch.tensor(advantages, dtype=torch.float32), \
           torch.tensor(returns, dtype=torch.float32)

# ── PPO Loss ──────────────────────────────────────────────────────────────────

def ppo_loss(new_log_probs: torch.Tensor,
             old_log_probs: torch.Tensor,
             advantages: torch.Tensor,
             values: torch.Tensor,
             returns: torch.Tensor,
             clip_eps: float,
             vf_coef: float):
    """
    new_log_probs: log πθ(aₜ | Sₜ)  [T]  — current policy
    old_log_probs: log πθ_old(aₜ | Sₜ) [T] — policy that generated the data
    advantages:    Aₜ  [T]
    values:        Vφ(Sₜ)  [T]
    returns:       Gₜ  [T]
    """
    # Importance sampling ratio: πθ / πθ_old
    ratio = torch.exp(new_log_probs - old_log_probs)

    # Clipped surrogate loss
    unclipped = ratio * advantages
    clipped   = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()

    # Value function loss (MSE)
    value_loss = F.mse_loss(values, returns)

    total_loss = policy_loss + vf_coef * value_loss
    return total_loss, policy_loss.item(), value_loss.item()

# ── Training Step ─────────────────────────────────────────────────────────────

def ppo_training_step(model, tokenizer, optimizer, prompts, ground_truths, cfg):
    model.train()
    device = next(model.parameters()).device

    all_metrics = []

    for prompt, gt in zip(prompts, ground_truths):

        # ── 1. Generate response (old policy) ────────────────────────────────
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        response_ids = generated[0][input_ids.shape[1]:]  # only new tokens
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        # ── 2. Compute reward ─────────────────────────────────────────────────
        final_reward = compute_reward(response_text, gt)

        # Build reward sequence: 0 for all steps except last
        T = len(response_ids)
        rewards = [0.0] * (T - 1) + [final_reward]

        # ── 3. Get old log probs and values (reference) ───────────────────────
        with torch.no_grad():
            full_ids = generated[0].unsqueeze(0)
            logits, values_all = model(full_ids)

            # Log probs for the generated tokens
            log_probs_all = F.log_softmax(logits[0], dim=-1)
            prompt_len = input_ids.shape[1]
            old_log_probs = log_probs_all[prompt_len - 1:-1].gather(
                1, response_ids.unsqueeze(1)
            ).squeeze(1)  # shape: [T]

            old_values = values_all[0, prompt_len - 1:-1]  # shape: [T]

        # ── 4. Compute GAE advantages and returns ─────────────────────────────
        advantages, returns = compute_gae(
            rewards, old_values.cpu().tolist(), cfg.gamma, cfg.lam
        )
        advantages = advantages.to(device)
        returns = returns.to(device)

        # Normalize advantages (reduces variance)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── 5. PPO update for multiple epochs ────────────────────────────────
        for epoch in range(cfg.ppo_epochs):
            logits, values_all = model(full_ids)
            log_probs_all = F.log_softmax(logits[0], dim=-1)
            new_log_probs = log_probs_all[prompt_len - 1:-1].gather(
                1, response_ids.unsqueeze(1)
            ).squeeze(1)
            new_values = values_all[0, prompt_len - 1:-1]

            loss, p_loss, v_loss = ppo_loss(
                new_log_probs, old_log_probs.detach(),
                advantages, new_values, returns,
                cfg.clip_eps, cfg.vf_coef
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            all_metrics.append({
                "reward": final_reward,
                "policy_loss": p_loss,
                "value_loss": v_loss,
                "ratio_mean": torch.exp(new_log_probs - old_log_probs.detach()).mean().item()
            })

    return all_metrics

# ── Main Training Loop ────────────────────────────────────────────────────────

def train_ppo():
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = ActorCritic(cfg.model_name)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)

    # Example math dataset (replace with real GSM8K or similar)
    prompts = [
        "Solve step by step: John has 3 apples and buys 5 more. How many?",
        "Solve step by step: A train travels 60 mph for 2 hours. How far?",
    ]
    ground_truths = ["8", "120"]

    for step in range(100):  # scale as needed
        metrics = ppo_training_step(
            model, tokenizer, optimizer, prompts, ground_truths, cfg
        )
        avg_reward = sum(m["reward"] for m in metrics) / len(metrics)
        avg_ratio  = sum(m["ratio_mean"] for m in metrics) / len(metrics)

        if step % 10 == 0:
            print(f"Step {step:3d} | Reward: {avg_reward:.3f} | "
                  f"Ratio: {avg_ratio:.3f}")

if __name__ == "__main__":
    train_ppo()
```

---

## Summary — How PPO Was Built Up Step by Step

```
Problem: Single reward → update billions of params → unstable
         ↓
Policy Gradient
  Loss = -Σ log π(aₜ|Sₜ) × Aₜ
  Need to compute Aₜ (advantage)
         ↓
Value Function Vφ
  Estimates expected future reward from each state
  Aₜ = Q(Sₜ,aₜ) - V(Sₜ)
  Adds a Critic alongside the Actor
         ↓
GAE (Generalized Advantage Estimate)
  Blends Monte Carlo (low bias, high variance)
  with TD bootstrapping (low variance, high bias)
  via exponentially weighted sum, controlled by λ
         ↓
Importance Sampling
  Allows multiple gradient updates per batch
  Ratio rₜ = πθ / πθ_old corrects for distribution shift
         ↓
PPO Clipping (the novel part)
  Clips rₜ to [1-ε, 1+ε]
  Prevents catastrophic updates
  Cheap alternative to TRPO's second-order methods
         ↓
Final PPO Loss:
  L = -min(rₜ×Aₜ, clip(rₜ,1-ε,1+ε)×Aₜ) + c₁×V_loss
```

### Key Hyperparameters

| Parameter | Typical Value | Controls |
|---|---|---|
| γ (gamma) | 0.99 | How much future rewards matter |
| λ (lambda) | 0.95 | Bias-variance trade-off in GAE |
| ε (epsilon) | 0.2 | Max allowed policy change per step |
| c₁ (vf_coef) | 0.5 | Weight of value function loss |
| ppo_epochs | 4 | How many times to reuse each batch |

---

> PPO builds on: Policy Gradient → Actor-Critic → GAE → Importance Sampling → Clipping.
> Each piece solves a specific instability problem from the piece before it.
