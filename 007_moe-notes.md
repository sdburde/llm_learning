# Mixture of Experts (MoE) — Complete Notes

> Used by Meta (Llama 4), DeepSeek, Mistral. The dominant strategy for scaling LLMs today.
> Full history from 1991 vowel recognition to 2 trillion parameter models.

---

## Why MoE Exists — The Scaling Problem

A 2020 paper put hard numbers on what everyone already intuited: **scale matters**. More compute, more data, more parameters → better models.

The industry scaled along all three axes, but hit walls:
- **Data** — we've scraped most of the internet. Not much left.
- **Compute** — being explored via RL and chain-of-thought
- **Parameters** — scaling parameters linearly scales inference latency too

MoE solves the parameters problem: **increase total parameter count without proportionally increasing inference cost**.

---

## Part 1 — The Original MoE (1991)

### The Task

Vowel recognition from speech. Input: two scalar values (formants) extracted from a spoken sound wave:
- **F₁** — how open the mouth is
- **F₂** — position of the tongue

Output: classify into one of four vowels.

### The Architecture — Dense MoE

```
Input X (2D vector)
      │
      ├─────────────────────────────────┐
      ▼                                 ▼
Gating Network                    Expert 1..7
  → probability distribution       → each independently outputs
    over 7 experts                   probability over 4 vowels
      │                                 │
      └─────────── weighted sum ────────┘
                        │
                     Output
```

**Gating network:** Takes input X, outputs a probability distribution over all 7 experts. "How likely is each expert to give the correct answer for this input?"

**Experts:** Each is a small neural network with its own parameters. Each independently predicts which vowel.

**Final output:** Sum of all expert predictions, weighted by gating probabilities.

This is called **dense MoE** — all 7 experts participate in every prediction, even if some get near-zero weight.

**Goal in 1991:** Not efficiency. Not memory savings. Just better classification accuracy. The model had a few hundred total parameters — tiny. MoE improved accuracy by allowing different experts to specialize on different regions of the input space.

**Key observation from the Colab reproduction:** Even though 7 experts are trained, only 2 end up truly active. The gating network learns on its own that the other 5 don't add value. This spontaneous specialization is the seed of everything that follows.

### Code: Dense MoE (1991 Style)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# ── Expert Network ────────────────────────────────────────────────────────────

class Expert(nn.Module):
    """
    A small feedforward network.
    One of several competing experts.
    Each has its own independent parameters.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),                          # old-school activation
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)  # [batch, output_dim]


# ── Gating Network ────────────────────────────────────────────────────────────

class GatingNetwork(nn.Module):
    """
    Takes input X, outputs a probability distribution over experts.
    Tells us how much to trust each expert for this particular input.
    """
    def __init__(self, input_dim: int, n_experts: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, n_experts),
        )

    def forward(self, x):
        logits = self.net(x)                    # [batch, n_experts]
        return F.softmax(logits, dim=-1)        # probability over experts


# ── Dense Mixture of Experts ──────────────────────────────────────────────────

class DenseMoE(nn.Module):
    """
    1991-style MoE: ALL experts participate in every prediction.
    Final output = weighted sum of all expert outputs.
    """
    def __init__(self, input_dim: int, n_experts: int,
                 hidden_dim: int, output_dim: int):
        super().__init__()
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim)
            for _ in range(n_experts)
        ])
        self.gate = GatingNetwork(input_dim, n_experts)

    def forward(self, x):
        # Gate: how much to trust each expert? [batch, n_experts]
        gate_weights = self.gate(x)

        # Run ALL experts [batch, output_dim] for each expert
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts],
            dim=1
        )  # [batch, n_experts, output_dim]

        # Weighted sum: gate_weights * expert_outputs
        # [batch, n_experts, 1] * [batch, n_experts, output_dim]
        gate_weights = gate_weights.unsqueeze(-1)
        output = (gate_weights * expert_outputs).sum(dim=1)  # [batch, output_dim]
        return output


# ── Training ──────────────────────────────────────────────────────────────────

def train_dense_moe():
    # Synthetic formant data: 4 vowel classes
    vowels = {'ae': [800, 1800], 'ah': [700, 1100],
              'ih': [450, 2100], 'uh': [600, 1400]}

    X_list, y_list = [], []
    for label, (f1_mean, f2_mean) in enumerate(vowels.values()):
        samples = np.random.randn(100, 2) * 50 + [f1_mean, f2_mean]
        X_list.append(samples)
        y_list.extend([label] * 100)

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list)

    # Normalize — important: backprop with raw Hz values is unstable
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = torch.tensor(X_train)
    X_test  = torch.tensor(X_test)
    y_train = torch.tensor(y_train)
    y_test  = torch.tensor(y_test)

    model = DenseMoE(input_dim=2, n_experts=7, hidden_dim=32, output_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(200):
        model.train()
        logits = model(X_train)
        loss = F.cross_entropy(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(X_test).argmax(dim=-1)
                acc = (preds == y_test).float().mean()
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Accuracy: {acc:.2%}")

train_dense_moe()
```

---

## Part 2 — Sparse MoE Revival (2017, Google)

### The Problem by 2017

Deep learning is flourishing. Models are getting large but hardware can't keep up. BERT (state-of-the-art that year) had 340M parameters — dense, all active at inference.

### The Solution: Sparsity

Google revived MoE but made it **sparse** — for any given input, only K experts activate. The rest are ignored.

```
1991 Dense MoE:   7 experts, all 7 active every time
2017 Sparse MoE:  hundreds/thousands of experts, only K=2 active per token
```

The 2017 model: **137 billion total parameters, only 15 million active per input**.
That's 400× more parameters than BERT, yet similar inference cost.

### How Sparse Gating Works

Instead of softmax over all experts (dense), we do top-K selection:

```
Input X
  → Gating network → logits [one per expert]
  → Pick top K logits → zero out the rest
  → Softmax on surviving K logits → weights for K experts
  → Weighted sum of K expert outputs → final output
```

**The "scary discontinuity" problem:** Top-K is not differentiable. The paper's response (literally): "while this form of sparsity creates some theoretically scary discontinuities in the output of a gating function, we have not yet observed this to be a problem in practice." Classic ML: it shouldn't work but it does.

### The Load Balancing Problem

At the start of training, gating network weights are random. Some experts get picked by random chance. Those experts get trained. They get slightly better. The gating network routes more inputs to them. **The rich get richer** — most experts never get trained.

**Fix: Add noise to the gating logits before top-K selection.**

The noise randomizes which experts get picked early in training, giving all experts a fair chance to learn. The noise variance is also learned per expert — some get more noise, some less.

```python
def noisy_top_k_gating(x, W_gate, W_noise, k, training=True):
    """
    Gating with learned noise for load balancing.
    x:       input embedding  [batch, dim]
    W_gate:  gate weight matrix  [dim, n_experts]
    W_noise: noise scale matrix  [dim, n_experts]
    k:       number of active experts
    """
    logits = x @ W_gate                              # [batch, n_experts]

    if training:
        # Learned noise: different variance per expert
        noise_scale = F.softplus(x @ W_noise)        # [batch, n_experts]
        noise = torch.randn_like(logits) * noise_scale
        logits = logits + noise

    # Top-K selection
    top_k_logits, top_k_indices = logits.topk(k, dim=-1)   # [batch, k]

    # Zero out everything except top K, then softmax
    sparse_logits = torch.full_like(logits, float('-inf'))
    sparse_logits.scatter_(1, top_k_indices, top_k_logits)
    gate_weights = F.softmax(sparse_logits, dim=-1)         # [batch, n_experts]

    return gate_weights, top_k_indices
```

---

## Part 3 — Switch Transformer & K=1 (2021, Google)

Scaled further to **1.6 trillion total parameters**. The key finding: you can reduce K from 2 down to **1 — a single active expert per token**.

This is counterintuitive. How does the router learn to route well if it never gets to compare two experts at once?

The explanation: training uses batches of N examples. Within one backpropagation step, the router makes N expert choices — just for N different inputs in the same batch. That's enough signal to learn.

### MoE Inside Transformers

In a standard transformer layer:
```
Self-Attention → Feed Forward Network (FFN)
```

MoE replaces the FFN:
```
Self-Attention → Mixture of Experts (collection of FFNs)
```

Each expert IS a feed forward network. So instead of one fixed FFN applied to all tokens, you pick the most suitable FFN for each token, individually.

**Self-attention vs FFN — the difference:**
- **Self-attention** = team meeting. Every token looks at every other token and updates its understanding based on context. "Bank" attends to "river" and updates to mean nature, not finance.
- **FFN** = solo work after the meeting. Each token's embedding is processed independently. No cross-token interaction.

MoE replaces the solo work phase. Each token gets processed by the expert most suited for it.

**Why not replace self-attention with MoE too?** The Switch Transformer tried this. Marginal improvement at the cost of unstable training. Not worth it. So 30 years after 1991, experts are still simple feed forward networks.

---

## Part 4 — Modern Open-Source MoEs

### Mixtral (Mistral, January 2024)

Very close to the Switch Transformer architecture. Minimal changes:
- 8 experts per FFN layer
- K=2 active per token (back to K>1)
- Standard transformer otherwise

```python
class MixtralMoELayer(nn.Module):
    """
    Mixtral-style sparse MoE layer.
    Replaces one FFN in a transformer layer.
    8 experts, top-2 selected per token.
    """
    def __init__(self, dim: int, n_experts: int = 8, k: int = 2,
                 ffn_hidden: int = 2048):
        super().__init__()
        self.k = k
        self.n_experts = n_experts

        # Gate: maps token embedding → expert selection
        self.gate = nn.Linear(dim, n_experts, bias=False)

        # Experts: each is an FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, ffn_hidden),
                nn.SiLU(),                      # modern activation
                nn.Linear(ffn_hidden, dim),
            )
            for _ in range(n_experts)
        ])

    def forward(self, x):
        """
        x: [batch, seq_len, dim]
        Returns: [batch, seq_len, dim]
        """
        batch, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)                    # [batch*seq, dim]

        # Gate: which experts to use for each token?
        gate_logits = self.gate(x_flat)             # [batch*seq, n_experts]
        gate_weights = F.softmax(gate_logits, dim=-1)

        # Top-K selection
        top_k_weights, top_k_indices = gate_weights.topk(self.k, dim=-1)
        # Renormalize selected weights to sum to 1
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Run each selected expert for each token
        output = torch.zeros_like(x_flat)
        for expert_idx in range(self.n_experts):
            # Which tokens route to this expert?
            mask = (top_k_indices == expert_idx).any(dim=-1)  # [batch*seq]
            if not mask.any():
                continue

            # Get the weight for this expert for those tokens
            expert_mask = (top_k_indices == expert_idx)        # [batch*seq, k]
            weight = (top_k_weights * expert_mask).sum(dim=-1) # [batch*seq]

            # Run expert on relevant tokens
            expert_out = self.experts[expert_idx](x_flat[mask])  # [n_routed, dim]
            output[mask] += weight[mask].unsqueeze(-1) * expert_out

        return output.view(batch, seq_len, dim)
```

### DeepSeek-MoE (January 2024)

Two key modifications on top of Mixtral:

**Modification 1 — Finer-grained experts:**
```
Mixtral:   8 experts, each full-size FFN
DeepSeek: 16 experts, each half-size FFN
```
Same total parameters. But more experts → more routing choices → more specialization.

This is the "generalist vs specialist" hiring trade-off. More specialists → better at their narrow domain. The right balance is empirical — you try things and see.

**Modification 2 — Shared Expert:**
```
DeepSeek MoE Layer:
  Input X
    ├── Shared Expert (always runs)     ← always active, handles common knowledge
    └── Routed Experts (top-K selected) ← handle specialized knowledge
  Output = shared_expert(X) + Σ weighted routed experts
```

**Why a shared expert?** All experts tend to learn certain fundamental capabilities just to be useful at all — basic grammar, common reasoning patterns, etc. This is redundant. Each expert reinvents the same wheel.

The shared expert absorbs all that common knowledge. Routed experts can then focus purely on their specialization without wasting capacity on basics. Analogy: everyone in a company needs communication skills — rather than each person separately developing them, you give everyone a common training.

Llama 4 adopted this same shared expert design from DeepSeek.

```python
class DeepSeekMoELayer(nn.Module):
    """
    DeepSeek-style MoE with shared expert + routed experts.
    Key differences from Mixtral:
    - 2× more experts, each 2× smaller
    - One shared expert always runs (never routed)
    """
    def __init__(self, dim: int,
                 n_routed_experts: int = 16,
                 k: int = 2,
                 ffn_hidden: int = 1024):      # half of Mixtral's 2048
        super().__init__()
        self.k = k

        # Shared expert: always active, learns common knowledge
        self.shared_expert = nn.Sequential(
            nn.Linear(dim, ffn_hidden),
            nn.SiLU(),
            nn.Linear(ffn_hidden, dim),
        )

        # Routed experts: specialized, selected by router
        self.routed_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, ffn_hidden),
                nn.SiLU(),
                nn.Linear(ffn_hidden, dim),
            )
            for _ in range(n_routed_experts)
        ])

        self.gate = nn.Linear(dim, n_routed_experts, bias=False)

    def forward(self, x):
        batch, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)

        # Shared expert always runs — no routing needed
        shared_out = self.shared_expert(x_flat)

        # Gating for routed experts
        gate_logits = self.gate(x_flat)
        gate_weights = F.softmax(gate_logits, dim=-1)
        top_k_weights, top_k_indices = gate_weights.topk(self.k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Run selected routed experts
        routed_out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.routed_experts):
            mask = (top_k_indices == i).any(dim=-1)
            if not mask.any():
                continue
            expert_mask = (top_k_indices == i)
            weight = (top_k_weights * expert_mask).sum(dim=-1)
            routed_out[mask] += weight[mask].unsqueeze(-1) * expert(x_flat[mask])

        # Combine: shared (always) + routed (selective)
        output = shared_out + routed_out
        return output.view(batch, seq_len, dim)
```

---

## Part 5 — Do Experts Actually Specialize?

**Short answer: yes, they do.** Here's the argument from first principles:

If experts did NOT specialize, we'd be in one of two absurd situations:

**Situation 1:** Any expert is equally good for any input → experts are redundant → we might as well have one expert → we're back to a dense model. But MoE models outperform dense models. Contradiction.

**Situation 2:** For any input, one expert is best, but the assignment is arbitrary → the gating network would need to memorize an arbitrary mapping from every possible input to the best expert → that's an incomprehensible amount of information to store → impossible. Contradiction.

Therefore specialization must exist.

### What Do They Specialize On?

**The confusing finding from the Mixtral paper:** they analyzed expert assignments across different topics (arXiv papers, GitHub code, etc.) and found roughly equal distribution regardless of topic. They concluded: "surprisingly, we do not observe obvious patterns in assignment of experts based on topic."

**But this doesn't mean no specialization.** The statement is "topic is not the right level of abstraction." It doesn't mean there's no pattern.

**Older papers (2017) found clear patterns at the linguistic level:**
- Expert 381: research-style vocabulary
- Expert 752: articles (a, an, the)
- Expert 2004: adjectives and descriptors

Specialization exists. It's just not at the topic level — it's at the syntactic/linguistic/functional level. Modern models are more complex and harder to interpret, so we haven't found the right lens to see it clearly yet.

---

## Part 6 — Parallelization (Brief)

In theory MoE is perfectly parallelizable: every token gets routed to a different expert, experts are independent.

In practice: MoE sits between self-attention and residual connections inside the transformer block. This tight coupling means you can't just parallelize experts in isolation — you need special graph optimization.

Existing tools: **GShard** (Google), **DeepSpeed** (Microsoft). Both require non-trivial setup. Not plug-and-play. A separate topic entirely.

---

## Full Evolution Timeline

```
1991 — Dense MoE
  7 experts, gating network, all experts active
  Goal: better accuracy on vowel classification
  Model: ~hundreds of parameters
       │
       ▼
2017 — Sparse MoE (Google)
  Hundreds/thousands of experts, K=2 active
  Goal: scale parameters without scaling latency
  Model: 137B total, 15M active
  New: noisy top-K gating for load balancing
       │
       ▼
2021 — Switch Transformer (Google)
  1.6 trillion total parameters
  K=1 (single active expert) — unintuitive but works
  MoE replaces FFN inside each transformer layer
       │
       ▼
Jan 2024 — Mixtral (Mistral)
  8 experts, K=2, standard transformer otherwise
       │
Jan 2024 — DeepSeek-MoE (DeepSeek)
  16 experts (finer-grained), K=2
  + Shared expert (always active) → reduces redundancy
       │
       ▼
2025 — Llama 4 (Meta), DeepSeek V3/R1
  Follows DeepSeek recipe (shared expert confirmed in blog post)
  Llama 4 Maverick: 400B total, <5% active
  Llama 4 Behemoth: 2T total
  DeepSeek V3/R1: 671B total, 37B active
```

---

## Summary Table

| Model | Year | Experts | Active (K) | Shared Expert | Total Params |
|---|---|---|---|---|---|
| Original MoE | 1991 | 7 | 7 (dense) | No | ~hundreds |
| Sparse MoE | 2017 | 100s–1000s | 2 | No | 137B |
| Switch Transformer | 2021 | 100s | 1 | No | 1.6T |
| Mixtral | 2024 | 8 | 2 | No | ~47B (active) |
| DeepSeek-MoE | 2024 | 16 (finer) | 2 | ✅ | varies |
| DeepSeek V3/R1 | 2024–25 | 256 | 8 (37B active) | ✅ | 671B |
| Llama 4 Maverick | 2025 | — | <5% | ✅ | 400B |

### Key Concepts

| Term | What It Means |
|---|---|
| Dense MoE | All experts participate in every output |
| Sparse MoE | Only top-K experts activate per token |
| Gating network | Learns which expert to route each token to |
| Load balancing | Ensuring all experts get trained equally (via noise) |
| Shared expert | Always-active expert for common knowledge |
| K | Number of active experts per token |

### The One Key Insight

> MoE lets you increase total knowledge stored (more parameters = more experts) without increasing per-token inference cost (only K activate). The router learns to match the right expert to each token — and experts do specialize, just not at the topic level.
