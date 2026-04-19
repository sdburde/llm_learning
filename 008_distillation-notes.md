# Knowledge Distillation — Complete Notes

> The technique behind Gemma 3, Llama 4 Scout & Maverick, DeepSeek-R1 distilled models.
> Nearly 20 years old, but more relevant than ever.

---

## Why Distillation Matters — The Scaling Problem

Three ways to scale LLMs: **compute**, **data**, **parameters**. All improve model quality.

- **Data** — we've scraped most of the internet. Distillation lets powerful LLMs generate more training data.
- **Parameters** — bigger models are slower at inference. Distillation decouples training from inference: use a giant model during training, deploy a small fast model.

```
Without distillation:
  Train big model → deploy big model → slow inference

With distillation:
  Train big teacher → distill into small student → deploy small student → fast inference
```

Real examples:
- Google: Gemini (proprietary) → Gemma 2 and Gemma 3 (open-source)
- Meta: Llama 4 Behemoth (2T params) → Scout and Maverick
- DeepSeek: R1 (671B MoE) → dense models (1.5B, 7B, 14B, 32B, 70B based on Llama and Qwen)

---

## Origin — Ensembles and Model Compression (2006)

### Why Ensembles?

In 2006, the best-performing models weren't single models — they were **ensembles**: collections of thousands of independently trained models whose predictions were averaged.

```
Spam classifier example:
  Train 1,000 separate classifiers
  For each email: average all 1,000 outputs
  Final verdict: 73% spam (averaged probability)
```

**Why were ensembles needed?** Training was noisy. Models were sensitive to random initialization and data shuffling. Some runs produced great models, others flopped. Averaging 1,000 runs smoothed this out.

**Ensemble vs Mixture of Experts (they're different):**
- **Ensembles:** models trained completely separately, become copies of each other
- **MoE:** experts trained jointly with a gating network, learn to be complementary and specialize

### The 2006 Idea — Model Compression

Problem: you can't ship 1,000 models onto a PDA (personal digital assistant). So researchers proposed:

**Step 1:** Train 1,000 classifiers on hard labels (true data labels). Freeze them.

**Step 2:** Train one small compressed model. But instead of teaching it to match the hard label, average the 1,000 frozen classifiers' outputs and use that average as the training target.

```
Hard label:  spam = 1.0, not_spam = 0.0
Soft label:  spam = 0.73, not_spam = 0.27   ← average of 1,000 classifiers
```

The compressed model learns from the soft label — the averaged wisdom of 1,000 models. Ship just the small model. The 1,000 originals are discarded.

**This is knowledge distillation.** The catchy name didn't come until 9 years later.

---

## Hinton's 2015 Paper — Soft Labels and Dark Knowledge

Hinton and Jeff Dean at Google revisited the 2006 idea for neural networks doing digit classification (MNIST: classify handwritten digits 0–9).

### The Key Discovery

Distillation is useful **even without an ensemble**. Even with a single large teacher model, training a small student on the teacher's soft labels is better than training the student directly on the one-hot hard labels.

```
Hard label for image of "5":
  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]   ← only digit 5 = 1, everything else = 0

Soft label from teacher for same image:
  [0.00, 0.00, 0.01, 0.07, 0.00, 0.85, 0.00, 0.00, 0.06, 0.01]
                           ↑                    ↑              ↑
                         "3"               "5" (correct)   "8"
```

**Why is the soft label better?** If the image is a messy 5 that looks a bit like a 3, the hard label says nothing about that. The soft label captures it — "3" gets a non-trivial probability. This extra information is what Hinton calls **dark knowledge**: knowledge about what the input is NOT, but might look like.

The teacher does the hard work of surfacing dark knowledge. The student gets it handed explicitly. With more signal, the student learns more with fewer parameters.

**Teacher-Student terminology** was coined in this paper and stuck.

### The Butterfly Analogy

Hinton's analogy for training vs inference: the butterfly. The larva and adult are completely different forms because the butterfly's needs change. A model needs a large parameter space during training to learn well, but at inference time it needs to be small, fast, and efficient.

---

## Temperature — Where the Term Actually Comes From

You use temperature as a "creativity knob" today. High temp = more surprising outputs. But temperature was **originally introduced for distillation**.

### The Problem It Solves

Soft labels sometimes have extremely small probabilities for wrong classes (e.g., 10⁻⁸). Gradients from probabilities this tiny are essentially zero — the student can't learn anything from them. Dark knowledge is invisible.

### The Fix

Divide the teacher's logits by a temperature T before applying softmax:

```
Standard softmax:
  p(i) = exp(logit_i) / Σ exp(logit_j)

Temperature-scaled softmax:
  p(i) = exp(logit_i / T) / Σ exp(logit_j / T)
```

- **T = 1** → standard distribution
- **T > 1** → softer, more uniform distribution → small probabilities get boosted → student sees more dark knowledge
- **T < 1** → sharper, more peaked distribution → standard behavior

The same temperature T is applied to both teacher (for soft labels) and student (during distillation training). After distillation, the student is deployed with T=1.

```python
import torch
import torch.nn.functional as F

def temperature_softmax(logits: torch.Tensor, temperature: float = 1.0):
    """
    Standard softmax when T=1.
    Softer distribution when T>1 — reveals dark knowledge.
    Sharper distribution when T<1 — more confident predictions.
    """
    return F.softmax(logits / temperature, dim=-1)

# Example: teacher logits for digit classification
logits = torch.tensor([−5.0, −3.0, −1.0, 2.0, −4.0,
                        8.0, −2.0, −1.5, 1.5, −3.5])

print("T=1 (standard):", temperature_softmax(logits, T=1.0).round(decimals=3))
print("T=4 (soft):    ", temperature_softmax(logits, T=4.0).round(decimals=3))
print("T=0.5 (sharp): ", temperature_softmax(logits, T=0.5).round(decimals=3))
# T=1:  [0.000, 0.000, 0.001, 0.030, 0.000, 0.951, 0.000, 0.001, 0.017, 0.000]
# T=4:  [0.012, 0.019, 0.038, 0.099, 0.015, 0.601, 0.027, 0.033, 0.092, 0.018]
# T=0.5:[0.000, 0.000, 0.000, 0.001, 0.000, 0.999, 0.000, 0.000, 0.000, 0.000]
# At T=4, wrong classes become visible — this is what the student learns from
```

---

## Distillation in Modern LLMs

### When to Apply Distillation

Three options in the LLM training pipeline:

```
Pre-training      → train on massive internet data, next word prediction
Mid/Post-training → fine-tune on instruction data, reasoning, preferences

Option 1: Distill during pre-training   (Llama 4)
Option 2: Distill during post-training  (DeepSeek)
Option 3: Distill during both stages    (Gemma 3)
```

---

## Proper Distillation vs Behavioral Cloning

This is the most important distinction in the video. Not all distillation is equal.

### Proper Distillation (Google, Meta)

At each training step, both teacher and student see the same input. Both produce full probability distributions over the vocabulary. The student is trained to minimize the gap between its distribution and the teacher's.

```
Document: "The cat sat on the ___"
                                 ↑ predicting "mat"

Teacher output: [mat: 0.71, rug: 0.12, floor: 0.08, bed: 0.04, ...]  ← SOFT LABEL
Student output: [mat: 0.40, rug: 0.05, floor: 0.20, bed: 0.15, ...]

Loss = KL divergence between student and teacher distributions
     + optionally cross-entropy with hard label

Student weights update to make its distribution closer to teacher's
```

**Dark knowledge is transferred.** The student learns not just "mat is correct" but "rug is a plausible second choice, floor is possible too."

### Behavioral Cloning (DeepSeek's Approach)

DeepSeek asks the teacher model to generate a dataset of outputs, then trains the student on those outputs using standard next-word prediction with hard labels.

```
Teacher R1 generates:
  Q: "What is 3+5?"
  A: "The answer is 8."   ← this becomes a data point

Student (Llama 7B) trains on:
  Input:  "What is 3+5?"
  Target: "The answer is 8."   ← ONE-HOT label, not teacher's distribution
```

The student learns to produce similar outputs to the teacher. But the teacher's soft probability distribution — the dark knowledge — is never seen by the student. The student imitates behavior, not internal reasoning.

**Why call it behavioral cloning instead of distillation?**
- No soft labels
- No teacher probability distributions
- No temperature
- No dark knowledge transfer
- Just: generate data with teacher, train student on that data

**Why does DeepSeek use behavioral cloning instead of proper distillation?**

Two reasons:

1. **White-box requirement:** Proper distillation requires access to the teacher's internal probability distributions. If you wanted to distill GPT-4, OpenAI would need to give you access to its logits — which they don't. Behavioral cloning only needs the teacher's outputs.

2. **Computational cost:** Proper distillation requires running the teacher model for every single token in the training dataset to get soft labels. This is extremely expensive at scale.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Proper Distillation Loss ──────────────────────────────────────────────────

def distillation_loss(student_logits: torch.Tensor,
                      teacher_logits: torch.Tensor,
                      hard_labels: torch.Tensor,
                      temperature: float = 4.0,
                      alpha: float = 0.7):
    """
    Combines soft label loss (from teacher) and hard label loss (from data).

    student_logits: [batch, vocab_size]
    teacher_logits: [batch, vocab_size]
    hard_labels:    [batch] — integer class indices
    temperature:    T > 1 amplifies dark knowledge
    alpha:          weight on soft label loss (1-alpha on hard label loss)
    """
    # Soft label loss — KL divergence between student and teacher
    # T applied to both so gradients are comparable
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    # KL(teacher || student) — minimize student's distance from teacher
    soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
    # Scale by T² to normalize gradients (standard practice from Hinton paper)
    soft_loss = soft_loss * (temperature ** 2)

    # Hard label loss — standard cross-entropy with ground truth
    hard_loss = F.cross_entropy(student_logits, hard_labels)

    # Combined loss
    total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
    return total_loss, soft_loss.item(), hard_loss.item()


# ── Behavioral Cloning Loss (DeepSeek style) ─────────────────────────────────

def behavioral_cloning_loss(student_logits: torch.Tensor,
                             teacher_generated_tokens: torch.Tensor):
    """
    Standard next-token prediction on teacher-generated data.
    No soft labels. No temperature. Just imitation.

    student_logits:          [batch, seq_len, vocab_size]
    teacher_generated_tokens: [batch, seq_len] — one-hot targets
    """
    batch, seq_len, vocab_size = student_logits.shape
    # Flatten and compute standard cross-entropy
    loss = F.cross_entropy(
        student_logits.view(-1, vocab_size),
        teacher_generated_tokens.view(-1)
    )
    return loss


# ── Simple Training Example ───────────────────────────────────────────────────

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.head(self.transformer(self.embed(x)))


def train_with_distillation(teacher, student, dataloader,
                             temperature=4.0, alpha=0.7, epochs=5):
    """
    Proper distillation training loop.
    Teacher is frozen. Student learns from teacher's soft labels.
    """
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student.train()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_tokens, hard_labels in dataloader:
            # Forward pass through both models
            with torch.no_grad():
                teacher_logits = teacher(batch_tokens)     # soft labels source

            student_logits = student(batch_tokens)

            # Distillation loss: soft (teacher KL) + hard (ground truth CE)
            loss, soft_l, hard_l = distillation_loss(
                student_logits[:, -1, :],   # predict last token
                teacher_logits[:, -1, :],
                hard_labels,
                temperature=temperature,
                alpha=alpha
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f}")
```

---

## Computational Cost of Proper Distillation

**Back of the napkin math:**

```
Training set: 8 trillion tokens
Vocabulary:   128,000 tokens

Soft labels needed: 8T × 128K = ~10¹⁸ values

Storage at FP8: equivalent to the entire contents of YouTube
```

That's too much to store. In practice, Gemma uses a trick: instead of the full distribution, sample **256 tokens** from the teacher's distribution per training example. Zero out everything else.

```
Full distribution:     128,000 values per token
Gemma's approach:      256 values per token  (top-256 or sampled)
Remaining storage:     ~10¹⁵ values — still massive, but manageable

Practical approach: compute soft labels on the fly, use immediately, discard
```

Even with sampling, running 8 trillion teacher forward passes is not trivial. This is why proper distillation is expensive and why behavioral cloning remains popular despite being less information-rich.

```python
def sparse_soft_labels(teacher_logits: torch.Tensor,
                        n_samples: int = 256,
                        temperature: float = 4.0):
    """
    Instead of storing full 128K-dim soft label distribution,
    sample only 256 tokens from teacher's distribution.
    Zero out the rest. Used by Gemma to reduce storage.
    """
    probs = F.softmax(teacher_logits / temperature, dim=-1)   # [batch, vocab]

    # Sample 256 token indices from the teacher distribution
    sampled_indices = torch.multinomial(probs, num_samples=n_samples)  # [batch, 256]

    # Sparse soft label: only 256 non-zero entries per example
    sparse_labels = torch.zeros_like(probs)
    sparse_probs  = probs.gather(1, sampled_indices)           # [batch, 256]
    sparse_labels.scatter_(1, sampled_indices, sparse_probs)

    # Renormalize to sum to 1
    sparse_labels = sparse_labels / sparse_labels.sum(dim=-1, keepdim=True)

    return sparse_labels   # [batch, vocab], 256 non-zero values per row
```

---

## Co-Distillation (Meta's Trick)

### The Standard Distillation Cost Problem

Standard distillation: **two forward passes per training example**.
1. Teacher forward pass (during teacher training — to produce teacher output)
2. Another teacher forward pass (during student training — to produce soft labels)

That doubles the compute. At 8 trillion tokens, this is brutal.

### Co-Distillation Solution

Train teacher and student **simultaneously**. When the teacher computes its output during its own training step, that output is immediately available as a soft label for the student.

```
Standard distillation:
  Phase 1: Train teacher on data → save teacher
  Phase 2: For each example, run frozen teacher → get soft label → train student
  Cost:    2N teacher forward passes

Co-distillation:
  One phase: Teacher trains on hard labels simultaneously with student training
             Teacher's output during its own training step = student's soft label
  Cost:    N teacher forward passes (saves the second N passes)
```

**The catch:** The teacher isn't fully trained yet when the student is learning from it. Early in training, the teacher's soft labels may be wrong — the blind leading the blind.

**The fix:** Train the student on a combination of soft labels (from the still-learning teacher) and hard labels (from the ground truth data). As training progresses, the teacher gets better and its soft labels become more reliable.

```python
def co_distillation_step(teacher, student, batch_tokens, hard_labels,
                          teacher_optimizer, student_optimizer,
                          temperature=4.0, alpha_schedule=None, step=0):
    """
    Co-distillation: teacher and student train simultaneously.
    Teacher's live output is immediately used as student's soft label.
    alpha_schedule: reduces soft label weight early in training
                    when teacher is still unreliable.
    """
    # Teacher forward pass (teacher is still learning)
    teacher_logits = teacher(batch_tokens)
    teacher_loss   = F.cross_entropy(teacher_logits[:, -1, :], hard_labels)

    # Update teacher
    teacher_optimizer.zero_grad()
    teacher_loss.backward(retain_graph=True)   # retain for soft labels below
    teacher_optimizer.step()

    # Soft label weight decreases early (teacher unreliable) → increases later
    alpha = alpha_schedule(step) if alpha_schedule else 0.7

    # Student forward pass — uses teacher's current (imperfect) output
    with torch.no_grad():
        teacher_soft = teacher_logits.detach()   # use as soft label immediately

    student_logits = student(batch_tokens)

    # Combined loss: soft (teacher) + hard (ground truth)
    loss, _, _ = distillation_loss(
        student_logits[:, -1, :],
        teacher_soft[:, -1, :],
        hard_labels,
        temperature=temperature,
        alpha=alpha
    )

    student_optimizer.zero_grad()
    loss.backward()
    student_optimizer.step()

    return teacher_loss.item(), loss.item()
```

---

## Summary — All Three Approaches Compared

| Approach | Who Uses It | Soft Labels? | Teacher Access | Cost |
|---|---|---|---|---|
| Proper distillation | Google (Gemma), Meta (Llama 4) | ✅ Full distribution | Need white-box teacher | Very high |
| Behavioral cloning | DeepSeek (R1 distills) | ❌ One-hot only | Only need outputs | Low |
| Co-distillation | Meta (Llama 4 hint) | ✅ Live from teacher | Need white-box teacher | Moderate |

### What Each Term Actually Means

| Term | Original Meaning | Common Misuse |
|---|---|---|
| Distillation | Transfer soft probability distributions from teacher to student | Any model trained on another model's outputs |
| Temperature | Softens teacher's distribution to amplify dark knowledge | Just "creativity/randomness knob" |
| Soft label | Full probability distribution from teacher | — |
| Dark knowledge | Information about what the input is NOT (non-zero probs on wrong classes) | — |
| Behavioral cloning | Imitating teacher's output behavior without soft labels | Called "distillation" by DeepSeek |
| Co-distillation | Training teacher and student simultaneously to save compute | — |

### The One Key Insight

> Soft labels carry strictly more information than hard labels. Hard labels say "this is the answer." Soft labels say "this is the answer, but here's how similar it looks to every other possible answer." That extra signal — dark knowledge — lets students learn more with fewer parameters.

---

> Distillation is nearly 20 years old. The implementation details — soft vs hard labels, temperature, when in the pipeline, white-box vs black-box teacher — are what separate effective distillation from just generating synthetic data.
