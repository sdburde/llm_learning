# DeepSeek — Complete Progression Notes

---

## Who Is DeepSeek?

DeepSeek is a Chinese AI company founded in July 2023 in Hangzhou, backed by the hedge fund High-Flyer. Founded by Liang Wenfeng (also co-founder of High-Flyer). The company's goal from the start was to build frontier AI models, not tied to any financial product. They recruit heavily from top Chinese universities and people outside traditional CS fields.

What makes them notable: they consistently match or beat Western frontier models at a fraction of the training cost, and they open-source almost everything under the MIT License.

---

## Background Concepts (Read First)

Before getting into each model, it helps to understand a few recurring terms.

### Dense vs Mixture-of-Experts (MoE)

A **dense model** activates all its parameters for every token it processes. A 67B dense model uses all 67 billion parameters every time.

A **Mixture-of-Experts (MoE)** model has many "experts" (sub-networks) but only activates a small subset of them per token. So a 671B MoE model might only activate 37B parameters per token. This means:
- Total parameters = massive (all knowledge stored)
- Active parameters per token = small (cheap per inference)
- You get the capacity of a huge model at the compute cost of a small one

### KV Cache Problem

When a model generates text, it needs to remember the keys and values from all previous tokens in the conversation. This is called the **KV cache**. For long conversations, this cache gets enormous — standard attention on a 128K context window would need ~488GB of memory just for the cache. This is a huge bottleneck.

### Reinforcement Learning (RL) in LLMs

Standard training: show the model examples, tell it the right answer, it learns by copying.
RL training: the model tries things, gets a reward for correct answers, and learns by trial and error. No need for labeled examples — just a scoring function.

---

## Model-by-Model Progression

---

### DeepSeek Coder — November 2023

**What it is:** DeepSeek's very first public release. A code-focused model trained on data from 80+ programming languages.

**Why it matters:** Proof that a Chinese lab could build a competitive coding model from scratch. It showed the team could train specialist models well.

**Size:** Multiple variants, focused on code completion and generation.

---

### DeepSeek LLM (V1) — November 2023

**What it is:** DeepSeek's first general-purpose language model. Two sizes: 7B and 67B parameters. Both come in Base and Chat versions.

**Architecture:** Standard dense transformer, very similar to LLaMA. Uses:
- RMSNorm for normalization
- SwiGLU in feedforward layers
- Rotary Position Embeddings (RoPE)
- 7B model uses Multi-Head Attention (MHA)
- 67B model uses Grouped-Query Attention (GQA) for efficiency

**Training data:** 2 trillion tokens in English and Chinese. Context length: 4096 tokens.

**What GQA does:** In standard Multi-Head Attention, every head has its own key and value matrices. This costs a lot of memory. GQA groups multiple query heads to share a single key-value head, cutting memory without much performance loss. DeepSeek uses this on the bigger 67B model where memory matters more.

**Performance:** The 67B model outperforms LLaMA-2 70B on reasoning, coding, math, and Chinese language tasks. Beats GPT-3.5 in Chinese.

**What changed from Llama:** Custom 102k vocabulary tokenizer (Llama uses ~32k). This larger vocab helps with Chinese since Chinese characters are more efficiently encoded. Also uses a multi-step learning rate schedule instead of cosine decay.

---

### DeepSeek MoE — January 2024

**What it is:** DeepSeek's first experiment with Mixture-of-Experts architecture. 16B total parameters, 2.7B activated per token. Still 4K context.

**The core MoE idea they introduced:** Standard MoE has N large experts. DeepSeek proposed using mN smaller, finer-grained experts instead — each expert specializes more narrowly, so knowledge is spread more cleanly. They also added **shared experts** — a set of experts that are always active for every token, handling common/core knowledge. Routed experts handle specialized knowledge. This prevents the shared experts from being wasteful and lets routed experts actually specialize.

**Why this matters:** They showed that a 16B MoE model performs comparably to their 7B dense model — but the MoE model is far cheaper to run because only 2.7B params are active per token. This validated MoE as the path forward.

**Expert balancing problem:** In normal MoE, some experts get overused while others rarely activate — wasting capacity. The shared-expert design helps route common patterns through shared experts and lets routed experts handle rarer, specific tasks.

---

### DeepSeek Math — April 2024

**What it is:** A math-specialized 7B model. Built by starting from DeepSeek Coder Base v1.5 and further pre-training on 500B math-heavy tokens.

**Why it matters:** This is where **GRPO was first introduced**. They used a process reward model (PRM) trained on step-by-step math solutions, then applied Group Relative Policy Optimization to train the model to get better at math via RL.

**Result:** 51.7% on MATH benchmarks — close to GPT-4 and Gemini Ultra performance at the time, in a 7B model.

**What GRPO is (first use):** Instead of comparing the model's answers to a fixed baseline, GRPO generates a group of answers for each question and scores them relative to each other. Better-than-average answers get reinforced, worse-than-average answers get penalized. No critic network needed. This was a proof of concept that GRPO works for math — and they'd scale it massively later in R1.

---

### DeepSeek V2 — May 2024

**What it is:** Major architectural leap. 236B total parameters, 21B activated per token. Context window: 128K tokens (extended from 4K using YaRN). Trained on 8.1T tokens.

**Two big inventions introduced here:**

#### 1. Multi-Head Latent Attention (MLA)

This is DeepSeek's most important architectural contribution.

**The problem it solves:** Standard attention stores full key and value vectors for every token in the KV cache. For DeepSeek-V3 at 128K context, standard attention would need ~488GB just for the cache. That's impractical.

**How MLA works:**
- Instead of storing separate full-dimension key and value vectors for every token, MLA compresses them into a single low-dimensional **latent vector**
- Compression ratio: full dimension → ~1/4 the size
- At inference time, the model reads this small latent vector and decompresses it back into keys and values only when needed
- The latent vector is much smaller: 512 dimensions vs what would be tens of thousands of floats

**Result:** KV cache reduced by **93.3%** compared to standard attention. But here's the surprising part — MLA actually slightly improves model quality too, not just efficiency. The low-rank factorization acts like a regularizer.

**Decoupled RoPE:** RoPE (Rotary Position Embeddings) encodes where each token is in the sequence. In MLA, the keys and values are compressed, but position information needs to be preserved separately. So DeepSeek uses a dedicated RoPE portion of the key vector that's not compressed — it carries positional info and is added back at attention time. This is called decoupled RoPE.

#### 2. DeepSeekMoE (refined)

Continued the MoE architecture from the January 2024 experiment but at scale. 160 routed experts + shared experts. The routing and specialization principles stay the same, just bigger.

**Numbers vs DeepSeek V1 67B:**
- 42.5% reduction in training costs
- 93.3% reduction in KV cache size
- 5.76× faster generation throughput
- Better performance on benchmarks despite being cheaper to run

---

### DeepSeek Coder V2 — June 2024

**What it is:** MoE coding model, 236B total / 21B active. Context: 128K. Trained on 6T tokens including massive code data. Supports 338 programming languages.

**Why it matters:** First open-source model to beat GPT-4 Turbo on coding benchmarks. Showed MoE + massive code pretraining = frontier coding performance at fraction of the cost.

---

### DeepSeek V2.5 — September 2024

**What it is:** Merged the best of DeepSeek-V2-0628 and DeepSeek-Coder-V2-0724 into one model. 238B parameters.

**Purpose:** A combined general + coding model. Topped open-source leaderboards at launch. Simpler to use than maintaining two separate models.

---

### DeepSeek V3 — December 2024

**What it is:** The big one before R1. 671B total parameters, 37B active per token. 256 experts total. Context: 128K. Trained on 14.8 trillion tokens. Cost: ~$5.6M (2.788M H800 GPU hours) — vs GPT-4's estimated $50–100M.

**Architecture carries forward:** MLA + DeepSeekMoE from V2. But adds two new innovations:

#### 1. Auxiliary-Loss-Free Load Balancing

**The problem:** In MoE, you want all experts to get roughly equal usage, otherwise some are wasted and some are overloaded. Previous approaches added an "auxiliary loss" to penalize imbalanced routing during training. But this auxiliary loss creates a tension — the model is optimizing two things at once (main task + balancing), and the balance objective slightly hurts performance.

**DeepSeek's solution:** Remove the auxiliary loss entirely. Instead, use a **bias term** on each expert. If an expert is getting too much traffic, its bias goes slightly negative, making the router less likely to send tokens to it. If an expert is underused, its bias goes slightly positive, attracting more tokens. This is a soft, dynamic correction that doesn't mess with the main training objective.

**Result:** Better performance than auxiliary-loss methods, with natural load balance.

#### 2. Multi-Token Prediction (MTP)

**Standard training:** At each position, the model predicts the next 1 token. Loss computed, weights updated.

**MTP:** At each position, the model also predicts the token 2 steps ahead (or more). This gives a denser training signal — every token position contributes to learning 2 future tokens instead of 1. Forces the model to "think ahead" and understand longer-range dependencies.

**Important:** The MTP module is only used during training. At inference time it's dropped — so inference cost stays the same. However, the second-token predictions can optionally be used for **speculative decoding** to speed up generation (85–90% acceptance rate on the second predicted token).

**Training efficiency:** FP8 mixed precision training — stores weights in 8-bit floating point instead of 16-bit. First time anyone had successfully validated FP8 training at this scale. Cuts memory and communication costs significantly.

**Performance:** Outperforms all other open-source models at time of release. Matches closed-source frontier models (GPT-4 class) on most benchmarks.

---

### DeepSeek R1-Zero — January 2025 (Released alongside R1)

**What it is:** An experiment, not a product. The base DeepSeek-V3 model trained with pure RL — no supervised fine-tuning at all, no human-labeled reasoning examples. Just: here are math/coding/logic problems, here's a reward for correct answers, figure it out.

**RL algorithm used:** GRPO (same one from DeepSeek-Math, now at massive scale).

**Reward function:** Two simple rules:
1. Is the final answer correct? (checked automatically against ground truth)
2. Is the answer formatted correctly inside `<think>` tags?

No neural reward models. No human preference data. Just those two checks.

**What emerged spontaneously:**
- The model started writing long step-by-step reasoning before answering
- It started re-reading its own reasoning and catching mistakes (self-verification)
- It started backtracking: "wait, I made an error in step 3, let me redo that"
- It started generating longer and longer responses as training progressed — not because it was told to, but because longer thinking correlated with correct answers

This is called the **"Aha moment"** — nobody programmed these behaviors. They emerged because RL rewarded correct answers, and the model discovered that thinking carefully was the best strategy for getting rewards.

**AIME 2024 result:** Base model: 15.6% → R1-Zero after RL: 71% → with majority voting: 86.7% (matches OpenAI o1).

**Problem:** Outputs were messy. Mixed languages. Hard to read. The model would think in a jumbled way that was hard for humans to follow. Good reasoning, bad readability.

---

### DeepSeek R1 — January 2025

**What it is:** The production reasoning model. Same 671B / 37B MoE architecture as V3. Fixes the readability problems of R1-Zero via a smarter training pipeline. Released with a chatbot app that hit #1 on the iOS App Store, surpassing ChatGPT.

**Training pipeline — 4 stages:**

**Stage 1 — Cold Start SFT**
Before any RL, the model is fine-tuned on a small set of thousands of high-quality, human-readable Chain-of-Thought examples. These examples show the model what good structured reasoning looks like — clear steps, proper language, no mixing. This gives the model a baseline of readability before RL begins. Without this, RL tends to produce messy outputs like R1-Zero.

**Stage 2 — Reasoning-Focused RL (GRPO)**
Apply GRPO on math, coding, and logic tasks. Same reward: correct answer + proper format. The model improves its reasoning ability substantially in this stage. GRPO works by generating 16 answers per question and comparing them to each other. No critic model needed — just group-relative scoring. A clipping function prevents the model from changing too drastically in one step, keeping training stable.

**Stage 3 — Rejection Sampling + General SFT**
Use the partially trained model from Stage 2 to generate many outputs for a wide range of tasks. Keep only the best outputs (rejection sampling). Then mix these with general-capability data from DeepSeek-V3 — things like writing, Q&A, factual recall, role-play. Fine-tune on this mixed dataset. This stage ensures the model is capable of more than just math and coding.

**Stage 4 — Final RL (Helpfulness + Safety)**
Second RL stage that covers both reasoning tasks and general tasks. Rewards are both rule-based (for verifiable tasks) and model-based (for subjective tasks like writing quality). This stage aligns the model with human preferences and safety requirements.

**Chain-of-Thought details:**
The model writes its reasoning inside `<think>...</think>` tags. This is visible to users. The final answer comes after the think block. The think block can be very long — the model can backtrack, verify, explore alternatives. Longer thinking generally = more accurate answers. This is "inference-time scaling" — spending more compute at answer time rather than training time.

**GRPO in detail:**
- For each question, generate 16 answers
- Score each answer: +1 for correct/well-formatted, 0 or negative otherwise
- Compute each answer's advantage = (its score - group average score)
- Update the model weights to increase probability of above-average answers
- Clip the update to prevent instability
- No separate critic model (unlike PPO, which needs one of the same size as the main model)
- Result: same RL quality at roughly half the memory cost of PPO

**Performance:** Matches OpenAI o1 on math and coding benchmarks. Beats its predecessor DeepSeek-V3 by large margins on reasoning tasks.

**Distilled models released alongside R1:**
DeepSeek generated 800K high-quality reasoning outputs from R1, then used those to fine-tune smaller models (Qwen-based and Llama-based) at sizes 1.5B, 7B, 8B, 14B, 32B, 70B. Key finding: distilling from R1 beats training a small model with RL from scratch. The 14B distilled model beats QwQ-32B-Preview on reasoning benchmarks. Even the 7B model scores 55.5% on AIME 2024 — far beyond what a 7B model trained normally would get.

**Impact:** R1's release caused an 18% drop in Nvidia's share price in a single day (January 27, 2025). Investors questioned whether Western AI companies' massive GPU investments were justified if DeepSeek could match them for $5.6M.

---

### DeepSeek R1-0528 — May 2025

**What it is:** An updated version of R1 with practical improvements.

**What changed:**
- Added support for system prompts (R1 originally had none)
- Added JSON output mode
- Added function calling / tool use

**Why this matters:** R1 was initially a pure reasoning model, hard to integrate into production apps. R1-0528 made it usable for agentic workflows where the model needs to call tools, follow system instructions, or return structured data.

---

### DeepSeek V3.1 — August 2025

**What it is:** 840B parameter base. Major upgrade focused on agentic and long-context tasks.

**Key new feature — Hybrid Thinking Mode:**
A single model that can operate in two modes:
- **Thinking mode:** like R1, writes long reasoning chains before answering (better for complex tasks)
- **Non-thinking mode:** like V3, answers directly without the reasoning overhead (better for simple/fast tasks)

Previously you had to use two different models (V3 for chat, R1 for reasoning). V3.1 combines both in one model. You toggle the mode with a parameter.

**Other improvements:**
- Significantly better tool calling and agent capabilities
- Better instruction following in multi-step workflows
- Surpasses V3 and R1 by over 40% on agent benchmarks like SWE-bench and Terminal-bench

Updated to V3.1-Terminus on September 22, 2025 with refinements.

---

### DeepSeek V3.2 — December 2025

**What it is:** Current flagship. Uses a new attention mechanism: **DeepSeek Sparse Attention**.

**Sparse Attention:** Standard attention (even MLA) looks at all tokens in context when computing attention. For 128K context, that's a lot. Sparse attention only attends to a selected subset of tokens — recent tokens plus tokens deemed important by a learned pattern. This makes long-context inference much faster and cheaper without much quality loss.

**V3.2-Speciale:** A variant of V3.2 with extra emphasis on reasoning, incorporating techniques from DeepSeekMath-V2 (a concurrent model focused on formal math proofs). Designed to push open-source reasoning to its limits.

**Status:** Powers both the `deepseek-chat` and `deepseek-reasoner` API endpoints. Current production model.

---

## Specialist Model Lines (Parallel to Main V-Series)

### DeepSeek-VL (Vision-Language)

Vision models for understanding images + text. DeepSeek-VL2 supports OCR, chart reading, document understanding. Separate from the main LLM line.

### Janus Series

Unified multimodal models that do both image understanding AND image generation in one model. Janus-Pro-7B is the current version — 7B parameters, supports text-to-image and image understanding.

### DeepSeek Prover

Formal theorem proving using Lean 4. Proves mathematical theorems by decomposing them into subgoals and generating verified proofs. DeepSeek-Prover-V2 (671B) achieves gold-level scores in math competitions.

---

## Architecture Evolution Summary

| Model | Params (Total) | Active/Token | Context | Key Innovation |
|---|---|---|---|---|
| LLM V1 | 67B | 67B (dense) | 4K | First model, GQA on large variant |
| MoE | 16B | 2.7B | 4K | Shared + routed expert design |
| V2 | 236B | 21B | 128K | MLA (93% KV cache reduction), DeepSeekMoE |
| V3 | 671B | 37B | 128K | Aux-loss-free balancing, MTP, FP8 training |
| R1 | 671B | 37B | 128K | GRPO at scale, multi-stage RL, cold-start SFT |
| V3.1 | 840B | ~37B | 128K | Hybrid thinking/non-thinking mode |
| V3.2 | ~840B | ~37B | 128K | Sparse Attention for long context |

---

## Core Technical Ideas, Simply Explained

### MoE — Why Not Just Use a Big Dense Model?
A 671B dense model activating all params per token is 18× more expensive per inference than a 671B MoE activating 37B params. MoE lets you store more knowledge (bigger total params) while keeping per-token compute cheap.

### MLA — Why Does KV Cache Matter?
Every token you've generated needs its keys and values remembered for future attention. Long conversations = huge cache. MLA stores a compressed version instead, 93% smaller, then decompresses on the fly. Makes 128K context practical on real hardware.

### GRPO — Why Not Just Use Human Feedback?
Human labeling is slow and expensive. GRPO doesn't need humans — it just needs a function that says "was this answer correct?" For math and code, that's easy to check automatically. Generate multiple answers, compare them to each other, push the model toward the better ones.

### Distillation — Why Not Just Use RL on Small Models?
When DeepSeek tried applying GRPO directly to a 32B model from scratch, it plateaued much lower than expected. When they instead fine-tuned the 32B model on 800K outputs from R1, it massively outperformed the RL-trained version. The reasoning patterns that emerge from large-scale RL on a giant model cannot be independently rediscovered by a small model. They must be transferred.

---

## Timeline

```
Jul 2023   DeepSeek founded
Nov 2023   DeepSeek Coder + DeepSeek LLM (V1) — 7B and 67B dense models
Jan 2024   DeepSeek MoE — first MoE experiment, shared+routed experts
Apr 2024   DeepSeek Math — GRPO introduced for first time
May 2024   DeepSeek V2 — MLA invented, 236B MoE, 128K context
Jun 2024   DeepSeek Coder V2 — beats GPT-4 Turbo on coding
Sep 2024   DeepSeek V2.5 — merged chat + coding model
Nov 2024   R1-Lite preview — first glimpse of reasoning model
Dec 2024   DeepSeek V3 — 671B MoE, MTP, aux-loss-free balancing, $5.6M training
Jan 2025   DeepSeek R1 + R1-Zero — reasoning model, GRPO at scale, #1 iOS app
Mar 2025   DeepSeek V3-0324 — minor V3 update, MIT License
May 2025   DeepSeek R1-0528 — R1 with tool use, system prompts, JSON mode
Aug 2025   DeepSeek V3.1 — hybrid thinking mode, 840B, agent-optimized
Dec 2025   DeepSeek V3.2 — sparse attention, current flagship
```

---

> All major models since R1 are open-weight under MIT License. Training data is not shared.
