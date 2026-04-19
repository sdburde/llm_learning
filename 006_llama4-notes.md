# Llama 4 — Architecture Deep Dive Notes

> Reverse-engineered from Meta's official blog post + referenced papers.
> No technical report yet — some details are best guesses based on prior Meta research.

**Key Papers:**
- Early Fusion (Chameleon): https://arxiv.org/pdf/2405.09818
- Ring Attention: https://arxiv.org/pdf/2310.01889
- RoPE: https://arxiv.org/pdf/2104.09864
- Position Interpolation: https://arxiv.org/pdf/2306.15595
- NoPE: https://arxiv.org/pdf/2305.19466

---

## The Three Models

| Model | Total Params | Active Params/Token | Context Window | Positioned Against |
|---|---|---|---|---|
| **Behemoth** | 2T | ~5% | — | Claude Sonnet 3.7, Gemini 2.0 Pro, GPT-4.5 |
| **Maverick** | 400B | <5% | — | Mid-tier production use |
| **Scout** | 109B | — | **10 million tokens** | Local / consumer use (needs H100) |

All three use **Mixture-of-Experts (MoE)** — a large total parameter count but only a small subset is activated per query.

### Why MoE?

Without MoE, a 400B parameter model would activate all 400B parameters for every single token — making inference extremely slow and memory-intensive.

With MoE:
- The network has many "experts" (sub-networks)
- A **router** decides which experts are relevant for each token
- Only selected experts activate — less than 5% of parameters for Maverick
- Not all knowledge is useful all the time: an English query doesn't need the Chinese language experts

MoE lets you store more total knowledge while keeping per-token compute cheap. DeepSeek used the same idea in V3 and R1.

---

## Part 1 — Native Multimodality

### How Llama 3 Did It (The Old Way — "Bolt-On" Multimodality)

Llama 3 added vision as an afterthought:

```
Step 1: Train LLM on text only → freeze it completely
Step 2: Train image encoder separately on image-caption pairs
         (similar to CLIP — maps image → continuous embedding)
Step 3: Train a vision adapter to bridge the two spaces
         → adapter learns to align image embeddings with LLM embeddings
Step 4: Decide where inside the LLM to inject visual embeddings
         (non-trivial design decision)
Step 5: Repeat for each new modality (video, speech = separate extensions)
```

Problems with this approach:
- LLM was never trained on images — the two embedding spaces don't naturally align
- Vision adapter is a patch, not a real solution
- Every new modality needs its own separate extension pipeline
- Can't train all modalities jointly — each is trained in isolation
- Very different from how humans perceive the world (we take in everything at once)

### How Llama 4 Does It — Early Fusion

Based on Meta's **Chameleon** paper (May 2024). Llama 4 is likely heavily based on it.

The key idea: **tokenize all modalities into a unified token ID space** before the model sees anything.

#### Text Tokenization (Standard)
```
Input: "What animal is this?"
→ Byte Pair Encoding (BPE) tokenizer
→ Token IDs: [482, 7922, 318, 428, 30]
```
Nothing new here.

#### Image Tokenization (The Interesting Part)

```
Input image
→ Split into 16×16 pixel patches
→ Pass each patch through a pre-trained image encoder (Meta's MetaCLIP)
→ Get one continuous embedding per patch

But now instead of keeping them continuous:
→ Look up each embedding in a CODEBOOK
→ Codebook = a discrete vocabulary for images
           = a set of "puzzle pieces" stored as continuous embeddings
           = each piece has a token ID
→ Pick the codebook entry with smallest distance to patch embedding
→ Get a token ID for each patch
```

**The codebook is what makes it work.** It converts continuous image embeddings into discrete token IDs — the same format as text tokens.

```
Text: "cat" → token ID 482
Image patch (cat's ear) → nearest codebook entry → token ID 8371
```

Both are just integers now. You can mix them freely:

```
Unified token sequence:
[482, 7922, 318, 8371, 9204, 7731, 428, 30]
 ^text^text^text ^img  ^img  ^img  ^text^text
```

This flat list goes into the transformer as a single sequence. The model doesn't need to know which are image tokens and which are text — they're all just token IDs.

#### Why This Is Better (Early Fusion Benefits)

| Llama 3 (Late Fusion) | Llama 4 (Early Fusion) |
|---|---|
| Separate encoder per modality | One unified token space |
| Adapter needed to align spaces | No adapter needed |
| LLM frozen during multimodal training | LLM trained jointly on all modalities |
| Text and image trained separately | Text + image + video + audio together |
| Sequential pipeline, brittle | Single end-to-end model |

**Richer training data:** With early fusion you can feed a document that interleaves text and images and the model sees both in the correct order — preserving the relationship between a figure and its caption, for example. With Llama 3, text and images had to be separated before training.

### Code: Image Tokenization via Codebook Lookup

```python
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# ── Codebook ──────────────────────────────────────────────────────────────────

class ImageCodebook(torch.nn.Module):
    """
    Discrete codebook for image patches.
    Maps continuous patch embeddings → token IDs.
    Similar to VQ-VAE quantization.
    """
    def __init__(self, codebook_size: int = 8192, embedding_dim: int = 768):
        super().__init__()
        # Each row = one codebook entry (a "visual puzzle piece")
        self.codebook = torch.nn.Embedding(codebook_size, embedding_dim)
        self.codebook_size = codebook_size

    def forward(self, patch_embeddings: torch.Tensor):
        """
        patch_embeddings: [N_patches, embedding_dim]
        Returns: token_ids [N_patches], quantized [N_patches, embedding_dim]
        """
        # Compute L2 distance between each patch and all codebook entries
        # distances: [N_patches, codebook_size]
        distances = torch.cdist(
            patch_embeddings,
            self.codebook.weight,
            p=2
        )

        # Pick nearest codebook entry for each patch
        token_ids = distances.argmin(dim=-1)   # [N_patches]

        # Look up the codebook entries
        quantized = self.codebook(token_ids)   # [N_patches, embedding_dim]

        return token_ids, quantized


# ── Patch Encoder ─────────────────────────────────────────────────────────────

class ImagePatchEncoder:
    """
    Splits image into 16×16 patches, encodes each with CLIP.
    """
    def __init__(self, patch_size: int = 16):
        self.patch_size = patch_size
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.clip.eval()

    def encode(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self.clip.get_image_features(**inputs)
        return features  # [1, embedding_dim]


# ── Multimodal Tokenizer ──────────────────────────────────────────────────────

class MultimodalTokenizer:
    """
    Converts text + images into a unified token ID sequence.
    Text tokens and image tokens share the same ID space.
    """
    def __init__(self, text_vocab_size: int = 32000, image_vocab_size: int = 8192):
        self.text_vocab_size  = text_vocab_size
        self.image_vocab_size = image_vocab_size
        # Image token IDs start after text vocabulary
        self.image_id_offset  = text_vocab_size
        self.codebook = ImageCodebook(image_vocab_size)

    def tokenize_text(self, text: str) -> list:
        """Standard BPE tokenization (simplified)."""
        # In practice: use the model's actual tokenizer
        return [ord(c) % self.text_vocab_size for c in text.split()]

    def tokenize_image(self, patch_embeddings: torch.Tensor) -> list:
        """Convert patch embeddings to image token IDs."""
        with torch.no_grad():
            token_ids, _ = self.codebook(patch_embeddings)
        # Offset to avoid collision with text token IDs
        return (token_ids + self.image_id_offset).tolist()

    def tokenize_multimodal(self, text: str, image: Image.Image = None) -> list:
        """
        Produces a unified flat token ID list.
        Example: "Look at <IMAGE_PATCHES> and describe it"
        """
        encoder = ImagePatchEncoder()

        tokens = []
        for part in text.split("<IMAGE>"):
            # Tokenize text segment
            tokens.extend(self.tokenize_text(part.strip()))

            # Insert image tokens if image provided
            if image is not None:
                img_embedding = encoder.encode(image)  # [1, D]
                # Simulate multiple patches (in practice, split image first)
                patch_embeddings = img_embedding.repeat(16, 1)  # [16, D]
                img_tokens = self.tokenize_image(patch_embeddings)
                tokens.extend(img_tokens)

        return tokens


# Usage
tokenizer = MultimodalTokenizer()
image = Image.open("cat.jpg")
tokens = tokenizer.tokenize_multimodal(
    "What animal is shown in <IMAGE> this photo?",
    image=image
)
print(f"Unified token sequence length: {len(tokens)}")
print(f"Token IDs: {tokens[:20]}...")
```

---

## Part 2 — 10 Million Token Context Window

Two separate challenges:

1. **Quadratic attention complexity** — O(N²) for N tokens
2. **Length generalization** — model trained on shorter sequences must work on longer ones

---

### Challenge 1 — Quadratic Attention & Ring Attention

Standard transformer attention requires every token to attend to every other token:

```
N = 1M tokens   → ~1 trillion  attention scores per layer
N = 10M tokens  → ~100 trillion attention scores per layer
```

That's per layer. A 32-layer transformer multiplies this by 32. Completely infeasible naively.

**Ring Attention** (also called Context Parallelism) — likely used by Llama 4 (not confirmed in blog post, but Ring Attention's original paper showed a 3B model supporting 10M context in 2023):

#### How Ring Attention Works

Split the 10M token sequence into chunks. Distribute chunks across devices.

```
Token sequence: [chunk_1 | chunk_2 | chunk_3 | chunk_4]
                  GPU_1     GPU_2     GPU_3     GPU_4
```

**Layer 1:** Each chunk computes self-attention *within itself* only.
```
chunk_1 attends to chunk_1 internally → O(chunk_size²) not O(N²)
```

**Layer 2:** Each chunk receives a summary from the previous chunk and incorporates it:
```
chunk_2 receives summary from chunk_1, attends internally + to chunk_1 context
```

**Layer 3:** This propagates further:
```
chunk_3 now carries info from chunk_2, which carried info from chunk_1
→ chunk_3 has transitive visibility into chunk_1
```

The "ring" structure: chunks pass summaries clockwise. After enough layers, all tokens have seen all other tokens — just not directly, but transitively through the chain.

**Connections count:**
- Naive: C² connections between layers (every chunk to every chunk)
- Ring: C connections (each chunk to its one neighbor)

This reduces attention from O(N²) toward O(N) effective complexity. And since each chunk lives on one GPU, the inter-chunk communication becomes inter-GPU communication — manageable because it's linear, not quadratic.

```python
def ring_attention_step(chunk_q, chunk_k, chunk_v, neighbor_kv=None):
    """
    Compute attention for one chunk, optionally incorporating neighbor's KV.
    In practice this happens across transformer layers via flash attention.
    """
    if neighbor_kv is not None:
        neighbor_k, neighbor_v = neighbor_kv
        # Extend keys and values with neighbor context
        k = torch.cat([neighbor_k, chunk_k], dim=1)
        v = torch.cat([neighbor_v, chunk_v], dim=1)
    else:
        k, v = chunk_k, chunk_v

    # Standard scaled dot-product attention
    scale = chunk_q.shape[-1] ** -0.5
    scores = torch.matmul(chunk_q, k.transpose(-2, -1)) * scale
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, v)
    return output
```

---

### Challenge 2 — Length Generalization

Scout was trained on 256K context windows. The 10M context window is only available at **inference time** — the model must generalize beyond what it saw during training.

This is the **length generalization** problem and it's deeply tied to positional embeddings.

#### RoPE (Rotary Position Embeddings)

Every Llama version uses RoPE. The idea: encode position by rotating word embeddings by an angle proportional to their position.

```
Word "cat" at position 2  → embedding rotated by angle θ₂
Word "mat" at position 6  → embedding rotated by angle θ₆  (bigger angle)
```

**Why RoPE is good:**
- No learned parameters per position — no model size increase for longer contexts
- Rotations are computed on-the-fly from a deterministic function
- Can encode arbitrarily long positions (the rotation just keeps going)

**Why RoPE fails at very long contexts:**
- Angles wrap around — multiple positions eventually share the same angle
- Position 1,000 and position 1,001,000 might have the same embedding angle
- Model gets confused: which position 1,000 is this?

#### Position Interpolation (Fix for RoPE Wrapping)

Instead of position 1 → angle θ₁, position 2 → angle θ₂, etc., **rescale all positions**:

```
If trained on context 1,024, deploying on context 4,096:
Rescale by factor 4:
  position 1    → now treated as position 0.25
  position 2    → now treated as position 0.5
  position 4096 → now treated as position 1024
```

Effect: slows down the rotation speed, so fewer positions wrap around and share the same angle.

**Trade-off:** Positions that are nearby now have very similar angles → harder to distinguish fine-grained position differences. Helps generalization, hurts precision.

#### NoPE (No Positional Embeddings) — Counterintuitive Finding

A 2023 paper tested models with absolutely no positional embeddings. Finding: for **long context generalization**, models with no positional embeddings outperformed models with RoPE or other explicit positional encodings.

Why? Positional embeddings encode human assumptions about positions. Those assumptions break down at very long ranges. No embeddings = no wrong assumptions = better generalization.

**But:** NoPE performs worse on short contexts that were seen during training. RoPE is better for short, NoPE is better for very long.

#### iRoPE — Llama 4's Solution (Best of Both)

Meta interleaves RoPE layers and NoPE layers across the transformer stack:

```
Layer 1:  RoPE  (good at short-range position encoding)
Layer 2:  NoPE  (good at long-range generalization)
Layer 3:  RoPE
Layer 4:  NoPE
...
```

Short-range relationships handled by RoPE layers. Long-range generalization handled by NoPE layers. One model, both capabilities.

```python
import torch
import torch.nn as nn
import math

class RoPEAttention(nn.Module):
    """Standard attention with Rotary Position Embeddings."""
    def __init__(self, dim: int, max_seq_len: int = 131072):
        super().__init__()
        self.dim = dim
        # Precompute rotation frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def rotate(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to query/key vectors."""
        # positions: [seq_len]
        freqs = torch.outer(positions.float(), self.inv_freq)   # [seq, dim/2]
        emb   = torch.cat([freqs, freqs], dim=-1)               # [seq, dim]
        cos   = emb.cos()[None, :, None, :]                     # [1, seq, 1, dim]
        sin   = emb.sin()[None, :, None, :]

        # Rotate: half dims rotated together
        x1, x2 = x[..., :self.dim // 2], x[..., self.dim // 2:]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin

    def forward(self, q, k, v, positions):
        q = self.rotate(q, positions)
        k = self.rotate(k, positions)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim)
        return torch.matmul(torch.softmax(scores, dim=-1), v)


class NoPEAttention(nn.Module):
    """Standard attention with NO positional embeddings."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, q, k, v, positions=None):
        # positions ignored — no positional info injected
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim)
        return torch.matmul(torch.softmax(scores, dim=-1), v)


class iRoPETransformer(nn.Module):
    """
    Interleaved RoPE + NoPE layers — Llama 4's iRoPE strategy.
    Odd layers use RoPE (short-range precision).
    Even layers use NoPE (long-range generalization).
    """
    def __init__(self, dim: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPEAttention(dim) if i % 2 == 0 else NoPEAttention(dim)
            for i in range(n_layers)
        ])
        self.layer_types = ["RoPE" if i % 2 == 0 else "NoPE" for i in range(n_layers)]

    def forward(self, q, k, v, positions):
        x = q
        for i, layer in enumerate(self.layers):
            if self.layer_types[i] == "RoPE":
                x = layer(x, k, v, positions)
            else:
                x = layer(x, k, v)   # NoPE ignores positions
        return x
```

---

## Part 3 — New Training Techniques

### Three-Stage Training (Pre → Mid → Post)

Standard LLMs use two stages: pre-training then post-training. Llama 4 adds a middle stage:

```
Pre-training
  → 30 trillion tokens (2× Llama 3, 2× DeepSeek V3)
  → Text + images + video mixed together (enabled by early fusion)
  → 200 languages (more than the number of countries)
       │
       ▼
Mid-training  ← NEW
  → Data with progressively longer contexts
  → Likely uses curriculum learning: gradually increase context length
  → Similar to how Llama 3 increased context in 6 stages up to 128K
  → Specifically dedicated to improving long-context behavior
       │
       ▼
Post-training
  → SFT on instruction data
  → RLVR on reasoning tasks (online setup — see below)
  → DPO on human preference data
  → Heavy data filtering: cut training datasets by half,
    keeping only examples an LLM judged as "hard"
```

### Online Reinforcement Learning

For reasoning fine-tuning, Llama 4 uses an alternating two-mode loop:

```
Mode 1: Train
  → Active RL training on current difficult examples
       ↕ alternates
Mode 2: Discover
  → Cycle through training data
  → Use model to identify hard examples it struggles with
  → Build the next batch for Mode 1
```

This is online RL + curriculum learning combined. The model actively participates in deciding what to train on next — only the hard problems, not the easy ones it already knows.

### FP8 Training

Llama 4 trained in **floating point 8** precision (like DeepSeek V3). US-based models had been using FP16. FP8 uses half the memory and is faster, but introduces training instability. DeepSeek figured out how to stabilize it first; Meta adopted the technique here.

---

## Part 4 — Is RAG Dead?

Short answer: **not yet, probably not soon.**

**The case for RAG dying:**
- 10M token context can fit entire codebases, books, multiple hours of video
- Avoids all the moving pieces of RAG: chunking, embedding models, re-rankers, vector stores

**The case against:**
- Large enterprises (Google, banks, hospitals) have 25+ years of internal documents
- Shoving an entire knowledge base into every employee query is expensive and impractical
- RAG is targeted retrieval — context window is brute force

**The realistic middle ground:**
RAG as currently implemented (brittle, many moving pieces) will be replaced — but not by "put everything in the context window." More likely by smarter retrieval that's less fragile than today's chunking + embedding + re-ranker pipelines.

---

## Part 5 — How Good Is It? (Honest Evaluation)

Meta's official benchmarks show all three models beating all competitors on all tasks. The online community is skeptical.

### Third-Party Benchmark Results

**Aider LLM Leaderboard** (225 hard coding exercises):
```
Maverick:       15.6%
Gemini 2.5 Pro: 72.9%   ← leader
```

**SimpleBench** (gotcha questions with distractors):
```
Humans:         83.7%
Gemini 2.0:     51.6%   ← best AI
Maverick:       27.7%
```

**Fiction.live long-context comprehension** (book understanding at 120K context):
```
Maverick:       28.1%
Gemini 2.5 Pro: 90.6%
```

The 10M context window doesn't help much if the model can't actually use the context correctly. Long-context comprehension appears to be a significant weakness compared to the marketing claims.

---

## Summary

| Innovation | What It Is | Why It Matters |
|---|---|---|
| MoE | Only activate <5% of parameters per token | Huge models at low inference cost |
| Early Fusion | Tokenize images and text into one unified ID space | True joint multimodal training, no adapters |
| Ring Attention | Split sequence into chunks, pass summaries in a ring | Reduces O(N²) attention to near-linear |
| iRoPE | Alternate RoPE (precision) and NoPE (generalization) layers | Best of both: short-range accuracy + long-range context |
| Mid-training | Extra stage between pre- and post-training | Progressively trains model on longer contexts |
| Online RL | Alternating train/discover modes | Curriculum learning — always training on hard examples |
| FP8 Training | 8-bit floating point precision | Cuts memory and compute vs FP16 |

### The Honest Picture

Architecture innovations are real and meaningful — especially early fusion and iRoPE. But benchmark performance on independent third-party tests tells a different story from the official blog post. The 10M context window is architecturally impressive but practically underutilized given the comprehension gaps shown on long-context benchmarks.

---

> No technical report available yet. Some architecture details are reverse-engineered from the Chameleon paper and Llama 3 technical report. Expect corrections when Meta publishes.
