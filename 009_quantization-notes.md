# Model Quantization — Fundamentals Notes

> The technique that lets you run DeepSeek R1 (671B params) on just two GPUs.
> Also enables ML on edge devices that don't support floating point at all.

---

## What Is Quantization?

**Quantization = mapping a continuous space to a discrete one.**

In signal processing: turning a real-world sound wave into digital data. You can't sample every infinite decimal moment, so you take snapshots at regular intervals (sampling) and round each amplitude to the nearest representable value.

In machine learning: converting model weights (and sometimes activations) from floating point values to integers.

```
Original weights:  [-0.87, 0.34, -0.12, 0.91, ...]   ← float, range [-1, 1]
Quantized weights: [-111,  43,   -15,   116,  ...]   ← int8, range [-128, 127]
```

**Why does this help?**
- DeepSeek R1 at full float16: ~720 GB
- After quantization: ~131 GB → fits across two Nvidia H100 GPUs
- That's over **80% memory reduction**

---

## Why Integers Are Faster Than Floats

### Integer Representation

An unsigned integer stores a number in base 2. Simple, direct.

Signed integers use **two's complement** (standardized in the 1960s) — just one extra bit for the sign, no additional overhead.

Adding two integers: a trivial bitwise XOR with carryover. On modern CPUs: **1 clock cycle**.

### Floating Point Representation

Floats store three components:
```
[sign bit | exponent bits | mantissa/significand bits]
```

Together these form a formula similar to scientific notation but with extra quirks (hidden bit, bias).

**IEEE 754 standard (most common):** sign + 8-bit exponent + 23-bit mantissa = 32 bits total.

**BFloat16 (Google Brain, for ML):** sign + 8-bit exponent + 7-bit mantissa = 16 bits. Allocates more bits to the exponent → larger **dynamic range** → can represent very small numbers near zero, which matters a lot during training.

Adding two floats: align exponents, add mantissas, normalize result. On modern CPUs: **3–4 clock cycles**.

**Summary:**

| Operation | Integer | Float |
|---|---|---|
| Addition | 1 clock cycle | 3–4 clock cycles |
| Representation | Simple bit pattern | Sign + exponent + mantissa |
| Memory per value | 1 byte (int8) | 2 bytes (fp16) or 4 bytes (fp32) |

Quantization = swap floats for integers → smaller memory footprint, faster operations.

**The trade-off:** you lose precision. Models have built-in redundancy so some precision loss is tolerable, but it's always a balance.

---

## When to Quantize

Two major milestones: **training** and **inference**.

### Training

During training, gradients require fine-grained precision. You can't avoid real values here.

Precision evolution over time:
```
FP32 (float32) → FP16 (float16) → BF16 → FP8
```

Recent models (DeepSeek R1, Llama 4) trained in FP8. Gradient descent likely can't go much lower — this trend has a natural floor.

### Inference

Inference = forward pass only. Much more resilient to precision loss. People have successfully run inference at:
- **INT8** — very common, minimal quality loss
- **INT4** — aggressive, needs careful handling
- **INT1** — experimental (1-bit LLMs)

### Two Strategies Based on Timing

**Post-Training Quantization (PTQ)**
- Quantize the model *after* training is done, before deployment
- Most common approach for open-weights LLMs today
- Meta and DeepSeek publish in float format → open-source communities (Unsloth, etc.) quantize them
- Recognizable by suffixes: GGUF, GPTQ, AWQ

**Quantization-Aware Training (QAT)**
- Build the model knowing it will be quantized
- Simulate quantization *during* training so the model adapts
- Required for extreme quantization (4-bit or lower) to avoid significant quality degradation
- How 1-bit LLMs are trained

```
PTQ: Train (fp32/bf16) → Full model → Quantize → Deploy (int8/int4)
QAT: Train (simulate quantization) → Quantization-resilient model → Quantize → Deploy
```

---

## How Quantization Works — The Math

### Symmetric Quantization (Simple Case)

Mapping float values from range `[-1, 1]` to int8 range `[-127, 127]`:

```
Scale s = (max_float - min_float) / (max_int - min_int)
        = 2 / 254 ≈ 0.00787

Quantize: Q = round(R / s)
Dequantize: R ≈ Q × s
```

Example: R = 0.42
```
Q = round(0.42 / 0.00787) = round(53.4) = 53
Dequantized: 53 × 0.00787 ≈ 0.417   ← approximation of 0.42
```

The rounding causes irreversible precision loss — dequantization only recovers an approximation.

### Asymmetric Quantization (General Case)

When the float range is not symmetric — for example, activations after ReLU are always ≥ 0, so the range is something like `[0.0, 3.7]` instead of `[-1, 1]`.

Two parameters are needed:
- **Scale s** — ratio of the two ranges
- **Zero point z** — which integer bucket corresponds to real value 0

```
s = (beta - alpha) / (Q_max - Q_min)
z = Q_min - round(alpha / s)

Quantize:   Q = round(R / s) + z
Dequantize: R ≈ s × (Q - z)
```

Example: float range `[0.0, 3.7]`, int range `[-7, 7]` (symmetric int4)
```
s = 3.7 / 14 ≈ 0.264
z = -7 - round(0.0 / 0.264) = -7

R = 1.0 → Q = round(1.0 / 0.264) + (-7) = round(3.79) - 7 = 4 - 7 = -3
Dequantize: 0.264 × (-3 - (-7)) = 0.264 × 4 ≈ 1.056
```

### Calibration

**Clipping range** `[alpha, beta]` = the range of real values the model actually uses. Values outside are clipped to the min/max bucket.

Finding the clipping range = **calibration**:
- For **weights**: easy — just find min and max of the weight tensor
- For **activations**: harder — must run inference on a calibration dataset because activations vary per input. Takes minutes to hours depending on model size

### Code: Quantization and Dequantization

```python
import torch
import torch.nn as nn

# ── Symmetric Quantization ────────────────────────────────────────────────────

def symmetric_quantize(x: torch.Tensor, n_bits: int = 8):
    """
    Symmetric quantization: assumes the float range is centered at 0.
    Maps [-abs_max, abs_max] → [-2^(n-1)+1, 2^(n-1)-1]
    """
    q_max = 2 ** (n_bits - 1) - 1    # e.g., 127 for int8

    # Scale = ratio of float range to int range
    abs_max = x.abs().max()
    scale   = abs_max / q_max

    # Quantize: divide by scale and round
    q = torch.clamp(torch.round(x / scale), -q_max, q_max).to(torch.int8)

    return q, scale


def symmetric_dequantize(q: torch.Tensor, scale: float):
    """Recover approximate float values from quantized integers."""
    return q.float() * scale


# ── Asymmetric Quantization ───────────────────────────────────────────────────

def asymmetric_quantize(x: torch.Tensor, n_bits: int = 8):
    """
    Asymmetric quantization: handles non-symmetric float ranges.
    Computes both scale AND zero_point.
    """
    q_min = -(2 ** (n_bits - 1))      # e.g., -128 for int8
    q_max =  (2 ** (n_bits - 1)) - 1  # e.g.,  127 for int8

    # Clipping range: actual min and max of the tensor
    alpha = x.min().item()
    beta  = x.max().item()

    # Scale: ratio of float range to int range
    scale = (beta - alpha) / (q_max - q_min)

    # Zero point: which int bucket corresponds to real 0
    zero_point = q_min - round(alpha / scale)
    zero_point = int(torch.clamp(torch.tensor(zero_point), q_min, q_max))

    # Quantize: Q = round(R / s) + z
    q = torch.clamp(
        torch.round(x / scale) + zero_point,
        q_min, q_max
    ).to(torch.int8)

    return q, scale, zero_point


def asymmetric_dequantize(q: torch.Tensor, scale: float, zero_point: int):
    """R ≈ s × (Q - z)"""
    return scale * (q.float() - zero_point)


# ── Demo ──────────────────────────────────────────────────────────────────────

weights = torch.randn(4, 4)
print("Original weights:\n", weights.round(decimals=3))

q, scale = symmetric_quantize(weights)
print("\nQuantized (int8):\n", q)

recovered = symmetric_dequantize(q, scale)
print("\nDequantized (approx):\n", recovered.round(decimals=3))

max_error = (weights - recovered).abs().max()
print(f"\nMax quantization error: {max_error:.5f}")
```

---

## Fixed-Point Arithmetic — Keeping Everything Integer

### The Problem

After quantizing, you still need to multiply by the scale `s` when computing outputs. But `s` is a real number — multiplying by it defeats the purpose of quantization (we're trying to avoid floats).

### The Solution: Fixed-Point Representation

Fixed-point = represent a fraction using only integers by tracking the decimal position.

```
Store $12.34 as:  1234  with implicit scale 1/100
Recover:          1234 / 100 = 12.34
```

In binary (for ML): use powers of 2 as the scale.

```
4-bit unsigned, fixed point:
Smallest step = 2^(-4) = 0.0625
To store 0.75: round(0.75 / 0.0625) = round(12.0) = 12  ← stored
Recover: 12 × 0.0625 = 0.75  ✓
```

### Multiplying x (integer) by s (real scale) → integer result

**Naive approach:** Store s as fixed-point, multiply. Problem: if s is small like 0.003, it rounds to zero → result is always zero. Too much precision loss.

**Better approach:** Decompose s = 2^(-n) × s₀ where s₀ > 0.5

```
s = 0.003
→ Find n such that s × 2^n > 0.5
→ 0.003 × 2^9 = 0.003 × 512 = 1.536 → n=9, s₀=1.536

Now: x × s = x × s₀ × 2^(-9)
           = (x × s₀) >> 9    ← integer multiply then bit shift
```

The bit shift (`>> 9`) is an extremely cheap operation. s₀ > 0.5 so it can be stored as fixed-point without rounding to zero.

This decomposition is precomputed once before deployment — no runtime cost.

```python
import math

def decompose_scale(s: float, precision_bits: int = 16):
    """
    Decompose scale s into (s0, n) where s = s0 × 2^(-n) and s0 > 0.5
    This allows integer multiply + bit shift instead of float multiply.
    """
    if s == 0:
        return 0, 0

    # Find n such that s × 2^n ∈ [0.5, 1.0)
    n = math.ceil(-math.log2(s))    # smallest n where s × 2^n >= 0.5

    # s0: scale shifted into [0.5, 1.0) range
    s0 = s * (2 ** n)

    # Store s0 as fixed-point integer (scaled by 2^precision_bits)
    s0_fixed = round(s0 * (2 ** precision_bits))

    return s0_fixed, n, precision_bits


def fixed_point_multiply(x: int, s: float) -> int:
    """
    Multiply integer x by real scale s, return integer result.
    Uses bit shifts instead of float multiplication.
    """
    s0_fixed, n, prec = decompose_scale(s)

    # x × s = (x × s0_fixed) >> (n + prec)
    result = (x * s0_fixed) >> (n + prec)
    return result

# Example
x = 53        # quantized value
s = 0.00787   # scale from int8 symmetric quantization
result = fixed_point_multiply(x, s)
print(f"x={x}, s={s}, x*s={x*s:.4f}, fixed_point result: {result}")
# Compare: 53 × 0.00787 ≈ 0.417 (the dequantized float)
```

---

## Matrix Multiplication in Integer Space

Scalar multiplication generalizes directly to matrices. For output element at position (i, j):

```
R3[i,j] = Σₖ R1[i,k] × R2[k,j]           ← float matmul

Q3[i,j] = round( (s1 × s2 / s3) × Σₖ (Q1[i,k] - z1)(Q2[k,j] - z2) ) + z3
```

The inner sum `Σₖ (Q1 - z1)(Q2 - z2)` is pure integer arithmetic.
The scale factor `s1×s2/s3` is applied once per output row — not per element.

### Why This Matters

A matrix of size 4,096 × 4,096 (typical in Llama 4):
- That's ~67 million multiply operations per layer
- Integer multiply: 1 cycle each
- Float multiply: 3–4 cycles each
- Net speedup per layer: 3–4×

And the scale correction `s1×s2/s3` only needs applying ~4,096 times (once per row) — a negligible cost.

**On modern GPUs (Nvidia):** float and int multiplications are comparably fast anyway. In those cases, the main gain from quantization isn't compute speed — it's **memory bandwidth**. Smaller weights = fewer bytes transferred between memory and compute units = faster inference.

```python
def quantized_matmul(Q1: torch.Tensor, Q2: torch.Tensor,
                     s1: float, z1: int,
                     s2: float, z2: int,
                     s3: float, z3: int,
                     output_bits: int = 8) -> torch.Tensor:
    """
    Quantized matrix multiplication entirely in integer space.
    Q1: [M, K] int8, Q2: [K, N] int8
    Returns: Q3 [M, N] int8
    """
    # Integer matmul: (Q1 - z1) @ (Q2 - z2)
    # Cast to int32 first to avoid overflow from int8 × int8 accumulation
    A = (Q1.int() - z1)    # [M, K]
    B = (Q2.int() - z2)    # [K, N]
    int_result = A @ B     # [M, N] — pure integer, no floats

    # Apply combined scale (only once per output, not per element)
    combined_scale = (s1 * s2) / s3

    # Requantize to output format
    q_max = 2 ** (output_bits - 1) - 1
    Q3 = torch.clamp(
        torch.round(int_result * combined_scale) + z3,
        -q_max, q_max
    ).to(torch.int8)

    return Q3


# ── Quantized Linear Layer ────────────────────────────────────────────────────

class QuantizedLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that operates in int8.
    Weights are pre-quantized. Activations quantized at forward time.
    """
    def __init__(self, in_features: int, out_features: int, n_bits: int = 8):
        super().__init__()
        self.n_bits = n_bits

        # Store weights as int8 after quantization
        # (initialized from random floats for demo)
        float_weights = torch.randn(out_features, in_features) * 0.1
        q_weights, self.w_scale = symmetric_quantize(float_weights, n_bits)
        self.register_buffer('q_weights', q_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize activations at runtime
        q_x, x_scale = symmetric_quantize(x, self.n_bits)

        # Integer matmul
        # q_x: [batch, in] × q_weights.T: [in, out] → [batch, out]
        int_out = q_x.int() @ self.q_weights.int().T  # pure integer

        # Dequantize output — apply combined scale once per output row
        combined_scale = x_scale * self.w_scale
        output = int_out.float() * combined_scale

        return output
```

---

## Granularity — One Scale Per Tensor? Per Row?

So far we assumed one pair of `(s, z)` for the entire model. In practice:

| Granularity | Scope | Quality | Cost |
|---|---|---|---|
| Per tensor | One scale for all weights | Lowest | Cheapest |
| Per layer | One scale per transformer layer | Medium | Medium |
| Per matrix | One scale per weight matrix | Good | Moderate |
| Per row | One scale per matrix row | Best | Highest |

Finer granularity = better accuracy = more scale values to store and apply.

This applies to both weights and activations. Activations require runtime calibration since they change per input.

---

## Practical Usage — You Don't Need to Implement This

Most ML frameworks handle quantization for you:

```python
# PyTorch built-in quantization
import torch.quantization

model = YourModel()

# Post-training quantization (PTQ)
model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # CPU
torch.quantization.prepare(model, inplace=True)

# Calibration: run inference on calibration data to determine clipping ranges
with torch.no_grad():
    for batch in calibration_dataloader:
        model(batch)

# Convert to quantized model
torch.quantization.convert(model, inplace=True)
```

For deployment, inference engines go further:
- **ONNX Runtime** — cross-platform optimized inference
- **Nvidia TensorRT** — hardware-specific GPU optimizations
- **Unsloth** — LLM-specific quantization for consumer hardware
- **llama.cpp** — GGUF format for CPU inference

---

## Summary

### What
Quantization = mapping float values to integers. Weights (and optionally activations) go from fp16/fp32 → int8/int4/int1.

### Why
- Integers are simpler, faster, and smaller than floats
- 80%+ memory reduction enables massive models on affordable hardware
- Edge devices (like Google Coral TPU) support integers only

### When
| Stage | Precision used | Why |
|---|---|---|
| Training | fp32 → fp16 → bf16 → fp8 | Gradients need precision |
| Inference | int8 / int4 / int1 | Forward pass is resilient |
| PTQ | After training | Common, easy, minimal quality loss |
| QAT | During training | Needed for extreme compression (4-bit, 1-bit) |

### How (Key Parameters)
| Term | Meaning |
|---|---|
| Scale (s) | Ratio of float range to integer range |
| Zero point (z) | Integer bucket corresponding to real value 0 |
| Clipping range [α, β] | Float range actually used by the model |
| Calibration | Process of finding α and β for activations |
| Fixed-point | Representing fractions with integers + tracked decimal position |
| Symmetric | Zero point = 0, simpler math |
| Asymmetric | Non-zero zero point, handles skewed distributions |

### The Core Formula
```
Quantize:   Q = clamp(round(R / s) + z,  Q_min, Q_max)
Dequantize: R ≈ s × (Q - z)
```

---

> Coming next: Post-Training Quantization (PTQ), Quantization-Aware Training (QAT), 1-bit LLMs.
