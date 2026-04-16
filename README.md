# llm_learning
**How LLMs Work by Dissecting Llama**

### 1. What is an LLM?
- **LLM** = Large Language Model
- It is a massive neural network trained to **predict the next token** (word or piece of word) in a sequence.
- Simple analogy: Extremely advanced autocomplete that has read almost the entire internet.
- It doesn’t truly “understand” — it is very good at recognizing and continuing patterns.

**Core Idea**: Everything an LLM does is based on next-token prediction.

---

### 2. Tokens – The Basic Units
- LLMs do not read letters or full words directly.
- Text is split into **tokens** using a tokenizer (Llama uses SentencePiece/BPE).
- Example:
  - "I love artificial intelligence" → ["I", " love", " artificial", " intelligence"]
  - Rare words are broken: "unbelievable" → ["un", "believ", "able"]
- Vocabulary size in Llama ≈ 128,000 tokens.
- Each token is converted into a numerical vector (embedding) for the model to process.

**Why important?** The model only sees and thinks in tokens.

---

### 3. Architecture: The Transformer (Decoder-Only)
Llama uses a **Decoder-only Transformer** architecture with multiple identical layers stacked on top of each other.

**Main Components in Each Layer:**

- **Self-Attention**  
  Allows the model to focus on relevant previous tokens.  
  Example: In “The cat sat because it was tired”, it connects “it” to “cat”.  
  Llama uses efficient **Grouped Query Attention (GQA)**.

- **Feed-Forward Network (MLP)**  
  Processes each token individually. Uses **SwiGLU** activation in Llama.

- **Layer Normalization + Residual Connections**  
  Help stabilize training of very deep networks.

**Simple Analogy**: Attention = “Paying attention to context”, MLP = “Thinking and updating knowledge”.

---

### 4. Pre-Training (How the Model Learns)
- **Objective**: Next-token prediction on massive data (trillions of tokens).
- Data: Internet text, books, code, articles, etc.
- Process: Predict next token → Calculate error → Update billions of weights using backpropagation.
- Result after pre-training = **Base Model** (raw LLM).

**Llama 3 Example**: Trained on ~15 trillion tokens.

---

### 5. Inference – How Text is Generated
Steps:
1. Convert prompt to tokens.
2. Pass through the model to get probability scores (logits) for next token.
3. Sample next token (using temperature, top-p, etc.).
4. Add new token to input and repeat.

**Important Optimization**: KV Caching (speeds up generation).

---

### 6. Post-Training (Base → Helpful Assistant)
- **Supervised Fine-Tuning (SFT)**: Train on high-quality question-answer pairs.
- **RLHF / DPO**: Use human feedback to make responses helpful, honest, and safe.
- Result = **Instruct / Chat Model**.

**Base Model vs Instruct Model**

| Feature         | Base Model                    | Instruct Model                     |
|-----------------|-------------------------------|------------------------------------|
| Purpose         | Text completion               | Follow instructions & chat         |
| Training        | Raw internet data             | Dialogues + Human feedback         |
| Behavior        | Can continue any text         | Helpful, polite, follows rules     |
| Hallucination   | Very high                     | Lower but still present            |

---

### 7. Key Limitations
- Hallucinations (makes up facts confidently)
- Weak in exact math, counting, and long reasoning
- Limited context length (e.g., 8K or 128K tokens)
- No real-world understanding — only statistical patterns

---

### 8. Best Hands-on Learning Approach (Most Important Part)

#### Step 1: Read the Model Config File

```python
from transformers import AutoConfig

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # Fully open model

config = AutoConfig.from_pretrained(model_name)

print("=== Model Configuration ===")
print(f"Hidden Size          : {config.hidden_size}")
print(f"Number of Layers     : {config.num_hidden_layers}")
print(f"Attention Heads      : {config.num_attention_heads}")
print(f"Vocabulary Size      : {config.vocab_size}")
print(f"Max Context Length   : {config.max_position_embeddings}")
```

#### Step 2: Run a Simple Forward Pass

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

next_token_id = torch.argmax(outputs.logits[0, -1, :]).item()
print("Predicted next token:", tokenizer.decode([next_token_id]))
```

#### Step 3: Generate Text with Different Settings

```python
def generate_text(prompt, max_tokens=200, temp=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temp,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test different settings
print(generate_text("Explain LLM in simple words:", temp=0.7))
```

#### Step 4: Explore Model Internals

```python
print(model)                          # Full model
print(model.model.layers[0])          # One decoder layer
print(model.model.layers[0].self_attn)   # Attention part
print(model.model.layers[0].mlp)         # Feed-forward part
```
##### Output
```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 2048)
    (layers): ModuleList(
      (0-21): 22 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (k_proj): Linear(in_features=2048, out_features=256, bias=False)
          (v_proj): Linear(in_features=2048, out_features=256, bias=False)
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2048, out_features=5632, bias=False)
          (up_proj): Linear(in_features=2048, out_features=5632, bias=False)
          (down_proj): Linear(in_features=5632, out_features=2048, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((2048,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=2048, out_features=32000, bias=False)
)
LlamaDecoderLayer(
  (self_attn): LlamaAttention(
    (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
    (k_proj): Linear(in_features=2048, out_features=256, bias=False)
    (v_proj): Linear(in_features=2048, out_features=256, bias=False)
    (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
  )
  (mlp): LlamaMLP(
    (gate_proj): Linear(in_features=2048, out_features=5632, bias=False)
    (up_proj): Linear(in_features=2048, out_features=5632, bias=False)
    (down_proj): Linear(in_features=5632, out_features=2048, bias=False)
    (act_fn): SiLUActivation()
  )
  (input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
  (post_attention_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
)
LlamaAttention(
  (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
  (k_proj): Linear(in_features=2048, out_features=256, bias=False)
  (v_proj): Linear(in_features=2048, out_features=256, bias=False)
  (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
)
LlamaMLP(
  (gate_proj): Linear(in_features=2048, out_features=5632, bias=False)
  (up_proj): Linear(in_features=2048, out_features=5632, bias=False)
  (down_proj): Linear(in_features=5632, out_features=2048, bias=False)
  (act_fn): SiLUActivation()
)
```
---
**Detailed Explanation of Model Internals (TinyLlama Architecture)**
---

### 1. Overall Structure: `LlamaForCausalLM`

This is the complete model. It has two main parts:

- **(model)**: LlamaModel → The actual brain (does all the heavy computation)
- **(lm_head)**: Final layer that converts internal understanding into word probabilities

---

### 2. Token Embeddings – `embed_tokens: Embedding(32000, 2048)`

- **What it does**: Converts each token ID into a vector of 2048 numbers.
- Vocabulary size = 32,000 tokens.
- Each token gets its own learned vector of length **2048** (this is called `hidden_size`).
- Think of it as: Every word/piece of word gets its own "ID card" with 2048 features.

**Size**: 32,000 × 2048 = ~65 million parameters.

---

### 3. Main Body: 22 Decoder Layers (`ModuleList (0-21): 22 x LlamaDecoderLayer`)

TinyLlama has **22 identical layers** stacked one after another. Each layer refines the understanding further.

Every `LlamaDecoderLayer` has 4 main parts:

#### A. Self-Attention (`LlamaAttention`)

This is the **most important part** of Transformers.

```python
(self_attn): LlamaAttention(
  (q_proj): Linear(2048 → 2048)
  (k_proj): Linear(2048 → 256)
  (v_proj): Linear(2048 → 256)
  (o_proj): Linear(2048 → 2048)
)
```

**Simple Explanation**:

- **q_proj (Query)**: Turns input into "What am I looking for?"
- **k_proj (Key)**: Turns input into "What do I contain?"
- **v_proj (Value)**: Turns input into "What information should I pass?"
- **o_proj (Output)**: Combines everything and gives final output.

**Why different sizes?**
- Hidden size = 2048
- But Key and Value are only 256 dimensions → This is **Grouped Query Attention (GQA)**. It makes the model faster and uses less memory while keeping good performance.

**Role of Attention**: Allows every token to "look at" all previous tokens and decide what is important.

---

#### B. MLP / Feed-Forward Network (`LlamaMLP`)

```python
(mlp): LlamaMLP(
  (gate_proj): Linear(2048 → 5632)
  (up_proj):   Linear(2048 → 5632)
  (down_proj): Linear(5632 → 2048)
  (act_fn):    SiLUActivation()
)
```

**How it works (Simple flow)**:

1. Input (2048 dim) goes to `gate_proj` and `up_proj`.
2. Both expand to **5632 dimensions** (much wider).
3. `gate_proj` decides which information should pass (like a gate).
4. Activation function **SiLU** (Smooth ReLU) is applied.
5. Then `down_proj` shrinks it back to 2048 dimensions.

**Purpose**: This is where the model actually "thinks" and does non-linear computation. It’s like the model’s brain doing calculations and pattern matching.

**Why expand to 5632?** This is roughly 2.75× hidden size — common in modern LLMs for better capacity.

---

#### C. Layer Normalization (`RMSNorm`)

```python
(input_layernorm)
(post_attention_layernorm)
```

- There are two RMSNorm layers in each decoder layer.
- **RMSNorm** is a faster and more stable version of Layer Normalization.
- It normalizes the values so training stays stable even with 22+ layers.

**Simple analogy**: Like adjusting the volume of audio before and after processing so nothing gets too loud or too quiet.

---

### 4. Final Layer Norm (`norm`: LlamaRMSNorm)

After all 22 layers finish processing, this final normalization is applied before the output head.

---

### 5. Rotary Embeddings (`rotary_emb`: LlamaRotaryEmbedding)

- This adds **positional information** to the tokens.
- Unlike older models that added fixed position numbers, Llama uses **RoPE (Rotary Position Embedding)** — it rotates the vectors based on position.
- Very effective for handling long sequences.

---

### 6. Language Modeling Head (`lm_head`: Linear(2048 → 32000))

- Takes the final 2048-dimensional vector of the last token.
- Converts it into **32,000 scores** (one score per token in vocabulary).
- These scores (logits) are turned into probabilities using softmax.
- The highest probability token is chosen as the next word.

---

### Full Data Flow Summary (Step by Step)

1. Input text → Tokens → `embed_tokens` → 2048-dim vectors
2. Add positional information using `rotary_emb`
3. For each of 22 layers:
   - Apply `input_layernorm`
   - Self-Attention (QKV + o_proj) → Residual connection
   - Apply `post_attention_layernorm`
   - MLP (expand → SiLU → shrink) → Residual connection
4. Final `norm`
5. `lm_head` → Convert to vocabulary scores
6. Sample next token → Repeat for generation

---

### Parameter Count Summary (TinyLlama 1.1B)

- Embeddings + lm_head: ~130M parameters
- Each layer: ~40M parameters
- 22 layers: ~880M parameters
- Total: **~1.1 Billion parameters**

---

**Quick Analogy of Whole Model**:

Think of the model as a **22-floor building**:
- Ground floor: Embeddings (gives meaning to words)
- Each floor: One Decoder Layer (Attention = discussion between words, MLP = thinking)
- Top floor: lm_head (decides which word to output next)

---

### Recommended Tools
- **Hugging Face Transformers** — Easiest
- **Ollama / llama.cpp** — Fast local running
- Start with **Llama-3.2-1B** or **3B** model

**Pro Tip**: Use Google Colab with GPU. Add `load_in_4bit=True` for lower memory usage.

---
