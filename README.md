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

---

### Recommended Tools
- **Hugging Face Transformers** — Easiest
- **Ollama / llama.cpp** — Fast local running
- Start with **Llama-3.2-1B** or **3B** model

**Pro Tip**: Use Google Colab with GPU. Add `load_in_4bit=True` for lower memory usage.

---
