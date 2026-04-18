# Tülu 3 — Post-Training Recipe Notes

> Tülu 3 is a fully open-source LLM by AI2 (Allen Institute of AI).
> Built on top of Llama 3.1. Fully open: model weights + datasets + training code + eval code + infrastructure.
> The goal: remove the guesswork from fine-tuning by publishing every detail.

**Links:**
- Paper: https://arxiv.org/abs/2411.15124
- GitHub: https://github.com/allenai/open-instruct
- Models: https://huggingface.co/collections/allenai/tulu-3-models
- Datasets: https://huggingface.co/collections/allenai/tulu-3-datasets

---

## Why Tülu 3 Exists

Post-training (everything after pre-training) is poorly documented even in "open source" projects. Llama releases model weights but not the fine-tuning recipe. Reproducing their reported accuracy improvements is extremely difficult — many people have tried and failed.

Tülu 3 publishes:
- All datasets used
- Exact chat templates
- All hyperparameters
- Full training + eval code

The name "Tülu" comes from a hybrid type of camel — fitting because the model uses a hybrid data approach (existing public data + synthetic data).

---

## The Full Pipeline (3 Stages)

```
Pre-trained Base Model (Llama 3.1)
          │
          ▼
[Stage 1] Instruction Fine-Tuning (SFT)
          │  learns to respond to instructions
          ▼
[Stage 2] Preference Fine-Tuning (DPO)
          │  learns which responses humans prefer
          ▼
[Stage 3] Reasoning Fine-Tuning (RLVR)
          │  learns to get verifiable answers correct
          ▼
     Final: Tülu 3
```

Each stage has two dimensions: **training data** and **learning objective**.

---

## Stage 1 — Instruction Fine-Tuning (SFT)

### Goal
Change the base model from "sentence completer" to "assistant that responds to instructions."

### What the base model does without SFT
```
User:   What is the capital of France?
Model:  What is the capital of Germany? What is the capital of Spain?...
```
It just continues the text. SFT fixes this.

### Training Data
Two sources:

**1. Existing public datasets**
AI2 published a spreadsheet of all public datasets used. Example: a coding dataset where each row has a problem (instruction) and a solution in C++/Python (reference response).

**2. Synthetic data** (covered in detail later)

### Learning Objective: Next Token Prediction
Same objective as pre-training — predict the next token. The difference is the data format.

Instructions and responses are concatenated into a flat conversation using a **chat template**:

```
<|system|>
You are a helpful assistant.
<|user|>
Write a Python function to reverse a string.
<|assistant|>
def reverse_string(s):
    return s[::-1]
```

Tülu's finding: **chat templates are surprisingly brittle**. Replacing the final newline character with an end-of-sentence token measurably improved performance. This is why publishing these tiny details matters.

### Why it's called "Supervised"
Both pre-training and SFT use next-token-prediction with cross-entropy loss. The difference: in SFT, the reference response is explicitly in the training text — you're supervising what the model should output. Pre-training discovers knowledge on its own from raw text.

### Code: SFT with Hugging Face

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

# Load base model
model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Load an instruction dataset
dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train")

# Chat template — flatten conversation to text
def format_example(example):
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

dataset = dataset.map(format_example)

# Training config
training_args = TrainingArguments(
    output_dir="./tulu3-sft",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,    # effective batch size = 16
    learning_rate=2e-5,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=2048,
)

trainer.train()
trainer.save_model("./tulu3-sft-final")
```

---

## Stage 2 — Preference Fine-Tuning (DPO)

### Goal
Teach the model to produce responses that humans prefer. Not just "correct" responses — but responses that are more helpful, better formatted, safer.

### Training Data Format
Each example has three things:
- An instruction
- A **chosen** response (preferred by a human)
- A **rejected** response (not preferred)

Example from the Alpaca Farm dataset:
```
Instruction: "Explain why the sky is blue"
Chosen:      "The sky appears blue due to Rayleigh scattering..."
Rejected:    "The sky is blue because it is."
```

You've seen this in practice if you've been asked by ChatGPT to pick between two responses — that's OpenAI collecting this preference data from users.

### Learning Objective: DPO (Direct Preference Optimization)

**Starting point:** The SFT checkpoint from Stage 1.

**What DPO does:**
- Feed the instruction to the model
- The model assigns a probability to the chosen response and to the rejected response
- DPO maximizes the probability of the chosen response
- DPO minimizes the probability of the rejected response
- Loss = negative of that → backpropagate → update weights

```
Loss = -log [ P(chosen) / P(rejected) ]
```

The model learns: given this instruction, rank the chosen response higher than the rejected one.

### Off-Policy vs On-Policy Responses

**Off-policy:** Responses in the training data were generated by some external LLM (like GPT-4), not by the model being trained. Easy to collect, but not ideal — you're teaching the model to prefer between other models' outputs.

**On-policy:** Sample two responses directly from the current SFT model. These are fresh responses the model itself generated. Since you don't have human labels for them, use an LLM judge (like GPT-4o) to estimate the preference.

On-policy is better in theory because you're aligning the model with preferences for its *own* outputs. Tülu 3 uses a **mix of both** off-policy and on-policy DPO in practice.

### Code: DPO with TRL

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Load SFT checkpoint (output of Stage 1)
model_name = "./tulu3-sft-final"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Preference dataset — needs "prompt", "chosen", "rejected" columns
dataset = load_dataset("allenai/tulu-3-pref-mixture", split="train")

# DPO config
dpo_config = DPOConfig(
    output_dir="./tulu3-dpo",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,               # much lower than SFT
    beta=0.1,                         # controls how far to deviate from SFT model
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
)

trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./tulu3-dpo-final")
```

**Key hyperparameter — `beta`:**
- Controls how much the model is allowed to deviate from the SFT checkpoint
- Low beta → aggressive preference optimization, may break other capabilities
- High beta → conservative, stays close to SFT behavior
- Tülu 3 uses `beta=0.1`

---

## Stage 3 — Reasoning Fine-Tuning (RLVR)

### Goal
Improve performance on tasks that have objectively verifiable answers: math, coding, logic. You either got it right or you didn't — no need for human preference judgments.

### Training Data Format
Each example has:
- An instruction (e.g., a math problem)
- A ground-truth answer to verify against

No "chosen/rejected" pairs needed — the reward is computed automatically.

### Learning Objective: RLVR (RL with Verifiable Rewards)

**Starting point:** The DPO checkpoint from Stage 2.

**How it works:**
1. Feed the instruction to the model
2. Model generates a response
3. Automatically check if the answer is correct
4. Reward = 1 if correct, 0 if wrong
5. Backpropagate the reward signal using PPO

```
Math problem → model answers → check against ground truth → reward → update weights
```

### The Algorithm: PPO (Proximal Policy Optimization)

PPO requires **two additional models** beyond the LLM being trained:

**Reward model:** Estimates reward at the token level, not just at the end of the response. Gives a signal for intermediate steps, not just the final answer.

**Value function:** Estimates how hard the current instruction is, so the model knows how much credit to take for a correct answer. An easy problem getting a correct answer should update weights less than a hard problem getting a correct answer.

This makes RLVR the most computationally expensive stage. You're effectively running three models simultaneously.

### Code: RLVR with TRL (using PPO)

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from datasets import load_dataset
import torch

model_name = "./tulu3-dpo-final"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# PPO needs a value head on top of the LLM
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

# Math/coding dataset with verifiable answers
dataset = load_dataset("allenai/tulu-3-rlvr-mixture", split="train")

ppo_config = PPOConfig(
    learning_rate=1e-6,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=4,
    ppo_epochs=4,
    kl_penalty="kl",       # penalize deviation from reference model
    init_kl_coef=0.1,
)

trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
)

# Verifiable reward function — 1 if correct, 0 if wrong
def compute_reward(response: str, ground_truth: str) -> float:
    # Extract the final answer from model response
    # (in practice, parse the boxed answer or code output)
    predicted = extract_answer(response)
    return 1.0 if predicted.strip() == ground_truth.strip() else 0.0

# Training loop
for batch in trainer.dataloader:
    instructions = batch["instruction"]
    ground_truths = batch["answer"]

    # Generate responses
    responses = trainer.generate(instructions, max_new_tokens=512)

    # Compute rewards
    rewards = [
        torch.tensor(compute_reward(r, gt))
        for r, gt in zip(responses, ground_truths)
    ]

    # PPO update
    trainer.step(instructions, responses, rewards)

trainer.save_pretrained("./tulu3-final")
```

---

## Synthetic Data Generation

Public datasets alone can't provide the ~1 million instruction-response pairs needed for effective post-training.

### The naive approach fails
Ask GPT-4o to generate 100,000 diverse prompts. Problem: at this scale, GPT-4o runs out of imagination. It starts repeating similar prompts. This is called **mode collapse** — the distribution of generated prompts collapses to a narrow set of common patterns.

### Tülu 3's solution: Persona-based generation

From the paper they reference:

**Step 1:** Generate 1 billion distinct personas. Examples:
- "a moving company driver"
- "a chemical kinetics researcher"
- "a high school student preparing for SATs"
- "a nurse in a rural clinic"

**Step 2:** For each persona, generate 2–3 prompt variations that this person might ask:
- Math problem variant
- Logical reasoning variant
- Domain-specific question variant

**Result:** 1 billion personas × 3 prompts = up to **3 billion unique prompts** in theory. Much more diverse than asking a single generic "generate a question" prompt.

**Step 3:** Generate responses with GPT-4o.
Answering a question requires far less creativity than inventing the question in the first place. GPT-4o can do this reliably at scale.

### Code: Persona-based synthetic data generation

```python
from openai import OpenAI

client = OpenAI()

# Step 1: Generate a persona
def generate_persona():
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": "Generate a unique, specific persona in one sentence. "
                       "Example: 'a retired marine biologist studying coral reef recovery'"
        }],
        max_tokens=50
    )
    return response.choices[0].message.content.strip()

# Step 2: Generate a prompt from this persona
def generate_prompt(persona: str, task_type: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"You are {persona}. "
                       f"Write a realistic {task_type} question or request "
                       f"that someone with your background might ask an AI assistant. "
                       f"Only output the question, nothing else."
        }],
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

# Step 3: Generate the response
def generate_response(instruction: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": instruction
        }],
        max_tokens=512
    )
    return response.choices[0].message.content.strip()

# Build a synthetic dataset
import json

task_types = ["math problem", "logical reasoning", "domain-specific question"]
dataset = []

for _ in range(1000):  # scale up as needed
    persona = generate_persona()
    for task in task_types:
        instruction = generate_prompt(persona, task)
        response = generate_response(instruction)
        dataset.append({
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response}
            ]
        })

with open("synthetic_sft_data.jsonl", "w") as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")

print(f"Generated {len(dataset)} examples")
```

### Data Decontamination

Before using any data for training, Tülu 3 removes any training examples that overlap with the evaluation set. Otherwise the eval is contaminated — the model may have seen the test questions during training, making benchmark scores misleading.

```python
from datasets import load_dataset
from difflib import SequenceMatcher

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def decontaminate(train_dataset, eval_dataset, threshold=0.8):
    eval_instructions = [ex["instruction"] for ex in eval_dataset]
    clean = []
    for ex in train_dataset:
        too_similar = any(
            similarity(ex["instruction"], eval_inst) > threshold
            for eval_inst in eval_instructions
        )
        if not too_similar:
            clean.append(ex)
    removed = len(train_dataset) - len(clean)
    print(f"Removed {removed} contaminated examples ({removed/len(train_dataset)*100:.1f}%)")
    return clean
```

---

## Evaluation Results

Each post-training stage improves performance on average across benchmarks (knowledge, reasoning, math, coding):

```
Base (Llama 3.1)  →  +SFT  →  +DPO  →  +RLVR  →  Tülu 3
     lowest              each step adds measurable gain
```

Tülu 3 is competitive with Llama 3.1 Instruct and DeepSeek V3 (note: not R1 — the comparison is against older versions).

Open-source models still trail proprietary models (GPT-4o, Claude) but the gap is narrowing.

---

## Summary

| Stage | Data Needed | Algorithm | What It Learns |
|---|---|---|---|
| SFT | instruction + response pairs | Next token prediction | How to respond to instructions |
| DPO | instruction + chosen + rejected | Direct preference optimization | Which responses humans prefer |
| RLVR | instruction + ground truth answer | PPO with verifiable rewards | How to get correct answers on math/code/logic |

### Key Insight from Each Stage

**SFT:** Chat templates are surprisingly brittle — tiny formatting changes measurably affect model behavior. Publish all details.

**DPO:** On-policy responses (from the model itself) are better to rank than off-policy responses (from external LLMs). Tülu 3 uses a mix.

**RLVR:** Verifiable tasks (math, code) are perfect for RL because you don't need human judges — just run the code or check the answer.

**Synthetic data:** Persona-based generation solves mode collapse at scale. 1B personas × 3 prompts = 3B diverse training examples.

---

> Full recipe, code, and datasets: https://github.com/allenai/open-instruct
