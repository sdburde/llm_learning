# DeepSeek R1 — Technical Notes

---

## Why This Matters

DeepSeek R1 reached parity with OpenAI's o1 reasoning model — which was considered the best on the market — at a fraction of the cost, open-sourced it, and published the full technical report. The US stock market lost $1 trillion in a single day because of this.

The core question: how did they do it so cheaply and openly?

Two directions:
1. **Scaling with compute** — at both training and inference time
2. **Implementation efficiency** — via smarter RL and distillation

---

## Part 1 — The Problem: Data as a Wall

For the past decade, AI progress meant one thing: train bigger models on more data. More data = smarter model.

But at the end of 2024, Ilya Sutskever (former Chief Scientist at OpenAI) said:

> "Pre-training as we know it will end. Data is not growing. We have but one internet."

We have essentially consumed the entire internet for training. You can't just add more data. So the industry needed a **new dimension of growth** — and that dimension is **compute**. Not more data, but more thinking.

OpenAI pointed at this direction with their o1 model but did it expensively and privately. DeepSeek did it efficiently and publicly.

Cost comparison:
- GPT-4 training: ~$80–100 million
- Unreleased GPT-5: reportedly ~$500 million
- DeepSeek R1 training: **less than $6 million**

---

## Part 2 — How LLMs Are Normally Trained

Before understanding DeepSeek's innovations, here's the standard pipeline:

**Step 1 — Pre-training**
Start with a randomly initialized model that can't generate coherent text. Show it the entirety of the internet. Train it to predict the next word in every piece of text. After this you have a **base model** — it has a lot of world knowledge but an inconvenient API: it just completes your sentence rather than having a conversation.

**Step 2 — Instruction Fine-Tuning (SFT)**
Train the model on pairs of instructions + desired human-written responses. This teaches the model to behave conversationally and follow directions. Some models stop here.

**Step 3 — Reinforcement Learning from Human Feedback (RLHF)**
You've probably seen ChatGPT give two answers and ask which you prefer. That's collecting human preference data. The model is refined to align with what humans like. This step is powerful but bottlenecked — you need humans for every piece of feedback.

**The bottleneck:** Both fine-tuning and RLHF are limited by human annotators. You can't scale them infinitely. DeepSeek's goal: remove the human bottleneck by having AI generate the training data instead.

---

## Part 3 — DeepSeek R1's Training (Where Things Get Different)

DeepSeek doesn't start from scratch. They use **DeepSeek V3 Base** as the starting point — an already-pretrained model. The two fine-tuning stages are preserved but redesigned.

The core philosophy: **replace human annotators with AI-generated data**.

The question then becomes: how do you train the AI that generates the training data? You don't have labeled examples — that's the whole reason you wanted a generator in the first place. This is the breakthrough.

---

## Part 4 — Reasoning-Oriented Reinforcement Learning

### Quick RL Recap

Reinforcement learning has:
- An **agent** — makes decisions
- An **environment** — the world the agent acts in
- A **policy** — the agent's brain, determines what action to take next
- A **reward** — a number from the environment saying how good or bad the last action was

The agent takes actions, gets rewards, and gradually updates its policy to get more rewards. Used in robots, chess, Go, Minecraft — now also LLMs.

### RL Applied to LLMs

- **Policy** = the LLM itself (this is what gets trained)
- **Action** = generating the next token
- **State** = the prompt + all tokens generated so far
- **Reward** = a number saying how good the output was

The key difference from standard supervised training: in standard training, the model sees the correct next word and learns to copy it. In RL, **there is no correct token shown**. The model only gets a reward signal at the end based on how good the final answer was. It figures out the path on its own.

### Why Reasoning Tasks Are Perfect for RL

Math and coding tasks are easy to verify automatically:
- Run the code in an interpreter
- Check syntax with a linter
- Run unit tests
- Compare a math answer to the known solution

Example: ask the model to write a Python function that returns the first N Fibonacci numbers. Run the unit tests. 1 out of 3 pass → reward = 1/3. The model gradually improves at writing correct code through this feedback loop — no human needed.

This is why DeepSeek named the process **Reasoning-Oriented Reinforcement Learning** — reasoning tasks (math, CS, logic) are exactly the kind of tasks where RL can give clean, automatic, verifiable rewards.

---

## Part 5 — R1-Zero: The Surprising Experiment

Before building R1, DeepSeek built **R1-Zero** — a model trained using RL alone, with zero supervised data. No human-written examples, no instruction fine-tuning. Just: here are math and coding problems, here's a reward when you get them right. Go.

The goal was modest: generate some training data for later fine-tuning.

The result was surprising: **R1-Zero turned out to be on par with OpenAI o1 on math benchmarks**.

In the training plots, R1-Zero's performance climbs through RL and by the end matches or beats o1 on the same metrics.

Why is this a big deal? For the first time, through **RL alone with no supervised data at all**, a model was brought from zero to ~90% performance on very hard reasoning tasks. No labeled examples. No human annotators. Just compute and a reward function.

This is the green energy of AI — compute scaling, not data scaling.

> R1-Zero showed us that data is not the only path. RL + compute can get you there too.

---

## Part 6 — The Full R1 Training Pipeline

Despite how powerful R1-Zero was, DeepSeek wanted to go further. They found that combining RL with supervised learning gives the best result. So the actual R1 training has more stages:

```
DeepSeek V3 Base
      │
      ▼
[Stage 1] Cold-Start Fine-Tuning
Generate "cold start" data using a good LLM
Fine-tune V3 Base on it → intermediate checkpoint
      │
      ▼
[Stage 2] Reasoning-Oriented RL
Apply RL on reasoning tasks (math, CS, logic)
Uses GRPO — more on this below
→ This becomes the DATA GENERATOR (R1-Zero equivalent)
      │
      ▼
[Stage 3] Use the Data Generator
Generate 600,000 instruction-response pairs for reasoning tasks
Pull 200,000 more non-reasoning pairs from DeepSeek V3
(creative writing, poems, marketing copy, etc.)
Total: 800,000 data points
      │
      ▼
[Stage 4] Instruction Fine-Tuning on 800K pairs
Fine-tune V3 Base again → instruction-tuned model
      │
      ▼
[Stage 5] Second RL Round (Human Preference)
For reasoning tasks → automatic rewards (linters, unit tests)
For creative tasks → a reward model trained on human preferences
Uses DeepSeek V3 as the judge for vibe-based tasks (helpfulness, language consistency)
      │
      ▼
Final Model: DeepSeek R1
```

The key insight here is that the human bottleneck is bypassed at the data generation stage. 600,000 reasoning examples that would have required enormous human effort were generated automatically. Only the preference RL stage still needs human input — and even there, a model trained on limited human preferences does the heavy lifting.

---

## Part 7 — Chain-of-Thought (Inference-Time Scaling)

R1 doesn't just scale at training time. It also uses more compute at **inference time** through Chain-of-Thought.

The idea is simple: instead of jumping to an answer, explicitly ask the model to write out its intermediate reasoning steps first.

Example without CoT:
```
Q: A car travels 60 miles in 2 hours. What is its speed?
A: 30 mph
```

Example with CoT:
```
Q: A car travels 60 miles in 2 hours. What is its speed?
<think>
Speed = distance ÷ time
Distance = 60 miles
Time = 2 hours
Speed = 60 ÷ 2 = 30
</think>
A: 30 mph
```

Writing out reasoning steps before the answer consistently improves the quality of the final answer. The model is forced to produce arguments that actually support the conclusion.

For R1 to do this at inference time, it needs to see this kind of data during training:
- In the fine-tuning stage: ask the data-generating LLM to include chains of thought in everything it produces
- In the RL stage: wrap every prompt with an instruction to include reasoning inside `<think>` tags

There is a powerful synergy between RL and CoT: in RL, the model never sees the correct next token — it only gets a reward when the final answer is right. This means the model can't just learn to produce good-sounding explanations that don't contribute to the answer. The reasoning has to actually help reach the correct answer, or it doesn't get rewarded. This forces genuinely useful thinking, not performative thinking.

---

## Part 8 — GRPO (Implementation Efficiency)

RL is notoriously expensive to implement. One reason: standard RL for LLMs requires training **two models simultaneously**.

**Model 1 — The Policy Model**
The LLM itself. Decides what token to generate next. This is what we actually want at the end.

**Model 2 — The Value Function**
A second model, just as large as the policy model. Its job: estimate how difficult the current instruction is. Why do you need this? Because if a task is very hard, getting a good reward means a lot — you should learn a lot from it. If the task is trivially easy, a good reward isn't surprising and shouldn't change your behavior much. The value function calibrates how much to learn from each reward.

The problem: if the LLM is 671 billion parameters, the value function adds another 671 billion parameters. You've doubled your compute and memory requirements.

**DeepSeek's solution — GRPO (Group Relative Policy Optimization):**

Instead of maintaining a value function model, use a hack:

For each instruction, instead of generating **one** response, generate **N** responses (e.g. 16 at once).

Each of the N responses gets a reward. The difficulty of the instruction is estimated as simply **the average reward across the group**. Above-average responses get positive reinforcement. Below-average responses get negative reinforcement.

This is a crude approximation of what the value function was doing — but it's good enough. And it eliminates the need for a second 671-billion-parameter model entirely.

Result: the same quality of RL training at roughly half the memory and compute cost.

---

## Part 9 — Distillation (Making It Accessible)

R1 is a 671 billion parameter model with a **Mixture-of-Experts (MoE)** architecture. In MoE, the network has many parallel paths ("experts"). Each input takes a specific path through a subset of experts, leaving the rest untouched. This is why it's called a sparse model — not every parameter is used for every input. It lets you have a massive number of total parameters without activating all of them, optimizing for quality while keeping per-token compute manageable.

But 671 billion total parameters is still enormous. Most people can't run this on their own hardware.

**Distillation:** Use the large R1 model as a teacher. Have it generate a large dataset of high-quality outputs. Use that dataset to fine-tune much smaller, dense student models — like Llama variants. The student learns to mimic the teacher's behavior without needing the teacher's architecture.

DeepSeek released distilled models at sizes: 1B, 7B, 8B, 14B, 32B, 70B.

You pick whichever fits your hardware, accepting the trade-off between size and quality. A 7B distilled model doesn't match R1, but it reasons far better than a 7B model trained from scratch without this process.

---

## Summary — The Full Picture

| Innovation | What It Does | Why It Matters |
|---|---|---|
| Reasoning-Oriented RL | Uses verifiable tasks (math/code) to generate automatic rewards | Removes need for human annotators in data generation |
| R1-Zero | Pure RL, no supervised data, still matches o1 | Proves compute alone can develop strong reasoning |
| Cold-Start SFT | Small supervised seed before RL | Makes RL more stable and outputs more readable |
| Chain-of-Thought | Model thinks step-by-step before answering | More compute at inference → better answers |
| GRPO | N outputs per question, average = baseline | Removes 671B value function model, halves RL cost |
| Distillation | Small models learn from R1's outputs | Makes frontier reasoning accessible on consumer hardware |

---

## The Bigger Picture

Ilya Sutskever said data is the fossil fuel of AI — finite, and we're running out of it.

R1-Zero's result suggests reinforcement learning and compute might be the green energy of AI — a renewable source of improvement that doesn't depend on consuming more data.

DeepSeek made profound discoveries and did so in public. In a field where most labs keep everything secret, that openness — combined with the efficiency — is what made headlines.

---

> Correction from the video: The $6M cost cited was for DeepSeek-V3, not R1. R1's training cost is not publicly known.
