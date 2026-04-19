# AI Model Training — 8 Tips That Never Go Out of Date

> These tips apply across all training stages — pre-training, instruction fine-tuning, preference fine-tuning, task fine-tuning.

---

## Tip 1 — Start From Evaluation

Before writing a single line of training code, define how you'll measure success. If you can't measure progress, there is no progress.

**But just having an eval set isn't enough — a bad eval is worse than none.**

### Example of a bad eval: Spam classifier
- Dataset: 99% not-spam, 1% spam
- A dumb classifier that always says "not spam" scores **99% accuracy**
- Looks great, completely useless

### More subtle example: Stanford NLI dataset
- Task: given two sentences (premise + hypothesis), classify as contradiction / entailment / neutral
- Random baseline should score ~33% (three equal classes)
- Researchers discovered a model shown **only the second sentence** scored **67%**
- Why? Human annotators writing contradictions overused explicit negation (e.g., "he's happy" vs "he's **not** happy"). The model learned: if the word "not" is in the second sentence → it's probably a contradiction. The eval was leaking the answer.

This benchmark was used for **years** before anyone noticed.

### What to do instead
- Use a **community-approved benchmark** — one that already has metrics, methodology, and runnable evaluation code
- Don't write your own eval code from scratch if you can avoid it — a tiny bug invalidates all your experiments
- If your use case is unique (e.g., company-specific task) and no benchmark fits perfectly, still start with an existing one. Use it until you trust your pipeline end-to-end, then swap in your own eval

---

## Tip 2 — Never Start From Scratch

There are too many moving pieces in LLM fine-tuning that all need to be configured correctly at the same time:
- Tokenization
- Prompt templates
- Model architecture
- Optimizer settings
- All of their hyperparameters

Getting all of these right through trial and error takes more time than it's worth.

**Always start from code written by a reputable source** — a Hugging Face Colab notebook, a GitHub repo accompanying a published paper, or a library like Unsloth.

Even if your goal is to learn by implementing something yourself, start from a working end-to-end pipeline first. Get it running and producing reasonable results. Then carve out the specific piece you want to rewrite and replace it while keeping everything else working. This way you always have a known-good baseline to compare against.

---

## Tip 3 — Move Away From Notebooks Quickly

Notebooks (Colab, Jupyter) are great at the start for:
- Visualizing data
- Figuring out library calls
- Checking tensor shapes
- Quickly iterating on setup

But they become dangerous once you start doing hyperparameter experiments, because of **state management**.

### The trap
You have:
- Cell 1: defines the model
- Cell 2: defines the trainer and starts training

When exploring hyperparameters, it's tempting to just change values and re-run Cell 2 only. **The model doesn't get re-initialized.** It continues training from where the last run left off — not from the base model.

This means two experiments are actually starting from different checkpoints and aren't comparable. You draw wrong conclusions about what the hyperparameter change did.

In a real example: the "tainted" run (only Cell 2 re-run) vs the "fresh" run (both cells re-run) show **very different training behavior** on a Weights & Biases dashboard — and the difference is entirely from the starting checkpoint, not the hyperparameter.

### Fix
Move to a plain Python script as soon as you're done with initial setup. A script always starts fresh when you run it. No hidden state, no surprises.

---

## Tip 4 — Understand the Training Loss

Training loss = the gap between what the model is doing and what it should be doing. Lower is better. But interpret it carefully.

### It will jiggle — that's normal
Every point on the training loss curve is computed on a **different batch** (a different small slice of your data). You're never comparing apples to apples. This variance is expected. Use the **smoothing feature** in your dashboard (W&B, TensorBoard) to see the real trend.

### Small batch sizes make this worse
As models get bigger, GPU memory fills up fast — not just from the weights but also from **activations** (every neuron's output for every example in the batch). Batch sizes get forced down, variance goes up. Diffusion models are an extreme case — a batch size of 2 can make the loss look completely random even when the model is clearly improving.

### It should not go down forever
If training loss reaches zero, the model has **memorized the training data exactly**. This is called **overfitting** or **catastrophic forgetting**:
- The model perfectly reproduces training examples
- But it forgets everything else it knew before
- Performance on anything outside training data collapses
- You've just built a very expensive storage format for your training set

---

## Tip 5 — Use a Validation Set

To know when to stop training, you need a **validation set**: 5–10% of your data that you never train on and never use for final evaluation. Its only purpose is to track how training is going.

### Why it's better than training loss for tracking

**Property 1 — Consistency:** Every point on the validation loss curve is computed on the **same fixed set**. Apples to apples. The curve is much smoother than training loss.

**Property 2 — Memorization detection:** Because the model never trains on this data, validation loss exposes overfitting. When the model starts memorizing training data:
- Training loss keeps going down (looks good)
- Validation loss flattens or starts going up (actually getting worse)

### Real example
In an overfitting experiment: training loss keeps dropping, looks like the model is improving. But validation loss eventually rises above the starting point — meaning fine-tuning has actually made the model **worse** than the base model it started from.

### When to stop
Stop when the validation loss stops improving. In a real experiment, that might be around 500–1500 steps — not at the global minimum of training loss.

---

## Tip 6 — Change One Thing at a Time

The single most productive habit in experimentation. Only change **one variable per experiment**.

### Why bundling changes is tempting but harmful
Say you're overfitting. You could:
- Lower the learning rate
- Increase warmup steps
- Increase batch size
- Increase gradient accumulation steps
- Add dropout

If you do all of these at once and the result improves, you have no idea what actually helped. If it still overfits, you have no idea where to go next.

### Real comparison
**Bad:** Two experiments with 5 changes between them — the curves are nearly impossible to interpret. You can maybe say one is slightly better, but you can't explain why or what to do next.

**Good:** Two experiments that differ only in learning rate (1e-4 vs 5e-4) — it's immediately obvious that the lower learning rate trains correctly and the higher one overfits.

### Important clarification
Changing one thing at a time doesn't mean running experiments sequentially. You can run many experiments in parallel. The rule is: **at evaluation time, each comparison should only differ in one setting.**

---

## Tip 7 — Check Boundary Conditions

When something in your training metrics doesn't make sense, you need to figure out: is this a **bug in my code** or just a **bad hyperparameter choice**? Boundary conditions let you tell the difference.

### Real example: Dropout not working
- Added 30% dropout to fight overfitting
- Training curves with and without dropout looked almost identical — suspicious
- Hypothesis: maybe the dropout setting isn't being applied at all, maybe there's a bug

**Boundary condition check:** Set dropout = 100% (all weights zeroed during training). If dropout is being applied correctly, training should completely break — loss shouldn't improve, gradients should be zero.

Result: with dropout = 100%, training loss stopped improving and all gradients were zero. ✅ Confirmed: no bug. Dropout is working. It just doesn't solve this particular overfitting problem.

**The principle:** Push a hyperparameter to an extreme value where you know exactly what the output should be. If you see that expected output → your code is correct. If you don't → there's a bug.

---

## Tip 8 — Be Patient

Training never works perfectly on the first run. Not for beginners, not for experienced researchers. Even the Transformer architecture — now universal for both language and vision — had a difficult start. Early training was unstable and it took the community a long time to find the right hyperparameter combinations.

### Be skeptical of tutorials
Many notebooks and tutorials "demonstrate" fine-tuning by just running inference on the fine-tuned model without ever showing whether it actually improves on the base model. Proving improvement takes real effort: proper evals, validation sets, controlled experiments.

---

## Summary

| Tip | One Line |
|---|---|
| Start from evaluation | A bad eval is worse than no eval — use community benchmarks |
| Never start from scratch | Too many moving pieces — start from proven working code |
| Move away from notebooks quickly | Hidden state causes misleading experiments — use Python scripts |
| Understand training loss | It jiggles naturally; zero loss = memorization = bad |
| Use a validation set | Smooth, consistent tracker; detects overfitting early |
| Change one thing at a time | Can't interpret results if multiple things changed |
| Check boundary conditions | Extreme values distinguish bugs from bad hyperparameters |
| Be patient | First run never works; that's normal for everyone |
