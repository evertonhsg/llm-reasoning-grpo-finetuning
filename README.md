# Teaching an LLM to Reason — GRPO Fine-Tuning Project

> **Fine-tuning Qwen 2.5 3B with Group Relative Policy Optimization (GRPO) to master letter-counting through step-by-step reasoning.**

---

## 📌 Overview

This project demonstrates how to fine-tune a large language model (LLM) using **reinforcement learning** — specifically the **GRPO (Group Relative Policy Optimization)** algorithm — to teach it a deceptively hard cognitive task: **counting how many times a specific letter appears in a word**.

While trivial for humans who spell things out manually, LLMs struggle with this task when asked directly. This project shows how **structured reasoning + reward-based training** transforms a model that guesses into one that reliably counts — letter by letter.

---

## 🧠 The Problem

Ask a vanilla LLM: *"How many g's are in the word 'engage'?"*

It will likely say **1** — when the correct answer is **2**.

The root cause? LLMs process tokens holistically rather than character-by-character. Without explicit step-by-step reasoning, they guess rather than count.

---

## 🔬 Approach

The project follows a three-stage methodology:

### 1. Prompt Engineering (Baseline)
Before any training, the model is guided using:
- **Chain-of-Thought (CoT) prompting** — instructing the model to spell the word letter by letter
- **Few-shot examples** — providing a worked example to show the expected format

This improves results but remains unreliable.

### 2. Dataset Creation
A synthetic dataset is generated from a curated word list (4–8+ letter words). For each word, every unique letter (and several absent letters) generates a labelled sample:

```
Question: How many of the letter "g" are there in the word "engage"?
Expected answer: 2
```

The dataset uses the HuggingFace `datasets` library and is structured with system + user prompts ready for training.

### 3. GRPO Fine-Tuning with Reward Functions
The model is fine-tuned using **GRPO** — a reinforcement learning technique that improves the model by comparing groups of its own completions against custom reward functions. Five reward functions guide training:

| Reward Function | What It Rewards |
|---|---|
| `numbering_reward_func` | Correctly numbered bullet points (1, 2, 3…) |
| `spelling_reward_func` | Accurate letter-by-letter spelling of the word |
| `counting_reward_func` | Correct running count at each step |
| `format_reward_func` | Using the required `<reasoning>` / `<answer>` XML tags |
| `correct_answer_reward_func` | Getting the final numeric answer exactly right |

---

## 🏗️ Architecture & Tools

| Component | Detail |
|---|---|
| **Base Model** | Qwen 2.5 3B Instruct |
| **Fine-tuning Method** | LoRA (Low-Rank Adaptation) via Unsloth |
| **Training Algorithm** | GRPO (Group Relative Policy Optimization) |
| **Inference** | vLLM for fast generation |
| **Dataset** | Custom synthetic letter-counting dataset |
| **Training Duration** | ~5 steps (quick check) → ~200 steps (~60 min full run) |
| **GPU Requirement** | ≥ 15 GB VRAM (e.g. NVIDIA T4) |

---

## 📂 Project Structure

```
gen_ai_fundamentals_project_everton.gomes.ipynb   # Main project notebook
README.md                                          # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- NVIDIA GPU with ≥ 15 GB VRAM
- CUDA installed

### Installation

The notebook installs its own dependencies. The key packages are:

```bash
pip install unsloth vllm trl datasets
```

### Running the Notebook

1. Open the notebook in **Jupyter Lab** or **Google Colab** (GPU runtime required)
2. Run cells sequentially — each section builds on the previous
3. Complete the `TODO` sections marked with `**********` — these are the learning exercises

---

## 📖 Notebook Walkthrough

| Section | Description |
|---|---|
| **Setup** | Install dependencies, load Qwen 2.5 3B with Unsloth + LoRA |
| **Prompt Engineering** | Experiment with CoT and few-shot prompts on the base model |
| **Dataset Creation** | Build a synthetic letter-counting dataset with HuggingFace `datasets` |
| **Reward Functions** | Define 5 reward functions covering numbering, spelling, counting, format, and correctness |
| **Quick Train (5 steps)** | Sanity-check run to validate reward functions and prompts |
| **Full Train (~200 steps)** | ~60-minute GRPO training session |
| **Evaluation** | Compare old vs. fine-tuned model side-by-side on counting and general knowledge |

---

## 📊 Expected Results

After ~200 training steps, you should observe:
- The **total reward** increasing over time
- The **`correct_answer_reward_func` mean** trending upward
- The fine-tuned model spelling words letter-by-letter and reaching the right count more consistently than the base model
- The model retaining general knowledge (e.g., capital cities) — confirming it didn't catastrophically forget

---

## 💡 Key Concepts Demonstrated

- **GRPO** — Reinforcement learning for LLMs without a separate critic model
- **LoRA fine-tuning** — Parameter-efficient adaptation using low-rank weight updates
- **Reward shaping** — Using multiple partial-credit rewards to guide structured reasoning
- **Chain-of-Thought prompting** — Eliciting step-by-step reasoning from LLMs
- **LLMOps** — Saving LoRA adapters and loading them for inference with vLLM

---

## 🧾 Output Format

The model is trained to respond in this structured XML format:

```xml
<reasoning>
Counting the number of g's in the word engage
1. e - 0 g's so far
2. n - 0 g's so far
3. g - 1 g's so far
4. a - 1 g's so far
5. g - 2 g's so far
6. e - 2 g's so far
</reasoning>
<answer>
2
</answer>
```

---

## 👤 Author

**Everton Gomes**
Senior Engineer — AI & Data | FedEx Europe
[GitHub](https://github.com/evertonhsg)

---

## 📚 References

- [Unsloth Documentation](https://docs.unsloth.ai)
- [TRL — GRPO Trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets)
- [Qwen 2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [vLLM Documentation](https://docs.vllm.ai)

---

*Part of the Generative AI Fundamentals curriculum.*
