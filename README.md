# Capacity-Gated Forgetting in LoRA Fine-Tuning
### Rank, Proximity, and Endogenous Replay in Medical LLMs

> Preliminary work submitted to ICML 2026. Do not distribute.

---

## Overview

This repository contains all experiments, results, and analysis notebooks for the paper **"Capacity-Gated Forgetting in LoRA Fine-Tuning: Rank, Proximity, and Endogenous Replay in Medical LLMs"**.

We run a controlled 11-experiment battery fine-tuning **Qwen3.5-9B-Base** on **MedQA-USMLE** with QLoRA, evaluating forgetting across all 57 MMLU subjects. The paper makes two main contributions:

- **Capacity-Gated Forgetting (CGF):** A falsifiable hypothesis that the capacity ratio `ρ = r_LoRA / d*(D_ft)` predicts two qualitatively distinct forgetting regimes — uniform forgetting below a critical threshold `ρ*`, and proximity-structured forgetting above it.
- **Endogenous Replay (SSRA):** A replay method in which the rehearsal corpus is generated from the base model itself. It achieves an **80% reduction in forgetting** over no replay and ~50% over real-text replay at matched sample budget, without sacrificing MedQA target accuracy.

---

## Key Results

| Experiment | LoRA Rank | Steps | Replay | Mean Forgetting (↓) |
|---|---|---|---|---|
| E01 | 4 | 500 | none | 0.163 |
| E02 | 16 | 500 | none | 0.352 |
| E03 | 64 | 500 | none | 0.540 |
| E04 | 128 | 500 | none | 0.545 |
| E09 | 16 | 500 | 50 real | 0.180 |
| E10 | 16 | 500 | 100 real | 0.142 |
| **E11** | **16** | **500** | **100 endogenous** | **0.070** |

- **Rank dominates forgetting:** Friedman χ²₃ = 166.5, p < 10⁻³⁵
- **Proximity gap appears only above rank 16:** The medical/non-medical forgetting gap flips sign between r=16 and r=64, reconciling Luo et al. (2024) and Biderman et al. (2024)
- **Endogenous Replay is Pareto-dominant:** E11 achieves 78.5% MedQA accuracy vs 78.0% (no replay) while reducing forgetting from 0.352 → 0.070

---

## Repository Structure

```
Major-Project-Group8/
├── experiments/
│   ├── e00-notebook.ipynb        # E00 (baseline) + E01–E04 (rank ablation)
│   ├── e09-notebook.ipynb        # E09 (50 real replay)
│   ├── e10-e11-notebook.ipynb    # E10 (100 real replay) + E11 (endogenous replay)
│   ├── gemma.ipynb               # K1 validation: Gemma 4 9B replication
│   ├── llama.ipynb               # Cross-family replication experiments
│   └── minilm-validations.ipynb  # H1 proximity scoring with MiniLM
│
└── results/
    ├── all_results.csv            # Master results table (E01–E11, all 57 MMLU subjects)
    ├── h1_proximity_per_exp.csv   # H1/H3 proximity stats per experiment
    └── h1_proximity_scores.csv    # MiniLM subject-to-corpus similarity scores
```

---

## Setup

### Requirements

```bash
pip install transformers peft trl bitsandbytes accelerate datasets \
            sentence-transformers scipy scikit-learn matplotlib seaborn
```

> **Note:** Qwen3.5 architecture requires installing `transformers` from the git main branch (as of early 2026, it is not yet in a PyPI release):
> ```bash
> pip install git+https://github.com/huggingface/transformers.git
> ```

### Hardware

All experiments were run on a single **Tesla T4 (15.6 GB VRAM)** via Kaggle. The K0 target-accuracy check used an **A100-40GB**. The model is loaded in 4-bit NF4 (QLoRA) to fit the 9B base within T4 memory.

### Hugging Face Authentication

A Hugging Face token is required to download Qwen3.5-9B-Base. Set it as an environment variable — **never hardcode tokens in notebooks**:

```python
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])
```

---

## Experiment Design

### Model & Data

| Component | Detail |
|---|---|
| Base model | `Qwen/Qwen3.5-9B-Base` (hybrid DeltaNet + Attention, 32 layers) |
| Quantisation | 4-bit NF4, double quantisation, bf16 compute (QLoRA) |
| Fine-tuning data | `GBaker/MedQA-USMLE-4-options-hf` — 4,096 examples, answer-token loss only |
| Evaluation | MMLU 57 subjects via `lm-evaluation-harness` v0.4.4, 5-shot, temperature 0 |
| Forgetting metric | `f_s = acc_base(s) − acc_ft(s)` (positive = forgot) |

### LoRA Configuration

- **Target modules:** `q_proj, k_proj, v_proj, o_proj` (attention) + `gate_proj, up_proj` (MLP)
- **α = r**, dropout = 0.05, batch size 4, gradient accumulation 4
- **LR:** 2×10⁻⁴ with cosine schedule, 10 warmup steps, seed 42

### Experiment Summary

| ID | Variable tested | Key finding |
|---|---|---|
| E00 | Baseline (no fine-tuning) | Mean MMLU = 0.776 |
| E01–E04 | Rank ablation (r = 4, 16, 64, 128) | Rank dominates; proximity gap appears only at r ≥ 64 |
| E05/E05b | Domain controls (GSM8K, CodeAlpaca) | Forgetting is domain-specific, not a training artefact |
| E06–E08 | Step ablation (100, 200, 500, 1000 steps) | Non-monotonic; 500 steps optimal due to cosine LR trough |
| E09–E11 | Replay ablation (50 real, 100 real, 100 endogenous) | Endogenous Replay: 80% reduction over no replay |

---

## Capacity-Gated Forgetting (CGF)

The capacity ratio is defined as:

```
ρ = r_LoRA / d*(D_ft)
```

where `d*(D_ft)` is the intrinsic dimension of the fine-tuning task. CGF predicts two regimes:

- **Sub-critical (ρ < ρ\*):** Forgetting is high in aggregate but uniformly distributed — no subject cluster is disproportionately harmed. Consistent with r = 4 and r = 16 in our data.
- **Super-critical (ρ ≥ ρ\*):** The adapter has capacity slack beyond the task subspace; forgetting inherits the task's topical structure. Medical MMLU subjects forget more than non-medical ones. Consistent with r = 64 and r = 128.

The transition is observed between r = 16 and r = 64, implying `d*(MedQA-USMLE) ∈ (16, 64)` for Qwen3.5-9B-Base.

---

## Endogenous Replay

Endogenous Replay (SSRA) generates the rehearsal corpus directly from the base model:

```python
# Algorithm: Endogenous Replay (SSRA)
R = []
for prompt in anchor_prompts:
    y = sample(base_model, prompt)   # single forward-sampling pass
    R.append((prompt, y))
train(lora_adapter, D_ft + R)
```

By construction, endogenous samples minimise the KL anchoring objective `E[KL(p_θ0(·|x) ‖ p_θ(·|x))]` exactly, making the method a sample-based anisotropic Fisher penalty — strictly more efficient per sample than real-text replay.

**K0 check (Pareto dominance at r=16, 500 steps):**

| Condition | MedQA Accuracy | MMLU Forgetting |
|---|---|---|
| E02 — no replay | 78.0% | 0.352 |
| E10 — 100 real | 77.5% | 0.142 |
| **E11 — 100 endogenous** | **78.5%** | **0.070** |

---

## Planned Validation Checks (K1–K4)

| Check | Description | Falsified by |
|---|---|---|
| K1 (Gemma) | Replicate E01–E04 on `google/gemma-4-9b` | Critical rank outside [16, 64] |
| K2 (Task complexity) | Simpler tasks should shift critical rank downward | No rank-window shift |
| K3 (Anchor scaling) | Replay benefit should improve sublinearly with budget | No monotone budget or diversity effect |
| K4 (CL baselines) | Endogenous Replay should match or beat EWC, LwF, L2-SP | Any baseline below 0.070 forgetting |

---

## Limitations

- **Single seed (42):** All runs use seed 42. Multi-seed replication (seeds 0 and 7) for E02/E09–E11 is the highest-priority follow-up.
- **Single base model family:** Qwen3.5-9B-Base only. The hybrid DeltaNet/Attention architecture is atypical; K1 addresses cross-family replication.
- **d\*(D_ft) not directly measured:** CGF is stated in terms of intrinsic dimension but this quantity is not estimated in the current submission (Open Problem 1).
- **No standard CL baselines yet:** EWC, LwF, and L2-SP comparisons are deferred to K4.
- **K0 target accuracy uses N_test = 100:** SE ≈ ±5pp per condition; doubling to N_test = 200 is planned.

---

## Citation

```bibtex
@inproceedings{cgf2026,
  title     = {Capacity-Gated Forgetting in LoRA Fine-Tuning: Rank, Proximity,
               and Endogenous Replay in Medical LLMs},
  author    = {Anonymous Authors},
  booktitle = {International Conference on Machine Learning (under review)},
  year      = {2026}
}
```

---

## References

- Biderman et al. (2024). *LoRA learns less and forgets less.* TMLR.
- Dettmers et al. (2023). *QLoRA: Efficient finetuning of quantized LLMs.* NeurIPS.
- Hu et al. (2022). *LoRA: Low-rank adaptation of large language models.* ICLR.
- Kirkpatrick et al. (2017). *Overcoming catastrophic forgetting in neural networks.* PNAS.
- Luo et al. (2024). *An empirical study of catastrophic forgetting in large language models during continual fine-tuning.* arXiv.
- Steele (2026). *Subspace geometry governs catastrophic forgetting in low-rank adaptation.* arXiv.
- Hendrycks et al. (2021). *Measuring massive multitask language understanding.* ICLR.
