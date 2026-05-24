"""
=============================================================================
NOTEBOOK A  —  Baseline (E00) + Rank Ablation (E01–E04)
Optimised for: Jarvis Labs RTX A5000 Pro (24 GB VRAM)

SPEED FIXES vs original
------------------------
1. eval_subject: batched inference (batch_size=16) instead of 1-at-a-time
   → ~10-15x faster eval (was ~5700 serial generate calls across 57 subjects)
2. per_device_train_batch_size: 1 → 4  (GPU was at 27% util)
3. gradient_accumulation_steps: 16 → 4  (same effective batch=16, fewer idle gaps)

Experiments
-----------
E00  Baseline MMLU — no fine-tuning, all 57 subjects
E01  LoRA rank=4,   MedQA, 500 steps
E02  LoRA rank=16,  MedQA, 500 steps   ← PRIMARY condition
E03  LoRA rank=64,  MedQA, 500 steps
E04  LoRA rank=128, MedQA, 500 steps

Usage
-----
  python notebook_A_baseline_rank.py               # run everything
  python notebook_A_baseline_rank.py --exp E00     # baseline only
  python notebook_A_baseline_rank.py --exp E01,E02 # specific ranks
  python notebook_A_baseline_rank.py --skip-e00    # skip if baseline exists
=============================================================================
"""

import argparse
import gc
import logging
import os
import re
import shutil
import time
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from scipy.stats import pearsonr, ttest_ind
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR     = Path("./lora_forgetting_research")
RESULTS_DIR  = BASE_DIR / "results"
ADAPTERS_DIR = BASE_DIR / "adapters"
RESULTS_CSV  = RESULTS_DIR / "all_results.csv"
BASELINE_CSV = RESULTS_DIR / "E00_baseline.csv"

for d in [RESULTS_DIR, ADAPTERS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MODEL_9B     = "Qwen/Qwen3.5-9B"
SEED_PRIMARY = 42

# ── A5000 Pro training hyperparameters (FIXED) ────────────────────────────────
# Original had per_device_train_batch_size=1, gradient_accumulation_steps=16
# → GPU sat at 27% util. Fixed: batch=4, accum=4 (same effective batch of 16).
A5000 = dict(
    per_device_train_batch_size=4,    # was 1 — A5000 has the headroom
    gradient_accumulation_steps=4,    # was 16 — fewer idle gaps, same eff. batch
    learning_rate=2e-4,
    fp16=False,
    bf16=True,                        # A5000 native bf16
    max_seq_length=1024,
    warmup_ratio=0.05,
    save_steps=50,
    save_total_limit=2,
    logging_steps=25,
    dataloader_num_workers=4,
    report_to="none",
)

# ── Eval batch size (separate from training) ──────────────────────────────────
EVAL_BATCH_SIZE = 16   # questions per forward pass during MMLU eval

MMLU_SUBJECTS = [
    "abstract_algebra","anatomy","astronomy","business_ethics",
    "clinical_knowledge","college_biology","college_chemistry",
    "college_computer_science","college_mathematics","college_medicine",
    "college_physics","computer_security","conceptual_physics",
    "econometrics","electrical_engineering","elementary_mathematics",
    "formal_logic","global_facts","high_school_biology",
    "high_school_chemistry","high_school_computer_science",
    "high_school_european_history","high_school_geography",
    "high_school_government_and_politics","high_school_macroeconomics",
    "high_school_mathematics","high_school_microeconomics",
    "high_school_physics","high_school_psychology","high_school_statistics",
    "high_school_us_history","high_school_world_history","human_aging",
    "human_sexuality","international_law","jurisprudence","logical_fallacies",
    "machine_learning","management","marketing","medical_genetics",
    "miscellaneous","moral_disputes","moral_scenarios","nutrition",
    "philosophy","prehistory","professional_accounting","professional_law",
    "professional_medicine","professional_psychology","public_relations",
    "security_studies","sociology","us_foreign_policy","virology",
    "world_religions",
]

MEDICAL_MMLU = {
    "anatomy","clinical_knowledge","college_biology","college_medicine",
    "high_school_biology","medical_genetics","professional_medicine",
    "virology","human_aging",
}

DOMAIN_DESCRIPTIONS = {
    "medqa":      "medicine clinical knowledge anatomy pharmacology pathology "
                  "diagnosis treatment disease symptoms",
    "gsm8k":      "mathematics arithmetic algebra word problems numerical computation",
    "codealpaca": "programming code software functions algorithms debugging python",
}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def log_result(exp_id, d):
    row = {"exp_id": exp_id, "timestamp": datetime.now().isoformat(), **d}
    df  = pd.DataFrame([row])
    if RESULTS_CSV.exists():
        df.to_csv(RESULTS_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(RESULTS_CSV, index=False)

def already_done(exp_id, n=50):
    if not RESULTS_CSV.exists(): return False
    df = pd.read_csv(RESULTS_CSV)
    done = df[(df["exp_id"]==exp_id) & (df["subject"].isin(MMLU_SUBJECTS))]
    if len(done) >= n:
        log.info(f"  ✅ {exp_id} already complete ({len(done)} subjects). Skipping.")
        return True
    return False

def gpu_status():
    if not torch.cuda.is_available(): return "CPU"
    f, t = torch.cuda.mem_get_info(0)
    return f"{torch.cuda.get_device_name(0)} | {t/1e9:.0f}GB total | {f/1e9:.1f}GB free"

def free_gpu(*objs):
    for o in objs:
        if o is not None: del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize(); time.sleep(2)
        f, _ = torch.cuda.mem_get_info(0)
        log.info(f"  GPU cleared — {f/1e9:.1f} GB free")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(model_id):
    log.info(f"Loading {model_id} (nf4 + bf16) ...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    kw = {}
    try:
        import flash_attn  # noqa
        kw["attn_implementation"] = "flash_attention_2"
        log.info("  Flash Attention 2 ✓")
    except ImportError:
        pass
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb,
        device_map="auto", trust_remote_code=True, **kw)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # left-pad for batched generation
    model.config.use_cache = False
    log.info(f"  Loaded {model.num_parameters()/1e9:.2f}B params | {gpu_status()}")
    return model, tok

# ═══════════════════════════════════════════════════════════════════════════════
# EVAL  (BATCHED — key speed fix)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_prompt(ex, tok):
    """Build chat-formatted prompt string for one MMLU example."""
    ch = "\n".join([f"{l}) {c}" for l,c in zip("ABCD", ex["choices"])])
    prompt = (f"Answer with only A, B, C, or D.\n\n"
              f"Question: {ex['question']}\n{ch}\n\nAnswer:")
    msgs = [{"role": "user", "content": prompt}]
    try:
        txt = tok.apply_chat_template(msgs, tokenize=False,
                                      add_generation_prompt=True,
                                      enable_thinking=False)
    except TypeError:
        try:
            txt = tok.apply_chat_template(
                [{"role":"user","content":prompt+" /no_think"}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            txt = prompt
    return txt

def eval_subject(model, tok, subject, max_samples=None, batch_size=EVAL_BATCH_SIZE):
    """
    Evaluate one MMLU subject using BATCHED inference.

    Previously: one model.generate() call per question  → GPU at ~27% util
    Now:        batch_size=16 questions per call         → much higher util
    """
    try:
        ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
    except Exception as e:
        log.warning(f"  {subject}: {e}"); return float("nan")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    lm    = {0:"A", 1:"B", 2:"C", 3:"D"}
    items = list(ds)
    correct = total = 0

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        prompts = [_build_prompt(ex, tok) for ex in batch]

        enc = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=A5000["max_seq_length"],
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )

        # decode only the newly generated tokens
        input_len = enc["input_ids"].shape[1]
        for j, ex in enumerate(batch):
            gen = tok.decode(out[j, input_len:], skip_special_tokens=True).strip().upper()
            m   = re.search(r"[ABCD]", gen)
            correct += int((m.group(0) if m else "X") == lm[ex["answer"]])
            total   += 1

    return correct / total if total else float("nan")

def eval_all(model, tok, tag="", batch_size=16):
    res = {}
    for i, s in enumerate(MMLU_SUBJECTS):
        res[s] = eval_subject(model, tok, s, batch_size=batch_size)
        if (i+1) % 5 == 0 or (i+1) == len(MMLU_SUBJECTS):
            log.info(f"  [{i+1:2d}/57] {s}: {res[s]:.3f}" + (f" [{tag}]" if tag else ""))
    return res

# ═══════════════════════════════════════════════════════════════════════════════
# FINE-TUNE
# ═══════════════════════════════════════════════════════════════════════════════

class CkptCB(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        log.info(f"  ckpt → step {state.global_step}")

def finetune(model_id, dataset, exp_id, rank=16, steps=500, seed=42):
    """LoRA fine-tune with automatic checkpoint resume (every 50 steps)."""
    set_seed(seed)
    out = str(ADAPTERS_DIR / exp_id)
    os.makedirs(out, exist_ok=True)

    ckpts = sorted([d for d in os.listdir(out) if d.startswith("checkpoint-")])
    resume = None
    if ckpts:
        step = int(ckpts[-1].split("-")[-1])
        resume = os.path.join(out, ckpts[-1])
        if step >= steps:
            log.info(f"✅ {exp_id}: training done (step {step}). Loading adapter.")
            m, t = load_model(model_id)
            m = PeftModel.from_pretrained(m, out)
            m.config.use_cache = False
            return m, t
        log.info(f"♻️  {exp_id}: resuming from step {step}/{steps} (rank={rank})")
    else:
        log.info(f"🔧 {exp_id} | rank={rank} | steps={steps} | seed={seed}")

    m, t = load_model(model_id)
    # left-pad was set for eval; switch to right for training
    t.padding_side = "right"
    m = get_peft_model(m, LoraConfig(
        r=rank, lora_alpha=rank*2,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM))
    m.print_trainable_parameters()

    SFTTrainer(
        model=m, processing_class=t, train_dataset=dataset, callbacks=[CkptCB()],
        args=SFTConfig(
            output_dir=out, max_steps=steps,
            per_device_train_batch_size=A5000["per_device_train_batch_size"],
            gradient_accumulation_steps=A5000["gradient_accumulation_steps"],
            warmup_ratio=A5000["warmup_ratio"],
            learning_rate=A5000["learning_rate"],
            fp16=A5000["fp16"], bf16=A5000["bf16"],
            logging_steps=A5000["logging_steps"],
            save_strategy="steps", save_steps=A5000["save_steps"],
            save_total_limit=A5000["save_total_limit"],
            dataset_text_field="text",
            dataloader_num_workers=A5000["dataloader_num_workers"],
            seed=seed, report_to="none",
        ),
    ).train(resume_from_checkpoint=resume)

    # restore left-pad for subsequent eval
    t.padding_side = "left"
    m.save_pretrained(out); t.save_pretrained(out)
    log.info(f"💾 Adapter saved: {out}")
    return m, t

# ═══════════════════════════════════════════════════════════════════════════════
# PROXIMITY TEST
# ═══════════════════════════════════════════════════════════════════════════════

_emb = None
def embedder():
    global _emb
    if _emb is None: _emb = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _emb

def proximity_test(domain_key, fgt_dict, exp_id):
    e   = embedder()
    dv  = e.encode(DOMAIN_DESCRIPTIONS[domain_key], normalize_embeddings=True)
    svs = e.encode([s.replace("_"," ") for s in MMLU_SUBJECTS],
                   normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    sims  = {s: float(np.dot(dv,v)) for s,v in zip(MMLU_SUBJECTS,svs)}
    valid = [s for s in MMLU_SUBJECTS if s in fgt_dict and not np.isnan(fgt_dict[s])]
    r, p  = pearsonr([sims[s] for s in valid], [fgt_dict[s] for s in valid])
    prox  = [fgt_dict[s] for s in valid if s in MEDICAL_MMLU]
    dist  = [fgt_dict[s] for s in valid if s not in MEDICAL_MMLU]
    ts, tp = (ttest_ind(prox,dist) if len(prox)>1 and len(dist)>1
              else (float("nan"), float("nan")))
    pm, dm = np.mean(prox), np.mean(dist)
    direction = ("proximal>distal (matches InternAL)" if pm<dm
                 else "proximal<=distal (reversal)")
    res = {"domain":domain_key,"h1_pearson_r":round(r,4),"h1_pearson_p":round(p,6),
           "h3_t_stat":round(ts,4) if not np.isnan(ts) else "nan",
           "h3_t_p":round(tp,6) if not np.isnan(tp) else "nan",
           "h3_proximal_mean_forgetting":round(pm,4) if not np.isnan(pm) else "nan",
           "h3_distal_mean_forgetting":round(dm,4) if not np.isnan(dm) else "nan",
           "h3_direction":direction,"n_subjects":len(valid)}
    log.info(f"  H1 r={r:.3f} p={p:.5f} | H3: {direction}")
    log.info(f"     proximal={pm:.4f} | distal={dm:.4f}")
    return res

# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_medqa(n=4000, seed=42):
    log.info("Loading MedQA ...")
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train",
                      trust_remote_code=True)
    df = ds.to_pandas()
    def opts(row):
        if "option_0" in df.columns: return [row[f"option_{i}"] for i in range(4)]
        if "options" in df.columns:
            o = row["options"]
            if isinstance(o,dict): return [o[k] for k in sorted(o)[:4]]
            if isinstance(o,list): return o[:4]
        return list("ABCD")
    if "meta_info" in df.columns and df["meta_info"].nunique()>1:
        n_cat = max(500, n//df["meta_info"].nunique())
        smp = df.groupby("meta_info",group_keys=False).apply(
            lambda x: x.sample(min(len(x),n_cat),random_state=seed))
    else:
        smp = df.sample(min(n,len(df)),random_state=seed)
    smp = smp.sample(frac=1,random_state=seed).reset_index(drop=True)
    rows = []
    for _,row in smp.iterrows():
        o = opts(row)
        os_ = "\n".join([f"{l}) {c}" for l,c in zip("ABCD",o)])
        a = str(row.get("answer","A")).strip().upper()
        if a not in "ABCD": a="A"
        rows.append({"text":f"Question: {row['question']}\n{os_}\nAnswer: {a}"})
    d = Dataset.from_list(rows)
    log.info(f"  MedQA ready: {len(d)} examples")
    return d

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════════

def run_e00(baseline, ebs=16):
    """Baseline: eval 57 MMLU subjects, no fine-tuning. Resumes per-subject."""
    if BASELINE_CSV.exists():
        df = pd.read_csv(BASELINE_CSV)
        ex = dict(zip(df["subject"], df["accuracy"]))
        if len(ex) >= 57:
            log.info(f"✅ E00 baseline already complete. Loaded {len(ex)} subjects.")
            return ex
        baseline.update(ex)
        log.info(f"  Resuming E00: {len(ex)}/57 done.")

    log.info("\n" + "="*60 + "\n▶  E00: Baseline evaluation\n" + "="*60)
    m, t = load_model(MODEL_9B); m.eval()

    for i, s in enumerate(MMLU_SUBJECTS):
        if s in baseline:
            log.info(f"  ⏭  {s} done ({baseline[s]:.3f})"); continue
        baseline[s] = eval_subject(m, t, s)
        log.info(f"  [{i+1:2d}/57] {s}: {baseline[s]:.3f}")
        pd.DataFrame([{"subject":k,"accuracy":v,"is_medical":k in MEDICAL_MMLU}
                      for k,v in baseline.items()]).to_csv(BASELINE_CSV, index=False)

    for s,a in baseline.items():
        log_result("E00",{"model":MODEL_9B,"domain":"none","lora_rank":0,"n_steps":0,
                          "replay_size":0,"seed":SEED_PRIMARY,"subject":s,
                          "accuracy":round(a,4) if not np.isnan(a) else "nan",
                          "is_medical":s in MEDICAL_MMLU,"delta_acc":0,"forgetting":0})
    free_gpu(m, t)
    log.info(f"\n📋 E00 COMPLETE | mean={np.nanmean(list(baseline.values())):.3f}")
    return baseline

def run_rank(exp_id, rank, steps, baseline, train_data, ebs=16):
    if already_done(exp_id): return

    apath = ADAPTERS_DIR / exp_id
    ckpts = sorted([d for d in os.listdir(apath) if d.startswith("checkpoint-")]) \
            if apath.exists() else []
    log.info("\n" + "="*60)
    if ckpts:
        log.info(f"♻️  {exp_id}: RESUMING from step "
                 f"{ckpts[-1].split('-')[-1]}/{steps} (rank={rank})")
    else:
        log.info(f"▶  {exp_id}: rank={rank}, steps={steps}")
    log.info("="*60)

    ft, ft_tok = finetune(MODEL_9B, train_data, exp_id, rank, steps, SEED_PRIMARY)
    ft.eval()

    log.info(f"\n📊 Evaluating MMLU after {exp_id} ...")
    post = eval_all(ft, ft_tok, tag=exp_id)

    fgt = {}
    for s in MMLU_SUBJECTS:
        b, p = baseline.get(s, float("nan")), post.get(s, float("nan"))
        d = p - b; fgt[s] = d
        log_result(exp_id,{"model":MODEL_9B,"domain":"medqa","lora_rank":rank,
                           "n_steps":steps,"replay_size":0,"seed":SEED_PRIMARY,
                           "subject":s,"is_medical":s in MEDICAL_MMLU,
                           "accuracy":round(p,4) if not np.isnan(p) else "nan",
                           "delta_acc":round(d,4) if not np.isnan(d) else "nan",
                           "forgetting":round(-d,4) if not np.isnan(d) else "nan"})

    prx = proximity_test("medqa", fgt, exp_id)
    log_result(f"{exp_id}_proximity",{"model":MODEL_9B,"domain":"medqa",
               "lora_rank":rank,"n_steps":steps,"replay_size":0,"seed":SEED_PRIMARY,
               "subject":"ALL_PROXIMITY_STATS",**prx,
               "delta_acc":0,"forgetting":0,"accuracy":0,"is_medical":False})

    mf = -np.nanmean(list(fgt.values()))
    mf_med = -np.nanmean([fgt[s] for s in MEDICAL_MMLU if s in fgt])
    mf_gen = -np.nanmean([v for s,v in fgt.items() if s not in MEDICAL_MMLU])
    log.info(f"\n📋 {exp_id} | rank={rank}")
    log.info(f"   Mean fgt:    {mf:+.4f}")
    log.info(f"   Medical fgt: {mf_med:+.4f}")
    log.info(f"   Non-med fgt: {mf_gen:+.4f}")
    log.info(f"   H1 r:        {prx['h1_pearson_r']:.3f}")
    log.info(f"   H3:          {prx['h3_direction']}")

    # Clean intermediate checkpoints
    for c in list(apath.iterdir()):
        if c.name.startswith("checkpoint-"):
            shutil.rmtree(c); log.info(f"   Cleaned {c.name}")

    free_gpu(ft, ft_tok)

def h2_summary():
    if not RESULTS_CSV.exists(): return
    df = pd.read_csv(RESULTS_CSV)
    log.info("\n" + "="*60 + "\n📊 H2 — Rank effect summary\n" + "="*60)
    log.info(f"{'Exp':6} {'Rank':6} {'Mean fgt':12} {'Med':12} {'Non-med':12} {'H1 r':8}")
    log.info("-"*60)
    vals = []
    for eid, rank in [("E01",4),("E02",16),("E03",64),("E04",128)]:
        de = df[(df["exp_id"]==eid)&(df["subject"].isin(MMLU_SUBJECTS))].copy()
        if len(de)==0: log.info(f"{eid:6} {rank:6}  ⚠️  not run"); continue
        de["f"] = pd.to_numeric(de["forgetting"],errors="coerce")
        de = de.dropna(subset=["f"])
        mf  = de["f"].mean()
        mmf = de[de["is_medical"]==True]["f"].mean()
        gmf = de[de["is_medical"]==False]["f"].mean()
        dp  = df[df["exp_id"]==f"{eid}_proximity"]
        r   = dp["h1_pearson_r"].values[0] if len(dp) else "n/a"
        vals.append((rank, mf))
        log.info(f"{eid:6} {rank:6} {mf:+12.4f} {mmf:+12.4f} {gmf:+12.4f} {str(r):8}")
    if len(vals)==4:
        v = [x for _,x in sorted(vals)]
        mono = all(v[i]<=v[i+1] for i in range(3))
        log.info(f"\n🔬 H2: Monotonic = {mono}")
        log.info("   → Biderman 2024 SUPPORTED" if mono
                 else "   → Steele 2026 SUPPORTED")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Notebook A — A5000 Pro (optimised)")
    ap.add_argument("--exp", default="all",
                    help="Experiments: E00,E01,E02,E03,E04 or 'all'")
    ap.add_argument("--skip-e00", action="store_true")
    ap.add_argument("--eval-batch-size", type=int, default=16,
                    help="Batch size for MMLU eval (default 16, lower if OOM)")
    args = ap.parse_args()

    ebs = args.eval_batch_size   # local var — no global needed

    run_these = ({"E00","E01","E02","E03","E04"} if args.exp.lower()=="all"
                 else {e.strip().upper() for e in args.exp.split(",")})

    log.info("="*60)
    log.info("NOTEBOOK A — Baseline + Rank Ablation (SPEED-OPTIMISED)")
    log.info(f"GPU: {gpu_status()}")
    log.info(f"Run: {sorted(run_these)}")
    log.info(f"Eval batch size: {ebs}")
    log.info(f"Train batch size: {A5000['per_device_train_batch_size']} "
             f"(grad_accum={A5000['gradient_accumulation_steps']})")
    log.info(f"Out: {RESULTS_CSV}")
    log.info("="*60)

    set_seed(SEED_PRIMARY)

    baseline = {}
    if "E00" in run_these and not args.skip_e00:
        baseline = run_e00(baseline, ebs)
    elif BASELINE_CSV.exists():
        df = pd.read_csv(BASELINE_CSV)
        baseline = dict(zip(df["subject"], df["accuracy"]))
        log.info(f"✅ Baseline loaded ({len(baseline)} subjects)")
    else:
        raise FileNotFoundError(
            f"Baseline missing: {BASELINE_CSV}\n"
            f"Run: python {__file__} --exp E00")

    rank_exps = {"E01","E02","E03","E04"} & run_these
    medqa = load_medqa(4000, SEED_PRIMARY) if rank_exps else None

    for eid, rank, steps in [("E01",4,500),("E02",16,500),
                               ("E03",64,500),("E04",128,500)]:
        if eid in run_these:
            run_rank(eid, rank, steps, baseline, medqa, ebs)

    if len(rank_exps) > 1:
        h2_summary()

    log.info(f"\n✅ Notebook A complete. Results → {RESULTS_CSV}")
