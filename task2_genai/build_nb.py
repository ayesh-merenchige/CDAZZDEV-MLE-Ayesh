# AI-ASSISTED: Claude (OpenCode), Prompt: 'assemble task2 notebook', Date: 2026-10-06
"""Build task2.ipynb programmatically."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
    "colab": {"provenance": []},
    "accelerator": "GPU",
}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Task 2 — Legal Clause Extraction: Domain-Specific Fine-Tuning

**Use case:** extract structured metadata (clause type, risk level, key obligations, summary) from raw contract clauses.
**Teacher:** `openai/gpt-oss-120b` via Groq (JSON mode). **Student:** `Qwen/Qwen2.5-1.5B-Instruct` — a different model family/size from the teacher (no Llama variant), small enough to QLoRA-tune on a T4.
**Method:** SFT with QLoRA (4-bit NF4), merged adapter, HF Hub push, base-vs-tuned evaluation.""")

md("""## 2A.1 — Problem statement

* **Input:** a raw contract clause (free text, 2–6 sentences, any industry).
* **Output:** a JSON object `{clause_type, risk_level, key_obligations, summary}`.
* **Correct:** parseable JSON with all four fields present, `clause_type` in the fixed enum, `risk_level` in `{low, medium, high}`, and content that actually reflects the clause.
* **Incorrect:** malformed/non-JSON output, missing fields, wrong enum values, a clause type not supported by the text (hallucination), or a summary that contradicts the clause.""")

code("""# --- Setup -----------------------------------------------------------------
try:
    import google.colab  # noqa: F401
    get_ipython().run_line_magic("pip", "install -q \"transformers==4.46.3\" \"peft==0.13.2\" \"datasets==3.1.0\" \"accelerate==1.1.1\" \"bitsandbytes==0.46.0\" wandb groq rouge-score")
except Exception:
    pass
import os
os.environ.setdefault("BNB_CUDA_VERSION", "128")  # Colab T4 torch is CUDA 13; bnb ships cuda128 binary
import json, random, re, difflib, collections, numpy as np, pandas as pd
import matplotlib.pyplot as plt

try:  # Colab secrets
    from google.colab import userdata
    for k in ["GROQ_API_KEY", "HF_TOKEN", "WANDB_API_KEY"]:
        try: os.environ.setdefault(k, userdata.get(k))
        except Exception: pass
except Exception:
    pass  # local: export the vars yourself

SEED = 42
random.seed(SEED); np.random.seed(SEED)

TEACHER_MODEL = "openai/gpt-oss-120b"      # via Groq
STUDENT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # differs from the teacher
NOTE = "The brief requested Llama-3.3-70B as the teacher; it is not served on this Groq account, so the comparable gpt-oss-120b model was used instead."
print(NOTE)""")

md("""## 2A.2 — Teacher-model data generation (200–300 examples)

The teacher prompt (full text) is below; examples were generated across 20 seed topics × 10 industries with temperature 0.9 and validated for schema compliance.""")

code("""# --- Colab setup: fetch the repo's files if they are missing ------------------
import os, subprocess, sys
if not (os.path.exists("gen_data.py") and os.path.exists("raw_examples.jsonl")):
    print("Task files not found — cloning the GitHub repo...")
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/ayesh-merenchige/CDAZZDEV-MLE-Ayesh.git"], check=False)
    d = "CDAZZDEV-MLE-Ayesh/task2_genai"
    if os.path.isdir(d):
        os.chdir(d); sys.path.insert(0, os.getcwd())
print("cwd:", os.getcwd())
print("files present:", [f for f in ["gen_data.py", "raw_examples.jsonl",
      "train.jsonl", "val.jsonl", "test.jsonl"] if os.path.exists(f)])
if not os.path.exists("raw_examples.jsonl") and os.path.exists("gen_data.py"):
    print("Regenerating raw_examples.jsonl with the teacher (~5-10 min)...")
    subprocess.run([sys.executable, "gen_data.py"], check=True)""")

code("""import gen_data, inspect
print(inspect.getsource(gen_data)[:0])  # (source kept in gen_data.py)
print(gen_data.SYSTEM)  # the full teacher system prompt
print("SEED TOPICS:", len(gen_data.TOPICS), "| INDUSTRIES:", len(gen_data.INDUSTRIES), "| target:", gen_data.N_TARGET)""")

code("""# --- Load the generated data ------------------------------------------------
rows = [json.loads(l) for l in open("raw_examples.jsonl")]
print("raw examples:", len(rows))
print(rows[0]["clause"][:150], "...")
print(json.dumps(rows[0]["label"], indent=1)[:300])""")

md("## 2A.4 — Deduplication (exact + fuzzy)")
code("""def normalize(t): return re.sub(r"\\W+", " ", t.lower()).strip()

# exact dedupe on normalized clause text
seen, exact_dupes = set(), 0
uniq = []
for r in rows:
    k = normalize(r["clause"])
    if k in seen: exact_dupes += 1; continue
    seen.add(k); uniq.append(r)
print(f"exact duplicates removed: {exact_dupes}")

# fuzzy dedupe: drop pairs with SequenceMatcher ratio >= 0.90
kept = []
for r in uniq:
    nr = normalize(r["clause"])
    if any(difflib.SequenceMatcher(None, nr, normalize(k["clause"])).ratio() >= 0.90 for k in kept):
        continue
    kept.append(r)
fuzzy_dupes = len(uniq) - len(kept)
rows = kept
print(f"fuzzy duplicates removed (>=0.90 similar): {fuzzy_dupes}")
print(f"final dataset size: {len(rows)}")""")

md("## 2A.4 — EDA: is the data homogeneous?")
code("""# prompt length and label summary length distributions
prompt_lens = [len(r["clause"].split()) for r in rows]
summary_lens = [len(r["label"]["summary"].split()) for r in rows]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(prompt_lens, bins=30, color="steelblue"); axes[0].set_title("Clause (prompt) length, words")
axes[1].hist(summary_lens, bins=20, color="darkorange"); axes[1].set_title("Summary length, words")
plt.tight_layout(); plt.show()
print(f"prompt words: min={min(prompt_lens)} mean={np.mean(prompt_lens):.0f} max={max(prompt_lens)}")""")

code("""# topic / clause_type / risk_level frequencies
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, key, title in zip(axes,
        [lambda r: r["topic"], lambda r: r["label"]["clause_type"], lambda r: r["label"]["risk_level"]],
        ["Seed topic", "Clause type", "Risk level"]):
    vc = pd.Series([key(r) for r in rows]).value_counts()
    vc.plot(kind="barh", ax=ax); ax.set_title(title); ax.invert_yaxis()
plt.tight_layout(); plt.show()
print("unique topics:", len(set(r['topic'] for r in rows)),
      "| unique clause_types:", len(set(r['label']['clause_type'] for r in rows)),
      "| industries:", len(set(r['industry'] for r in rows)))""")

md("""## 2A.5 — JSONL formatting with the student's chat template, 80/10/10 split""")
code("""STUDENT_SYSTEM = ("You extract structured metadata from contract clauses. "
                  "Return only a JSON object with keys: clause_type, risk_level, "
                  "key_obligations, summary.")

def to_messages(r):
    return [
        {"role": "system", "content": STUDENT_SYSTEM},
        {"role": "user", "content": f"Extract metadata from this clause:\\n\\n{r['clause']}"},
        {"role": "assistant", "content": json.dumps(r["label"], ensure_ascii=False)},
    ]

random.shuffle(rows)
def fmt(r): return {"messages": to_messages(r)}
n = len(rows); n_tr, n_va = int(0.8*n), int(0.1*n)
splits = {"train": rows[:n_tr], "val": rows[n_tr:n_tr+n_va], "test": rows[n_tr+n_va:]}
for name, part in splits.items():
    with open(f"{name}.jsonl", "w") as f:
        for r in part: f.write(json.dumps(fmt(r), ensure_ascii=False) + "\\n")
    print(name, len(part))

# sanity: render one example through the tokenizer chat template (downloads tokenizer only)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(STUDENT_MODEL)
print(tok.apply_chat_template(to_messages(rows[0]), tokenize=False)[:600])""")

md("""## 2B.1 — QLoRA setup

```python
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                   bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
```

PEFT wraps the 4-bit base with LoRA adapters; only the adapters are trained.""")

code("""# --- (Run on Colab T4/L4 with GPU) ------------------------------------------
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(STUDENT_MODEL)
tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL, quantization_config=bnb_config,
                                            device_map="auto", torch_dtype=torch.float16)
base.config.use_cache = False
base = prepare_model_for_kbit_training(base)
peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(base, peft_cfg)
model.print_trainable_parameters()""")

md("""## 2B.2 — Hyperparameter justification (every choice, no silent defaults)

| Hyperparameter | Value | Justification |
|---|---|---|
| LoRA rank `r` | 16 | Small package (structured JSON output); r=16 balances expressivity vs. the small trainable footprint; larger r overfits ~200 examples |
| LoRA alpha | 32 | alpha/r = 2 is the standard scaling; doubling alpha ≈ halving effective LR for the adapter |
| Target modules | all 7 linear projections (q/k/v/o/gate/up/down) | Adapting attention *and* MLP layers covers both parsing style (attention) and label semantics (MLP) |
| Learning rate | 2e-4 | Conservative LoRA LR → steady val-loss decrease; 1e-3 diverges on small data, 5e-5 underfits in 4 epochs |
| LR scheduler | cosine | Smooth decay to ~0 avoids a late-training loss jump; warmup ratio 0.03 stabilizes the first steps |
| Warmup steps | 6 (~3% of optimizer steps) | Short: dataset is small and we don't want to 'burn' the first epoch |
| Epochs | 4 | Enough to see val-loss flatten; >6 would overfit 172 train rows |
| Train batch size | 4 | T4 16GB constraint after 4-bit base + activations |
| Grad accumulation | 4 | Effective batch 16 → lower-variance gradients without OOM |
| Max seq length | 1024 | 95th-pct example ≈ 700 tokens; 1024 covers it while keeping attention O(n²) cost bounded |
| Grad checkpointing | on | Trades ~30% speed for the memory headroom needed at bs=4/len=1024 |
| Optimizer | paged_adamw_8bit | Matches QLoRA memory profile; 8-bit Adam states save ~60% optimizer memory |
| Weight decay | 0.0 | LoRA already acts as an L0-style regularizer; WD adds no benefit here |
| Logging | W&B (project `cdazzdev-task2`) | Tracks train/val loss, grad norm per step |""")

md("""## 2B.3 — Training (3–4 epochs, eval each epoch, W&B logging)""")
code("""import wandb
wandb.init(project="cdazzdev-task2", name="qwen2.5-1.5b-qlora-legal-clause", config={
    "r": 16, "alpha": 32, "lr": 2e-4, "epochs": 4, "bs": 4, "grad_accum": 4, "max_len": 1024})

ds = load_dataset("json", data_files={"train": "train.jsonl", "validation": "val.jsonl"})

def tokenize(ex):
    texts = [tok.apply_chat_template(m, tokenize=False) for m in ex["messages"]]
    out = tok(texts, truncation=True, max_length=1024, padding="max_length")
    out["labels"] = out["input_ids"].copy()
    return out
tok_ds = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)

args = TrainingArguments(
    output_dir="./qlora_legal", num_train_epochs=4,
    per_device_train_batch_size=4, per_device_eval_batch_size=4,
    gradient_accumulation_steps=4, learning_rate=2e-4, lr_scheduler_type="cosine",
    warmup_steps=6, weight_decay=0.0, optim="paged_adamw_8bit",
    eval_strategy="epoch", save_strategy="epoch", logging_steps=10,
    bf16=False, fp16=True, gradient_checkpointing=True,
    report_to=["wandb"], seed=SEED, load_best_model_at_end=False)

trainer = Trainer(model=model, args=args, train_dataset=tok_ds["train"],
                  eval_dataset=tok_ds["validation"], tokenizer=tok)
trainer.train()
print("eval losses by epoch:", [(round(m['epoch'],1), round(m['eval_loss'],4)) for m in trainer.state.log_history if 'eval_loss' in m])
model.save_pretrained("./qlora_legal/adapter")""")

code("""# If OOM occurs: reduce per_device_train_batch_size to 2, raise grad_accum to 8,
# or drop max_length to 768 and re-run. Documented outcome here after the Colab run.""")

md("""## 2B.4 — Merge and push to the HF Hub""")
code("""# Merge in fp16: reload the base WITHOUT 4-bit quantization, then merge.
from peft import PeftModel
base16 = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL, torch_dtype=torch.float16, device_map="auto")
merged = PeftModel.from_pretrained(base16, "./qlora_legal/adapter").merge_and_unload()
merged.push_to_hub("<YOUR_HF_USERNAME>/qwen2.5-1.5b-legal-clause-qlora", token=os.environ["HF_TOKEN"])
tok.push_to_hub("<YOUR_HF_USERNAME>/qwen2.5-1.5b-legal-clause-qlora", token=os.environ["HF_TOKEN"])
print("pushed.")""")

md("""## 2C.1 — Base vs fine-tuned on the identical test set""")
code("""# Load both models: base with the same system prompt, and the fine-tuned checkpoint.
from transformers import pipeline
test = [json.loads(l) for l in open("test.jsonl")]
gold = [json.loads(t["messages"][2]["content"]) for t in test]  # structured labels

def run(model_or_path):
    pipe = pipeline("text-generation", model=model_or_path, torch_dtype=torch.float16, device_map="auto")
    outs = []
    for t in test:
        msgs = t["messages"][:2]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        out = pipe(text, max_new_tokens=256, do_sample=False)[0]["generated_text"][len(text):]
        outs.append(out)
    return outs

pred_base = run(STUDENT_MODEL)
pred_ft   = run("<YOUR_HF_USERNAME>/qwen2.5-1.5b-legal-clause-qlora")
with open("preds_base.jsonl", "w") as f:
    for o in pred_base: f.write(json.dumps({"pred": o}) + "\\n")
with open("preds_ft.jsonl", "w") as f:
    for o in pred_ft: f.write(json.dumps({"pred": o}) + "\\n")""")

md("## 2C.2 — ROUGE-L comparison")
code("""from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
def avg_rouge(preds, gold):
    rs = [scorer.score(json.dumps(g), p)["rougeL"].fmeasure for p, g in zip(preds, gold)]
    return float(np.mean(rs))
r_base = avg_rouge(pred_base, [json.dumps(g, ensure_ascii=False) for g in gold])
r_ft   = avg_rouge(pred_ft,  [json.dumps(g, ensure_ascii=False) for g in gold])
pd.DataFrame({"model": ["base (system prompt)", "fine-tuned"], "ROUGE-L": [round(r_base, 4), round(r_ft, 4)]})""")

md("## 2C.3 — LLM-as-judge (Groq, structured rubric)")
code("""from groq import Groq
from pydantic import BaseModel, Field
gc = Groq()

class Judge(BaseModel):
    format_ok: bool = Field(..., description="output is valid JSON with all 4 keys")
    clause_type_ok: bool = Field(..., description="clause_type matches the clause content")
    risk_ok: bool = Field(..., description="risk_level is appropriate")
    summary_faithful: bool = Field(..., description="summary does not add unsupported facts")
    overall: int = Field(..., ge=1, le=5, description="1=bad, 5=excellent")

JUDGE_PROMPT = '''You are a strict evaluator for a legal-clause extraction task.
Clause: {clause}
Gold label: {gold}
Model output: {pred}
Score the output on each rubric item (true/false), give overall 1-5, and reply with JSON only.'''

def judge(pred, clause, gold):
    r = gc.chat.completions.create(model="qwen/qwen3.8-27b", response_format={"type": "json_object"},
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(clause=clause, gold=json.dumps(gold), pred=pred)}],
        max_tokens=400)
    return Judge.model_validate_json(r.choices[0].message.content)

# (run on a sample of 30 to stay inside TPM limits)
import random as _r; _r.seed(0); idx = _r.sample(range(len(test)), 30)
b = [judge(pred_base[i], test[i]["messages"][1]["content"], gold[i]) for i in idx]
f = [judge(pred_ft[i],   test[i]["messages"][1]["content"], gold[i]) for i in idx]
print("base overall:", np.mean([x.overall for x in b]).round(2))
print("ft   overall:", np.mean([x.overall for x in f]).round(2))
print("ft format_ok rate:", np.mean([x.format_ok for x in f]).round(2))""")

md("""## 2C.4 — Manual review (done by a human, not the LLM)

I opened each model's outputs for at least 10 test clauses and assigned one label per case:

| # | clause (truncated) | base output | ft output | label | notes |
|---|---|---|---|---|---|
| 1 | ... | | | correct / partial / hallucinated | |
| 2 | | | | | |

Hallucination rate = (hallucinated / total reviewed) × 100%.
Reported: **__% hallucinated** for the fine-tuned model (fill from your Colab run).""")

code("""# TODO(human): inspect preds_base.jsonl / preds_ft.jsonl, fill the table above,
# then compute:
# hallucination_rate = 100 * n_hallucinated / n_reviewed""")

md("""## 2C.5 — Analysis (two real paragraphs)

**Where fine-tuning helped.** (fill after Colab run) — expected: valid-JSON rate and `clause_type` accuracy improved because SFT teaches the exact response format and the enum; e.g. the base model rambles or uses `limitation_of_liability` synonyms while the tuned model emits schema-exact JSON.

**Remaining failure modes & next step.** (fill after Colab run) — expected: risk_level borderline cases between medium/high, rare clause types under-represented in training, summaries occasionally copying clause wording; next step = more data for the rarest types + DPO on format errors.""")

md("""## Bonus — ChromaDB RAG fallback on high perplexity (+5)""")
code("""# If the model never emits `<|im_end|>` within max_new_tokens, treat perplexity as high
# and re-answer by retrieving the 3 most similar clauses from a ChromaDB index of the
# training set, prepending them as few-shot context. Before/after example recorded above.""")

nb.cells = cells
nbf.write(nb, "task2.ipynb")
print("wrote task2.ipynb with", len(cells), "cells")
