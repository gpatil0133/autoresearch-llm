# Survey Text Analytics Autoresearch — Implementation Plan

## Overview

This document outlines the comprehensive adaptation of the **autoresearch framework** to automate the research workflow for **survey response text analytics** using 3-4B parameter LLMs.

### Current State
- **Original autoresearch**: Language model pre-training on open-domain data (next-token prediction)
- **Our adaptation**: Fine-tuning on custom labeled survey data for structured multi-task analysis
  - Sentiment classification
  - Named Entity Recognition (NER)
  - Theme extraction
  - Emotion identification
  - Forced structured JSON output

### Goal
Transform autoresearch into an autonomous research loop:
```
Model Selection → Fine-tune → Inference → Evaluate → Benchmark → Repeat
```

With fixed 5-minute training budget per experiment, systematically improving survey analysis quality.

---

## Part 1: Architecture & Workflow

### 1.1 New Overall Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    STARTUP PHASE                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. prepare.py (NEW - Survey-specific)                  │
│     ├─ Load custom survey CSV/JSON                      │
│     ├─ Create label splits (train/val)                  │
│     ├─ Define JSON schemas for structured output        │
│     ├─ Generate task-specific dataloaders               │
│     └─ Pre-compute baseline metrics                     │
│                                                          │
│  2. model_selector.py (NEW)                             │
│     ├─ Strategies: round-robin, random, UCB, evolutionary
│     └─ Auto-select next model to experiment with        │
│                                                          │
│  3. train.py (MODIFIED - Multi-task fine-tuning)        │
│     ├─ Load base model (Phi-3, DistilBERT, Mistral)    │
│     ├─ Apply LoRA for parameter efficiency              │
│     ├─ Add multi-task heads                             │
│     ├─ Train for fixed 5-minute budget                  │
│     └─ Output metrics (F1, VRAM, etc.)                  │
│                                                          │
│  4. inference.py (NEW - Structured generation)          │
│     ├─ Use Outlines library for JSON schema enforcement │
│     ├─ Run inference on test data                       │
│     └─ Extract structured analysis                      │
│                                                          │
│  5. eval.py (NEW - Benchmarking)                        │
│     ├─ Compute F1 scores per task                       │
│     ├─ Validate JSON schema compliance                  │
│     ├─ Log results to results.tsv                       │
│     └─ Update leaderboard.json                          │
│                                                          │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                  EXPERIMENT LOOP                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  REPEAT UNTIL MANUAL STOP:                              │
│                                                          │
│  Step 1: SELECT MODEL                                   │
│    └─ ModelSelector.select(step, strategy)              │
│                                                          │
│  Step 2: GET CONFIG                                     │
│    └─ Auto hyperparams based on model size/type         │
│                                                          │
│  Step 3: HACK train.py                                  │
│    └─ Inject MODEL_NAME, LR, batch_size, task_weights  │
│                                                          │
│  Step 4: GIT COMMIT                                     │
│    └─ git add train.py && git commit -m "exp N: desc"   │
│                                                          │
│  Step 5: RUN TRAINING                                   │
│    └─ uv run train.py > run.log 2>&1                    │
│                                                          │
│  Step 6: PARSE METRICS                                  │
│    └─ grep "^val_metric:\|^peak_vram_mb:" run.log       │
│                                                          │
│  Step 7: RUN INFERENCE                                  │
│    └─ uv run inference.py                               │
│                                                          │
│  Step 8: EVALUATE & LOG                                 │
│    └─ eval.py → results.tsv, leaderboard.json           │
│                                                          │
│  Step 9: DECIDE                                         │
│    ├─ If val_metric improved: keep commit              │
│    └─ Else: git reset --hard HEAD~1                     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Key Differences from Original Autoresearch

| Aspect | Original | Survey Analytics |
|--------|----------|------------------|
| **Data** | HuggingFace climbmix (open domain) | Custom labeled surveys |
| **Task** | Next-token prediction (language modeling) | Multi-task: sentiment, NER, themes, emotions |
| **Models** | GPT-like decoder (pre-training) | Decoder (LLM) + Encoder (BERT) variants |
| **Metric** | Bits-per-byte (BPB) | F1-macro (sentiment), Token-level F1 (NER), Multi-label F1 (themes) |
| **Output** | Text generation (free-form) | **Structured JSON** (Outlines enforced) |
| **Optimization** | Reduce validation loss | Maximize task-specific F1 scores |
| **Model sizes** | 50M+ (reported in results) | 3-4B parameters |
| **Inference** | Decoder-only generation | Encoder or generative with schema constraints |

---

## Part 2: Model Selection Strategy

### 2.1 Preferred Models for 3-4B Parameter Range

We'll experiment with a diverse set of models, each with unique strengths for survey analytics:

#### **Category 1: Instruction-Tuned Generative Models (Best for structured output)**

| Model | Size | Fit | Reason |
|-------|------|-----|--------|
| **Phi-3-mini** | 3.8B | ⭐⭐⭐⭐⭐ | Excellent instruction-following, compact, designed for structured reasoning |
| **Qwen2-3B** | 3B | ⭐⭐⭐⭐⭐ | Strong multilingual support, good instruction tuning, smaller variant |
| **Llama-3.2-3B** | 3B | ⭐⭐⭐⭐ | Solid baseline, well-established, good community support |

#### **Category 2: Encoder-Only Models (Fast, lightweight classification)**

| Model | Size | Fit | Reason |
|-------|------|-----|--------|
| **DistilBERT** | 66M | ⭐⭐⭐⭐ | **Fastest inference**, only ~40% parameters vs BERT, perfect for adapter-based tuning |
| **BERT-base** | 110M | ⭐⭐⭐ | Standard baseline, strong performance but slower |
| **RoBERTa-base** | 125M | ⭐⭐⭐⭐ | Improved BERT pretraining, slightly better on tasks |

#### **Category 3: Decoder + Encoder (Seq2Seq for structured output)**

| Model | Size | Fit | Reason |
|-------|------|-----|--------|
| **T5-base** | 220M | ⭐⭐⭐ | Text-to-text framework, can format output as text-to-JSON |
| **FLAN-T5-base** | 250M | ⭐⭐⭐⭐ | Instruction tuned, better generalization |

### 2.2 Recommended Experimentation Path

**Phase 1: Quick Baselines (Models)**
```
Experiment 1: Phi-3-mini (baseline)
Experiment 2: DistilBERT (speed)
Experiment 3: Qwen2-3B (multilingual)
Experiment 4: Llama-3.2-3B (alternative)
```

**Phase 2: Architecture Choices**
```
Experiment 5-7: LoRA ranks (r=8, 16, 32)
Experiment 8-10: Task weight combinations
Experiment 11-13: Shared vs. task-specific heads
```

**Phase 3: Training Hyper-parameters**
```
Experiment 14-16: Learning rate (1e-5, 5e-5, 1e-4)
Experiment 17-19: Batch size (8, 16, 32, 64)
Experiment 20-22: Scheduler type (linear, cosine, warmup)
```

**Phase 4: Output Strategy**
```
Experiment 23-25: Full schema vs. minimal JSON
Experiment 26-28: Grammar-guided decoding vs. free-form
```

---

## Part 3: Structured Output with Outlines

### 3.1 Why Outlines?

**Problem**: LLMs can drift from JSON format, hallucinate fields, or produce invalid JSON.

**Solution**: Use **Outlines** library to enforce a Pydantic schema during token generation itself.

```python
# Without Outlines (unreliable)
output = model.generate("Analyze survey...")
# Result: Could be valid JSON, partial JSON, or no JSON at all

# With Outlines (guaranteed valid)
output = outlines.generate.json(model, SurveyAnalysisSchema)
results = output("Analyze survey...")  # Always valid Pydantic model
```

### 3.2 Schema Definitions

**File: prepare.py (schemas section)**

```python
from pydantic import BaseModel
from typing import List, Optional

# ─────────────────────────────────────
# Task 1: Sentiment Analysis
# ─────────────────────────────────────
class SentimentAnalysis(BaseModel):
    """Forced structured output for sentiment analysis."""
    paragraph_sentiment: str  # "positive" | "negative" | "neutral"
    sentiment_score: float    # -1.0 to 1.0
    main_themes: List[dict]   # [{theme: str, score: float, relevance: float}]
    sentences: List[dict]     # [{text: str, sentiment: str, score: float}]
    action_plan: Optional[str] = None


# ─────────────────────────────────────
# Task 2: Entity Recognition
# ─────────────────────────────────────
class EntityAnalysis(BaseModel):
    """NER with metadata sentiment."""
    entities: List[dict]      # [{name: str, type: str (Person/Product/Company/Location), 
                              #   sentiment: str, score: float}]
    entity_count: int


# ─────────────────────────────────────
# Task 3: Theme Extraction
# ─────────────────────────────────────
class ThemeAnalysis(BaseModel):
    """Multi-label theme extraction with emotions."""
    themes: List[dict]        # [{theme: str, sentiment: str, 
                              #   relevance: float, emotion: str, 
                              #   emotion_intensity: float}]
    primary_emotion: str
    emotion_intensity: float


# ─────────────────────────────────────
# Combined Output
# ─────────────────────────────────────
class SurveyAnalysisOutput(BaseModel):
    """Complete analysis output."""
    sentiment: SentimentAnalysis
    entities: EntityAnalysis
    themes: ThemeAnalysis
```

### 3.3 Usage in Inference

**File: inference.py (example)**

```python
import outlines.models as models
import outlines.samplers as samplers
from prepare import SurveyAnalysisOutput
import json

# Load model with Outlines
model = models.transformers("microsoft/phi-3-mini-4k-instruct")

# Create generator with schema constraint
generator = outlines.generate.json(model, SurveyAnalysisOutput)

# Define prompt template
SYSTEM_PROMPT = """You are an expert NLP analyst. Analyze the survey response and provide:
1. Sentiment classification with score (-1.0 to 1.0)
2. Named entities with sentiment
3. Main themes with relevance scores
4. Emotional tones
Return valid JSON matching the schema."""

USER_TEMPLATE = """Survey Response:
{response}

Provide analysis in JSON format."""

# Run inference
test_response = "The product quality exceeded expectations but delivery took too long."

full_prompt = f"{SYSTEM_PROMPT}\n\n{USER_TEMPLATE.format(response=test_response)}"

# Generate with constraint
output = generator(full_prompt, max_tokens=500)

# Parse and validate
analysis = SurveyAnalysisOutput.parse_raw(output)
print(analysis.dict())
```

---

## Part 4: File Structure & Implementation

### 4.1 Files to Create/Modify

#### **NEW: prepare.py (Survey-specific data prep)**

**Responsibilities:**
- Load labeled survey CSV
- Create train/val/test splits
- Define Pydantic schemas (above)
- Build task-specific dataloaders
- Pre-compute evaluation metrics framework
- Define ALL constants (TIME_BUDGET, label mappings, etc.)

**Must NOT be modified in train.py**

**Key Functions:**
```python
def load_survey_data(csv_path) → Dataset
def make_sentiment_dataloader(texts, labels, tokenizer, batch_size)
def make_ner_dataloader(texts, entities, tokenizer, batch_size)
def make_theme_dataloader(texts, themes, tokenizer, batch_size)
def evaluate_sentiment(model, val_loader) → float (F1-macro)
def evaluate_ner(model, val_loader) → float (Token-level F1)
def evaluate_themes(model, val_loader) → float (Multi-label F1)
def evaluate_multi_task(model, all_loaders) → float (combined metric)
def get_model_config(model_name) → dict (LR, batch_size, LoRA rank)
```

---

#### **MODIFIED: train.py (Fine-tuning driver)**

**Can modify:** Everything

**Cannot modify:** 
- Import statements that rely on prepare.py constants/functions
- The evaluation harness

**Structure:**
```python
# ─── Hyperparameter Variables (MODIFY THESE PER EXPERIMENT) ───
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"  # Change per round
USE_LORA = True                                   # Experiment with this
LORA_RANK = 16                                    # Change per round
TASK_WEIGHTS = {"sentiment": 0.4, "ner": 0.3, "themes": 0.3}  # Tune
LEARNING_RATE = 5e-5                             # Experiment
BATCH_SIZE = 16                                   # Experiment

# ─── Load model & tokenizer ───
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from prepare import evaluate_multi_task, TIME_BUDGET

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ─── Apply LoRA if enabled ───
if USE_LORA:
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)

# ─── Multi-task heads ───
class MultiTaskHeads(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.sentiment_head = torch.nn.Linear(hidden_dim, 3)      # 3 classes
        self.ner_head = torch.nn.Linear(hidden_dim, num_ner_tags)
        self.theme_head = torch.nn.Linear(hidden_dim, num_themes)
    
    def forward(self, hidden_states, task_id):
        if task_id == 0:
            return self.sentiment_head(hidden_states)
        elif task_id == 1:
            return self.ner_head(hidden_states)
        else:
            return self.theme_head(hidden_states)

heads = MultiTaskHeads(hidden_dim=model.config.hidden_size)

# ─── Optimizer & Scheduler ───
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TIME_BUDGET)

# ─── Training loop (fixed 5-minute budget) ───
elapsed = 0
step = 0
peak_vram = 0
t0 = time.time()

loaders = [sentiment_loader, ner_loader, theme_loader]
weights = [TASK_WEIGHTS["sentiment"], TASK_WEIGHTS["ner"], TASK_WEIGHTS["themes"]]

while elapsed < TIME_BUDGET:
    for task_id, (loader, weight) in enumerate(zip(loaders, weights)):
        for batch in loader:
            logits = heads(model(batch), task_id)
            loss = criterion(logits, batch['labels']) * weight
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            step += 1
            elapsed = time.time() - t0
            peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / 1e9)
            
            if elapsed > TIME_BUDGET:
                break
        if elapsed > TIME_BUDGET:
            break

# ─── Evaluation ───
val_metric = evaluate_multi_task(model, val_sentiment_loader, val_ner_loader, val_theme_loader)

# ─── Print summary (grep-able) ───
print("---")
print(f"val_metric:       {val_metric:.6f}")
print(f"training_seconds: {elapsed:.1f}")
print(f"total_seconds:    {time.time() - t0:.1f}")
print(f"peak_vram_mb:     {peak_vram * 1024:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {sum(p.numel() for p in model.parameters()) / 1e6:.1f}")
print(f"model_name:       {MODEL_NAME}")
print(f"lora_rank:        {LORA_RANK if USE_LORA else 'none'}")
```

---

#### **NEW: inference.py (Structured output generation)**

**Responsibilities:**
- Load trained checkpoint
- Set up Outlines generators with schema
- Run inference on test set
- Validate JSON compliance
- Save results

**Key Functions:**
```python
def load_model_and_tokenizer(checkpoint_path, model_name)
def setup_generators(model, tokenizer)
def run_inference_batch(test_texts, generators)
def validate_json_schemas(outputs, schemas)
def save_results(results, output_file)
```

**Example flow:**
```python
import outlines.models as models
from prepare import SurveyAnalysisOutput

# Load fine-tuned model
checkpoint = "experiments/best_model"
model = models.transformers(checkpoint)

# Set up structured generator
generator = outlines.generate.json(model, SurveyAnalysisOutput)

# Inference loop
results = []
for survey_response in test_data:
    prompt = format_prompt(survey_response)
    output = generator(prompt, max_tokens=500)
    results.append(output)

# Save
save_results(results, "inference_results.json")
```

---

#### **NEW: eval.py (Benchmarking & tracking)**

**Responsibilities:**
- Compute F1 scores for each task
- Validate JSON schema compliance
- Log results to TSV
- Maintain leaderboard

**Key Functions:**
```python
def evaluate_sentiment_task(predictions, gold_labels) → float (F1)
def evaluate_ner_task(predictions, gold_labels) → float (Token-level F1)
def evaluate_themes_task(predictions, gold_labels) → float (Multi-label F1)
def validate_json_schemas(outputs, schema) → float (% valid)
def log_result(commit, metric, vram, status, description, model_name, config)
def update_leaderboard()
def generate_report()
```

**TSV Schema:**
```
commit          val_metric       memory_gb       status          description                           model_name                    config_json
a1b2c3d         0.742300         8.2             keep            baseline phi-3-mini                   microsoft/phi-3-mini-4k      {"lr":5e-5,"lora_r":16}
b2c3d4e         0.751200         8.4             keep            increase lr to 1e-4                  microsoft/phi-3-mini-4k      {"lr":1e-4,"lora_r":16}
c3d4e5f         0.739000         8.1             discard         reduce task_weight_ner to 0.2        microsoft/phi-3-mini-4k      {"lr":5e-5,"ner_weight":0.2}
```

---

#### **NEW: model_selector.py (Automated model selection)**

**Responsibilities:**
- Provide strategies for selecting which model to test next
- Track model performance over time
- Suggest next experiment

**Strategies:**
```python
class ModelSelector:
    models = [
        "microsoft/phi-3-mini-4k-instruct",
        "distilbert-base-uncased",
        "Qwen/Qwen2-3B-Instruct",
        "meta-llama/Llama-3.2-3B",
    ]
    
    @staticmethod
    def round_robin(step):
        """Cycle through models sequentially."""
        return ModelSelector.models[step % len(ModelSelector.models)]
    
    @staticmethod
    def random(step):
        """Random selection (useful for hyperparameter tuning)."""
        return random.choice(ModelSelector.models)
    
    @staticmethod
    def ucb(step):
        """Upper Confidence Bound: balance exploration vs. exploitation."""
        # Track performance of each model, select best + unexplored
        pass
    
    @staticmethod
    def evolutionary(step):
        """Select variants of best-performing model."""
        # Identify best model, then perturb hyperparams
        pass

# Usage:
next_model = ModelSelector.round_robin(step=5)
```

---

#### **NEW: benchmark_tracker.py (Leaderboard & metrics)**

**Responsibilities:**
- Track model performance across runs
- Identify Pareto frontier (best models by F1 vs speed vs memory)
- Export leaderboard

**Key Functions:**
```python
class BenchmarkTracker:
    def log_run(self, model_name, config, val_metric, vram_mb, inference_time)
    def get_best_models(self, top_k=5)
    def get_pareto_frontier(self)
    def compare_models(self)
    def export_leaderboard(self, format="json"|"csv"|"markdown")
```

---

### 4.2 Updated program.md

Key changes to document:

```markdown
## Modified Sections

### Setup
4. **Verify data exists**: Check that survey data CSV is at `./data/surveys.csv`
5. **Verify model cache**: HF models will be cached at `~/.cache/huggingface/`

### Experimentation
**What you CAN modify in train.py:**
- `MODEL_NAME` — which 3-4B model to use
- `USE_LORA` — enable/disable LoRA
- `LORA_RANK` — LoRA rank (8, 16, 32)
- `LEARNING_RATE` — optimizer LR
- `BATCH_SIZE` — training batch size
- `TASK_WEIGHTS` — sentiment/NER/themes balance
- Optimizer choice, scheduler, dropout, etc.

**Metric:**
- Changed from `val_bpb` to `val_metric` (F1-macro for sentiment + weighted NER + themes F1)
- Lower is better → WAIT, no! F1 is "higher is better", so we report `1 - val_metric` to keep < comparison

### Output format
New fields in results.tsv:
```
commit	val_metric	memory_gb	status	description	model_name	config_json
```

### Results Tracking
- results.tsv: per-experiment results (NOT committed to git)
- leaderboard.json: auto-generated top-5 models
```

---

## Part 5: Task-Specific Details

### 5.1 Multi-Task Learning Setup

**Why multi-task?**
- Shared representations reduce total parameters
- Prevents overfitting on small datasets
- Better generalization across related tasks

**Architecture:**
```
Input Text
    ↓
[Shared Transformer Backbone]
    ↓
    ├─→ [Sentiment Head] → 3 classes → Cross-Entropy Loss
    ├─→ [NER Head] → BIO tags → Token Classification Loss
    └─→ [Theme Head] → Multi-hot → BCE Loss
    
Combined Loss = w_s * loss_sentiment + w_n * loss_ner + w_t * loss_theme
```

**Task Weights:**
- Recommendation: `{sentiment: 0.4, ner: 0.3, themes: 0.3}`
- Experiment with: `{0.5, 0.25, 0.25}`, `{0.33, 0.33, 0.33}`, `{0.6, 0.2, 0.2}`

### 5.2 LoRA Configuration

**Why LoRA?**
- Reduces trainable parameters by 90%+
- Keeps model memory low
- Faster training

**Recommended configs:**
```python
# For smaller models (DistilBERT)
LoRA_RANK = 8
LORA_ALPHA = 16  # = 2 * rank

# For mid-size models (Phi-3-mini)
LoRA_RANK = 16
LORA_ALPHA = 32

# For larger models (Mistral-7B)
LoRA_RANK = 32
LORA_ALPHA = 64
```

---

## Part 6: Implementation Checklist

### Phase 0: Preparation
- [ ] Create labeled survey CSV with columns: `id`, `text`, `sentiment_label`, `ner_tags`, `themes`, `emotions`
- [ ] Place at `./data/surveys.csv`
- [ ] Verify pyproject.toml has dependencies: `transformers`, `peft`, `torch`, `pandas`, `scikit-learn`, `outlines`

### Phase 1: Core Infrastructure
- [ ] Write `prepare.py` with:
  - [ ] `load_survey_data()` function
  - [ ] All label mappings (sentiment → 0/1/2, NER → BIO, etc.)
  - [ ] Pydantic schemas for structured output
  - [ ] Dataloader creation functions
  - [ ] Evaluation functions (F1 metrics)
  - [ ] `get_model_config()` for auto hyperparams
  
- [ ] Write `model_selector.py`
  - [ ] `ModelSelector.round_robin()`
  - [ ] `ModelSelector.random()`
  - [ ] Strategy options

- [ ] Write `benchmark_tracker.py`
  - [ ] `log_run()` to TSV
  - [ ] `update_leaderboard()`
  - [ ] `generate_report()`

### Phase 2: Training Pipeline
- [ ] Write `train.py` template
  - [ ] Hyperparameter variables at top
  - [ ] Model loading (AutoModel)
  - [ ] LoRA setup
  - [ ] Multi-task heads definition
  - [ ] Training loop (5-min budget)
  - [ ] Metrics printing

- [ ] Write `inference.py`
  - [ ] Checkpoint loading
  - [ ] Outlines generator setup
  - [ ] Batch inference loop
  - [ ] JSON validation
  - [ ] Results saving

- [ ] Write `eval.py`
  - [ ] Per-task evaluation functions
  - [ ] Result logging
  - [ ] Leaderboard updates

### Phase 3: Experimentation Loop
- [ ] Set up git branch: `git checkout -b autoresearch/survey-v1`
- [ ] Initialize results.tsv header
- [ ] Create run script (bash/powershell) for automated loop
- [ ] Run first baseline
- [ ] Log results
- [ ] Test experiment loop (modify → commit → train → eval → decide)

### Phase 4: Automation
- [ ] Create `run_experiment.py` (orchestrates full loop)
  - [ ] Selects model
  - [ ] Updates train.py
  - [ ] Commits
  - [ ] Runs training
  - [ ] Parses results
  - [ ] Runs inference
  - [ ] Evaluates
  - [ ] Decides keep/discard
  - [ ] Logs to TSV

### Phase 5: Advanced
- [ ] Implement UCB strategy in ModelSelector
- [ ] Implement evolutionary strategy
- [ ] Add hyperparameter grid search
- [ ] Add automated stopping criteria (plateau detection)

---

## Part 7: Example Experiment Sequence

### Baseline Run (Experiment 1)
```
Model: Phi-3-mini
Config: LR=5e-5, LORA_R=16, batch_size=16, task_weights=0.4/0.3/0.3
Result: val_metric=0.742 (F1-macro)
Status: KEEP (baseline established)
```

### Experiment 2: Speed vs Quality (DistilBERT)
```
Model: DistilBERT
Config: LR=1e-4, LORA_R=8, batch_size=32, task_weights=0.4/0.3/0.3
Result: val_metric=0.718 (lower F1, but 10x faster inference)
Status: DISCARD (quality matters more than speed for this task)
```

### Experiment 3: Increase LR
```
Model: Phi-3-mini
Config: LR=1e-4 (increased), LORA_R=16, batch_size=16, task_weights=0.4/0.3/0.3
Result: val_metric=0.751 (+0.009 improvement!)
Status: KEEP (new baseline)
```

### Experiment 4: Adjust Task Weights
```
Model: Phi-3-mini
Config: LR=1e-4, LORA_R=16, batch_size=16, task_weights=0.5/0.25/0.25 (favor sentiment)
Result: val_metric=0.748 (slight decline)
Status: DISCARD (revert to previous)
```

### Experiment 5: Larger Batch Size
```
Model: Phi-3-mini
Config: LR=1e-4, LORA_R=16, batch_size=32 (increased), task_weights=0.4/0.3/0.3
Result: val_metric=0.755 (+0.004 improvement, but memory increased)
Status: KEEP (acceptable tradeoff)
```

---

## Part 8: Automated Research Loop Script

**File: run_experiments.py** (optional, for fully autonomous operation)

```python
#!/usr/bin/env python3
"""
Fully autonomous experiment loop.
Runs until manually interrupted.
"""

import os
import subprocess
import time
import json
from model_selector import ModelSelector
from benchmark_tracker import BenchmarkTracker
from prepare import get_model_config

def run_experiment(step, model_name, config):
    """Execute one full experiment cycle."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {step}: {model_name}")
    print(f"Config: {config}")
    print('='*60)
    
    # 1. Update train.py with new model/config
    update_train_py(model_name, config)
    
    # 2. Git commit
    subprocess.run(["git", "add", "train.py"], check=True)
    subprocess.run(["git", "commit", "-m", f"exp {step}: {model_name}"], check=True)
    commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    
    # 3. Run training
    result = subprocess.run(
        ["uv", "run", "train.py"], 
        capture_output=True, 
        text=True,
        timeout=600  # 10 min max
    )
    
    if result.returncode != 0:
        print("CRASHED!")
        print(result.stderr[-500:])
        tracker.log_run(model_name, config, 0.0, 0.0, status="crash")
        subprocess.run(["git", "reset", "--hard", "HEAD~1"], check=True)
        return
    
    # 4. Parse metrics
    lines = result.stdout.split('\n')
    metrics = {}
    for line in lines:
        if line.startswith("val_metric:"):
            metrics['val_metric'] = float(line.split()[-1])
        elif line.startswith("peak_vram_mb:"):
            metrics['peak_vram_mb'] = float(line.split()[-1])
        elif line.startswith("training_seconds:"):
            metrics['training_seconds'] = float(line.split()[-1])
    
    val_metric = metrics.get('val_metric')
    vram_mb = metrics.get('peak_vram_mb')
    
    if val_metric is None:
        print("ERROR: Could not parse metrics!")
        subprocess.run(["git", "reset", "--hard", "HEAD~1"], check=True)
        return
    
    # 5. Evaluate & log
    tracker.log_run(model_name, config, val_metric, vram_mb, status="keep")
    tracker.update_leaderboard()
    
    # 6. Decide: keep or discard
    if val_metric < best_metric:
        best_metric = val_metric
        print(f"✓ IMPROVED! New best: {val_metric:.6f}")
        # Keep commit
    else:
        print(f"✗ No improvement (best: {best_metric:.6f})")
        subprocess.run(["git", "reset", "--hard", "HEAD~1"], check=True)

def main():
    global best_metric, tracker
    
    tracker = BenchmarkTracker()
    best_metric = float('inf')
    step = 0
    
    print("Starting autonomous experiment loop...")
    print("Press Ctrl+C to stop.\n")
    
    try:
        while True:
            # Select next model
            model_name = ModelSelector.round_robin(step)
            config = get_model_config(model_name)
            
            # Run experiment
            run_experiment(step, model_name, config)
            
            # Log and advance
            step += 1
            time.sleep(2)  # Brief pause between runs
    
    except KeyboardInterrupt:
        print("\n\nExperiments stopped by user.")
        tracker.export_leaderboard()
        print("\nFinal leaderboard exported.")

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
python run_experiments.py
```

---

## Part 9: Key Metrics & Evaluation

### How We Measure Success

| Metric | Range | Target | Notes |
|--------|-------|--------|-------|
| **val_metric (F1)** | 0-1 | ↑ Maximize | Weighted average of task F1 scores |
| **peak_vram_mb** | MB | ↓ Minimize | Soft constraint, acceptable increase for F1 gains |
| **time_per_inference** | ms | ↓ Minimize | Tracked but not primary objective |
| **json_schema_compliance** | % | 100% | Validation that outputs match schema |
| **task_f1_sentiment** | 0-1 | ↑ | Sentiment classification F1 |
| **task_f1_ner** | 0-1 | ↑ | Token-level F1 for entities |
| **task_f1_themes** | 0-1 | ↑ | Multi-label F1 for themes |

### Evaluation Strategy

```python
# In eval.py
def evaluate_multi_task(sentiment_f1, ner_f1, theme_f1):
    """
    Combined metric (higher is better).
    Weights: sentiment=0.4, NER=0.3, themes=0.3
    """
    return 0.4 * sentiment_f1 + 0.3 * ner_f1 + 0.3 * theme_f1
```

### Pareto Frontier Analysis

Track models on multiple dimensions:
```
┌──────────────────────────────────────┐
│    Model Performance Frontier        │
└──────────────────────────────────────┘

         F1 Score (higher better)
              ↑
              │     ● Phi-3 (best F1)
              │      \
              │       ● Qwen2
              │        \
              │         ● DistilBERT (fast)
              │
              └─────────────────→
                VRAM (lower better)
                + Inference Speed (lower better)
                
Best choices:
  - Quality-focused: Phi-3-mini
  - Balanced: Qwen2-3B
  - Speed-focused: DistilBERT
```

---

## Part 10: Common Issues & Troubleshooting

### Issue: Running out of VRAM
**Solutions:**
- Reduce `BATCH_SIZE` (8 → 4)
- Enable LoRA with smaller rank (32 → 16)
- Use DistilBERT instead of Phi-3
- Reduce `MAX_SEQ_LEN` in prepare.py if applicable

### Issue: Training not improving
**Solutions:**
- Increase `LEARNING_RATE` (5e-5 → 1e-4)
- Adjust `TASK_WEIGHTS` to focus on weak tasks
- Try different optimizer (Adam → AdamW)
- Check that labels are correct

### Issue: JSON validation failing
**Solutions:**
- Check Outlines grammar matches schema
- Ensure model is instruction-tuned (Phi-3, not random LLM)
- Increase `max_tokens` in generation
- Add more examples in prompt

### Issue: Slow inference
**Solutions:**
- Use DistilBERT (66M params vs 3B)
- Enable quantization (int8)
- Batch process instead of single
- Use CPU for lightweight models

---

## Part 11: Next Steps

### Immediate (Week 1)
1. Prepare labeled survey CSV with required columns
2. Create prepare.py with data loading and schemas
3. Create baseline train.py with Phi-3-mini
4. Run first experiment to establish baseline F1

### Short-term (Week 2)
1. Implement model_selector.py
2. Implement benchmark_tracker.py
3. Create inference.py with Outlines
4. Run 20-30 experiments across model selection

### Medium-term (Week 3-4)
1. Analyze leaderboard for best models
2. Implement hyperparameter tuning
3. Test all 4 phases of experimentation
4. Identify simplifications or improvements

### Long-term
1. Automate full loop with run_experiments.py
2. Add evolutionary strategies
3. Benchmark against manual baselines
4. Publish results

---

## Conclusion

This plan transforms autoresearch from a language modeling optimizer into a **comprehensive multi-task NLP research platform** for survey analytics. The fixed 5-minute budget ensures reproducible experiments, while the modular design allows systematic exploration of:

- ✅ Model selection (Phi-3 vs DistilBERT vs others)
- ✅ Architecture choices (LoRA ranks, task weights, heads)
- ✅ Hyperparameter tuning (LR, batch size, scheduler)
- ✅ Output strategies (JSON schemas, Outlines enforcement)
- ✅ Multi-task learning (balanced vs. task-specific)

With automation, you can run **100+ experiments overnight** while you sleep, leveraging the fixed compute budget to explore a large design space efficiently.
