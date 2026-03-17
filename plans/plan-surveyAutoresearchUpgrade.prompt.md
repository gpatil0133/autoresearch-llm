## Plan: Survey Autoresearch Upgrade

Replace the current single-metric GPT pretraining loop with a survey-text fine-tuning pipeline that supports both generative structured-output LLMs and encoder baselines, keeps the fixed 5-minute experiment budget, and evaluates experiments with task-specific metrics plus JSON-schema compliance. The recommended path is a staged refactor: first convert data prep and evaluation around your labeled CSV and required JSON schema, then introduce a baseline training path for decoder and encoder families, then automate the train -> inference -> eval -> benchmark loop.

**Steps**
1. Phase 1 - Reframe core repo responsibilities. Redefine prepare.py from climbmix/tokenizer prep into survey data + schema + metric utilities while preserving the useful fixed constants pattern such as TIME_BUDGET from the current implementation. This step blocks all later steps.
2. In prepare.py, define the canonical dataset contract for the initial local CSV under data/. Include required columns for raw text, overall sentiment, sentence-level sentiments, metadata/entity spans and types with sentiment, themes, and emotions. Normalize these into one in-memory example schema so train/inference/eval all consume the same representation. Depends on 1.
3. In prepare.py, define the structured-output schema that exactly matches the requested text_analysis JSON envelope. Use strict enum-like validation for sentiment fields, score bounds for sentiment/relevance/intensity values, and preserve original-language text in sentence and phrase fields. Depends on 1 and 2.
4. In prepare.py, replace the current pretraining dataloader/evaluate_bpb utilities with task utilities for: dataset split loading, tokenizer/model-family aware collation, sentiment targets, token-level metadata targets, theme/emotion multi-label targets, and combined validation scoring. Reuse the current pattern of central shared helpers in prepare.py rather than scattering evaluation logic. Depends on 2 and 3.
5. Phase 2 - Establish experiment abstractions. Implement model_selector.py with a single selection interface and start with deterministic round-robin plus random strategies only. Add model registry entries for at least one decoder baseline and one encoder baseline, with per-family defaults for learning rate, batch size, LoRA usage, and max sequence length. Depends on 4.
6. Add a benchmark tracker module or fold equivalent logic into eval.py to own results.tsv writes, leaderboard.json generation, and experiment summaries. Keep results.tsv append-only and make leaderboard generation derived, not hand-maintained. This can run in parallel with 5 once 4 is done.
7. Phase 3 - Build a new training baseline in train.py. Replace the current custom GPT pretraining stack with a fine-tuning driver that loads Hugging Face backbones, applies family-appropriate adapters, trains within the existing fixed wall-clock budget, and prints grep-friendly summary metrics. Reuse the current wall-clock stop condition, VRAM tracking, and summary-print pattern; do not carry over the Muon/custom GPT stack. Depends on 4 and 5.
8. In train.py, support two baseline paths behind one config surface: decoder/instruction LLM path for JSON generation training and encoder path for classification/token-label baselines. Keep shared experiment knobs near the top of the file so the autonomous loop can mutate them cleanly. Depends on 7.
9. Decide and document the optimization metric as higher-is-better weighted validation score, not the original lower-is-better BPB. Avoid inverting the metric; instead update the loop logic and result comparison semantics everywhere to maximize val_metric directly. Depends on 6 and 7.
10. Phase 4 - Implement structured inference in inference.py. Load the trained checkpoint, build prompts for survey analysis, enforce the requested JSON schema during generation, validate outputs, and persist per-example predictions. Decoder models should use constrained structured generation; encoder models should use a deterministic post-processor that assembles the same JSON contract from head outputs so both families are benchmarked against one output schema. Depends on 3, 4, 5, and 7.
11. In inference.py, keep original-language text spans by deriving sentence_sentiments and theme_associated_phrases from source text spans rather than regenerated paraphrases whenever labels/spans are available. This is important for your stated output requirement. Depends on 10.
12. Phase 5 - Implement evaluation in eval.py. Score each task independently, compute JSON-schema compliance, compute an aggregate validation metric, and log the full experiment record including model family and config. Treat missing/invalid JSON as a measurable failure mode rather than silently dropping records. Depends on 6 and 10.
13. Define evaluation granularity explicitly: overall sentiment macro F1, metadata/entity extraction span or token F1 plus type correctness, theme multi-label F1, optional emotion classification F1, sentence-level sentiment accuracy/F1, and JSON compliance percentage. The aggregate score should weight only the tasks you want to optimize globally, while still logging all submetrics for diagnosis. Depends on 12.
14. Phase 6 - Automate the loop. Add a run_experiments orchestrator that selects a model/config, updates train.py or a dedicated experiment config, runs training, runs inference, runs eval, logs outcomes, and decides keep/discard without destructive history rewriting. Prefer recording experiment states instead of using git reset --hard; this repo may be dirty and the autonomous loop should not discard unrelated work. Depends on 5, 7, 10, and 12.
15. Update program.md to reflect the new contract: full-repo refactor is now allowed, prepare.py is no longer immutable, dependencies can expand, the target metric is val_metric, and the loop includes inference plus structured-output validation rather than train-only BPB evaluation. Depends on 14.
16. Update README.md and dependency declarations in pyproject.toml so setup reflects survey CSV preparation, model downloads, new packages, and the baseline commands for prepare, train, inference, eval, and automated experimentation. Depends on 7, 10, 12, and 14.
17. Phase 7 - Verification and rollout. First verify data loading and schema validation on a tiny hand-checked sample CSV, then run one decoder baseline and one encoder baseline end-to-end under the 5-minute budget, then verify leaderboard/result logging, then enable unattended experiment cycling. Depends on all previous steps.

**Relevant files**
- f:\autoresearch-llm\autoresearch\prepare.py - currently owns constants, data prep, dataloader, and evaluation; reuse that central utility role but replace climbmix/BPE/BPB logic.
- f:\autoresearch-llm\autoresearch\train.py - currently owns the entire experimental training loop and fixed-budget stop logic; replace the custom GPT stack with fine-tuning flows.
- f:\autoresearch-llm\autoresearch\eval.py - currently empty; implement scoring, logging, and aggregate metric computation here.
- f:\autoresearch-llm\autoresearch\inference.py - currently empty; implement schema-constrained structured output generation and prediction export here.
- f:\autoresearch-llm\autoresearch\model_selector.py - currently empty; implement model registry and selection strategies here.
- f:\autoresearch-llm\autoresearch\program.md - currently still describes the original single-file train.py mutation loop and must be rewritten for the new workflow.
- f:\autoresearch-llm\autoresearch\README.md - currently documents the original pretraining setup and must be updated to the survey analytics workflow.
- f:\autoresearch-llm\autoresearch\pyproject.toml - currently lacks Hugging Face fine-tuning, schema, and metric dependencies.
- f:\autoresearch-llm\autoresearch\SURVEY_ANALYTICS_PLAN.md - use as the product-level design input, but tighten decisions around metric semantics, data contract, and non-destructive experiment retention.

**Verification**
1. Validate the local CSV parser against a small fixture containing all selected labels and multilingual text; confirm sentence text and theme-associated phrases remain identical to source substrings.
2. Validate the schema layer by intentionally feeding invalid sentiment labels and out-of-range scores and confirming they are rejected before inference/eval logging.
3. Run one decoder baseline end-to-end and confirm train.py prints val_metric, peak_vram_mb, model_name, and config values in a stable parseable form.
4. Run one encoder baseline end-to-end and confirm inference.py can still emit the exact same JSON envelope through deterministic assembly.
5. Confirm eval.py logs per-task metrics plus aggregate val_metric and JSON compliance into results.tsv and regenerates leaderboard.json deterministically.
6. Dry-run the experiment orchestrator on two short experiments and verify keep/discard decisions do not rewrite unrelated git history.

**Decisions**
- Input source: initial implementation assumes a local CSV under data/.
- Label availability: plan assumes overall sentiment, sentence-level sentiment, metadata/entity spans with types and sentiment, theme labels, and emotion labels already exist in the dataset.
- Baseline strategy: compare decoder and encoder families from the start rather than treating encoder models only as a later optimization.
- Scope: full repo refactor is allowed, including prepare.py changes, new modules, and dependency expansion.
- Output contract: every inference path must emit the exact requested text_analysis JSON structure.
- Included: training, inference, evaluation, model selection, benchmark logging, and autonomous experiment orchestration.
- Excluded for MVP: UCB/evolutionary selection, large hyperparameter search, distributed training, and production serving APIs.

**Further Considerations**
1. Decoder training target recommendation: start with supervised JSON generation from labeled examples, not separate multi-head modeling inside the decoder path. Use multi-head classification only for encoder baselines.
2. Metadata evaluation choice: prefer span-level evaluation if offsets are available in the CSV; fall back to token-level only if annotations are BIO-tag based.
3. Experiment configuration recommendation: use a small config object or sidecar file instead of mutating train.py text directly once the basic loop works, because it is safer for unattended runs and easier to compare across model families.