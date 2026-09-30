# RKBQA

Code and revision datasets for **RKBQA: Revise-Once Correction for Generate-then-Retrieve KBQA** (Wenbo Huang and Lihui Liu, IEEE Big Data 2025).

[Paper](https://doi.org/10.1109/BigData66926.2025.11401949)

RKBQA inserts an optional single correction step between logical-form generation and knowledge-base retrieval. A relation-based prescreen can skip correction for high-confidence predictions. Otherwise, a fine-tuned language model checks the top predicted S-expression and may propose a revision. The revision is prepended to the candidate list, preserving the original candidates for downstream execution.

## Repository contents

| File | Purpose |
| --- | --- |
| `simple_corrector.py` | Build a relation-based prescreen from labeled JSONL records. Returning `True` means skip model correction. |
| `train_bash.py` | Launch `llmtuner.train.tuner.run_exp()` using externally supplied training arguments; disable W&B reporting. |
| `train_pred_best_k_llama2.py` | Load a Llama-2-7B base model and LoRA adapter, prescreen the top candidate, and run correction when needed. |
| `train_pred_best_k_llama3.py` | Load Llama-3.1-8B-Instruct and a LoRA adapter, then run correction for each sample. Prescreening is commented out in this version. |
| `repair_sexpression.py` | Normalize spacing in the first candidate S-expression of each sample. |
| `result_analysis/result_analysis.py` | Join execution results, per-question scores, and WebQSP questions into a readable analysis JSON. |
| `data/ORS_revision_set.json` | Revision dataset. |
| `data/llama3_sft_alpaca_3000questions_allerrors.jsonl` | Supervised fine-tuning dataset. |

The six Python files are preserved from the original experiment scripts. The two `train_pred_best_k_*` files perform **inference**, despite their names; they do not train a model.

## Environment and configuration

- Use Python 3.10 or newer for the scripts in this repository.
- The correction scripts require `tqdm`, a compatible `llmtuner` environment, model weights, and trained LoRA adapters. The training entry point also depends on `llmtuner`. These dependencies and model artifacts are not bundled here, and this release does not pin a tested dependency stack.
- `simple_corrector.py`, `repair_sexpression.py`, and `result_analysis/result_analysis.py` use only the Python standard library.
- Before running correction, edit `DATA_DIR`, `INPUT_JSON`, `OUTPUT_JSON`, `model_name_or_path`, and `adapter_name_or_path` in the selected script. The included `/mnt/public/algm/wb/...` paths refer to the original experiment machine.
- Llama2 prescreening also requires `DIRECT_STATS_JSON`, a JSONL file containing `pred_lf` and a Boolean `judge` per record. The referenced `gpt4o_cot_responses_allerrors_direct.jsonl` is not included in this repository. Do not substitute another dataset without checking its schema.
- Training data selection, hyperparameters, and output paths must be supplied to `train_bash.py` through the compatible `llmtuner` configuration or command-line interface.

Initial candidate generation, entity/relation retrieval, knowledge-base execution, and the original answer-scoring implementation are external to these six scripts. They are intended to be used with the surrounding ChatKBQA pipeline.

## Correction inference

Both correction scripts read a JSON array with at least these fields:

```json
[
  {
    "question": "What language do people in Jamaica speak?",
    "predictions": [
      "( JOIN [ location , country , languages spoken ] [ Jamaica ] )"
    ]
  }
]
```

After configuring the paths and installing the compatible model environment, run one of:

```bash
python train_pred_best_k_llama2.py
python train_pred_best_k_llama3.py
```

Each script checks only `predictions[0]`. If the first model response contains a nonempty `<revised_logic_form>...</revised_logic_form>` tag, the extracted revision is inserted at index 0. Existing candidates and other sample fields are retained. The output also records `prescreen_skip`.

### Experiment settings preserved in the scripts

| Setting | Llama2 | Llama3 |
| --- | --- | --- |
| Base model | Llama-2-7b-hf | Meta-Llama-3.1-8B-Instruct |
| Prescreen | Enabled: threshold `0.85`, minimum frequency `5` | Disabled in the current script |
| Initialization `num_beams` | `15` | `40` |
| `chat()` call `num_beams` | `15` | **`50`** |
| `do_sample` | `False` | `False` |

The Llama3 file contains both the initialization value of 40 and the per-call value of 50. Their application depends on the installed `llmtuner` implementation. Beam search settings here control correction generation; the script reads only the first returned response and does **not** export 50 revised candidates per question.

### Prescreen behavior

- `make_prescreen_func_from_direct()` requires every extracted relation to meet `min_freq`, then compares the **minimum** relation score with `threshold`.
- `make_prescreen_func_from_revised()` averages the scores of relations that meet `min_freq`; it returns `False` when no usable relations are found.
- A relation score is the fraction of labeled predictions containing that relation that were marked correct. It is not an independently annotated relation-level correctness label.

The two prescreen functions therefore use different aggregation rules; the current Llama2 script uses the `direct` version.

## Normalize S-expression spacing

```bash
python repair_sexpression.py --input predictions.json --output predictions_fixed.json
```

This utility accepts either a top-level `predictions` list or a nested `pred.predictions` list per sample. It changes only the first candidate. It normalizes whitespace and punctuation spacing; it does not validate semantics or repair missing parentheses.

## Build per-question analysis

After running the external knowledge-base execution and scoring pipeline:

```bash
python result_analysis/result_analysis.py \
  --beam execution_results.json \
  --score per_question_scores.json \
  --test WebQSP.test.json \
  --out analysis/webqsp_test_eval_summary.json
```

The execution and score files are joined by `qid`. The question file must use WebQSP's `Questions`, `QuestionId`, and `RawQuestion` fields. The analysis selects the normalized prediction using `execute_index` and marks a sample correct only when its existing precision, recall, F1, and hit scores all equal 1. It does not calculate those input scores itself.

This repository is a release of experiment scripts and datasets, not a self-contained end-to-end runtime. Model inference and full reproduction require the external resources and configuration described above.
