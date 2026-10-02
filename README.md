# Digital Persona Agents

This repo provides a CLI workflow for predicting survey answers for one selected user at cutoff year `2019` or `2023`.

## Research Goal

We want a deep and thorough analysis of the prediction results. The goal is not only to compare models, but to understand:

- when agents perform well
- when agents fail
- which questions are easy or difficult to predict
- which demographic groups lead to better or worse performance
- whether certain studies or survey types are more predictable
- whether certain agent architectures are consistently better
- whether lexical or semantic retrieval improves performance, and under which conditions
- whether some LLMs are systematically better for specific question types
- where digital personas produce answers close to real participants, and where they do not

The analysis should identify the conditions under which digital personas can reliably approximate real human responses and produce conclusions similar to those we would obtain from surveying real people.

The ultimate research objective is to understand when digital personas can serve as a scientifically reliable substitute for real human participants, and when they cannot.

This project should support a rigorous, insight-driven analysis focused on extracting actionable conclusions, failure modes, and practical recommendations for future use of digital personas in scientific research.

## Repository Layout

```
scripts/   — all pipeline scripts (step1_ … step9_, exp_*)
docs/      — workflow and design documentation (excluding README)
```

## Entry Points

- [scripts/step5_ask_user_question.py](scripts/step5_ask_user_question.py) is the interactive runner. It prompts for dataset source, cutoff year, question set, input scope, agent mode, batch size, and model choices at runtime.
- [scripts/step6_run_single.py](scripts/step6_run_single.py) is the non-interactive fixed runner. It always uses:
  - `dataset_source = datasets_categories_only`
  - `cutoff_year = 2023`
  and takes `question_set` and profile-based `input_scope` from the command line.
- [scripts/step1_sample_users.py](scripts/step1_sample_users.py) builds stratified 500-user sample CSVs for the categories-only 2023/2024 sampling workflow. It can run one selected sample definition or all sample definitions.

## Data Layout

- `step5_ask_user_question.py` lets you choose one source dataset at runtime:
  - `datasets`
  - `datasets_categories_only`
- `datasets` is the full source dataset.
- `datasets_categories_only` contains only closed-ended questions.
- User data lives under:
  - `datasets/nomem_encr_cutoff_2019/{user_id}/`
  - `datasets/nomem_encr_cutoff_2023/{user_id}/`
  - `datasets_categories_only/nomem_encr_categories_only_cutoff_2019/{user_id}/`
  - `datasets_categories_only/nomem_encr_categories_only_cutoff_2023/{user_id}/`
- Valid selectable `user_id` values:
  - for `datasets`, come from `datasets/shared_both_overlap_2019_2023.json` -> key `both_overlap`
  - for `datasets_categories_only`, are computed from the overlap of the available 2019 and 2023 user folders
- Per-user files:
  - `{user_id}_bg_for_cutoff_<year>.csv`
  - `{user_id}_core_cutoff_<year>_before.csv`
  - `{user_id}_core_cutoff_<year>_latest_before.csv`
  - `{user_id}_core_cutoff_<year>_target.csv`
  - `{user_id}_single_cutoff_<year>_before.csv`
  - `{user_id}_single_cutoff_<year>_target.csv`
- For the 2023/2024 categories-only sampling workflow, some 2023 user folders also include:
  - `{user_id}_single_cutoff_2023_2024_target.csv`
- Input scope options:
  - `A`: background only
  - `B`: background + core before
  - `C`: background + single before
  - `D`: background + core before + single before
  - `E`: background + core latest before
  - `F`: background + core latest before + single before
- Target metadata files:
  - `standalone/core_all_waves_questions_metadata_standalone_v2_cutoff_<year>_target.csv`
  - `standalone/filtered_table_single_wave_cutoff_<year>_target.csv`
  - `standalone_categories_only/core_all_waves_questions_metadata_standalone_v2_categories_only_cutoff_<year>_target.csv`
  - `standalone_categories_only/filtered_table_single_wave_categories_only_cutoff_<year>_target.csv`
- Overlap questions to predict:
  - If dataset source is `datasets_categories_only` and question set is `core`, the overlap is between:
    `datasets_categories_only/nomem_encr_categories_only_cutoff_<year>/{user_id}/{user_id}_core_cutoff_<year>_target.csv`
    and
    `standalone_categories_only/core_all_waves_questions_metadata_standalone_v2_categories_only_cutoff_<year>_target.csv`
  - If dataset source is `datasets_categories_only` and question set is `single`, the overlap is between:
    `datasets_categories_only/nomem_encr_categories_only_cutoff_<year>/{user_id}/{user_id}_single_cutoff_<year>_target.csv`
    and
    `standalone_categories_only/filtered_table_single_wave_categories_only_cutoff_<year>_target.csv`
  - If dataset source is `datasets` and question set is `core`, the overlap is between:
    `datasets/nomem_encr_cutoff_<year>/{user_id}/{user_id}_core_cutoff_<year>_target.csv`
    and
    `standalone/core_all_waves_questions_metadata_standalone_v2_cutoff_<year>_target.csv`
  - If dataset source is `datasets` and question set is `single`, the overlap is between:
    `datasets/nomem_encr_cutoff_<year>/{user_id}/{user_id}_single_cutoff_<year>_target.csv`
    and
    `standalone/filtered_table_single_wave_cutoff_<year>_target.csv`

Dataset source details and JSON ID-list descriptions are documented in [DATASET.md](datasets/DATASET.md).

## Agent Modes

- `agent-baseline`
- `agent-profile`
- `agent-profile-topK-lexical`
- `agent-profile-topK-rows-semantic`
- `agent-profile-topK-para-semantic`

Agent-mode selection behavior:

- If you choose a specific mode, the script runs only that mode.
- If you press Enter without choosing a specific mode, the script runs two automatic parallel phases:
  - phase 1: `agent-baseline` and `agent-profile`
  - phase 2: `agent-profile-topK-lexical` and `agent-profile-topK-rows-semantic`

Detailed workflow, prediction behavior, validation, and linked implementation functions are documented in [docs/AGENT_WORKFLOW.md](docs/AGENT_WORKFLOW.md).

## Precompute Embeddings

Use [scripts/step2_precompute_embeddings.py](scripts/step2_precompute_embeddings.py) before running:

- `agent-profile-topK-rows-semantic`

Embed all metadata `*_before.csv` and `*_target.csv` files in both folders:

```bash
.venv312/bin/python scripts/step2_precompute_embeddings.py
```

Embed only `*_before.csv` files:

```bash
.venv312/bin/python scripts/step2_precompute_embeddings.py --file-kind before
```

Embed only `*_target.csv` files in `standalone_categories_only/`:

```bash
.venv312/bin/python scripts/step2_precompute_embeddings.py --directory-scope standalone_categories_only --file-kind target
```

Embed specific CSV files only:

```bash
.venv312/bin/python scripts/step2_precompute_embeddings.py \
  --csv standalone_categories_only/filtered_table_single_wave_categories_only_cutoff_2023_before.csv \
  --csv standalone_categories_only/core_all_waves_questions_metadata_standalone_v2_categories_only_cutoff_2023_target.csv
```

The script writes sibling embedding files ending in `*_embeddings.json`, where each file maps `variable_name` to its embedding vector.

Runtime embedding notes:

- `agent-profile-topK-rows-semantic` needs both `*_before.csv` and `*_target.csv` metadata embeddings.

## Run

### Install dependencies:

```bash
.venv312/bin/pip install -r requirements.txt
```

### Run the interactive CLI:

```bash
.venv312/bin/python scripts/step5_ask_user_question.py
```

When prompted for agent mode, press Enter to run the default grouped parallel workflow, or enter `1`, `2`, `3`, or `4` to run only one specific mode.

When prompted for answer prediction, you now also choose a prediction reasoning mode:

- `none`: do not send an explicit reasoning setting to the prediction model
- `medium`: request medium reasoning for the prediction model

### Run the fixed non-interactive runner:

```bash
.venv312/bin/python scripts/step6_run_single.py <user_id> <question_set> <input_scope> <batch_target_size> <profile_provider> <profile_model> <prediction_provider> <prediction_model> [--prediction-reasoning none|medium]
```

Example:

```bash
.venv312/bin/python scripts/step6_run_single.py 899923 single d 10 openai gpt-5.4 openai gpt-5.4 --prediction-reasoning none
```

Optional single-mode run for the fixed runner:

```bash
.venv312/bin/python scripts/step6_run_single.py 899923 core e 10 openai gpt-5.4 openai gpt-5.4 --agent-mode profile --prediction-reasoning none
```

Notes for the fixed runner:

- `question_set` must be `core` or `single`.
- `input_scope` can be `A`, `B`, `C`, `D`, `E`, or `F`:
  - `A`: background only
  - `B`: background + core before
  - `C`: background + single before
  - `D`: background + core before + single before
  - `E`: background + core latest before
  - `F`: background + core latest before + single before
- `agent-baseline` still always uses background-only scope `A`.
- Profile-based modes use the provided `input_scope`.

Crash recovery behavior for the fixed runner:

- successful mode runs write a `.done.json` marker next to the output CSV
- rerunning the same command skips completed modes and resumes only the unfinished ones

### Run the fixed batch runner for a JSON user list:

```bash
.venv312/bin/python scripts/step7_run_batch.py <user_ids_json> <question_set> <input_scope> <batch_target_size> <profile_provider> <profile_model> <prediction_provider> <prediction_model> --parallel-users <n> [--prediction-reasoning none|medium]
```

**Example with cs->sw:**

```bash
.venv312/bin/python scripts/step7_run_batch.py sample_500_2023_2024_categories_only/cs_to_sw_500users.json single e 20 openai gpt-5.4 openai gpt-5.4 --parallel-users 4 --prediction-reasoning none
```
**Example with sw->cs:**

```bash
.venv312/bin/python scripts/step7_run_batch.py sample_500_2023_2024_categories_only/sw_to_cs_500users.json core c 20 openai gpt-5.4 openai gpt-5.4 --parallel-users 4 --prediction-reasoning none
```

**Batch runner notes:**

- `--parallel-users` is adjustable. Start with `4` on this laptop and tune later if needed.
- `--limit` is optional if you want to run only the first N user IDs from the JSON list for a small testcase.
- The JSON file can be changed at runtime, so you can switch between files like `part_1`, `part_2`, `part_3`, or `part_4` without editing code.
- `--agent-mode` is optional if you want to run only one fixed mode instead of the default two-phase workflow.
- `--prediction-reasoning` is optional and defaults to `none`. Set it to `medium` to request medium reasoning from the prediction model.
- `--fail-fast` is optional if you want the batch launcher to stop scheduling new users after the first failure.
- Per-user logs and a batch `summary.json` are written under `batch_run_logs/`.
- Each batch log subfolder is named as a sanitized version of `<user_ids_json>_<date>_<time>`.
- See [docs/BATCH_RUN_LOGS.md](docs/BATCH_RUN_LOGS.md) for a brief guide to tracking batch results and inspecting failed users.

**Testcase example using only the first 12 user IDs from part 1:**

```bash
.venv312/bin/python scripts/step7_run_batch.py sample_500_2023_2024_categories_only/cs_to_sw_500users.json single e 20 openai gpt-5.4 openai gpt-5.4 --parallel-users 4 --limit 2 --prediction-reasoning none
```

### Build 500-user stratified samples:

[scripts/step1_sample_users.py](scripts/step1_sample_users.py) builds respondent-level metadata, allocates a stratified sample, and writes sample CSVs under:

```text
sample_500_2023_2024_categories_only/
```

Run it with:

```bash
python3 scripts/step1_sample_users.py
```

If Python tries to write bytecode outside the workspace, use:

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache python3 scripts/step1_sample_users.py
```

At the top of `step1_sample_users.py`, choose which sample definitions to run:

```python
samples_to_run = ["cs_to_sw"]
```

Valid sample keys:

- `sw_to_cs`: Sample 1, single-wave before -> core target
- `sw_to_sw`: Sample 2, single-wave before -> single-wave target
- `cssw_to_sw`: Sample 3, core before + single-wave before -> single-wave target
- `cs_to_sw`: Sample 4, core latest before -> single-wave 2023/2024 target
- `all`: run every sample definition

Examples:

```python
samples_to_run = ["cs_to_sw"]
samples_to_run = ["sw_to_cs", "cs_to_sw"]
samples_to_run = ["all"]
```

Current Sample 4 file mapping:

- left/profile side: `{user_id}_core_cutoff_2023_latest_before.csv`
- right/target side: `{user_id}_single_cutoff_2023_2024_target.csv`
- output key: `cs_to_sw`

`step1_sample_users.py` classifies non-background files separately so these files are not accidentally combined:

- `core_before`
- `core_latest_before`
- `core_target`
- `single_before`
- `single_target`
- `single_2023_2024_target`

Background fields used for stratification:

- `gender`: uses `geslacht` first, then `gender`, then `d1`. This avoids treating `gender = "Missing - question was never asked"` as `Other` when `geslacht` has `Male` or `Female`.
- `age_group`: derives from numeric `age` / `leeftijd` when available.
- `children_count`: derives from `aantalki`.
- `household_stage`: derives from partner, `aantalki`, and `woonvorm`.

The `household_stage` labels are mutually exclusive because each respondent receives the first matching label:

- `Family_with_children`
- `Couple_no_children`
- `Single_no_children`
- `Other_multi_person`
- `Unknown`

Sampling notes:

- `target_n = 500`.
- `min_stratum_size = 5` is used to identify very small primary strata before fallback labeling. It is not a guarantee that every final allocated stratum receives at least 5 sampled users.
- Allocation is proportional across `stratum_final`, then respondents within each stratum are ranked by previous-answer and target-answer coverage.

For each selected sample key, the script writes:

- `allocation_<sample_key>.csv`
- `strata_compare_<sample_key>.csv`
- `sample_<sample_key>.csv`
- `sample_<sample_key>_ranked_check.csv`

It also writes shared metadata:

- `eligible_metadata_from_json.csv`

For the current Sample 4 run, useful generated files include:

- `sample_500_2023_2024_categories_only/sample_cs_to_sw.csv`
- `sample_500_2023_2024_categories_only/sample_cs_to_sw_ranked_check.csv`
- `sample_500_2023_2024_categories_only/allocation_cs_to_sw.csv`
- `sample_500_2023_2024_categories_only/strata_compare_cs_to_sw.csv`
- `sample_500_2023_2024_categories_only/sample_cs_to_sw_summary_tables.csv`
- `sample_500_2023_2024_categories_only/sample_cs_to_sw_summary_tables.xlsx`

### Combine prediction CSVs for one user, question set, cutoff year, and input scope:

```bash
.venv312/bin/python combine_prediction_csvs.py 899923 single 2019 C
```

### Aggregate exact-match accuracy across all users for one question set, cutoff year, and input scope:

```bash
.venv312/bin/python combine_prediction_csvs.py all single 2023 D
```

Optional filters and custom output location:

```bash
.venv312/bin/python combine_prediction_csvs.py all single 2023 D --dataset-source datasets_categories_only --output pred_ouput/_aggregated_accuracy/my_run
```

## Output

- Output files are written to `pred_ouput/{user_id}/{source_dataset}_{year}_{input_scope}_{question_set}/`
- Filename format:
  - `{user_id}_{dataset_source}_{core_or_single}_cutoff_<year>_<input_option>_b<batch_target_size>_<agent_label>_profile-[<LLM for the structured profile>]_pred-[<LLM version>].csv`
- Files are grouped so each subfolder contains only runs sharing the same `source_dataset`, cutoff `year`, input `scope`, and `question_set`.
- [combine_prediction_csvs.py](combine_prediction_csvs.py) supports two modes:
  - single-user mode: combine only the prediction CSVs in `pred_ouput/{user_id}/` that match the provided `question_set` (`core` or `single`), cutoff `year`, and input `scope` (`A`, `B`, `C`, `D`, `E`, or `F`)
  - all-user mode: scan every `pred_ouput/{user_id}/` folder that matches the provided `question_set`, cutoff `year`, and input `scope`, then compute exact-match accuracy across all rows
- For `datasets_categories_only`, `single`, cutoff year `2023`, all-user mode also adds a fifth comparison line called `Baseline (no bg)`.
- `Baseline (no bg)` is built from `500sample/sample_cssw_to_sw_natural_answer_500times.csv` by mapping:
  - `prediction_001` ... `prediction_500`
  - to the user IDs listed in order across `500sample/sample_cssw_to_sw_folder_name_part_1.json` through `part_4.json`
- For each mapped user, the script compares that prediction column to the true `answer` values in:
  - `datasets_categories_only/nomem_encr_categories_only_cutoff_2023/{user_id}/{user_id}_single_cutoff_2023_target.csv`
- Single-user default combined output filename:
  - `{user_id}_combined_predictions_{question_set}_cutoff_<year>_<scope>.csv`
- Single-user optional filter:
  - `--dataset-source <dataset_source>` if more than one dataset source exists under the same user
- All-user default output directory:
  - `pred_ouput/_aggregated_accuracy/{dataset_source}_{question_set}_cutoff_<year>_<scope}/`
- All-user generated files:
  - `all_prediction_exact_match_rows.csv`
- `accuracy_by_agent_model.csv` and `accuracy_by_agent_model.svg`
- `accuracy_by_study_id.csv`
- `accuracy_by_question_type.csv`
- `accuracy_by_agent_model_and_study_id.csv`
- `accuracy_by_agent_model_and_question_type.csv`
- In the `datasets_categories_only_single_cutoff_2023_D` aggregation, these summaries and plots now include `Baseline (no bg)` alongside the existing agent outputs, so the by-agent views reflect five agent labels in total.
- Study-based outputs enrich `study_id` values such as `[348]` by looking them up in `liss_study_metadata.csv`, then keep `study_id`, `study_name`, and a 2-3 word `study_name_concise` summary in the CSV files while using `study_name_concise` on the plots.
- Exact-match accuracy compares `Pred Answer` to the true answer with light normalization, so values like `1` and `1.0` are treated as the same answer.

## API Keys

- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Gemini: `GEMINI_API_KEY`
