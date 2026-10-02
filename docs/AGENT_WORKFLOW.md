# Agent Workflow

This file documents how each agent mode in [scripts/step5_ask_user_question.py](scripts/step5_ask_user_question.py) is implemented and which functions are responsible for each step.

## Runner Types

- [scripts/step5_ask_user_question.py](scripts/step5_ask_user_question.py) is the interactive runner. It prompts the user for runtime selections.
- [scripts/step6_run_single.py](scripts/step6_run_single.py) is the non-interactive fixed runner. It always uses `datasets_categories_only`, cutoff year `2023`, question set `single`, and input scope `D`, then runs either:
  - the default grouped parallel workflow
  - or one specific agent mode when `--agent-mode` is provided

Fixed runner command:

```bash
.venv312/bin/python scripts/step6_run_single.py <user_id> <batch_target_size> <profile_provider> <profile_model> <prediction_provider> <prediction_model> [--prediction-reasoning none|medium]
```

Example:

```bash
.venv312/bin/python scripts/step6_run_single.py 899923 10 openai gpt-5.4 openai gpt-5.4 --prediction-reasoning medium
```

## Shared Flow

All agent modes share these high-level steps:

1. Select the source dataset in [choose_dataset_source](scripts/step5_ask_user_question.py):
   - `datasets` for the full source
   - `datasets_categories_only` for closed-ended questions only
2. Select the agent mode in [choose_agent_mode](scripts/step5_ask_user_question.py).
   - If the user chooses one specific mode, the script runs only that mode.
   - If the user presses Enter without choosing a specific mode, the script runs two automatic parallel phases:
     - phase 1: `agent-baseline` and `agent-profile`
     - phase 2: `agent-profile-topK-lexical` and `agent-profile-topK-rows-semantic`
3. Select the `user_id` from the chosen dataset source via [list_available_users](scripts/step5_ask_user_question.py) and [choose_user_dir](scripts/step5_ask_user_question.py).
4. Select input files and split them into background and `*_before.csv` groups in [get_selected_file_groups](scripts/step5_ask_user_question.py#L610).
5. Load CSV content into prompt-ready context blocks with [build_context_from_files](scripts/step5_ask_user_question.py#L479).
6. Choose the LLM for answer prediction.
7. Choose the prediction reasoning mode:
   - `none` to omit an explicit reasoning setting
   - `medium` to request medium reasoning for the prediction model
8. Filter target questions to overlapping `variable_name` values between metadata and the user target file.
9. Group the overlapping target question set by `project_number` for core targets or `study_id` for single-wave targets.
10. Ask the user for batch parameter `b`, then split each `project_number` or `study_id` group into smaller prediction sub-batches using:
   - `No_q_s` = number of questions in that `project_number` or `study_id`
   - `No_B_s = ceil(No_q_s / b)`
   - `No_q_B_s = ceil(No_q_s / No_B_s)`
   - each sub-batch contains up to `No_q_B_s` questions from that group
11. Route each batch through [prepare_batch_prediction_prompt](scripts/step5_ask_user_question.py).
12. Send each batch prompt to the model and validate the batch response with [run_prediction_for_batch](scripts/step5_ask_user_question.py).

## `agent-baseline`

Purpose:
Use the selected raw CSV context directly.

Main functions:

- [choose_agent_mode](scripts/step5_ask_user_question.py#L193)
- [build_context_from_files](scripts/step5_ask_user_question.py#L471)
- [build_batch_prompt](scripts/step5_ask_user_question.py)
- [prepare_batch_prediction_prompt](scripts/step5_ask_user_question.py)
- [run_prediction_for_batch](scripts/step5_ask_user_question.py)
- [get_prediction_output_path](scripts/step5_ask_user_question.py#L523)

Workflow:

1. The user selects `agent-baseline`.
2. The selected scope files are converted into one raw prompt context.
3. The user selects one LLM/model for answer prediction and chooses whether prediction reasoning is `none` or `medium`.
4. The overlapping target question set is first grouped by `project_number` for core targets or `study_id` for single targets, then split into smaller sub-batches using the user-specified batch parameter `b`.
5. For each batch, [build_batch_prompt](scripts/step5_ask_user_question.py) creates one batch prompt covering every question in that group.
6. [run_prediction_for_batch](scripts/step5_ask_user_question.py) executes the model call and validates every answer in the batch.
7. [get_prediction_output_path](scripts/step5_ask_user_question.py#L523) writes an output filename in the form `{user_id}_{core_or_single}_cutoff_<year>_<input_option>_b<batch_target_size>_<agent_label>_profile-[none]_pred-[<prediction LLM>].csv`.

## `agent-profile`

Purpose:
Build one structured predictive user profile from the selected `*_before.csv` rows, then predict using background plus that profile.

Main functions:

- [get_selected_file_groups](scripts/step5_ask_user_question.py#L602)
- [build_profile_prompt](scripts/step5_ask_user_question.py#L628)
- [run_profile_builder](scripts/step5_ask_user_question.py#L659)
- [get_structured_profile_output_path](scripts/step5_ask_user_question.py#L567)
- [load_structured_profile](scripts/step5_ask_user_question.py#L596)
- [save_structured_profile](scripts/step5_ask_user_question.py#L585)
- [build_profile_prediction_prompt](scripts/step5_ask_user_question.py)
- [prepare_batch_prediction_prompt](scripts/step5_ask_user_question.py)
- [run_prediction_for_batch](scripts/step5_ask_user_question.py)
- [get_prediction_output_path](scripts/step5_ask_user_question.py#L523)

Workflow:

1. The user selects `agent-profile`.
2. The script isolates the selected `*_before.csv` files with [get_selected_file_groups](scripts/step5_ask_user_question.py#L602).
3. The user selects one LLM/model for structured profile generation.
4. [build_profile_prompt](scripts/step5_ask_user_question.py#L628) creates the profile-building prompt from the `*_before.csv` context.
5. [get_structured_profile_output_path](scripts/step5_ask_user_question.py#L567) computes the expected cached profile JSON path in `pred_ouput/{user_id}/{source_dataset}_{year}_{input_scope}_{question_set}/`.
6. If that JSON file already exists for the same `user_id`, year, input option, agent label, question set, and selected profile LLM, [load_structured_profile](scripts/step5_ask_user_question.py#L596) loads it and the script skips regeneration.
7. If that JSON file does not exist, [run_profile_builder](scripts/step5_ask_user_question.py#L659) receives the selected profile-generation LLM as an input parameter and sends the same prompt to that chosen model, then [save_structured_profile](scripts/step5_ask_user_question.py#L585) saves the result.
8. The user selects one LLM/model for answer prediction and chooses whether prediction reasoning is `none` or `medium`.
9. The overlapping target question set is split into batches by `project_number` for core targets or `study_id` for single targets.
10. For each batch, [build_profile_prediction_prompt](scripts/step5_ask_user_question.py) creates the final prediction prompt using:
   - background context
   - structured profile
11. [run_prediction_for_batch](scripts/step5_ask_user_question.py) executes the batch prediction and validates the result.
12. [get_prediction_output_path](scripts/step5_ask_user_question.py#L523) writes an output filename in the form `{user_id}_{core_or_single}_cutoff_<year>_<input_option>_b<batch_target_size>_<agent_label>_profile-[<structured profile LLM>]_pred-[<prediction LLM>].csv`.

## `agent-profile-topK-lexical`

Purpose:
Do everything in `agent-profile`, then add the top-`K` most related prior answered rows for each batch.

Main functions:

- [get_selected_file_groups](scripts/step5_ask_user_question.py#L602)
- [build_profile_prompt](scripts/step5_ask_user_question.py#L628)
- [run_profile_builder](scripts/step5_ask_user_question.py#L659)
- [get_structured_profile_output_path](scripts/step5_ask_user_question.py#L567)
- [load_structured_profile](scripts/step5_ask_user_question.py#L596)
- [save_structured_profile](scripts/step5_ask_user_question.py#L585)
- [tokenize_for_similarity](scripts/step5_ask_user_question.py#L674)
- [compute_relatedness_score](scripts/step5_ask_user_question.py#L680)
- [get_top_k_related_rows](scripts/step5_ask_user_question.py#L713)
- [build_relevant_rows_context](scripts/step5_ask_user_question.py#L731)
- [build_profile_topk_prediction_prompt](scripts/step5_ask_user_question.py)
- [prepare_batch_prediction_prompt](scripts/step5_ask_user_question.py)
- [run_prediction_for_batch](scripts/step5_ask_user_question.py)
- [get_prediction_output_path](scripts/step5_ask_user_question.py#L523)

Workflow:

1. The user selects `agent-profile-topK-lexical`.
2. The user selects one LLM/model for structured profile generation.
3. [get_structured_profile_output_path](scripts/step5_ask_user_question.py#L567) computes the expected cached profile JSON path in `pred_ouput/{user_id}/{source_dataset}_{year}_{input_scope}_{question_set}/`.
4. If that JSON file already exists for the same `user_id`, year, input option, agent label, question set, and selected profile LLM, [load_structured_profile](scripts/step5_ask_user_question.py#L596) loads it and the script skips regeneration.
5. If that JSON file does not exist, the script builds the same structured profile flow as `agent-profile`: [build_profile_prompt](scripts/step5_ask_user_question.py#L628) keeps the prompt fixed, [run_profile_builder](scripts/step5_ask_user_question.py#L659) uses the selected profile-generation LLM to generate the structured profile, and [save_structured_profile](scripts/step5_ask_user_question.py#L585) saves it.
6. The user selects one LLM/model for answer prediction.
7. The overlapping target question set is first grouped by `project_number` for core targets or `study_id` for single targets, then split into smaller sub-batches using the user-specified batch parameter `b`.
8. For each original `project_number` or `study_id` group:
   - `No_q_s` is the total number of questions in that group
   - `No_B_s = ceil(No_q_s / b)`
   - `No_q_B_s = ceil(No_q_s / No_B_s)`
   - `No_q_B_s` is used as the dynamic top-K retrieval count for every sub-batch from that group
9. For each sub-batch, [get_top_k_related_rows_for_batch](scripts/step5_ask_user_question.py) ranks prior answered rows from the selected `*_before.csv` data against the full sub-batch and keeps the top `No_q_B_s` rows.
10. [build_profile_topk_prediction_prompt](scripts/step5_ask_user_question.py) creates the final batch prediction prompt using:
   - background context
   - structured profile
   - dynamically selected related prior answered rows for the batch
11. [run_prediction_for_batch](scripts/step5_ask_user_question.py) executes the batch prediction and validates the result.
12. [get_prediction_output_path](scripts/step5_ask_user_question.py#L523) writes an output filename in the form `{user_id}_{core_or_single}_cutoff_<year>_<input_option>_b<batch_target_size>_<agent_label>_profile-[<structured profile LLM>]_pred-[<prediction LLM>].csv`.

**Top-K selection workflow:**

1. For each target question, the script compares that question against every answered row from the selected `*_before.csv` files.
2. [tokenize_for_similarity](scripts/step5_ask_user_question.py#L674) tokenizes:
   - target question label
   - target categories
   - prior row label
   - prior row categories
3. It removes simple stopwords and keeps lowercase alphanumeric tokens.
4. [compute_relatedness_score](scripts/step5_ask_user_question.py#L680) scores each prior row using:
   - `label_overlap * 3`
   - `category_overlap`
   - `+2` if `Representation Type` matches
   - `+1` if `question type` matches
5. [get_top_k_related_rows_for_batch](scripts/step5_ask_user_question.py) sorts rows by descending batch-level score and keeps the first `No_q_B_s`.
6. [build_relevant_rows_context](scripts/step5_ask_user_question.py#L731) formats those selected rows into the prompt block used by [build_profile_topk_prediction_prompt](scripts/step5_ask_user_question.py#L843).

**Important note:**

- This is a lexical heuristic ranker, not embedding-based retrieval. It works best when related questions share similar wording, categories, or answer formats.

## `agent-profile-topK-rows-semantic`

Purpose:
Do everything in `agent-profile`, then add the top-`K` most semantically similar prior answered rows for each batch using precomputed embeddings.

Workflow:

1. Precompute embeddings with [scripts/step2_precompute_embeddings.py](scripts/step2_precompute_embeddings.py).
2. The script loads sibling `*_embeddings.json` files for the matching metadata `*_before.csv` files in `standalone/` or `standalone_categories_only/`, then joins those embeddings onto the selected user's answered rows by `variable_name`.
3. The script loads the sibling `*_embeddings.json` file for the selected target metadata `*_target.csv` file.
4. The overlapping target question set is first grouped by `project_number` for core targets or `study_id` for single targets, then split into smaller sub-batches using the user-specified batch parameter `b`.
5. For each original `project_number` or `study_id` group:
   - `No_q_s` is the total number of questions in that group
   - `No_B_s = ceil(No_q_s / b)`
   - `No_q_B_s = ceil(No_q_s / No_B_s)`
   - `No_q_B_s` is used as the dynamic top-K retrieval count for every sub-batch from that group
6. For each sub-batch, it compares each prior answered-row embedding against all target-question embeddings in that sub-batch and sums cosine similarity across the sub-batch.
7. It keeps the top `No_q_B_s` rows for the sub-batch, formats them into the prompt, and runs the same batch prediction flow as the lexical top-K agent.

## `agent-profile-topK-para-semantic`

Purpose:
Do everything in `agent-profile`, then retrieve the top-`K` most semantically similar structured-profile paragraphs for each batch using paragraph embeddings.

Note:
`agent-profile-topK-para-semantic` is not included in the current simulation run scope.

Workflow:

1. Precompute target-question embeddings with [scripts/step2_precompute_embeddings.py](scripts/step2_precompute_embeddings.py).
2. The script builds or loads the structured profile text exactly as in `agent-profile`.
3. It splits that structured profile into paragraph-sized retrieval chunks:
   - the summary paragraph
   - each bullet item under the profile sections
4. It embeds those profile paragraphs with OpenAI embeddings and caches them beside the saved profile text file.
5. The script loads the sibling `*_embeddings.json` file for the selected target metadata `*_target.csv` file.
6. For each batch, it compares every profile-paragraph embedding against all target-question embeddings in that batch and sums cosine similarity across the batch.
7. It keeps the top-`K` profile paragraphs for the batch, formats them into the prompt, and runs batch prediction using only those retrieved profile paragraphs plus background context.

## Prompt Routing

The central dispatcher is [prepare_batch_prediction_prompt](scripts/step5_ask_user_question.py).

It decides:

- baseline -> [build_batch_prompt](scripts/step5_ask_user_question.py)
- profile -> [build_profile_prediction_prompt](scripts/step5_ask_user_question.py)
- top-K lexical -> [build_profile_topk_prediction_prompt](scripts/step5_ask_user_question.py)
- top-K row semantic -> [build_profile_topk_prediction_prompt](scripts/step5_ask_user_question.py)
- top-K paragraph semantic -> [build_profile_topk_paragraph_prediction_prompt](scripts/step5_ask_user_question.py)

That means prompt construction differs by agent mode, but final answer generation still goes through [run_prediction_for_batch](scripts/step5_ask_user_question.py). For profile-based agents, the workflow now also separates the LLM used for structured profile generation from the LLM used for final answer prediction.

## Prediction Behavior

The script does not manually select one target question at a time. Instead:

1. It loads the selected target metadata file:
   - `standalone/core_all_waves_questions_metadata_standalone_v2_cutoff_<year>_target.csv`
   - or `standalone/filtered_table_single_wave_cutoff_<year>_target.csv`
2. It loads the matching user target file:
   - `{user_id}_core_cutoff_<year>_target.csv`
   - or `{user_id}_single_cutoff_<year>_target.csv`
3. It keeps only the overlapping `variable_name` values that exist in both files.
4. It groups that overlapping question set by:
   - core target file -> `project_number`
   - single target file -> `study_id`
5. It asks the user for batch parameter `b`, then splits each group into smaller sub-batches using:
   - `No_q_s` = number of questions in the group
   - `No_B_s = ceil(No_q_s / b)`
   - `No_q_B_s = ceil(No_q_s / No_B_s)`
6. It predicts answers sub-batch by sub-batch rather than one question at a time.
7. It writes the results by adding `Pred Answer` next to the original `answer` or `Answer` column in the output CSV.

Context behavior by agent mode:

- `agent-baseline`: uses the selected raw context files directly
- `agent-profile`: uses `bg_for_cutoff + structured profile`
- `agent-profile-topK-lexical`: uses `bg_for_cutoff + structured profile + dynamically selected related prior answered rows`, where the retrieval count per sub-batch is `No_q_B_s`
- `agent-profile-topK-rows-semantic`: uses `bg_for_cutoff + structured profile + dynamically selected semantically related prior answered rows`, where the retrieval count per sub-batch is `No_q_B_s`
- `agent-profile-topK-para-semantic`: uses `bg_for_cutoff + top-K semantically related structured profile paragraphs`
  Note: this mode is not included in the current simulation run scope.

Output behavior:

- Output folder: `pred_ouput/{user_id}/{source_dataset}_{year}_{input_scope}_{question_set}/`
- Output filename format:
  - `{user_id}_{core_or_single}_cutoff_<year>_<input_option>_b<batch_target_size>_<agent_label>_profile-[<LLM for the structured profile>]_pred-[<LLM version>].csv`
- For `agent-profile-topK-para-semantic`, the script also writes a companion JSON file containing the retrieved profile paragraphs for each batch.

## Output Validation

Answer validation happens in [run_prediction_for_batch](scripts/step5_ask_user_question.py) together with:

- [build_answer_spec](scripts/step5_ask_user_question.py#L293)
- [normalize_answer](scripts/step5_ask_user_question.py#L354)
- [normalize_numeric](scripts/step5_ask_user_question.py#L316)

Validation rules:

- Categories: the response must be a valid category code
- Numeric: the response must satisfy the numeric type and any range constraints
- Text: the response must be plain text

If the model returns an invalid batch format or an invalid answer for any item in the batch, the script retries with stricter feedback before failing that batch.
