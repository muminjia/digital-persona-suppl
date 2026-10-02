# When Can Digital Personas Reliably Approximate Human Survey Findings?

This repository contains the code used to construct and evaluate digital personas against held-out responses from the Longitudinal Internet Studies for the Social Sciences (LISS) panel.

Paper: [When Can Digital Personas Reliably Approximate Human Survey Findings?](https://arxiv.org/abs/2605.10659)

The experiment follows a temporal holdout design. Background variables and pre-2023 survey history are used to construct a persona. The persona then predicts the same respondent's post-cutoff answers. The repository supports two prediction tasks, four persona architectures, lexical and semantic retrieval, multiple LLM providers, stratified respondent sampling, and batch execution.

## Repository Structure

```text
scripts/
  step1_sample_users.py          Build stratified respondent samples
  step2_precompute_embeddings.py Cache semantic-retrieval embeddings
  step3_llm_api.py               Minimal LLM API example
  step4_ai_agent.py              Provider-independent single-node agent
  step4b_run_ai_agent.py         CLI wrapper for the single-node agent
  step5_ask_user_question.py     Interactive prediction runner
  step6_run_single.py            Fixed single-respondent runner
  step7_run_batch.py             Parallel batch runner

docs/
  AGENT_WORKFLOW.md              Detailed agent and prompt workflow

requirements.txt                 Python dependencies
```

## Experimental Design

The paper uses the LISS panel with a 2023 temporal cutoff and samples 500 eligible respondents for each prediction task. Eligible respondents have background data, the required pre-cutoff history, and at least one evaluable target answer.

The two prediction tasks are:

- `single`: pre-2023 core-study history -> 2023-2024 single-wave targets
- `core`: pre-2023 single-wave history -> 2023 core-study targets

The implemented persona architectures are:

1. `agent-baseline`: background variables only
2. `agent-profile`: background variables plus a structured profile of prior answers
3. `agent-profile-topK-lexical`: profile plus lexically retrieved prior answers
4. `agent-profile-topK-rows-semantic`: profile plus semantically retrieved prior answers

The profile is generated once per respondent and cached. Prediction questions are grouped by study and split into batches. Retrieval uses a dynamic `K` equal to the realized batch size, as described in Appendix B of the paper.

The evaluation framework in the paper reports question-level weighted F1, respondent-level exact-match rate, question-distribution Jensen-Shannon divergence, respondent-distribution MMD, demographic equity, and respondent clustering agreement.

## Data Access

The LISS data are not redistributed in this repository. Researchers must request access from [Centerdata](https://www.lissdata.nl/) and comply with the LISS data user statement.

Each respondent directory should contain background, prior-answer, and target CSV files. The categories-only source is recommended for reproducing the paper because the paper restricts evaluation to closed-ended questions with finite answer spaces.

## Getting Started

Install the dependencies in a virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Set only the API keys required by the selected providers:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GEMINI_API_KEY="..."
```

Do not commit keys or `.env` files.

## Reproduce the Sampling

Configure `samples_to_run` near the top of [scripts/step1_sample_users.py](scripts/step1_sample_users.py), then run:

```bash
python scripts/step1_sample_users.py
```

The supported sample definitions are:

- `sw_to_cs`: single-wave history -> core target
- `sw_to_sw`: single-wave history -> single-wave target
- `cssw_to_sw`: combined core and single-wave history -> single-wave target
- `cs_to_sw`: latest core history -> 2023-2024 single-wave target
- `all`: build every configured sample

The script writes allocation tables, respondent metadata, and sampled user lists under `sample_500_2023_2024_categories_only/`.

## Reproduce Semantic Retrieval

Semantic retrieval requires offline metadata embeddings. Generate them before running `agent-profile-topK-rows-semantic`:

```bash
python scripts/step2_precompute_embeddings.py
```

Useful narrower runs are:

```bash
python scripts/step2_precompute_embeddings.py --file-kind before
python scripts/step2_precompute_embeddings.py --directory-scope standalone_categories_only --file-kind target
```

Embeddings are written as sibling `*_embeddings.json` files and to `embedding_cache/` when required by the runtime lookup.

## Run Predictions

### Interactive run

```bash
python scripts/step5_ask_user_question.py
```

The interactive runner asks for the dataset source, cutoff year, question set, input scope, persona architecture, batch size, profile model, and prediction model.

### Single respondent

```bash
python scripts/step6_run_single.py \
  <user_id> <question_set> <input_scope> <batch_target_size> \
  <profile_provider> <profile_model> \
  <prediction_provider> <prediction_model> \
  --prediction-reasoning none
```

Example:

```bash
python scripts/step6_run_single.py \
  899923 single e 20 \
  openai gpt-5.4 \
  openai gpt-5.4 \
  --prediction-reasoning none
```

Run one architecture with `--agent-mode`:

```bash
python scripts/step6_run_single.py \
  899923 core c 20 \
  openai gpt-5.4 \
  openai gpt-5.4 \
  --agent-mode profile_topk_semantic
```

Supported input scopes are:

- `A`: background only
- `B`: background + core history
- `C`: background + single-wave history
- `D`: background + core + single-wave history
- `E`: background + latest core history
- `F`: background + latest core + single-wave history

The background-only `agent-baseline` always uses scope `A`; profile-based agents use the selected scope.

### Batch run

```bash
python scripts/step7_run_batch.py \
  <user_ids.json> <question_set> <input_scope> <batch_target_size> \
  <profile_provider> <profile_model> \
  <prediction_provider> <prediction_model> \
  --parallel-users 4
```

Example:

```bash
python scripts/step7_run_batch.py \
  sample_500_2023_2024_categories_only/cs_to_sw_500users.json \
  single e 20 \
  openai gpt-5.4 \
  openai gpt-5.4 \
  --parallel-users 4
```

For a cheap smoke test, add `--limit 2`.


## Reproducibility Notes

- The paper reports model calls without explicit chain-of-thought reasoning; use `--prediction-reasoning none`.
- Structured profiles use GPT-5.4 and are cached per respondent, task, cutoff, scope, and profile model.
- Semantic retrieval uses `text-embedding-3-small` embeddings computed offline.
- API responses and generated profiles depend on provider availability and model-version changes.
- The full run is expensive. The paper reports approximately 80 hours and about $3,000 in API usage.
- The current repository contains prediction and sampling runners. Metric aggregation should be performed only after checking output completeness and matching each prediction file to its corresponding target rows.

## Citation

If you use this code or the experimental design, cite:

```bibtex
@article{jia2026digitalpersonas,
  title   = {When Can Digital Personas Reliably Approximate Human Survey Findings?},
  author  = {Jia, Mumin and Chen, Yilin and Sharma, Divya and Diaz-Rodriguez, Jairo},
  journal = {arXiv preprint arXiv:2605.10659},
  year    = {2026},
  doi     = {10.48550/arXiv.2605.10659}
}
```

For implementation details, see [docs/AGENT_WORKFLOW.md](docs/AGENT_WORKFLOW.md).