from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import step5_ask_user_question as ask_user_question_module
from step5_ask_user_question import (
    DEFAULT_AGENT_MODE_PHASES,
    get_batch_group_column,
    get_dataset_source_config,
    get_agent_label,
    get_source_options,
    get_prediction_output_path,
    get_selected_file_groups,
    get_target_rows_by_variable,
    get_user_target_path_for_source,
    group_questions_for_batching,
    load_answered_embedding_rows,
    load_target_embeddings_by_variable,
    mode_name,
    mode_requires_semantic_embeddings,
    mode_requires_structured_profile,
    read_csv,
    resolve_api_key,
    run_agent_mode_execution,
    semantic_precompute_available,
    select_questions_for_user,
    split_question_groups_into_prediction_batches,
    UserContextCache,
    collect_answered_rows,
    get_metadata_path_for_source,
)


FIXED_DATASET_SOURCE = "datasets_categories_only"
FIXED_YEAR = "2023"
FIXED_AGENT_MODE_PHASES = [list(phase) for phase in DEFAULT_AGENT_MODE_PHASES]
FIXED_PROFILE_TEXT_OUTPUT_DIR = Path("profile_output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed datasets_categories_only / 2023 workflow with configurable "
            "question_set and input_scope. "
            "agent-baseline always uses background-only scope A, while the profile-based modes "
            "use the configured input scope. "
            "By default this executes two parallel phases: "
            "agent-baseline + agent-profile, then "
            "agent-profile-topK-lexical + agent-profile-topK-rows-semantic."
        )
    )
    parser.add_argument("user_id", help="User ID to run.")
    parser.add_argument(
        "question_set",
        choices=["core", "single"],
        help="Question set to run.",
    )
    parser.add_argument(
        "input_scope",
        choices=["a", "b", "c", "d", "e", "f", "A", "B", "C", "D", "E", "F"],
        help="Profile-based input scope to use. agent-baseline still uses scope A.",
    )
    parser.add_argument(
        "batch_target_size",
        type=int,
        help="Target questions per sub-batch within each project_number/study_id.",
    )
    parser.add_argument(
        "profile_provider",
        help="Structured profile generation provider: openai/gpt, anthropic/claude, or gemini.",
    )
    parser.add_argument("profile_model", help="Structured profile generation model id.")
    parser.add_argument(
        "prediction_provider",
        help="Answer prediction provider: openai/gpt, anthropic/claude, or gemini.",
    )
    parser.add_argument("prediction_model", help="Answer prediction model id.")
    parser.add_argument(
        "--prediction-reasoning",
        choices=["none", "medium"],
        default="none",
        help="Prediction reasoning mode. Use 'medium' to request medium reasoning, or 'none' to omit an explicit reasoning setting.",
    )
    parser.add_argument(
        "--output-root",
        help=(
            "Optional prediction output root directory override. "
            "Profile text caching still uses profile_output."
        ),
    )
    parser.add_argument(
        "--agent-mode",
        choices=["current", "profile", "profile_topk_lexical", "profile_topk_semantic"],
        help=(
            "Optional single agent mode to run. If omitted, the script runs the default grouped "
            "two-phase workflow."
        ),
    )
    return parser.parse_args()


def normalize_provider(value: str) -> str:
    normalized = value.strip().lower()
    mapping = {
        "gpt": "openai",
        "openai": "openai",
        "claude": "anthropic",
        "anthropic": "anthropic",
        "gemini": "gemini",
    }
    if normalized not in mapping:
        raise ValueError(
            "Unsupported provider. Use one of: openai, gpt, anthropic, claude, gemini."
        )
    return mapping[normalized]


def require_api_key(provider: str) -> str:
    api_key = resolve_api_key(provider)
    if api_key:
        return api_key
    env_name = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }.get(provider, "API_KEY")
    raise ValueError(f"Missing API key for {provider}. Set environment variable {env_name}.")


def get_mode_completion_marker(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.done.json")


def write_mode_completion_marker(
    marker_path: Path,
    result: Dict[str, object],
    batch_target_size: int,
    profile_provider: str,
    profile_model: str,
    prediction_provider: str,
    prediction_model: str,
    prediction_reasoning_mode: str,
) -> None:
    payload = {
        "agent_mode": result["agent_mode"],
        "output_path": str(result["output_path"]),
        "predictions_generated": result["predictions_generated"],
        "rows_with_pred_answer": result["rows_with_pred_answer"],
        "failure_count": len(result["failures"]),
        "batch_target_size": batch_target_size,
        "profile_provider": profile_provider,
        "profile_model": profile_model,
        "prediction_provider": prediction_provider,
        "prediction_model": prediction_model,
        "prediction_reasoning_mode": prediction_reasoning_mode,
    }
    marker_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def is_valid_success_marker(
    marker_path: Path,
    output_path: Path,
    agent_mode: str,
    batch_target_size: int,
    profile_provider: str,
    profile_model: str,
    prediction_provider: str,
    prediction_model: str,
    prediction_reasoning_mode: str,
) -> bool:
    if not marker_path.exists() or not output_path.exists():
        return False

    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    if not isinstance(payload, dict):
        return False

    expected_payload = {
        "agent_mode": agent_mode,
        "output_path": str(output_path),
        "failure_count": 0,
        "batch_target_size": batch_target_size,
        "profile_provider": profile_provider,
        "profile_model": profile_model,
        "prediction_provider": prediction_provider,
        "prediction_model": prediction_model,
        "prediction_reasoning_mode": prediction_reasoning_mode,
    }
    for key, expected_value in expected_payload.items():
        if payload.get(key) != expected_value:
            return False

    return True


def resolve_agent_mode_phases(selected_agent_mode: Optional[str]) -> List[List[str]]:
    if selected_agent_mode is not None:
        return [[selected_agent_mode]]
    return [list(phase) for phase in FIXED_AGENT_MODE_PHASES]


def main() -> None:
    args = parse_args()
    if args.batch_target_size <= 0:
        raise ValueError("batch_target_size must be a positive integer.")
    question_set = args.question_set.strip().lower()
    input_scope = args.input_scope.strip().lower()

    profile_provider = normalize_provider(args.profile_provider)
    prediction_provider = normalize_provider(args.prediction_provider)
    profile_api_key = require_api_key(profile_provider)
    prediction_api_key = require_api_key(prediction_provider)
    if args.output_root:
        ask_user_question_module.PRED_OUTPUT_ROOT = Path(args.output_root).expanduser().resolve()

    dataset_config = get_dataset_source_config(FIXED_DATASET_SOURCE)
    user_roots = dataset_config["user_roots"]
    if not isinstance(user_roots, dict):
        raise ValueError(f"Dataset source config is missing user_roots: {FIXED_DATASET_SOURCE}")
    users_root_dir = user_roots[FIXED_YEAR]
    user_dir = users_root_dir / args.user_id
    if not user_dir.exists():
        raise FileNotFoundError(f"User directory not found: {user_dir}")

    metadata_path = get_metadata_path_for_source(
        FIXED_DATASET_SOURCE,
        FIXED_YEAR,
        question_set,
    )
    user_target_path = get_user_target_path_for_source(
        user_dir,
        FIXED_DATASET_SOURCE,
        question_set,
        FIXED_YEAR,
    )
    agent_mode_phases = resolve_agent_mode_phases(args.agent_mode)
    selected_agent_modes = [mode for phase in agent_mode_phases for mode in phase]

    context_cache = UserContextCache(user_dir)
    selected_files, bg_files, before_files = get_selected_file_groups(
        user_dir,
        input_scope,
        FIXED_YEAR,
    )
    user_context, input_files, newly_loaded_sources, reused_sources = context_cache.build_context_from_files(
        selected_files
    )
    bg_context, bg_input_files, bg_newly_loaded_sources, bg_reused_sources = (
        context_cache.build_context_from_files(bg_files)
    )
    before_context = ""
    before_rows: List[Dict[str, str]] = []
    before_embedding_rows: List[Dict[str, str]] = []
    if before_files:
        before_context, _, _, _ = context_cache.build_context_from_files(before_files)
        before_rows = collect_answered_rows(before_files)
        if any(mode_requires_semantic_embeddings(mode) for mode in selected_agent_modes):
            if semantic_precompute_available(question_set, input_scope):
                before_embedding_rows = load_answered_embedding_rows(
                    before_files,
                    dataset_source=FIXED_DATASET_SOURCE,
                    year=FIXED_YEAR,
                    question_set=question_set,
                    scope=input_scope,
                )

    metadata_rows = read_csv(metadata_path)
    if not metadata_rows:
        raise ValueError(f"No question rows found in metadata file: {metadata_path}")

    target_rows = read_csv(user_target_path)
    if not target_rows:
        raise ValueError(f"No target rows found in user target file: {user_target_path}")

    target_embeddings_by_variable: Dict[str, List[float]] = {}
    if (
        any(mode_requires_semantic_embeddings(mode) for mode in selected_agent_modes)
        and semantic_precompute_available(question_set, input_scope)
    ):
        target_embeddings_by_variable = load_target_embeddings_by_variable(
            metadata_path,
            target_rows,
            dataset_source=FIXED_DATASET_SOURCE,
            question_set=question_set,
            year=FIXED_YEAR,
        )

    selected_question_rows = select_questions_for_user(metadata_rows, target_rows)
    if not selected_question_rows:
        raise ValueError(
            "No overlapping variable_name values were found between the selected metadata file "
            "and the selected user target file."
        )

    target_rows_by_variable = get_target_rows_by_variable(target_rows)
    grouped_question_rows = group_questions_for_batching(
        selected_question_rows,
        target_rows_by_variable,
        question_set,
    )
    prediction_batches = split_question_groups_into_prediction_batches(
        grouped_question_rows,
        batch_target_size=args.batch_target_size,
    )
    grouping_column = get_batch_group_column(question_set)
    profile_llm_name = f"{profile_provider}:{args.profile_model}"

    print("--- Fixed Configuration ---")
    print(f"user_id: {args.user_id}")
    print(f"dataset_source: {FIXED_DATASET_SOURCE}")
    print(f"cutoff_year: {FIXED_YEAR}")
    print(f"question_set: {question_set}")
    print(f"agent-baseline input_scope: A ({get_source_options(FIXED_YEAR)['a']})")
    print(
        f"profile-based input_scope: {input_scope.upper()} "
        f"({get_source_options(FIXED_YEAR)[input_scope]})"
    )
    print(f"batch_target_size_b: {args.batch_target_size}")
    print(f"profile_provider: {profile_provider}")
    print(f"profile_model: {args.profile_model}")
    print(f"profile_text_output_dir: {FIXED_PROFILE_TEXT_OUTPUT_DIR}")
    print(f"prediction_provider: {prediction_provider}")
    print(f"prediction_model: {args.prediction_model}")
    print(f"prediction_reasoning_mode: {args.prediction_reasoning}")
    print(f"prediction_output_root: {ask_user_question_module.PRED_OUTPUT_ROOT}")
    if args.agent_mode is None:
        for phase_index, phase_modes in enumerate(agent_mode_phases, start=1):
            print(f"phase_{phase_index}: {', '.join(mode_name(mode) for mode in phase_modes)}")
    else:
        print(f"selected_agent_mode: {mode_name(args.agent_mode)}")

    shared_profile_state: Dict[str, object] = {"lock": threading.Lock()}
    print_lock = threading.Lock()
    run_results: List[Dict[str, object]] = []

    for phase_index, phase_modes in enumerate(agent_mode_phases, start=1):
        pending_modes: List[str] = []
        print(f"\n=== Phase {phase_index}/{len(agent_mode_phases)} ===")
        for agent_mode in phase_modes:
            output_path = get_prediction_output_path(
                user_id=args.user_id,
                dataset_source=FIXED_DATASET_SOURCE,
                question_set=question_set,
                year=FIXED_YEAR,
                scope=input_scope,
                batch_target_size=args.batch_target_size,
                agent_label=get_agent_label(agent_mode),
                profile_llm=profile_llm_name if mode_requires_structured_profile(agent_mode) else "none",
                prediction_llm=f"{prediction_provider}:{args.prediction_model}",
                prediction_reasoning_mode=args.prediction_reasoning,
            )
            marker_path = get_mode_completion_marker(output_path)
            if is_valid_success_marker(
                marker_path=marker_path,
                output_path=output_path,
                agent_mode=agent_mode,
                batch_target_size=args.batch_target_size,
                profile_provider=profile_provider,
                profile_model=args.profile_model,
                prediction_provider=prediction_provider,
                prediction_model=args.prediction_model,
                prediction_reasoning_mode=args.prediction_reasoning,
            ):
                print(f"[{mode_name(agent_mode)}] skip_existing_success_marker: {marker_path}")
                run_results.append(
                    {
                        "agent_mode": agent_mode,
                        "output_path": output_path,
                        "predictions_generated": 0,
                        "rows_with_pred_answer": 0,
                        "failures": [],
                        "skipped": True,
                    }
                )
            else:
                if marker_path.exists() and not output_path.exists():
                    print(
                        f"[{mode_name(agent_mode)}] ignore_stale_marker_missing_output: {marker_path}"
                    )
                elif marker_path.exists():
                    print(f"[{mode_name(agent_mode)}] ignore_mismatched_or_failed_marker: {marker_path}")
                pending_modes.append(agent_mode)

        if not pending_modes:
            print("phase_status: all modes already completed")
            continue

        with ThreadPoolExecutor(max_workers=len(pending_modes)) as executor:
            future_to_mode = {
                executor.submit(
                    run_agent_mode_execution,
                    agent_mode=agent_mode,
                    user_id=args.user_id,
                    dataset_source=FIXED_DATASET_SOURCE,
                    question_set=question_set,
                    year=FIXED_YEAR,
                    scope=input_scope,
                    batch_target_size=args.batch_target_size,
                    metadata_path=metadata_path,
                    user_target_path=user_target_path,
                    input_files=input_files,
                    bg_input_files=bg_input_files,
                    newly_loaded_sources=newly_loaded_sources,
                    reused_sources=reused_sources,
                    bg_newly_loaded_sources=bg_newly_loaded_sources,
                    bg_reused_sources=bg_reused_sources,
                    user_context=user_context,
                    bg_context=bg_context,
                    before_context=before_context,
                    before_rows=before_rows,
                    before_embedding_rows=before_embedding_rows,
                    profile_paragraph_embedding_rows=[],
                    target_embeddings_by_variable=target_embeddings_by_variable,
                    metadata_rows=metadata_rows,
                    target_rows=target_rows,
                    selected_question_rows=selected_question_rows,
                    grouped_question_rows=grouped_question_rows,
                    prediction_batches=prediction_batches,
                    grouping_column=grouping_column,
                    prediction_provider=prediction_provider,
                    prediction_model=args.prediction_model,
                    prediction_reasoning_mode=args.prediction_reasoning,
                    prediction_api_key=prediction_api_key,
                    profile_provider=profile_provider,
                    profile_model=args.profile_model,
                    profile_api_key=profile_api_key,
                    profile_llm_name=profile_llm_name,
                    profile_text_output_dir=FIXED_PROFILE_TEXT_OUTPUT_DIR,
                    shared_profile_state=shared_profile_state,
                    print_lock=print_lock,
                    selection_header="✨ --- Selection --- ✨",
                    output_header="🚀 --- Output --- 🚀",
                ): agent_mode
                for agent_mode in pending_modes
            }

            for future in as_completed(future_to_mode):
                agent_mode = future_to_mode[future]
                try:
                    result = future.result()
                    marker_path = get_mode_completion_marker(result["output_path"])
                    if len(result["failures"]) == 0:
                        write_mode_completion_marker(
                            marker_path=marker_path,
                            result=result,
                            batch_target_size=args.batch_target_size,
                            profile_provider=profile_provider,
                            profile_model=args.profile_model,
                            prediction_provider=prediction_provider,
                            prediction_model=args.prediction_model,
                            prediction_reasoning_mode=args.prediction_reasoning,
                        )
                        print(f"[{mode_name(agent_mode)}] completion_marker_written: {marker_path}")
                    else:
                        print(
                            f"[{mode_name(agent_mode)}] completion_marker_skipped_due_to_failures: "
                            f"{len(result['failures'])}"
                        )
                    result["skipped"] = False
                    run_results.append(result)
                except Exception as exc:
                    print(f"[{mode_name(agent_mode)}] mode_failed: {exc}")
                    run_results.append(
                        {
                            "agent_mode": agent_mode,
                            "output_path": None,
                            "predictions_generated": 0,
                            "rows_with_pred_answer": 0,
                            "failures": [str(exc)],
                            "skipped": False,
                        }
                    )

    print("\n--- Run Summary ---")
    for result in run_results:
        output_path = result.get("output_path")
        output_text = str(output_path) if output_path is not None else "<not written>"
        skipped_text = "yes" if result.get("skipped") else "no"
        print(
            f"{mode_name(str(result['agent_mode']))}: "
            f"output={output_text}, "
            f"skipped={skipped_text}, "
            f"predictions_generated={result['predictions_generated']}, "
            f"rows_with_pred_answer={result['rows_with_pred_answer']}, "
            f"failures={len(result['failures'])}"
        )


if __name__ == "__main__":
    main()
