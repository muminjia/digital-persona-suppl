from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DEFAULT_RUNNER_PATH = SCRIPT_DIR / "step6_run_single.py"
DEFAULT_LOG_ROOT = BASE_DIR / "batch_run_logs"
FIXED_PROFILE_TEXT_OUTPUT_DIR = Path("pred_ouput") / "cssw_to_sw_profile"
VALID_AGENT_MODES = {
    "current",
    "profile",
    "profile_topk_lexical",
    "profile_topk_semantic",
}
DATE_STAMP_PATTERN = re.compile(r"_(\d{8})_\d{6}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run run_fixed_categories_only_2023_single.py for each user ID in a JSON file "
            "with adjustable parallelism. The delegated fixed runner always uses "
            "background-only scope A for agent-baseline and the configured fixed scope "
            "for the profile-based modes."
        )
    )
    parser.add_argument(
        "user_ids_json",
        help="Path to a JSON file containing a list of user IDs.",
    )
    parser.add_argument(
        "question_set",
        choices=["core", "single"],
        help="Question set to pass to each single-user run.",
    )
    parser.add_argument(
        "input_scope",
        choices=["a", "b", "c", "d", "e", "f", "A", "B", "C", "D", "E", "F"],
        help="Profile-based input scope to pass to each single-user run.",
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
        help="Prediction reasoning mode to pass to each single-user run.",
    )
    parser.add_argument(
        "--parallel-users",
        type=int,
        default=4,
        help="How many users to process simultaneously. Default: 4.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional number of user IDs to take from the start of the JSON list.",
    )
    parser.add_argument(
        "--agent-mode",
        choices=sorted(VALID_AGENT_MODES),
        help="Optional single agent mode to run. If omitted, the default two-phase workflow runs.",
    )
    parser.add_argument(
        "--runner-path",
        default=str(DEFAULT_RUNNER_PATH),
        help="Path to the single-user runner script.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable to use for launching the single-user runner.",
    )
    parser.add_argument(
        "--log-root",
        default=str(DEFAULT_LOG_ROOT),
        help="Directory where per-user logs and the batch summary will be written.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop launching new users after the first failure.",
    )
    return parser.parse_args()


def load_user_ids(path: Path) -> List[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in file: {path}") from exc

    candidates: object = payload
    if isinstance(payload, dict):
        list_values = [value for value in payload.values() if isinstance(value, list)]
        if len(list_values) == 1:
            candidates = list_values[0]
        else:
            raise ValueError(
                "JSON must be either a top-level list or a dict containing exactly one list value."
            )

    if not isinstance(candidates, list):
        raise ValueError("JSON payload must resolve to a list of user IDs.")

    normalized_ids: List[str] = []
    for index, value in enumerate(candidates):
        user_id = str(value).strip()
        if not user_id:
            raise ValueError(f"User ID at index {index} is empty.")
        normalized_ids.append(user_id)

    unique_ids = list(dict.fromkeys(normalized_ids))
    if not unique_ids:
        raise ValueError("No user IDs were found in the JSON file.")
    return unique_ids


def build_command(args: argparse.Namespace, runner_path: Path, user_id: str) -> List[str]:
    command = [
        args.python_executable,
        str(runner_path),
        user_id,
        args.question_set,
        args.input_scope,
        str(args.batch_target_size),
        args.profile_provider,
        args.profile_model,
        args.prediction_provider,
        args.prediction_model,
    ]
    if args.agent_mode:
        command.extend(["--agent-mode", args.agent_mode])
    if args.prediction_reasoning != "none":
        command.extend(["--prediction-reasoning", args.prediction_reasoning])
    if getattr(args, "prediction_output_root", None):
        command.extend(["--output-root", str(args.prediction_output_root)])
    return command


def sanitize_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def build_batch_label(user_ids_json_arg: str, user_ids_json: Path, timestamp: str) -> str:
    raw_label = user_ids_json_arg.strip()
    if not raw_label:
        try:
            raw_label = str(user_ids_json.relative_to(BASE_DIR))
        except ValueError:
            raw_label = user_ids_json.name
    else:
        candidate = Path(raw_label).expanduser()
        if candidate.is_absolute():
            try:
                raw_label = str(user_ids_json.relative_to(BASE_DIR))
            except ValueError:
                raw_label = user_ids_json.name

    return f"{sanitize_filename(raw_label)}_{timestamp}"


def build_log_dir(log_root: Path, batch_label: str, timestamp: str) -> Path:
    run_date = timestamp.split("_", 1)[0]
    return log_root / run_date / batch_label


def migrate_existing_log_dirs(log_root: Path) -> None:
    if not log_root.exists():
        return

    for child in sorted(log_root.iterdir()):
        if not child.is_dir():
            continue
        if re.fullmatch(r"\d{8}", child.name):
            continue

        match = DATE_STAMP_PATTERN.search(child.name)
        if match is None:
            continue

        dated_parent = log_root / match.group(1)
        dated_parent.mkdir(parents=True, exist_ok=True)
        target = dated_parent / child.name
        if target.exists():
            continue
        child.rename(target)


def run_single_user(
    args: argparse.Namespace,
    runner_path: Path,
    log_dir: Path,
    user_id: str,
) -> Tuple[str, int, float, Path]:
    command = build_command(args, runner_path, user_id)
    started_at = time.time()
    result = subprocess.run(
        command,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    duration_seconds = time.time() - started_at

    log_path = log_dir / f"{sanitize_filename(user_id)}.log"
    log_lines = [
        f"command: {' '.join(command)}",
        f"exit_code: {result.returncode}",
        f"duration_seconds: {duration_seconds:.2f}",
        "",
        "=== STDOUT ===",
        result.stdout,
        "",
        "=== STDERR ===",
        result.stderr,
    ]
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    return user_id, result.returncode, duration_seconds, log_path


def write_summary(
    summary_path: Path,
    args: argparse.Namespace,
    user_ids_json: Path,
    results: Sequence[Tuple[str, int, float, Path]],
    total_duration_seconds: float,
) -> None:
    payload = {
        "user_ids_json": str(user_ids_json),
        "total_users": len(results),
        "question_set": args.question_set,
        "input_scope": args.input_scope.upper(),
        "parallel_users": args.parallel_users,
        "batch_target_size": args.batch_target_size,
        "profile_provider": args.profile_provider,
        "profile_model": args.profile_model,
        "prediction_provider": args.prediction_provider,
        "prediction_model": args.prediction_model,
        "prediction_reasoning_mode": args.prediction_reasoning,
        "prediction_output_root": str(args.prediction_output_root) if args.prediction_output_root else None,
        "profile_text_output_dir": str(FIXED_PROFILE_TEXT_OUTPUT_DIR),
        "agent_mode": args.agent_mode,
        "runner_path": args.runner_path,
        "python_executable": args.python_executable,
        "fail_fast": args.fail_fast,
        "total_duration_seconds": round(total_duration_seconds, 2),
        "results": [
            {
                "user_id": user_id,
                "exit_code": exit_code,
                "duration_seconds": round(duration_seconds, 2),
                "log_path": str(log_path),
            }
            for user_id, exit_code, duration_seconds, log_path in results
        ],
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_target_size <= 0:
        raise ValueError("batch_target_size must be a positive integer.")
    if args.parallel_users <= 0:
        raise ValueError("--parallel-users must be a positive integer.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer.")

    user_ids_json = Path(args.user_ids_json).expanduser().resolve()
    runner_path = Path(args.runner_path).expanduser().resolve()
    log_root = Path(args.log_root).expanduser().resolve()
    if not user_ids_json.exists():
        raise FileNotFoundError(f"User ID JSON file not found: {user_ids_json}")
    if not runner_path.exists():
        raise FileNotFoundError(f"Runner script not found: {runner_path}")

    migrate_existing_log_dirs(log_root)

    user_ids = load_user_ids(user_ids_json)
    if args.limit is not None:
        user_ids = user_ids[: args.limit]
        if not user_ids:
            raise ValueError("--limit resulted in an empty user list.")
        args.prediction_output_root = (BASE_DIR / "pred_output_test").resolve()
    else:
        args.prediction_output_root = None
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    batch_label = build_batch_label(args.user_ids_json, user_ids_json, run_timestamp)
    log_dir = build_log_dir(log_root, batch_label, run_timestamp)
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "summary.json"

    print("--- Batch Run Configuration ---")
    print(f"user_ids_json: {user_ids_json}")
    print(f"user_count: {len(user_ids)}")
    print(f"limit: {args.limit if args.limit is not None else 'all'}")
    print(f"question_set: {args.question_set}")
    print(f"input_scope: {args.input_scope.upper()}")
    print(f"parallel_users: {args.parallel_users}")
    print(f"batch_target_size: {args.batch_target_size}")
    print(f"profile_provider: {args.profile_provider}")
    print(f"profile_model: {args.profile_model}")
    print(f"prediction_provider: {args.prediction_provider}")
    print(f"prediction_model: {args.prediction_model}")
    print(f"prediction_reasoning_mode: {args.prediction_reasoning}")
    print(f"prediction_output_root: {args.prediction_output_root or (BASE_DIR / 'pred_ouput')}")
    print(f"profile_text_output_dir: {FIXED_PROFILE_TEXT_OUTPUT_DIR}")
    print(f"agent_mode: {args.agent_mode or 'default-phases'}")
    print(f"runner_path: {runner_path}")
    print(f"python_executable: {args.python_executable}")
    print(f"log_dir: {log_dir}")

    print_lock = threading.Lock()
    results: List[Tuple[str, int, float, Path]] = []
    stop_event = threading.Event()
    started_at = time.time()

    def run_and_log(user_id: str) -> Tuple[str, int, float, Path]:
        if stop_event.is_set():
            raise RuntimeError("Batch halted before this user was launched.")

        with print_lock:
            print(f"[{user_id}] started")

        result = run_single_user(
            args=args,
            runner_path=runner_path,
            log_dir=log_dir,
            user_id=user_id,
        )
        _, exit_code, duration_seconds, log_path = result

        with print_lock:
            status = "success" if exit_code == 0 else "failed"
            print(
                f"[{user_id}] {status} exit_code={exit_code} "
                f"duration_seconds={duration_seconds:.2f} log={log_path}"
            )

        if exit_code != 0 and args.fail_fast:
            stop_event.set()
        return result

    with ThreadPoolExecutor(max_workers=args.parallel_users) as executor:
        user_iter = iter(user_ids)
        future_to_user = {}

        while len(future_to_user) < args.parallel_users:
            try:
                user_id = next(user_iter)
            except StopIteration:
                break
            future_to_user[executor.submit(run_and_log, user_id)] = user_id

        while future_to_user:
            for future in as_completed(list(future_to_user)):
                user_id = future_to_user.pop(future)
                try:
                    results.append(future.result())
                except Exception as exc:
                    with print_lock:
                        print(f"[{user_id}] wrapper_failed: {exc}")
                    results.append((user_id, 1, 0.0, log_dir / f"{sanitize_filename(user_id)}.log"))
                    if args.fail_fast:
                        stop_event.set()

                if stop_event.is_set():
                    continue

                try:
                    next_user_id = next(user_iter)
                except StopIteration:
                    pass
                else:
                    future_to_user[executor.submit(run_and_log, next_user_id)] = next_user_id
                break

    total_duration_seconds = time.time() - started_at
    results.sort(key=lambda item: item[0])
    write_summary(
        summary_path=summary_path,
        args=args,
        user_ids_json=user_ids_json,
        results=results,
        total_duration_seconds=total_duration_seconds,
    )

    success_count = sum(1 for _, exit_code, _, _ in results if exit_code == 0)
    failure_count = len(results) - success_count
    print("\n--- Batch Run Summary ---")
    print(f"completed_users: {len(results)}")
    print(f"successes: {success_count}")
    print(f"failures: {failure_count}")
    print(f"total_duration_seconds: {total_duration_seconds:.2f}")
    print(f"summary_path: {summary_path}")

    if failure_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
