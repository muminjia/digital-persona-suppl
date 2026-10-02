from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from step4_ai_agent import AgentConfig, SingleNodeAgent
from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
PRED_OUTPUT_ROOT = BASE_DIR / "pred_ouput"
DATASET_SOURCE_CONFIG = {
    "datasets": {
        "label": "datasets",
        "description": "full dataset source",
        "user_roots": {
            "2019": BASE_DIR / "datasets" / "nomem_encr_cutoff_2019",
            "2023": BASE_DIR / "datasets" / "nomem_encr_cutoff_2023",
        },
        "shared_user_list_path": BASE_DIR / "datasets" / "shared_both_overlap_2019_2023.json",
        "metadata": {
            "core": {
                "2019": BASE_DIR / "standalone" / "core_all_waves_questions_metadata_standalone_v2_cutoff_2019_target.csv",
                "2023": BASE_DIR / "standalone" / "core_all_waves_questions_metadata_standalone_v2_cutoff_2023_target.csv",
            },
            "single": {
                "2019": BASE_DIR / "standalone" / "filtered_table_single_wave_cutoff_2019_target.csv",
                "2023": BASE_DIR / "standalone" / "filtered_table_single_wave_cutoff_2023_target.csv",
            },
        },
    },
    "datasets_categories_only": {
        "label": "datasets_categories_only",
        "description": "closed-ended questions only",
        "user_roots": {
            "2019": BASE_DIR / "datasets_categories_only" / "nomem_encr_categories_only_cutoff_2019",
            "2023": BASE_DIR / "datasets_categories_only" / "nomem_encr_categories_only_cutoff_2023",
        },
        "shared_user_list_path": None,
        "metadata": {
            "core": {
                "2019": BASE_DIR / "standalone_categories_only" / "core_all_waves_questions_metadata_standalone_v2_categories_only_cutoff_2019_target.csv",
                "2023": BASE_DIR / "standalone_categories_only" / "core_all_waves_questions_metadata_standalone_v2_categories_only_cutoff_2023_target.csv",
            },
            "single": {
                "2019": BASE_DIR / "standalone_categories_only" / "filtered_table_single_wave_categories_only_cutoff_2019_target.csv",
                "2023": BASE_DIR / "standalone_categories_only" / "filtered_table_single_wave_categories_only_cutoff_2023_2024_target.csv",
            },
        },
    },
}
SHARED_USER_LIST_KEY = "both_overlap"
METADATA_FOLDER_BY_DATASET_SOURCE = {
    "datasets": BASE_DIR / "standalone",
    "datasets_categories_only": BASE_DIR / "standalone_categories_only",
}
QUESTION_SET_CONFIG = {
    "core": {"user_target_suffix": "core"},
    "single": {"user_target_suffix": "single"},
}
API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
BEFORE_SOURCE_NAMES = {"core_before", "core_latest_before", "single_before"}
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 100
MISSING_PRECOMPUTE_EMBEDDINGS_MESSAGE = "missing precompute embeddings....."
EMBEDDING_CACHE_DIR = Path(os.environ.get("EMBEDDING_CACHE_DIR", "embedding_cache"))
SPECIAL_CORE_LATEST_BEFORE_EMBEDDING_PATH = (
    EMBEDDING_CACHE_DIR
    / "core_latest_waves_questions_metadata_standalone_v2_categories_only_cutoff_2023_before_embeddings.json"
)
SPECIAL_SINGLE_2023_BEFORE_EMBEDDING_PATH = (
    EMBEDDING_CACHE_DIR
    / "filtered_table_single_wave_categories_only_cutoff_2023_before_embeddings.json"
)
SPECIAL_CORE_TARGET_EMBEDDING_PATH = (
    EMBEDDING_CACHE_DIR
    / "core_all_waves_questions_metadata_standalone_v2_categories_only_cutoff_2023_target_embeddings.json"
)
SPECIAL_SINGLE_2023_TARGET_EMBEDDING_PATH = (
    EMBEDDING_CACHE_DIR
    / "filtered_table_single_wave_categories_only_cutoff_2023_2024_target_embeddings.json"
)
DEFAULT_AGENT_MODE_PHASES = [
    ["current", "profile"],
    ["profile_topk_lexical", "profile_topk_semantic"],
]
MODE_DISPLAY_NAME = {
    "current": "agent-baseline",
    "profile": "agent-profile",
    "profile_topk_lexical": "agent-profile-topK-lexical",
    "profile_topk_semantic": "agent-profile-topK-rows-semantic",
    "profile_topk_para_semantic": "agent-profile-topK-para-semantic",
}
PREDICTION_REASONING_MODES = {
    "none": "without explicit reasoning",
    "medium": "with medium reasoning",
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "did",
    "for",
    "from",
    "have",
    "how",
    "if",
    "in",
    "is",
    "it",
    "last",
    "many",
    "much",
    "of",
    "on",
    "or",
    "per",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "which",
    "who",
    "with",
    "would",
    "you",
    "your",
}

PROFILE_SECTION_HEADERS = [
    "1. Personality traits",
    "2. Reasoning style",
    "3. Knowledge profile",
    "4. Values and motivations",
    "5. Biases and heuristics",
    "6. Decision patterns",
    "7. Confidence patterns",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_row_value(row: Dict[str, str], candidates: List[str]) -> str:
    for key in candidates:
        if key in row:
            value = row.get(key, "")
            if value is not None:
                return str(value).strip()

    lowered = {k.lower(): k for k in row.keys()}
    for key in candidates:
        found = lowered.get(key.lower())
        if found:
            value = row.get(found, "")
            if value is not None:
                return str(value).strip()
    return ""


def choose_cutoff_year() -> str:
    print("Choose cutoff year:")
    print("1. 2019")
    print("2. 2023")

    while True:
        choice = input("Enter 1/2 (or 2019/2023): ").strip().lower()
        if choice in {"1", "2019"}:
            return "2019"
        if choice in {"2", "2023"}:
            return "2023"
        print("Invalid choice. Please enter 1, 2, 2019, or 2023.")


def choose_dataset_source() -> str:
    print("Choose source dataset:")
    print("1. datasets")
    print("2. datasets_categories_only")

    while True:
        choice = input(
            "Enter 1/2 (or datasets/datasets_categories_only): "
        ).strip().lower()
        if choice in {"1", "datasets"}:
            return "datasets"
        if choice in {"2", "datasets_categories_only", "categories_only"}:
            return "datasets_categories_only"
        print("Invalid choice. Please enter 1, 2, datasets, or datasets_categories_only.")


def get_dataset_source_config(dataset_source: str) -> Dict[str, object]:
    config = DATASET_SOURCE_CONFIG.get(dataset_source)
    if config is None:
        raise ValueError(f"Unsupported dataset source: {dataset_source}")
    return config


def list_shared_users_from_roots(user_roots: Dict[str, Path]) -> List[str]:
    valid_user_sets: List[set[str]] = []
    for path in user_roots.values():
        if not path.exists():
            raise FileNotFoundError(f"Users directory not found: {path}")
        valid_user_sets.append({child.name for child in path.iterdir() if child.is_dir()})
    if not valid_user_sets:
        return []
    shared_ids = set.intersection(*valid_user_sets)
    return sorted(shared_ids)


def list_available_users(users_root_dir: Path, year: str, dataset_source: str) -> List[str]:
    if not users_root_dir.exists():
        return []

    dataset_config = get_dataset_source_config(dataset_source)
    shared_ids_path = dataset_config.get("shared_user_list_path")
    if isinstance(shared_ids_path, Path):
        if not shared_ids_path.exists():
            raise FileNotFoundError(f"Shared user list not found: {shared_ids_path}")

        data = read_json(shared_ids_path)
        if not isinstance(data, dict):
            raise ValueError(f"Shared user list must be a JSON object: {shared_ids_path}")

        both_ids = data.get(SHARED_USER_LIST_KEY, [])
        if not isinstance(both_ids, list):
            raise ValueError(
                f"Shared user list '{SHARED_USER_LIST_KEY}' must be a JSON array: {shared_ids_path}"
            )

        valid_dirs = {path.name for path in users_root_dir.iterdir() if path.is_dir()}
        return sorted(str(user_id) for user_id in both_ids if str(user_id) in valid_dirs)

    user_roots = dataset_config["user_roots"]
    if not isinstance(user_roots, dict):
        raise ValueError(f"Dataset source config is missing user_roots: {dataset_source}")
    return list_shared_users_from_roots(user_roots)


def choose_user_dir(users_root_dir: Path, year: str, dataset_source: str) -> Tuple[str, Path]:
    if not users_root_dir.exists():
        raise FileNotFoundError(f"Users directory not found: {users_root_dir}")

    user_ids = list_available_users(users_root_dir, year, dataset_source)
    if not user_ids:
        raise ValueError(f"No user folders found in {users_root_dir}")

    print("Available users:")
    print(", ".join(user_ids))

    while True:
        selected = input("Enter user id: ").strip()
        path = users_root_dir / selected
        if path.exists() and path.is_dir():
            return selected, path
        print(f"User '{selected}' not found. Please choose from the listed ids.")


def choose_question_set() -> str:
    print("Choose question set:")
    print("1. Core questions")
    print("2. Single wave questions")

    while True:
        choice = input("Enter 1/2 (or core/single): ").strip().lower()
        if choice in {"1", "core"}:
            return "core"
        if choice in {"2", "single", "single wave", "single_wave"}:
            return "single"
        print("Invalid choice. Please enter 1, 2, core, or single.")


def get_source_options(year: str) -> Dict[str, str]:
    return {
        "a": f"A) {year} background only",
        "b": f"B) {year} background + core before",
        "c": f"C) {year} background + single before",
        "d": f"D) {year} background + core before + single before",
        "e": f"E) {year} background + core latest before",
        "f": f"F) {year} background + core latest before + single before",
    }


def choose_input_scope(year: str, allow_bg_only: bool = True) -> str:
    print("Choose input information scope:")
    if allow_bg_only:
        print(f"A. {year} background only")
    print(f"B. {year} background + core before")
    print(f"C. {year} background + single before")
    print(f"D. {year} background + core before + single before")
    print(f"E. {year} background + core latest before")
    print(f"F. {year} background + core latest before + single before")

    allowed_choices = {"b", "c", "d", "e", "f"}
    prompt = "Enter B/C/D/E/F: "
    if allow_bg_only:
        allowed_choices = {"a", "b", "c", "d", "e", "f"}
        prompt = "Enter A/B/C/D/E/F: "

    while True:
        choice = input(prompt).strip().lower()
        if choice in allowed_choices:
            return choice
        if allow_bg_only:
            print("Invalid choice. Please enter A, B, C, D, E, or F.")
        else:
            print("Invalid choice. Please enter B, C, D, E, or F.")


def choose_agent_mode() -> Optional[str]:
    print("Choose agent mode:")
    print("1. agent-baseline")
    print("2. agent-profile")
    print("3. agent-profile-topK-lexical")
    print("4. agent-profile-topK-rows-semantic")
    print("Press Enter to run the default grouped sequence:")
    print("  phase 1 -> agent-baseline + agent-profile")
    print("  phase 2 -> agent-profile-topK-lexical + agent-profile-topK-rows-semantic")

    while True:
        choice = input("Enter 1/2/3/4, or press Enter for the default grouped run: ").strip().lower()
        if not choice or choice in {"0", "all", "default"}:
            return None
        if choice in {"1", "current", "baseline", "agent", "agent-baseline"}:
            return "current"
        if choice in {"2", "new1", "profile", "agent with user profile", "agent-profile"}:
            return "profile"
        if choice in {
            "3",
            "new2",
            "profile_topk",
            "profile_topk_lexical",
            "topk",
            "lexical",
            "agent-profile-topk-lexical",
            "agent with user profile and top-k",
            "anget-profile-topk",
        }:
            return "profile_topk_lexical"
        if choice in {
            "4",
            "profile_topk_semantic",
            "semantic",
            "agent-profile-topk-rows-semantic",
            "agent-profile-topk-semantic",
            "agent with user profile and semantic top-k",
        }:
            return "profile_topk_semantic"
        print("Invalid choice. Please enter 1, 2, 3, 4, or press Enter.")


def choose_top_k() -> int:
    while True:
        selected = input("Enter top-k context items to retrieve per batch (e.g. 5): ").strip()
        if selected.isdigit() and int(selected) > 0:
            return int(selected)
        print("Invalid value. Please enter a positive integer.")


def choose_batch_target_size() -> int:
    while True:
        selected = input(
            "Enter target questions per sub-batch within each project_number/study_id (b, e.g. 5): "
        ).strip()
        if selected.isdigit() and int(selected) > 0:
            return int(selected)
        print("Invalid value. Please enter a positive integer.")


def choose_llm_provider() -> str:
    print("Choose LLM:")
    print("1. GPT")
    print("2. Claude")
    print("3. Gemini")

    while True:
        choice = input("Enter 1/2/3 (or GPT/Claude/Gemini): ").strip().lower()
        if choice in {"1", "gpt"}:
            return "openai"
        if choice in {"2", "claude"}:
            return "anthropic"
        if choice in {"3", "gemini"}:
            return "gemini"
        print("Invalid choice. Please enter 1, 2, 3, GPT, Claude, or Gemini.")


def choose_model(provider: str) -> str:
    suggestions = {
        "openai": ["gpt-5.4", "gpt-5.2", "gpt-5-mini", "gpt-5-nano"],
        "anthropic": ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"],
        "gemini": [
            "gemini-3.1-pro-preview",
            "gemini-3-flash-preview",
            "gemini-3.1-flash-lite-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        ],
    }
    options = suggestions.get(provider, [])

    if options:
        print("Suggested model versions:")
        for idx, model in enumerate(options, start=1):
            print(f"{idx}. {model}")

    while True:
        selected = input("Enter exact model id (or option number): ").strip()
        if not selected:
            print("Model id cannot be empty.")
            continue

        if options:
            match = re.match(r"^\s*(\d+)\s*\.?\s*(.*)$", selected)
            if match:
                option_idx = int(match.group(1))
                trailing_text = match.group(2).strip()
                if 1 <= option_idx <= len(options):
                    return options[option_idx - 1]
                if trailing_text:
                    return trailing_text

        return selected


def choose_prediction_reasoning_mode() -> str:
    print("Choose prediction reasoning mode:")
    print("1. none")
    print("2. medium")

    while True:
        choice = input("Enter 1/2 (or none/medium): ").strip().lower()
        if choice in {"1", "none", "no", "without reasoning"}:
            return "none"
        if choice in {"2", "medium", "med"}:
            return "medium"
        print("Invalid choice. Please enter 1, 2, none, or medium.")


def resolve_api_key(provider: str) -> Optional[str]:
    return os.getenv(API_KEY_ENV.get(provider, ""))


def parse_categories(categories: str) -> Dict[str, str]:
    options: Dict[str, str] = {}
    for part in categories.split(";"):
        item = part.strip()
        if not item or ":" not in item:
            continue
        key, value = item.split(":", 1)
        code = key.strip()
        label = value.strip()
        if code:
            options[code] = label
    return options


def parse_numeric_representation(rep: str) -> Optional[Dict[str, Optional[str]]]:
    text = rep.strip()
    if not text.startswith("{") or "Numeric Type" not in text:
        return None
    try:
        data = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(data, dict):
        return None
    return {
        "numeric_type": str(data.get("Numeric Type", "")).strip(),
        "decimals": str(data.get("Decimals", "")).strip(),
        "range_low": str(data.get("Range Low", "")).strip(),
        "range_high": str(data.get("Range High", "")).strip(),
    }


def build_answer_spec(question_meta: Dict[str, str]) -> Dict[str, object]:
    question_type = get_row_value(question_meta, ["question type", "Question Type"]).lower()
    if question_type == "others":
        return {"type": "text"}

    categories = get_row_value(question_meta, ["Categories", "categories"])
    if categories:
        options = parse_categories(categories)
        if options:
            return {"type": "categories", "options": options}

    rep_type = get_row_value(question_meta, ["Representation Type", "representation_type"])
    if rep_type.lower() == "text":
        return {"type": "text"}

    numeric = parse_numeric_representation(rep_type)
    if numeric:
        return {"type": "numeric", **numeric}

    return {"type": "text"}


def normalize_numeric(value: str, spec: Dict[str, object]) -> Tuple[Optional[str], Optional[str]]:
    raw = value.strip()
    number_text = raw
    if raw.lower().startswith("answer:"):
        number_text = raw.split(":", 1)[1].strip()

    try:
        number = float(number_text)
    except ValueError:
        return None, "Response is not numeric."

    numeric_type = str(spec.get("numeric_type", "")).lower()
    if numeric_type == "integer" and not number.is_integer():
        return None, "Response must be an integer."

    low = str(spec.get("range_low", "")).strip()
    high = str(spec.get("range_high", "")).strip()

    if low and low != "-":
        try:
            low_val = float(low)
            if number < low_val:
                return None, f"Response is below minimum {low}."
        except ValueError:
            pass
    if high and high != "-":
        try:
            high_val = float(high)
            if number > high_val:
                return None, f"Response is above maximum {high}."
        except ValueError:
            pass

    if numeric_type == "integer":
        return str(int(number)), None

    decimals = str(spec.get("decimals", "")).strip()
    if decimals.isdigit():
        return f"{number:.{int(decimals)}f}", None
    return str(number), None


def normalize_answer(raw_answer: str, spec: Dict[str, object]) -> Tuple[Optional[str], Optional[str]]:
    answer = raw_answer.strip()
    if not answer:
        return None, "Response is empty."

    if answer.lower().startswith("answer:"):
        answer = answer.split(":", 1)[1].strip()

    spec_type = spec.get("type")
    if spec_type == "categories":
        options = spec.get("options", {})
        if not isinstance(options, dict):
            return None, "Internal options format error."

        if answer in options:
            return answer, None

        for code, label in options.items():
            if answer.lower() == str(label).lower():
                return str(code), None

        for code in options:
            if re.match(rf"^{re.escape(str(code))}\b", answer):
                return str(code), None

        return None, f"Response must be one of these codes: {', '.join(options.keys())}."

    if spec_type == "numeric":
        return normalize_numeric(answer, spec)

    if spec_type == "text":
        return answer, None

    return None, "Unknown answer type."


def build_context_lines(rows: List[Dict[str, str]], source_name: str) -> List[str]:
    lines: List[str] = [f"[{source_name}]"]
    has_content = False

    for row in rows:
        qid = get_row_value(row, ["question_id", "variable_name", "Variable Name"])
        answer = get_row_value(row, ["answer", "Answer"])
        if not answer:
            continue

        label = get_row_value(row, ["variable_label", "Variable Label"])
        categories = get_row_value(row, ["categories", "Categories"])
        year = get_row_value(row, ["year", "Year"])

        lines.append(f"question_id: {qid}")
        if label:
            lines.append(f"variable_label: {label}")
        lines.append(f"answer: {answer}")
        if categories:
            lines.append(f"categories: {categories}")
        if year:
            lines.append(f"year: {year}")
        lines.append("")
        has_content = True

    if not has_content:
        lines.append("No non-empty answers available.")
    return lines


def get_selected_input_files(user_dir: Path, scope: str, year: str) -> List[Tuple[str, Path]]:
    user_id = user_dir.name
    bg_path = user_dir / f"{user_id}_bg_for_cutoff_{year}.csv"
    core_before_path = user_dir / f"{user_id}_core_cutoff_{year}_before.csv"
    core_latest_before_path = user_dir / f"{user_id}_core_cutoff_{year}_latest_before.csv"
    single_before_path = user_dir / f"{user_id}_single_cutoff_{year}_before.csv"

    selected: List[Tuple[str, Path]] = [("bg_for_cutoff", bg_path)]
    if scope in {"b", "d"}:
        selected.append(("core_before", core_before_path))
    if scope in {"e", "f"}:
        selected.append(("core_latest_before", core_latest_before_path))
    if scope in {"c", "d", "f"}:
        selected.append(("single_before", single_before_path))

    for name, path in selected:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing expected file for scope {scope.upper()}: {name} -> {path}"
            )
    return selected


class UserContextCache:
    def __init__(self, user_dir: Path) -> None:
        self.user_dir = user_dir
        self.source_context_by_key: Dict[str, str] = {}

    def build_context_from_files(
        self, selected_files: List[Tuple[str, Path]]
    ) -> Tuple[str, List[Path], List[str], List[str]]:
        context_blocks: List[str] = []
        file_paths: List[Path] = []
        newly_loaded: List[str] = []
        reused: List[str] = []

        for source_name, path in selected_files:
            cache_key = f"{source_name}:{path.name}"
            if cache_key in self.source_context_by_key:
                source_block = self.source_context_by_key[cache_key]
                reused.append(source_name)
            else:
                rows = read_csv(path)
                source_block = "\n".join(build_context_lines(rows, source_name)).strip()
                self.source_context_by_key[cache_key] = source_block
                newly_loaded.append(source_name)

            context_blocks.append(source_block)
            file_paths.append(path)

        return "\n\n".join(context_blocks).strip(), file_paths, newly_loaded, reused

    def build_user_context(
        self, scope: str, year: str
    ) -> Tuple[str, List[Path], List[str], List[str]]:
        selected_files = get_selected_input_files(self.user_dir, scope, year)
        return self.build_context_from_files(selected_files)


def get_metadata_path(year: str, question_set: str) -> Path:
    return get_metadata_path_for_source("datasets", year, question_set)


def get_metadata_path_for_source(dataset_source: str, year: str, question_set: str) -> Path:
    dataset_config = get_dataset_source_config(dataset_source)
    metadata_config = dataset_config.get("metadata", {})
    question_config = metadata_config.get(question_set, {})
    path = question_config.get(year)
    if not isinstance(path, Path):
        raise FileNotFoundError(
            f"Metadata path is not configured for source={dataset_source}, "
            f"question_set={question_set}, year={year}"
        )
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    return path


def get_user_target_path(user_dir: Path, question_set: str, year: str) -> Path:
    user_id = user_dir.name
    suffix = QUESTION_SET_CONFIG[question_set]["user_target_suffix"]
    path = user_dir / f"{user_id}_{suffix}_cutoff_{year}_target.csv"
    if not path.exists():
        raise FileNotFoundError(f"User target file not found: {path}")
    return path


def get_user_target_path_for_source(
    user_dir: Path,
    dataset_source: str,
    question_set: str,
    year: str,
) -> Path:
    user_id = user_dir.name
    suffix = QUESTION_SET_CONFIG[question_set]["user_target_suffix"]
    if dataset_source == "datasets_categories_only" and year == "2023" and question_set == "single":
        path = user_dir / f"{user_id}_{suffix}_cutoff_{year}_2024_target.csv"
    else:
        path = user_dir / f"{user_id}_{suffix}_cutoff_{year}_target.csv"
    if not path.exists():
        raise FileNotFoundError(f"User target file not found: {path}")
    return path


def sanitize_filename_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-") or "model"


def get_user_prediction_output_dir(
    user_id: str,
    dataset_source: str,
    year: str,
    scope: str,
    question_set: str,
) -> Path:
    user_output_dir = PRED_OUTPUT_ROOT / user_id / (
        f"{sanitize_filename_component(dataset_source)}_{year}_{scope.upper()}_{question_set}"
    )
    user_output_dir.mkdir(parents=True, exist_ok=True)
    return user_output_dir


def get_prediction_output_path(
    user_id: str,
    dataset_source: str,
    question_set: str,
    year: str,
    scope: str,
    batch_target_size: int,
    agent_label: str,
    profile_llm: str,
    prediction_llm: str,
    prediction_reasoning_mode: str,
) -> Path:
    user_output_dir = get_user_prediction_output_dir(
        user_id, dataset_source, year, scope, question_set
    )
    return user_output_dir / (
        f"{user_id}_{sanitize_filename_component(dataset_source)}_{question_set}_cutoff_{year}_"
        f"{scope.upper()}_"
        f"b{batch_target_size}_"
        f"{agent_label}_"
        f"profile-[{sanitize_filename_component(profile_llm)}]_"
        f"pred-[{sanitize_filename_component(prediction_llm)}]_"
        f"reasoning-[{sanitize_filename_component(prediction_reasoning_mode)}].csv"
    )


def get_retrieved_profile_paragraphs_output_path(prediction_output_path: Path) -> Path:
    return prediction_output_path.with_name(
        f"{prediction_output_path.stem}_retrieved-profile-paragraphs.json"
    )


def get_profile_text_output_path(
    user_id: str,
    dataset_source: str,
    question_set: str,
    year: str,
    scope: str,
    profile_llm: str,
    output_dir: Optional[Path] = None,
) -> Path:
    if output_dir is None:
        output_dir = get_user_prediction_output_dir(
            user_id, dataset_source, year, scope, question_set
        )
    return output_dir / (
        f"{user_id}_{sanitize_filename_component(dataset_source)}_cutoff_{year}_"
        f"{scope.upper()}_"
        f"profile-[{sanitize_filename_component(profile_llm)}].txt"
    )


def find_existing_profile_text_path(
    user_id: str,
    dataset_source: str,
    question_set: str,
    year: str,
    scope: str,
    profile_llm: str,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    canonical_path = get_profile_text_output_path(
        user_id=user_id,
        dataset_source=dataset_source,
        question_set=question_set,
        year=year,
        scope=scope,
        profile_llm=profile_llm,
        output_dir=output_dir,
    )
    if canonical_path.exists():
        return canonical_path

    llm_component = sanitize_filename_component(profile_llm)
    dataset_component = sanitize_filename_component(dataset_source)
    canonical_filename = canonical_path.name
    legacy_filename = (
        f"{user_id}_{dataset_component}_{question_set}_cutoff_{year}_{scope.upper()}_"
        f"profile-[{llm_component}].txt"
    )
    if output_dir is not None:
        legacy_path = output_dir / legacy_filename
        if legacy_path.exists():
            return legacy_path
        return None

    user_output_dir = PRED_OUTPUT_ROOT / user_id
    if not user_output_dir.exists():
        return None

    grouped_dir_names = [
        f"{dataset_component}_{year}_{scope.upper()}_{question_set}",
        f"{dataset_component}_{year}",
    ]
    for grouped_dir_name in grouped_dir_names:
        grouped_output_dir = user_output_dir / grouped_dir_name
        if grouped_output_dir.exists():
            grouped_canonical_path = grouped_output_dir / canonical_filename
            if grouped_canonical_path.exists():
                return grouped_canonical_path
            grouped_legacy_path = grouped_output_dir / legacy_filename
            if grouped_legacy_path.exists():
                return grouped_legacy_path

    legacy_path = user_output_dir / legacy_filename
    if legacy_path.exists():
        return legacy_path
    return None


def get_embedding_output_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_embeddings.json")


def semantic_precompute_available(question_set: str, scope: str) -> bool:
    normalized_question_set = question_set.strip().lower()
    normalized_scope = scope.strip().lower()
    return (
        (normalized_question_set == "single" and normalized_scope == "e")
        or (normalized_question_set == "core" and normalized_scope == "c")
    )


def resolve_answered_embedding_path(
    dataset_source: str,
    before_source_name: str,
    year: str,
    question_set: str,
    scope: str,
) -> Path:
    normalized_scope = scope.strip().lower()
    if (
        dataset_source == "datasets_categories_only"
        and before_source_name == "core_latest_before"
        and year == "2023"
        and normalized_scope == "e"
    ):
        return SPECIAL_CORE_LATEST_BEFORE_EMBEDDING_PATH
    if (
        dataset_source == "datasets_categories_only"
        and before_source_name == "single_before"
        and year == "2023"
        and normalized_scope == "c"
    ):
        return SPECIAL_SINGLE_2023_BEFORE_EMBEDDING_PATH

    metadata_before_path = get_before_metadata_path_for_source(dataset_source, before_source_name, year)
    return get_embedding_output_path(metadata_before_path)


def resolve_target_embedding_path(
    dataset_source: str,
    metadata_path: Path,
    question_set: str,
    year: str,
) -> Path:
    normalized_question_set = question_set.strip().lower()
    if (
        dataset_source == "datasets_categories_only"
        and year == "2023"
        and normalized_question_set == "core"
    ):
        return SPECIAL_CORE_TARGET_EMBEDDING_PATH
    if (
        dataset_source == "datasets_categories_only"
        and year == "2023"
        and normalized_question_set == "single"
    ):
        return SPECIAL_SINGLE_2023_TARGET_EMBEDDING_PATH
    return get_embedding_output_path(metadata_path)


def get_profile_paragraph_embedding_output_path(profile_text_path: Path, model: str) -> Path:
    model_component = sanitize_filename_component(model)
    return profile_text_path.with_name(
        f"{profile_text_path.stem}_paragraph_embeddings_[{model_component}].json"
    )


def save_profile_text(profile_text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(profile_text.rstrip())
        f.write("\n")


def format_path_for_display(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def load_profile_text(output_path: Path) -> str:
    return output_path.read_text(encoding="utf-8").strip()


def compute_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_embedding_vector(raw_value: str) -> List[float]:
    value = raw_value.strip()
    if not value:
        raise ValueError("Embedding value is empty.")
    parsed = json.loads(value)
    if not isinstance(parsed, list) or any(not isinstance(item, (int, float)) for item in parsed):
        raise ValueError("Embedding must be a JSON array of numbers.")
    return [float(item) for item in parsed]


def load_embedding_lookup(embedding_path: Path) -> Dict[str, List[float]]:
    payload = read_json(embedding_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Embedding file must be a JSON object: {embedding_path}")

    embeddings_by_variable: Dict[str, List[float]] = {}
    for variable_name, embedding in payload.items():
        if not isinstance(variable_name, str):
            raise ValueError(f"Embedding key must be a string in {embedding_path}")
        if not isinstance(embedding, list) or any(not isinstance(item, (int, float)) for item in embedding):
            raise ValueError(
                f"Embedding value for variable_name={variable_name} must be a numeric array in {embedding_path}"
            )
        embeddings_by_variable[variable_name.strip()] = [float(item) for item in embedding]
    return embeddings_by_variable


def build_embedding_vectors(
    client: OpenAI,
    texts: List[str],
    model: str,
    batch_size: int,
) -> List[List[float]]:
    if batch_size <= 0:
        raise ValueError("Embedding batch size must be a positive integer.")
    if not texts:
        return []

    vectors: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        response = client.embeddings.create(model=model, input=batch_texts)
        vectors.extend([[float(value) for value in item.embedding] for item in response.data])

    if len(vectors) != len(texts):
        raise ValueError("Embedding response count did not match input text count.")
    return vectors


def get_before_metadata_path_for_source(dataset_source: str, before_source_name: str, year: str) -> Path:
    metadata_dir = METADATA_FOLDER_BY_DATASET_SOURCE.get(dataset_source)
    if metadata_dir is None:
        raise ValueError(f"Unsupported dataset source for metadata embeddings: {dataset_source}")

    if before_source_name == "core_before":
        filename = f"core_all_waves_questions_metadata_standalone_v2"
        if dataset_source == "datasets_categories_only":
            filename += "_categories_only"
        filename += f"_cutoff_{year}_before.csv"
        path = metadata_dir / filename
    elif before_source_name == "core_latest_before":
        filename = "core_latest_waves_questions_metadata_standalone_v2"
        if dataset_source == "datasets_categories_only":
            filename += "_categories_only"
        filename += f"_cutoff_{year}_before.csv"
        path = metadata_dir / filename
    elif before_source_name == "single_before":
        filename = "filtered_table_single_wave"
        if dataset_source == "datasets_categories_only":
            filename += "_categories_only"
        filename += f"_cutoff_{year}_before.csv"
        path = metadata_dir / filename
    else:
        raise ValueError(f"Unsupported before source name: {before_source_name}")

    if not path.exists():
        raise FileNotFoundError(f"Before metadata file not found: {path}")
    return path


def load_answered_embedding_rows(
    selected_files: List[Tuple[str, Path]],
    dataset_source: str,
    year: str,
    question_set: str,
    scope: str,
) -> List[Dict[str, str]]:
    rows_with_source: List[Dict[str, str]] = []
    for source_name, path in selected_files:
        embedding_path = resolve_answered_embedding_path(
            dataset_source=dataset_source,
            before_source_name=source_name,
            year=year,
            question_set=question_set,
            scope=scope,
        )
        if not embedding_path.exists():
            raise FileNotFoundError(
                f"Missing embedding file for {source_name}: {embedding_path}. "
                "Run the embedding precompute script first."
            )
        embedding_by_variable = load_embedding_lookup(embedding_path)

        for row in read_csv(path):
            answer = get_row_value(row, ["answer", "Answer"])
            variable_name = get_row_value(row, ["variable_name", "Variable Name"])
            if not answer or not variable_name:
                continue
            embedding = embedding_by_variable.get(variable_name)
            if not embedding:
                raise KeyError(
                    f"Missing precomputed embedding for variable_name={variable_name} "
                    f"in {embedding_path}"
                )
            row_with_source = dict(row)
            row_with_source["__source_name"] = source_name
            row_with_source["embedding"] = json.dumps(embedding, separators=(",", ":"))
            rows_with_source.append(row_with_source)
    return rows_with_source


def load_target_embeddings_by_variable(
    metadata_path: Path,
    target_rows: List[Dict[str, str]],
    dataset_source: str,
    question_set: str,
    year: str,
) -> Dict[str, List[float]]:
    embedding_path = resolve_target_embedding_path(
        dataset_source=dataset_source,
        metadata_path=metadata_path,
        question_set=question_set,
        year=year,
    )
    if not embedding_path.exists():
        raise FileNotFoundError(
            f"Missing target embedding file for {metadata_path.name}: {embedding_path}. "
            "Run the embedding precompute script first."
        )

    all_embeddings_by_variable = load_embedding_lookup(embedding_path)
    selected_variable_names = {
        get_row_value(row, ["variable_name", "Variable Name"])
        for row in target_rows
        if get_row_value(row, ["variable_name", "Variable Name"])
    }

    filtered_embeddings: Dict[str, List[float]] = {}
    for variable_name in selected_variable_names:
        embedding = all_embeddings_by_variable.get(variable_name)
        if embedding is None:
            raise KeyError(
                f"Missing target embedding for variable_name={variable_name} in {embedding_path}"
            )
        filtered_embeddings[variable_name] = embedding
    return filtered_embeddings


def split_structured_profile_into_paragraphs(structured_profile: str) -> List[Dict[str, str]]:
    paragraphs: List[Dict[str, str]] = []
    current_section = "PROFILE"
    paragraph_lines: List[str] = []

    def add_paragraph(text: str) -> None:
        cleaned_text = text.strip()
        if not cleaned_text:
            return
        paragraphs.append(
            {
                "paragraph_id": f"profile_paragraph_{len(paragraphs) + 1}",
                "section": current_section,
                "text": cleaned_text,
            }
        )

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if paragraph_lines:
            add_paragraph(" ".join(paragraph_lines))
            paragraph_lines = []

    for raw_line in structured_profile.splitlines():
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            continue
        if line.endswith(":") and line.upper() == line and not line.startswith("- "):
            flush_paragraph()
            current_section = line[:-1].strip() or "PROFILE"
            continue
        if line.startswith("- "):
            flush_paragraph()
            add_paragraph(line[2:].strip())
            continue
        paragraph_lines.append(line)

    flush_paragraph()
    return paragraphs


def build_profile_paragraph_embedding_text(paragraph: Dict[str, str]) -> str:
    section = paragraph.get("section", "").strip()
    text = paragraph.get("text", "").strip()
    if section:
        return f"section: {section}\ntext: {text}"
    return text


def write_profile_paragraph_embedding_cache(
    output_path: Path,
    structured_profile: str,
    model: str,
    paragraph_rows: List[Dict[str, str]],
) -> None:
    payload = {
        "profile_text_sha256": compute_text_sha256(structured_profile),
        "embedding_model": model,
        "paragraphs": [
            {
                "paragraph_id": row["paragraph_id"],
                "section": row["section"],
                "text": row["text"],
                "embedding": parse_embedding_vector(row["embedding"]),
            }
            for row in paragraph_rows
        ],
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True)


def load_profile_paragraph_embedding_cache(
    output_path: Path,
    structured_profile: str,
    model: str,
) -> List[Dict[str, str]]:
    payload = read_json(output_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Profile paragraph embedding cache must be a JSON object: {output_path}")

    expected_hash = compute_text_sha256(structured_profile)
    if payload.get("profile_text_sha256") != expected_hash:
        raise ValueError("Cached paragraph embeddings do not match the current structured profile.")
    if payload.get("embedding_model") != model:
        raise ValueError("Cached paragraph embeddings were created with a different embedding model.")

    raw_paragraphs = payload.get("paragraphs")
    if not isinstance(raw_paragraphs, list):
        raise ValueError("Cached paragraph embeddings must include a 'paragraphs' array.")

    paragraph_rows: List[Dict[str, str]] = []
    for item in raw_paragraphs:
        if not isinstance(item, dict):
            raise ValueError("Each cached paragraph embedding entry must be a JSON object.")
        paragraph_id = str(item.get("paragraph_id", "")).strip()
        section = str(item.get("section", "")).strip()
        text = str(item.get("text", "")).strip()
        embedding = item.get("embedding")
        if not paragraph_id or not text:
            raise ValueError("Cached paragraph embedding entries must include paragraph_id and text.")
        if not isinstance(embedding, list) or any(not isinstance(value, (int, float)) for value in embedding):
            raise ValueError("Cached paragraph embedding vectors must be numeric arrays.")
        paragraph_rows.append(
            {
                "paragraph_id": paragraph_id,
                "section": section,
                "text": text,
                "embedding": json.dumps([float(value) for value in embedding], separators=(",", ":")),
            }
        )
    return paragraph_rows


def build_profile_paragraph_embedding_rows(
    structured_profile: str,
    embedding_client: OpenAI,
    model: str,
    batch_size: int,
) -> List[Dict[str, str]]:
    paragraph_rows = split_structured_profile_into_paragraphs(structured_profile)
    if not paragraph_rows:
        raise ValueError("Structured profile did not produce any retrievable paragraphs.")

    embedding_texts = [build_profile_paragraph_embedding_text(row) for row in paragraph_rows]
    embedding_vectors = build_embedding_vectors(embedding_client, embedding_texts, model, batch_size)

    rows_with_embeddings: List[Dict[str, str]] = []
    for row, embedding in zip(paragraph_rows, embedding_vectors):
        row_with_embedding = dict(row)
        row_with_embedding["embedding"] = json.dumps(embedding, separators=(",", ":"))
        rows_with_embeddings.append(row_with_embedding)
    return rows_with_embeddings


def get_or_create_profile_paragraph_embedding_rows(
    profile_text_path: Path,
    structured_profile: str,
    embedding_client: OpenAI,
    model: str,
    batch_size: int,
) -> Tuple[List[Dict[str, str]], bool]:
    cache_path = get_profile_paragraph_embedding_output_path(profile_text_path, model)
    if cache_path.exists():
        try:
            return load_profile_paragraph_embedding_cache(
                output_path=cache_path,
                structured_profile=structured_profile,
                model=model,
            ), True
        except ValueError:
            pass

    paragraph_rows = build_profile_paragraph_embedding_rows(
        structured_profile=structured_profile,
        embedding_client=embedding_client,
        model=model,
        batch_size=batch_size,
    )
    write_profile_paragraph_embedding_cache(
        output_path=cache_path,
        structured_profile=structured_profile,
        model=model,
        paragraph_rows=paragraph_rows,
    )
    return paragraph_rows, False


def select_questions_for_user(
    metadata_rows: List[Dict[str, str]],
    target_rows: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    target_variable_names = {
        get_row_value(row, ["variable_name", "Variable Name"])
        for row in target_rows
        if get_row_value(row, ["variable_name", "Variable Name"])
    }
    return [
        row
        for row in metadata_rows
        if get_row_value(row, ["variable_name", "Variable Name"]) in target_variable_names
    ]


def get_target_rows_by_variable(
    target_rows: List[Dict[str, str]],
) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    for row in target_rows:
        variable_name = get_row_value(row, ["variable_name", "Variable Name"])
        if variable_name:
            mapping[variable_name] = row
    return mapping


def get_batch_group_column(question_set: str) -> str:
    if question_set == "core":
        return "project_number"
    if question_set == "single":
        return "study_id"
    raise ValueError(f"Unsupported question set for batching: {question_set}")


def ceil_division(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be a positive integer.")
    return (numerator + denominator - 1) // denominator


def group_questions_for_batching(
    selected_question_rows: List[Dict[str, str]],
    target_rows_by_variable: Dict[str, Dict[str, str]],
    question_set: str,
) -> List[Tuple[str, List[Dict[str, str]]]]:
    group_column = get_batch_group_column(question_set)
    grouped: Dict[str, List[Dict[str, str]]] = {}

    for question_meta in selected_question_rows:
        variable_name = get_row_value(question_meta, ["variable_name", "Variable Name"])
        target_row = target_rows_by_variable.get(variable_name, {})
        group_value = get_row_value(target_row, [group_column, group_column.title()])
        if not group_value:
            group_value = f"missing-{variable_name}"
        grouped.setdefault(group_value, []).append(question_meta)

    ordered_groups: List[Tuple[str, List[Dict[str, str]]]] = []
    for group_value in sorted(grouped.keys()):
        ordered_groups.append((group_value, grouped[group_value]))
    return ordered_groups


def split_question_groups_into_prediction_batches(
    grouped_question_rows: List[Tuple[str, List[Dict[str, str]]]],
    batch_target_size: int,
) -> List[Dict[str, object]]:
    if batch_target_size <= 0:
        raise ValueError("batch_target_size must be a positive integer.")

    prediction_batches: List[Dict[str, object]] = []
    for group_value, question_rows in grouped_question_rows:
        question_count = len(question_rows)
        if question_count == 0:
            continue

        batch_count = ceil_division(question_count, batch_target_size)
        dynamic_batch_size = ceil_division(question_count, batch_count)

        for batch_index_in_group, start in enumerate(range(0, question_count, dynamic_batch_size), start=1):
            batch_rows = question_rows[start:start + dynamic_batch_size]
            prediction_batches.append(
                {
                    "group_value": group_value,
                    "group_question_count": question_count,
                    "batch_index_in_group": batch_index_in_group,
                    "batch_count_in_group": batch_count,
                    "question_rows": batch_rows,
                    "dynamic_top_k": dynamic_batch_size,
                }
            )

    return prediction_batches


def build_answer_rule(answer_spec: Dict[str, object], question_meta: Dict[str, str]) -> str:
    spec_type = answer_spec.get("type")
    if spec_type == "categories":
        options = answer_spec.get("options", {})
        allowed_codes = ", ".join(options.keys()) if isinstance(options, dict) else ""
        return (
            "Return only one category code. "
            f"Allowed codes: {allowed_codes}. "
            "Do not return labels, words, or explanations."
        )
    if spec_type == "numeric":
        return (
            "Return only one numeric value. "
            f"numeric_type={answer_spec.get('numeric_type', '')}, "
            f"decimals={answer_spec.get('decimals', '')}, "
            f"range_low={answer_spec.get('range_low', '')}, "
            f"range_high={answer_spec.get('range_high', '')}. "
            "Do not include units or explanation."
        )

    question_type = get_row_value(question_meta, ["question type", "Question Type"])
    if question_type.strip().lower() == "others":
        return (
            "Return a specific substantive answer to the question text, not a missing-value code. "
            "Do not use placeholders such as -8 or -9."
        )
    return "Return only plain text for the answer. No extra prefixes."


def build_batch_answer_rule(variable_name: str, answer_spec: Dict[str, object], question_meta: Dict[str, str]) -> str:
    return f"{variable_name}: {build_answer_rule(answer_spec, question_meta)}"


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return cleaned


def build_batch_question_block(
    question_meta: Dict[str, str],
    answer_spec: Dict[str, object],
) -> str:
    variable_name = get_row_value(question_meta, ["Variable Name", "variable_name"])
    variable_label = get_row_value(question_meta, ["Variable Label", "variable_label"])
    categories = get_row_value(question_meta, ["Categories", "categories"])
    rep_type = get_row_value(question_meta, ["Representation Type", "representation_type"])
    question_type = get_row_value(question_meta, ["question type", "Question Type"])

    lines = [
        f"variable_name: {variable_name}",
        f"question_text: {variable_label}",
        f"representation_type: {rep_type if rep_type else 'N/A'}",
        f"question_type: {question_type if question_type else 'N/A'}",
        f"options/categories: {categories if categories else 'N/A'}",
        f"answer_rule: {build_batch_answer_rule(variable_name, answer_spec, question_meta)}",
    ]
    return "\n".join(lines)


def build_batch_questions_text(
    batch_question_rows: List[Dict[str, str]],
    answer_specs: Dict[str, Dict[str, object]],
) -> str:
    blocks: List[str] = []
    for question_meta in batch_question_rows:
        variable_name = get_row_value(question_meta, ["Variable Name", "variable_name"])
        answer_spec = answer_specs[variable_name]
        block = build_batch_question_block(question_meta, answer_spec)
        blocks.append(block)
    return "\n\n".join(blocks)


def get_agent_label(agent_mode: str, top_k: Optional[int] = None) -> str:
    if agent_mode == "current":
        return "agent-baseline"
    if agent_mode == "profile":
        return "agent-profile"
    if agent_mode == "profile_topk_para_semantic":
        if top_k is None:
            return "agent-profile-topK-para-semantic"
        return f"agent-profile-topK-para-semantic{top_k}"
    if agent_mode == "profile_topk_semantic":
        return "agent-profile-topK-rows-semantic"
    return "agent-profile-topK-lexical"


def get_selected_file_groups(
    user_dir: Path, scope: str, year: str
) -> Tuple[List[Tuple[str, Path]], List[Tuple[str, Path]], List[Tuple[str, Path]]]:
    selected_files = get_selected_input_files(user_dir, scope, year)
    bg_files = [(name, path) for name, path in selected_files if name == "bg_for_cutoff"]
    before_files = [(name, path) for name, path in selected_files if name in BEFORE_SOURCE_NAMES]
    return selected_files, bg_files, before_files


def collect_answered_rows(selected_files: List[Tuple[str, Path]]) -> List[Dict[str, str]]:
    rows_with_source: List[Dict[str, str]] = []
    for source_name, path in selected_files:
        for row in read_csv(path):
            answer = get_row_value(row, ["answer", "Answer"])
            if not answer:
                continue
            row_with_source = dict(row)
            row_with_source["__source_name"] = source_name
            rows_with_source.append(row_with_source)
    return rows_with_source


def validate_profile_text(candidate: str) -> str:
    text = strip_code_fences(candidate).strip()
    if not text:
        raise ValueError("Profile response is empty.")

    missing_headers = [header for header in PROFILE_SECTION_HEADERS if header not in text]
    if missing_headers:
        raise ValueError(
            "Profile response is missing required section headers: "
            + ", ".join(missing_headers)
        )

    if len(text.split()) > 1500:
        raise ValueError("Profile response exceeds the 1500-word limit.")

    return text


def build_profile(history_text: str) -> str:
    return f"""
You are analyzing a set of questions and answers from a single person.

Your goal is to build a predictive persona profile that can estimate how this person would answer future questions on any topic.

From the examples provided, infer stable patterns in:
- beliefs and worldview
- personality traits
- reasoning style
- decision-making strategy
- knowledge domains and expertise
- handling of uncertainty
- biases or recurring assumptions
- communication style
- priorities and values
- reactions to disagreement

Output a structured profile that can be used by another AI to simulate this person.

Do NOT summarize answers. Extract consistent patterns and behavioral rules.

Format:

1. Personality traits
2. Reasoning style
3. Knowledge profile
4. Values and motivations
5. Biases and heuristics
6. Decision patterns
7. Confidence patterns

Past Q&A:
{history_text}
"""


def run_profile_builder(
    agent: SingleNodeAgent, history_text: str, llm: str
) -> str:
    prompt = build_profile(history_text)
    for attempt in range(3):
        candidate = agent.run(prompt)
        if not candidate:
            prompt = (
                f"{prompt}\n\nPrevious response was empty.\n"
                f"Profile generation LLM: {llm}\n"
                "Return a valid persona profile using the required 10-section format now."
            )
            continue

        try:
            return validate_profile_text(candidate)
        except ValueError as exc:
            prompt = (
                f"{prompt}\n\nPrevious response was invalid.\n"
                f"Profile generation LLM: {llm}\n"
                f"Validation error: {exc}\n"
                "Return only a valid persona profile with exactly the required 10 section headers now."
            )

    raise ValueError("Profile builder could not produce a valid persona profile.")


def tokenize_for_similarity(*parts: str) -> List[str]:
    text = " ".join(part for part in parts if part).lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


def compute_relatedness_score(
    target_question: Dict[str, str], before_row: Dict[str, str]
) -> Tuple[float, str]:
    target_label = get_row_value(target_question, ["Variable Label", "variable_label"])
    target_categories = get_row_value(target_question, ["Categories", "categories"])
    target_rep = get_row_value(target_question, ["Representation Type", "representation_type"])
    target_qtype = get_row_value(target_question, ["question type", "Question Type"])

    before_label = get_row_value(before_row, ["Variable Label", "variable_label"])
    before_categories = get_row_value(before_row, ["Categories", "categories"])
    before_rep = get_row_value(before_row, ["Representation Type", "representation_type"])
    before_qtype = get_row_value(before_row, ["question type", "Question Type"])

    target_label_tokens = set(tokenize_for_similarity(target_label))
    before_label_tokens = set(tokenize_for_similarity(before_label))
    target_category_tokens = set(tokenize_for_similarity(target_categories))
    before_category_tokens = set(tokenize_for_similarity(before_categories))

    label_overlap = len(target_label_tokens & before_label_tokens)
    category_overlap = len(target_category_tokens & before_category_tokens)

    score = float(label_overlap * 3 + category_overlap)
    if target_rep and before_rep and target_rep.strip().lower() == before_rep.strip().lower():
        score += 2.0
    if target_qtype and before_qtype and target_qtype.strip().lower() == before_qtype.strip().lower():
        score += 1.0

    reason_parts: List[str] = []
    if label_overlap:
        reason_parts.append(f"label_overlap={label_overlap}")
    if category_overlap:
        reason_parts.append(f"category_overlap={category_overlap}")
    if target_rep and before_rep and target_rep.strip().lower() == before_rep.strip().lower():
        reason_parts.append("same_representation")
    if target_qtype and before_qtype and target_qtype.strip().lower() == before_qtype.strip().lower():
        reason_parts.append("same_question_type")

    return score, ", ".join(reason_parts) if reason_parts else "fallback_rank"


def get_top_k_related_rows(
    target_question: Dict[str, str], before_rows: List[Dict[str, str]], top_k: int
) -> List[Dict[str, str]]:
    scored_rows: List[Tuple[float, int, Dict[str, str], str]] = []
    for index, row in enumerate(before_rows):
        score, reason = compute_relatedness_score(target_question, row)
        scored_rows.append((score, index, row, reason))

    scored_rows.sort(key=lambda item: (-item[0], item[1]))
    top_rows: List[Dict[str, str]] = []
    for score, _, row, reason in scored_rows[:top_k]:
        row_with_score = dict(row)
        row_with_score["__match_score"] = str(score)
        row_with_score["__match_reason"] = reason
        top_rows.append(row_with_score)
    return top_rows


def get_top_k_related_rows_for_batch(
    batch_question_rows: List[Dict[str, str]], before_rows: List[Dict[str, str]], top_k: int
) -> List[Dict[str, str]]:
    scored_rows: List[Tuple[float, int, Dict[str, str], str]] = []
    for index, row in enumerate(before_rows):
        total_score = 0.0
        reason_parts: List[str] = []
        for question_meta in batch_question_rows:
            variable_name = get_row_value(question_meta, ["Variable Name", "variable_name"])
            score, reason = compute_relatedness_score(question_meta, row)
            total_score += score
            if score > 0:
                reason_parts.append(f"{variable_name}: {reason}")

        if not reason_parts:
            reason_parts.append("fallback_rank")
        scored_rows.append((total_score, index, row, "; ".join(reason_parts)))

    scored_rows.sort(key=lambda item: (-item[0], item[1]))
    top_rows: List[Dict[str, str]] = []
    for score, _, row, reason in scored_rows[:top_k]:
        row_with_score = dict(row)
        row_with_score["__match_score"] = str(score)
        row_with_score["__match_reason"] = reason
        top_rows.append(row_with_score)
    return top_rows


def cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
    if len(vector_a) != len(vector_b):
        raise ValueError("Embedding vectors must have the same dimension.")
    norm_a = math.sqrt(sum(value * value for value in vector_a))
    norm_b = math.sqrt(sum(value * value for value in vector_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    dot_product = sum(left * right for left, right in zip(vector_a, vector_b))
    return dot_product / (norm_a * norm_b)


def get_top_k_semantic_related_rows_for_batch(
    batch_question_rows: List[Dict[str, str]],
    before_embedding_rows: List[Dict[str, str]],
    target_embeddings_by_variable: Dict[str, List[float]],
    top_k: int,
) -> List[Dict[str, str]]:
    scored_rows: List[Tuple[float, int, Dict[str, str], str]] = []
    for index, row in enumerate(before_embedding_rows):
        row_embedding = parse_embedding_vector(get_row_value(row, ["embedding"]))
        similarities: List[Tuple[str, float]] = []
        for question_meta in batch_question_rows:
            variable_name = get_row_value(question_meta, ["Variable Name", "variable_name"])
            target_embedding = target_embeddings_by_variable.get(variable_name)
            if target_embedding is None:
                raise KeyError(f"Missing target embedding for variable_name={variable_name}")
            similarities.append((variable_name, cosine_similarity(target_embedding, row_embedding)))

        total_score = sum(score for _, score in similarities)
        top_matches = sorted(similarities, key=lambda item: item[1], reverse=True)[:3]
        reason = "; ".join(
            f"{variable_name}: cosine={score:.4f}" for variable_name, score in top_matches
        )
        row_with_score = dict(row)
        row_with_score["__match_score"] = f"{total_score:.6f}"
        row_with_score["__match_reason"] = reason
        scored_rows.append((total_score, index, row_with_score, reason))

    scored_rows.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row, _ in scored_rows[:top_k]]


def get_top_k_semantic_profile_paragraphs_for_batch(
    batch_question_rows: List[Dict[str, str]],
    profile_paragraph_embedding_rows: List[Dict[str, str]],
    target_embeddings_by_variable: Dict[str, List[float]],
    top_k: int,
) -> List[Dict[str, str]]:
    scored_rows: List[Tuple[float, int, Dict[str, str], str]] = []
    for index, row in enumerate(profile_paragraph_embedding_rows):
        paragraph_embedding = parse_embedding_vector(get_row_value(row, ["embedding"]))
        similarities: List[Tuple[str, float]] = []
        for question_meta in batch_question_rows:
            variable_name = get_row_value(question_meta, ["Variable Name", "variable_name"])
            target_embedding = target_embeddings_by_variable.get(variable_name)
            if target_embedding is None:
                raise KeyError(f"Missing target embedding for variable_name={variable_name}")
            similarities.append((variable_name, cosine_similarity(target_embedding, paragraph_embedding)))

        total_score = sum(score for _, score in similarities)
        top_matches = sorted(similarities, key=lambda item: item[1], reverse=True)[:3]
        reason = "; ".join(
            f"{variable_name}: cosine={score:.4f}" for variable_name, score in top_matches
        )
        row_with_score = dict(row)
        row_with_score["__match_score"] = f"{total_score:.6f}"
        row_with_score["__match_reason"] = reason
        scored_rows.append((total_score, index, row_with_score, reason))

    scored_rows.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row, _ in scored_rows[:top_k]]


def build_relevant_rows_context(rows: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for index, row in enumerate(rows, start=1):
        variable_name = get_row_value(row, ["variable_name", "Variable Name"])
        label = get_row_value(row, ["variable_label", "Variable Label"])
        answer = get_row_value(row, ["answer", "Answer"])
        categories = get_row_value(row, ["categories", "Categories"])
        year = get_row_value(row, ["year", "Year"])
        source_name = row.get("__source_name", "")
        match_reason = row.get("__match_reason", "")

        lines.append(f"[related_row_{index}]")
        if source_name:
            lines.append(f"source: {source_name}")
        lines.append(f"question_id: {variable_name}")
        if label:
            lines.append(f"variable_label: {label}")
        lines.append(f"answer: {answer}")
        if categories:
            lines.append(f"categories: {categories}")
        if year:
            lines.append(f"year: {year}")
        if match_reason:
            lines.append(f"match_reason: {match_reason}")
        match_score = row.get("__match_score", "")
        if match_score:
            lines.append(f"match_score: {match_score}")
        lines.append("")

    return "\n".join(lines).strip() if lines else "No related prior answered rows available."


def build_profile_paragraphs_context(paragraph_rows: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for index, row in enumerate(paragraph_rows, start=1):
        section = row.get("section", "")
        paragraph_id = row.get("paragraph_id", "")
        text = row.get("text", "")
        match_reason = row.get("__match_reason", "")
        match_score = row.get("__match_score", "")

        lines.append(f"[profile_paragraph_{index}]")
        if paragraph_id:
            lines.append(f"paragraph_id: {paragraph_id}")
        if section:
            lines.append(f"section: {section}")
        lines.append(f"text: {text}")
        if match_reason:
            lines.append(f"match_reason: {match_reason}")
        if match_score:
            lines.append(f"match_score: {match_score}")
        lines.append("")

    return "\n".join(lines).strip() if lines else "No related profile paragraphs available."


def build_batch_prompt(
    batch_question_rows: List[Dict[str, str]],
    user_context: str,
    answer_specs: Dict[str, Dict[str, object]],
    question_set: str,
    year: str,
    group_label: str,
    group_value: str,
) -> str:
    questions_text = build_batch_questions_text(batch_question_rows, answer_specs)
    return f"""You are simulating a specific person answering a survey.

Your task is to answer each new question the way this person would most likely answer, using:
1.⁠ ⁠the respondent background data as supporting context

Background data:
{user_context}

Instructions:
- ⁠Answer as the person.
- ⁠Use the background data to infer the most likely response.
- ⁠If the exact answer is unclear, give the response most consistent with the available evidence.
- ⁠Give one answer for every question.

Output requirements:
- ⁠Return valid JSON only.
- ⁠Do not include markdown, code fences, comments, or extra text.
- ⁠Use this exact structure:
{{
  "predictions": [
    {{"variable_name": "<id>", "predicted_answer": "<answer only>"}},
    {{"variable_name": "<id>", "predicted_answer": "<answer only>"}}
  ]
}}

New survey questions:
{questions_text}

"""


def build_profile_prediction_prompt(
    batch_question_rows: List[Dict[str, str]],
    bg_context: str,
    structured_profile: str,
    answer_specs: Dict[str, Dict[str, object]],
    question_set: str,
    year: str,
    group_label: str,
    group_value: str,
) -> str:
    questions_text = build_batch_questions_text(batch_question_rows, answer_specs)

    return f"""You are simulating a specific person answering a survey.

Your task is to answer each new question the way this person would most likely answer, using:
1.⁠ ⁠the behavioral profile below
2.⁠ ⁠the respondent background data as supporting context

Background data:
{bg_context if bg_context else 'No background data available.'}

Profile:
{structured_profile}

Instructions:
- ⁠Answer as the person.
- ⁠Use the profile and background data to infer the most likely response.
- ⁠If the exact answer is unclear, give the response most consistent with the available evidence.
- ⁠Give one answer for every question.

Output requirements:
- ⁠Return valid JSON only.
- ⁠Do not include markdown, code fences, comments, or extra text.
- ⁠Use this exact structure:
{{
  "predictions": [
    {{"variable_name": "<id>", "predicted_answer": "<answer only>"}},
    {{"variable_name": "<id>", "predicted_answer": "<answer only>"}}
  ]
}}

New survey questions:
{questions_text}
"""


def build_profile_topk_prediction_prompt(
    batch_question_rows: List[Dict[str, str]],
    bg_context: str,
    structured_profile: str,
    answer_specs: Dict[str, Dict[str, object]],
    batch_related_rows: List[Dict[str, str]],
    question_set: str,
    year: str,
    retrieval_mode: str,
    group_label: str,
    group_value: str,
) -> str:
    questions_text = build_batch_questions_text(batch_question_rows, answer_specs)
    related_rows_text = build_relevant_rows_context(batch_related_rows)
    retrieved_count = len(batch_related_rows)
    similarity_adverb = "semantically" if retrieval_mode == "semantic" else "lexically"

    return f"""You are simulating a specific person answering a survey.

Your task is to answer each new question the way this person would most likely answer, using:
1.⁠ ⁠the behavioral profile below
2. the previously answered questions as behavioral examples
3.⁠ ⁠the respondent background data as supporting context

Background data:
{bg_context if bg_context else 'No background data available.'}

Profile:
{structured_profile}

Previously answered questions:
{related_rows_text}

Instructions:
- ⁠Answer as the person.
- ⁠Use the profile, background data, and prior answers to infer the most likely response.
- ⁠If the exact answer is unclear, give the response most consistent with the available evidence.
- ⁠Give one answer for every question.

Output requirements:
- ⁠Return valid JSON only.
- ⁠Do not include markdown, code fences, comments, or extra text.
- ⁠Use this exact structure:
{{
  "predictions": [
    {{"variable_name": "<id>", "predicted_answer": "<answer only>"}},
    {{"variable_name": "<id>", "predicted_answer": "<answer only>"}}
  ]
}}

New survey questions:
{questions_text}
"""


def build_profile_topk_paragraph_prediction_prompt(
    batch_question_rows: List[Dict[str, str]],
    bg_context: str,
    answer_specs: Dict[str, Dict[str, object]],
    batch_profile_paragraphs: List[Dict[str, str]],
    question_set: str,
    year: str,
    group_label: str,
    group_value: str,
) -> str:
    questions_text = build_batch_questions_text(batch_question_rows, answer_specs)
    profile_paragraphs_text = build_profile_paragraphs_context(batch_profile_paragraphs)
    retrieved_count = len(batch_profile_paragraphs)

    return f"""You are answering a batch of survey questions for a specific user.
Use only the respondent background data and the top {retrieved_count} most semantically relevant structured profile paragraphs for this batch.
Prefer direct evidence from the retrieved profile paragraphs when available.

Question set: {question_set}
Cutoff year: {year}
Batch grouping column: {group_label}
Batch grouping value: {group_value}

Background data:
{bg_context if bg_context else 'No background data available.'}

Retrieved structured profile paragraphs for this batch:
{profile_paragraphs_text}

Questions to answer in this batch:
{questions_text}

Instructions:
1. Prioritize the retrieved structured profile paragraphs that are most semantically similar to this batch.
2. Use the background data as supporting context.
3. Do not assume facts that are not supported by the retrieved profile paragraphs or the background data.
4. Infer conservatively when exact evidence is missing.
5. Return exactly one prediction for every listed variable_name.
6. Return JSON only in this format:
{{
  "predictions": [
    {{"variable_name": "<id>", "predicted_answer": "<answer only>"}},
    {{"variable_name": "<id>", "predicted_answer": "<answer only>"}}
  ]
}}
7. Do not include explanations or extra keys.
"""


def build_agent(
    provider: str,
    model: str,
    api_key: str,
    reasoning_effort: Optional[str] = None,
) -> SingleNodeAgent:
    return SingleNodeAgent(
        AgentConfig(
            provider=provider,
            model=model,
            system_prompt="You map user profile data to survey answers carefully and transparently.",
            api_key=api_key,
            reasoning_effort=reasoning_effort,
        )
    )


def parse_batch_prediction_response(candidate: str) -> Dict[str, str]:
    parsed = json.loads(strip_code_fences(candidate))
    if not isinstance(parsed, dict):
        raise ValueError("Batch response must be a JSON object.")

    predictions = parsed.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("Batch response must contain a 'predictions' array.")

    answers_by_variable: Dict[str, str] = {}
    for item in predictions:
        if not isinstance(item, dict):
            raise ValueError("Each prediction must be a JSON object.")
        variable_name = str(item.get("variable_name", "")).strip()
        if not variable_name:
            raise ValueError("Each prediction must include variable_name.")
        if "predicted_answer" not in item:
            raise ValueError(f"Prediction for {variable_name} is missing predicted_answer.")
        predicted_answer = item.get("predicted_answer")
        answers_by_variable[variable_name] = "" if predicted_answer is None else str(predicted_answer).strip()
    return answers_by_variable


def run_prediction_for_batch(
    agent: SingleNodeAgent,
    batch_question_rows: List[Dict[str, str]],
    prompt: str,
    answer_specs: Dict[str, Dict[str, object]],
) -> Dict[str, str]:
    expected_variables = [
        get_row_value(question_meta, ["Variable Name", "variable_name"])
        for question_meta in batch_question_rows
    ]

    validation_attempts = 0
    while True:
        candidate = agent.run(prompt).strip()
        try:
            parsed_answers = parse_batch_prediction_response(candidate)
        except (json.JSONDecodeError, ValueError) as exc:
            validation_error = str(exc)
        else:
            missing = [variable for variable in expected_variables if variable not in parsed_answers]
            unexpected = [variable for variable in parsed_answers if variable not in expected_variables]
            if missing:
                validation_error = f"Missing predictions for: {', '.join(missing)}"
            elif unexpected:
                validation_error = f"Unexpected predictions for: {', '.join(unexpected)}"
            else:
                normalized_answers: Dict[str, str] = {}
                item_errors: List[str] = []
                for question_meta in batch_question_rows:
                    variable_name = get_row_value(question_meta, ["Variable Name", "variable_name"])
                    normalized, error = normalize_answer(
                        parsed_answers.get(variable_name, ""),
                        answer_specs[variable_name],
                    )
                    if error:
                        item_errors.append(f"{variable_name}: {error}")
                    else:
                        normalized_answers[variable_name] = normalized or ""
                if not item_errors:
                    return normalized_answers
                validation_error = "; ".join(item_errors)

        validation_attempts += 1
        if validation_attempts >= 4:
            raise ValueError(f"Model could not produce a valid batch response: {validation_error}")

        prompt = (
            f"{prompt}\n\nPrevious invalid response: {candidate}\n"
            f"Validation error: {validation_error}\n"
            "Return only a valid JSON object with one prediction for each requested variable_name now."
        )


def prepare_batch_prediction_prompt(
    agent_mode: str,
    batch_question_rows: List[Dict[str, str]],
    answer_specs: Dict[str, Dict[str, object]],
    question_set: str,
    year: str,
    raw_user_context: str,
    bg_context: str,
    structured_profile: Optional[str],
    before_rows: List[Dict[str, str]],
    before_embedding_rows: List[Dict[str, str]],
    profile_paragraph_embedding_rows: List[Dict[str, str]],
    target_embeddings_by_variable: Dict[str, List[float]],
    retrieval_k: Optional[int],
    group_label: str,
    group_value: str,
) -> Tuple[str, List[Dict[str, str]]]:
    if agent_mode == "current":
        return (
            build_batch_prompt(
                batch_question_rows,
                raw_user_context,
                answer_specs,
                question_set,
                year,
                group_label,
                group_value,
            ),
            [],
        )

    if not structured_profile:
        raise ValueError("Structured profile is required for this agent mode.")

    if agent_mode == "profile":
        return (
            build_profile_prediction_prompt(
                batch_question_rows=batch_question_rows,
                bg_context=bg_context,
                structured_profile=structured_profile,
                answer_specs=answer_specs,
                question_set=question_set,
                year=year,
                group_label=group_label,
                group_value=group_value,
            ),
            [],
        )

    if retrieval_k is None:
        raise ValueError("retrieval_k is required for the profile top-k agent modes.")

    if agent_mode == "profile_topk_lexical":
        batch_related_rows = get_top_k_related_rows_for_batch(
            batch_question_rows,
            before_rows,
            retrieval_k,
        )
        retrieval_mode = "lexical"
    elif agent_mode == "profile_topk_semantic":
        if not before_embedding_rows:
            raise ValueError(
                "Precomputed prior-question embeddings are required for agent-profile-topK-rows-semantic."
            )
        if not target_embeddings_by_variable:
            raise ValueError(
                "Precomputed target-question embeddings are required for agent-profile-topK-rows-semantic."
            )
        batch_related_rows = get_top_k_semantic_related_rows_for_batch(
            batch_question_rows=batch_question_rows,
            before_embedding_rows=before_embedding_rows,
            target_embeddings_by_variable=target_embeddings_by_variable,
            top_k=retrieval_k,
        )
        retrieval_mode = "semantic"
    elif agent_mode == "profile_topk_para_semantic":
        if not profile_paragraph_embedding_rows:
            raise ValueError(
                "Structured profile paragraph embeddings are required for agent-profile-topK-para-semantic."
            )
        if not target_embeddings_by_variable:
            raise ValueError(
                "Precomputed target-question embeddings are required for agent-profile-topK-para-semantic."
            )
        batch_profile_paragraphs = get_top_k_semantic_profile_paragraphs_for_batch(
            batch_question_rows=batch_question_rows,
            profile_paragraph_embedding_rows=profile_paragraph_embedding_rows,
            target_embeddings_by_variable=target_embeddings_by_variable,
            top_k=retrieval_k,
        )
        return (
            build_profile_topk_paragraph_prediction_prompt(
                batch_question_rows=batch_question_rows,
                bg_context=bg_context,
                answer_specs=answer_specs,
                batch_profile_paragraphs=batch_profile_paragraphs,
                question_set=question_set,
                year=year,
                group_label=group_label,
                group_value=group_value,
            ),
            batch_profile_paragraphs,
        )
    else:
        raise ValueError(f"Unsupported top-k agent mode: {agent_mode}")
    return (
        build_profile_topk_prediction_prompt(
            batch_question_rows=batch_question_rows,
            bg_context=bg_context,
            structured_profile=structured_profile,
            answer_specs=answer_specs,
            batch_related_rows=batch_related_rows,
            question_set=question_set,
            year=year,
            retrieval_mode=retrieval_mode,
            group_label=group_label,
            group_value=group_value,
        ),
        batch_related_rows,
    )


def write_prediction_output(
    target_rows: List[Dict[str, str]],
    predictions_by_variable: Dict[str, str],
    output_path: Path,
) -> int:
    fieldnames = list(target_rows[0].keys()) if target_rows else []
    if "Pred Answer" not in fieldnames:
        inserted = False
        for answer_column in ["Answer", "answer"]:
            if answer_column in fieldnames:
                insert_at = fieldnames.index(answer_column) + 1
                fieldnames.insert(insert_at, "Pred Answer")
                inserted = True
                break
        if not inserted:
            fieldnames.append("Pred Answer")

    written = 0
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in target_rows:
            output_row = dict(row)
            variable_name = get_row_value(row, ["variable_name", "Variable Name"])
            if variable_name in predictions_by_variable:
                output_row["Pred Answer"] = predictions_by_variable[variable_name]
                written += 1
            else:
                output_row["Pred Answer"] = ""
            writer.writerow(output_row)
    return written


def write_retrieved_profile_paragraphs_output(
    output_path: Path,
    batches: List[Dict[str, object]],
) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"batches": batches}, f, ensure_ascii=True, indent=2)
        f.write("\n")


def prompt_for_api_key(provider: str, current_api_key: Optional[str]) -> str:
    api_key = resolve_api_key(provider) or current_api_key
    while not api_key:
        env_name = API_KEY_ENV.get(provider, "API_KEY")
        typed_key = input(
            f"Missing API key for {provider} ({env_name}). Paste key now: "
        ).strip()
        if typed_key:
            api_key = typed_key
        else:
            print("API key cannot be empty.")
    return api_key


def mode_requires_structured_profile(agent_mode: str) -> bool:
    return agent_mode != "current"


def mode_requires_semantic_embeddings(agent_mode: str) -> bool:
    return agent_mode in {"profile_topk_semantic", "profile_topk_para_semantic"}


def get_agent_mode_phases(selected_agent_mode: Optional[str]) -> List[List[str]]:
    if selected_agent_mode is not None:
        return [[selected_agent_mode]]
    return [list(phase) for phase in DEFAULT_AGENT_MODE_PHASES]


def mode_name(agent_mode: str) -> str:
    return MODE_DISPLAY_NAME.get(agent_mode, agent_mode)


def log_with_prefix(print_lock: threading.Lock, prefix: str, message: str) -> None:
    with print_lock:
        print(f"[{prefix}] {message}")


def ensure_structured_profile(
    shared_profile_state: Dict[str, object],
    before_context: str,
    profile_agent: SingleNodeAgent,
    profile_llm_name: str,
    user_id: str,
    dataset_source: str,
    question_set: str,
    year: str,
    scope: str,
    profile_text_output_dir: Optional[Path],
    print_lock: threading.Lock,
    prefix: str,
) -> Dict[str, object]:
    lock = shared_profile_state["lock"]
    if not hasattr(lock, "acquire") or not hasattr(lock, "release"):
        raise ValueError("Shared profile state is missing a valid lock.")

    with lock:
        cached_profile = shared_profile_state.get("structured_profile")
        if isinstance(cached_profile, str) and cached_profile:
            return {
                "structured_profile": cached_profile,
                "structured_profile_text_output_path": shared_profile_state.get(
                    "structured_profile_text_output_path"
                ),
                "structured_profile_loaded_from_path": shared_profile_state.get(
                    "structured_profile_loaded_from_path"
                ),
                "structured_profile_reused": True,
                "structured_profile_cache_source": "in-memory",
            }

        cached_error = shared_profile_state.get("error")
        if cached_error is not None:
            raise ValueError(str(cached_error))

        if not before_context:
            raise ValueError("No *_before.csv context was found for the selected input scope.")

        structured_profile_text_output_path = get_profile_text_output_path(
            user_id=user_id,
            dataset_source=dataset_source,
            question_set=question_set,
            year=year,
            scope=scope,
            profile_llm=profile_llm_name,
            output_dir=profile_text_output_dir,
        )
        existing_profile_text_path = find_existing_profile_text_path(
            user_id=user_id,
            dataset_source=dataset_source,
            question_set=question_set,
            year=year,
            scope=scope,
            profile_llm=profile_llm_name,
            output_dir=profile_text_output_dir,
        )
        structured_profile: Optional[str] = None
        structured_profile_loaded_from_path: Optional[Path] = None
        structured_profile_reused = False
        structured_profile_cache_source = "generated"
        should_rebuild_profile = existing_profile_text_path is None

        if existing_profile_text_path is not None:
            log_with_prefix(print_lock, prefix, "Loading existing structured profile from cache...")
            try:
                structured_profile = load_profile_text(existing_profile_text_path)
                structured_profile_loaded_from_path = existing_profile_text_path
                structured_profile_reused = True
                structured_profile_cache_source = "disk-cache"
            except ValueError as exc:
                log_with_prefix(print_lock, prefix, f"cached_profile_invalid: {exc}")
                log_with_prefix(print_lock, prefix, "Rebuilding structured profile text...")
                should_rebuild_profile = True

        if should_rebuild_profile:
            log_with_prefix(print_lock, prefix, "Building structured profile from *_before.csv...")
            try:
                structured_profile = run_profile_builder(
                    profile_agent,
                    before_context,
                    profile_llm_name,
                )
                save_profile_text(structured_profile, structured_profile_text_output_path)
            except Exception as exc:
                shared_profile_state["error"] = exc
                raise

        if not structured_profile:
            error = ValueError("Structured profile text could not be prepared.")
            shared_profile_state["error"] = error
            raise error

        shared_profile_state["structured_profile"] = structured_profile
        shared_profile_state["structured_profile_text_output_path"] = structured_profile_text_output_path
        shared_profile_state["structured_profile_loaded_from_path"] = structured_profile_loaded_from_path
        shared_profile_state["structured_profile_reused"] = structured_profile_reused
        shared_profile_state["structured_profile_cache_source"] = structured_profile_cache_source
        return {
            "structured_profile": structured_profile,
            "structured_profile_text_output_path": structured_profile_text_output_path,
            "structured_profile_loaded_from_path": structured_profile_loaded_from_path,
            "structured_profile_reused": structured_profile_reused,
            "structured_profile_cache_source": structured_profile_cache_source,
        }


def run_agent_mode_execution(
    agent_mode: str,
    user_id: str,
    dataset_source: str,
    question_set: str,
    year: str,
    scope: str,
    batch_target_size: int,
    metadata_path: Path,
    user_target_path: Path,
    input_files: List[Path],
    bg_input_files: List[Path],
    newly_loaded_sources: List[str],
    reused_sources: List[str],
    bg_newly_loaded_sources: List[str],
    bg_reused_sources: List[str],
    user_context: str,
    bg_context: str,
    before_context: str,
    before_rows: List[Dict[str, str]],
    before_embedding_rows: List[Dict[str, str]],
    profile_paragraph_embedding_rows: List[Dict[str, str]],
    target_embeddings_by_variable: Dict[str, List[float]],
    metadata_rows: List[Dict[str, str]],
    target_rows: List[Dict[str, str]],
    selected_question_rows: List[Dict[str, str]],
    grouped_question_rows: List[Tuple[str, List[Dict[str, str]]]],
    prediction_batches: List[Dict[str, object]],
    grouping_column: str,
    prediction_provider: str,
    prediction_model: str,
    prediction_reasoning_mode: str,
    prediction_api_key: str,
    profile_provider: Optional[str],
    profile_model: Optional[str],
    profile_api_key: Optional[str],
    profile_llm_name: str,
    profile_text_output_dir: Optional[Path],
    shared_profile_state: Dict[str, object],
    print_lock: threading.Lock,
    selection_header: str = "--- Selection ---",
    output_header: str = "--- Output ---",
) -> Dict[str, object]:
    prefix = mode_name(agent_mode)
    log_with_prefix(print_lock, prefix, "Starting mode run...")
    if agent_mode == "profile_topk_semantic" and not semantic_precompute_available(question_set, scope):
        output_path = get_prediction_output_path(
            user_id=user_id,
            dataset_source=dataset_source,
            question_set=question_set,
            year=year,
            scope=scope,
            batch_target_size=batch_target_size,
            agent_label=get_agent_label(agent_mode),
            profile_llm=profile_llm_name,
            prediction_llm=f"{prediction_provider}:{prediction_model}",
            prediction_reasoning_mode=prediction_reasoning_mode,
        )
        log_with_prefix(print_lock, prefix, MISSING_PRECOMPUTE_EMBEDDINGS_MESSAGE)
        return {
            "agent_mode": agent_mode,
            "output_path": output_path,
            "predictions_generated": 0,
            "rows_with_pred_answer": 0,
            "failures": [MISSING_PRECOMPUTE_EMBEDDINGS_MESSAGE],
            "prediction_reasoning_mode": prediction_reasoning_mode,
        }

    baseline_only_mode = agent_mode == "current"
    effective_scope = "a" if baseline_only_mode else scope
    effective_input_files = bg_input_files if baseline_only_mode else input_files
    effective_newly_loaded_sources = (
        bg_newly_loaded_sources if baseline_only_mode else newly_loaded_sources
    )
    effective_reused_sources = bg_reused_sources if baseline_only_mode else reused_sources
    effective_user_context = bg_context if baseline_only_mode else user_context

    prediction_reasoning_effort = (
        "medium" if prediction_reasoning_mode == "medium" else None
    )
    prediction_agent = build_agent(
        prediction_provider,
        prediction_model,
        prediction_api_key,
        reasoning_effort=prediction_reasoning_effort,
    )
    structured_profile: Optional[str] = None
    structured_profile_text_output_path: Optional[Path] = None
    structured_profile_loaded_from_path: Optional[Path] = None
    structured_profile_reused = False
    structured_profile_cache_source = "not-used"
    top_k: Optional[int] = None
    output_profile_llm = "none"

    if mode_requires_structured_profile(agent_mode):
        if profile_provider is None or profile_model is None or profile_api_key is None:
            raise ValueError(f"{prefix} requires a configured structured profile LLM.")
        profile_agent = build_agent(profile_provider, profile_model, profile_api_key)
        structured_profile_info = ensure_structured_profile(
            shared_profile_state=shared_profile_state,
            before_context=before_context,
            profile_agent=profile_agent,
            profile_llm_name=profile_llm_name,
            user_id=user_id,
            dataset_source=dataset_source,
            question_set=question_set,
            year=year,
            scope=scope,
            profile_text_output_dir=profile_text_output_dir,
            print_lock=print_lock,
            prefix=prefix,
        )
        structured_profile = structured_profile_info["structured_profile"]
        structured_profile_text_output_path = structured_profile_info[
            "structured_profile_text_output_path"
        ]
        structured_profile_loaded_from_path = structured_profile_info[
            "structured_profile_loaded_from_path"
        ]
        structured_profile_reused = bool(structured_profile_info["structured_profile_reused"])
        structured_profile_cache_source = str(
            structured_profile_info.get("structured_profile_cache_source", "unknown")
        )
        output_profile_llm = profile_llm_name

    log_with_prefix(print_lock, prefix, selection_header)
    log_with_prefix(print_lock, prefix, f"user_id: {user_id}")
    log_with_prefix(print_lock, prefix, f"dataset_source: {dataset_source}")
    log_with_prefix(print_lock, prefix, f"cutoff_year: {year}")
    log_with_prefix(print_lock, prefix, f"question_set: {question_set}")
    log_with_prefix(print_lock, prefix, f"agent_mode: {agent_mode}")
    log_with_prefix(
        print_lock,
        prefix,
        f"input_scope: {effective_scope.upper()} ({get_source_options(year)[effective_scope]})",
    )
    log_with_prefix(print_lock, prefix, f"metadata_file: {metadata_path.name}")
    log_with_prefix(print_lock, prefix, f"user_target_file: {user_target_path.name}")
    if effective_newly_loaded_sources:
        log_with_prefix(
            print_lock,
            prefix,
            f"context_loaded_now: {', '.join(effective_newly_loaded_sources)}",
        )
    if effective_reused_sources:
        log_with_prefix(
            print_lock,
            prefix,
            f"context_reused_from_memory: {', '.join(effective_reused_sources)}",
        )
    for path in effective_input_files:
        log_with_prefix(print_lock, prefix, f"input_file: {path.name}")
    if profile_provider and profile_model and mode_requires_structured_profile(agent_mode):
        log_with_prefix(print_lock, prefix, f"profile_provider: {profile_provider}")
        log_with_prefix(print_lock, prefix, f"profile_model: {profile_model}")
    log_with_prefix(print_lock, prefix, f"prediction_provider: {prediction_provider}")
    log_with_prefix(print_lock, prefix, f"prediction_model: {prediction_model}")
    log_with_prefix(
        print_lock,
        prefix,
        f"prediction_reasoning_mode: {prediction_reasoning_mode}",
    )
    log_with_prefix(print_lock, prefix, f"metadata_questions_total: {len(metadata_rows)}")
    log_with_prefix(print_lock, prefix, f"user_target_rows_total: {len(target_rows)}")
    log_with_prefix(print_lock, prefix, f"questions_to_answer: {len(selected_question_rows)}")
    log_with_prefix(print_lock, prefix, f"batch_grouping_column: {grouping_column}")
    log_with_prefix(print_lock, prefix, f"batch_target_size_b: {batch_target_size}")
    log_with_prefix(print_lock, prefix, f"original_group_count: {len(grouped_question_rows)}")
    log_with_prefix(print_lock, prefix, f"prediction_batches: {len(prediction_batches)}")
    if agent_mode in {"profile_topk_lexical", "profile_topk_semantic"}:
        log_with_prefix(
            print_lock,
            prefix,
            "top_k_retrieved_items_per_batch: dynamic (matches derived per-group sub-batch size)",
        )
    if agent_mode == "profile_topk_semantic":
        log_with_prefix(print_lock, prefix, f"before_embedding_rows_count: {len(before_embedding_rows)}")
        log_with_prefix(print_lock, prefix, f"target_embedding_rows_count: {len(target_embeddings_by_variable)}")
    if structured_profile is not None:
        log_with_prefix(print_lock, prefix, f"profile_before_rows_count: {len(before_rows)}")
    if structured_profile_text_output_path is not None:
        log_with_prefix(
            print_lock,
            prefix,
            "profile_text_file: "
            f"{format_path_for_display(structured_profile_loaded_from_path or structured_profile_text_output_path)}",
        )
        log_with_prefix(
            print_lock,
            prefix,
            f"profile_text_cache_target: {format_path_for_display(structured_profile_text_output_path)}",
        )
        log_with_prefix(print_lock, prefix, f"structured_profile_reused: {structured_profile_reused}")
        log_with_prefix(
            print_lock,
            prefix,
            f"structured_profile_cache_source: {structured_profile_cache_source}",
        )

    predictions_by_variable: Dict[str, str] = {}
    failures: List[str] = []

    for batch_index, batch_info in enumerate(prediction_batches, start=1):
        group_value = str(batch_info["group_value"])
        group_question_count = int(batch_info["group_question_count"])
        batch_index_in_group = int(batch_info["batch_index_in_group"])
        batch_count_in_group = int(batch_info["batch_count_in_group"])
        batch_question_rows = list(batch_info["question_rows"])
        dynamic_top_k = int(batch_info["dynamic_top_k"])
        batch_variable_names = [
            get_row_value(question_meta, ["Variable Name", "variable_name"])
            for question_meta in batch_question_rows
        ]
        retrieval_k = top_k
        if agent_mode in {"profile_topk_lexical", "profile_topk_semantic"}:
            retrieval_k = dynamic_top_k

        log_with_prefix(
            print_lock,
            prefix,
            f"[{batch_index}/{len(prediction_batches)}] Predicting batch "
            f"{grouping_column}={group_value} (sub-batch {batch_index_in_group}/{batch_count_in_group})",
        )
        log_with_prefix(print_lock, prefix, f"group_question_count: {group_question_count}")
        log_with_prefix(print_lock, prefix, f"batch_size: {len(batch_question_rows)}")
        log_with_prefix(print_lock, prefix, f"batch_variables: {', '.join(batch_variable_names)}")
        if agent_mode in {"profile_topk_lexical", "profile_topk_semantic"}:
            log_with_prefix(print_lock, prefix, f"dynamic_top_k_for_batch: {retrieval_k}")

        try:
            answer_specs = {
                get_row_value(question_meta, ["Variable Name", "variable_name"]): build_answer_spec(question_meta)
                for question_meta in batch_question_rows
            }
            prompt, retrieved_context_items = prepare_batch_prediction_prompt(
                agent_mode=agent_mode,
                batch_question_rows=batch_question_rows,
                answer_specs=answer_specs,
                question_set=question_set,
                year=year,
                raw_user_context=effective_user_context,
                bg_context=bg_context,
                structured_profile=structured_profile,
                before_rows=before_rows,
                before_embedding_rows=before_embedding_rows,
                profile_paragraph_embedding_rows=profile_paragraph_embedding_rows,
                target_embeddings_by_variable=target_embeddings_by_variable,
                retrieval_k=retrieval_k,
                group_label=grouping_column,
                group_value=(
                    f"{group_value} (sub-batch {batch_index_in_group}/{batch_count_in_group})"
                    if batch_count_in_group > 1
                    else group_value
                ),
            )
            if agent_mode in {"profile_topk_lexical", "profile_topk_semantic"}:
                log_with_prefix(print_lock, prefix, f"related_rows_used: {len(retrieved_context_items)}")
            batch_predictions = run_prediction_for_batch(
                agent=prediction_agent,
                batch_question_rows=batch_question_rows,
                prompt=prompt,
                answer_specs=answer_specs,
            )
            predictions_by_variable.update(batch_predictions)
            log_with_prefix(print_lock, prefix, f"batch_predictions_generated: {len(batch_predictions)}")
        except Exception as exc:
            failures.append(f"{grouping_column}={group_value}: {exc}")
            log_with_prefix(print_lock, prefix, f"prediction_failed: {exc}")

    output_path = get_prediction_output_path(
        user_id=user_id,
        dataset_source=dataset_source,
        question_set=question_set,
        year=year,
        scope=scope,
        batch_target_size=batch_target_size,
        agent_label=get_agent_label(agent_mode, top_k),
        profile_llm=output_profile_llm,
        prediction_llm=f"{prediction_provider}:{prediction_model}",
        prediction_reasoning_mode=prediction_reasoning_mode,
    )
    matched_rows = write_prediction_output(target_rows, predictions_by_variable, output_path)

    log_with_prefix(print_lock, prefix, output_header)
    log_with_prefix(print_lock, prefix, f"output_file: {output_path}")
    log_with_prefix(print_lock, prefix, f"predictions_generated: {len(predictions_by_variable)}")
    log_with_prefix(print_lock, prefix, f"rows_with_pred_answer: {matched_rows}")
    log_with_prefix(print_lock, prefix, f"prediction_failures: {len(failures)}")
    if failures:
        log_with_prefix(print_lock, prefix, "Failed questions:")
        for failure in failures:
            log_with_prefix(print_lock, prefix, failure)

    return {
        "agent_mode": agent_mode,
        "output_path": output_path,
        "predictions_generated": len(predictions_by_variable),
        "rows_with_pred_answer": matched_rows,
        "failures": failures,
        "prediction_reasoning_mode": prediction_reasoning_mode,
    }


def main() -> None:
    dataset_source = choose_dataset_source()
    dataset_config = get_dataset_source_config(dataset_source)
    year = choose_cutoff_year()
    user_roots = dataset_config["user_roots"]
    if not isinstance(user_roots, dict):
        raise ValueError(f"Dataset source config is missing user_roots: {dataset_source}")
    users_root_dir = user_roots[year]
    user_id, user_dir = choose_user_dir(users_root_dir, year, dataset_source)
    question_set = choose_question_set()
    metadata_path = get_metadata_path_for_source(dataset_source, year, question_set)
    user_target_path = get_user_target_path(user_dir, question_set, year)
    selected_agent_mode = choose_agent_mode()
    agent_mode_phases = get_agent_mode_phases(selected_agent_mode)
    selected_agent_modes = [mode for phase in agent_mode_phases for mode in phase]
    allow_bg_only = all(not mode_requires_structured_profile(mode) for mode in selected_agent_modes)
    scope = choose_input_scope(year, allow_bg_only=allow_bg_only)
    batch_target_size = choose_batch_target_size()

    context_cache = UserContextCache(user_dir)
    selected_files, bg_files, before_files = get_selected_file_groups(user_dir, scope, year)
    user_context, input_files, newly_loaded_sources, reused_sources = context_cache.build_context_from_files(
        selected_files
    )
    bg_context, bg_input_files, bg_newly_loaded_sources, bg_reused_sources = (
        context_cache.build_context_from_files(bg_files)
    )
    before_context = ""
    before_rows: List[Dict[str, str]] = []
    before_embedding_rows: List[Dict[str, str]] = []
    profile_paragraph_embedding_rows: List[Dict[str, str]] = []
    if before_files:
        before_context, _, _, _ = context_cache.build_context_from_files(before_files)
        before_rows = collect_answered_rows(before_files)
        if any(mode_requires_semantic_embeddings(mode) for mode in selected_agent_modes):
            if semantic_precompute_available(question_set, scope):
                before_embedding_rows = load_answered_embedding_rows(
                    before_files,
                    dataset_source=dataset_source,
                    year=year,
                    question_set=question_set,
                    scope=scope,
                )

    profile_provider: Optional[str] = None
    profile_model: Optional[str] = None
    profile_api_key: Optional[str] = None
    profile_llm_name = "none"
    if any(mode_requires_structured_profile(mode) for mode in selected_agent_modes):
        print("\nChoose LLM for structured profile generation:")
        profile_provider = choose_llm_provider()
        profile_model = choose_model(profile_provider)
        profile_api_key = prompt_for_api_key(profile_provider, resolve_api_key(profile_provider))
        profile_llm_name = f"{profile_provider}:{profile_model}"

    print("\nChoose LLM for answer prediction:")
    provider = choose_llm_provider()
    model = choose_model(provider)
    prediction_reasoning_mode = choose_prediction_reasoning_mode()
    api_key = prompt_for_api_key(provider, resolve_api_key(provider))
    target_embeddings_by_variable: Dict[str, List[float]] = {}

    metadata_rows = read_csv(metadata_path)
    if not metadata_rows:
        raise ValueError(f"No question rows found in metadata file: {metadata_path}")

    target_rows = read_csv(user_target_path)
    if not target_rows:
        raise ValueError(f"No target rows found in user target file: {user_target_path}")
    if (
        any(mode_requires_semantic_embeddings(mode) for mode in selected_agent_modes)
        and semantic_precompute_available(question_set, scope)
    ):
        target_embeddings_by_variable = load_target_embeddings_by_variable(
            metadata_path,
            target_rows,
            dataset_source=dataset_source,
            question_set=question_set,
            year=year,
        )

    selected_question_rows = select_questions_for_user(metadata_rows, target_rows)
    if not selected_question_rows:
        raise ValueError(
            "No overlapping variable_name values were found between the selected "
            "metadata file and the selected user target file."
        )
    target_rows_by_variable = get_target_rows_by_variable(target_rows)
    grouped_question_rows = group_questions_for_batching(
        selected_question_rows,
        target_rows_by_variable,
        question_set,
    )
    prediction_batches = split_question_groups_into_prediction_batches(
        grouped_question_rows,
        batch_target_size=batch_target_size,
    )
    grouping_column = get_batch_group_column(question_set)

    print("\n--- Run Plan ---")
    if selected_agent_mode is None:
        for phase_index, phase_modes in enumerate(agent_mode_phases, start=1):
            print(
                f"phase_{phase_index}: "
                f"{', '.join(mode_name(agent_mode) for agent_mode in phase_modes)}"
            )
    else:
        print(f"selected_agent_mode: {mode_name(selected_agent_mode)}")
    print(
        "prediction_reasoning_mode: "
        f"{prediction_reasoning_mode} "
        f"({PREDICTION_REASONING_MODES[prediction_reasoning_mode]})"
    )

    shared_profile_state: Dict[str, object] = {"lock": threading.Lock()}
    print_lock = threading.Lock()
    run_results: List[Dict[str, object]] = []

    for phase_index, phase_modes in enumerate(agent_mode_phases, start=1):
        print(
            f"\n=== Phase {phase_index}/{len(agent_mode_phases)} === "
            f"{', '.join(mode_name(agent_mode) for agent_mode in phase_modes)}"
        )
        if len(phase_modes) == 1:
            run_results.append(
                run_agent_mode_execution(
                    agent_mode=phase_modes[0],
                    user_id=user_id,
                    dataset_source=dataset_source,
                    question_set=question_set,
                    year=year,
                    scope=scope,
                    batch_target_size=batch_target_size,
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
                    profile_paragraph_embedding_rows=profile_paragraph_embedding_rows,
                    target_embeddings_by_variable=target_embeddings_by_variable,
                    metadata_rows=metadata_rows,
                    target_rows=target_rows,
                    selected_question_rows=selected_question_rows,
                    grouped_question_rows=grouped_question_rows,
                    prediction_batches=prediction_batches,
                    grouping_column=grouping_column,
                    prediction_provider=provider,
                    prediction_model=model,
                    prediction_reasoning_mode=prediction_reasoning_mode,
                    prediction_api_key=api_key,
                    profile_provider=profile_provider,
                    profile_model=profile_model,
                    profile_api_key=profile_api_key,
                    profile_llm_name=profile_llm_name,
                    profile_text_output_dir=None,
                    shared_profile_state=shared_profile_state,
                    print_lock=print_lock,
                )
            )
            continue

        with ThreadPoolExecutor(max_workers=len(phase_modes)) as executor:
            future_to_mode = {
                executor.submit(
                    run_agent_mode_execution,
                    agent_mode,
                    user_id,
                    dataset_source,
                    question_set,
                    year,
                    scope,
                    batch_target_size,
                    metadata_path,
                    user_target_path,
                    input_files,
                    bg_input_files,
                    newly_loaded_sources,
                    reused_sources,
                    bg_newly_loaded_sources,
                    bg_reused_sources,
                    user_context,
                    bg_context,
                    before_context,
                    before_rows,
                    before_embedding_rows,
                    profile_paragraph_embedding_rows,
                    target_embeddings_by_variable,
                    metadata_rows,
                    target_rows,
                    selected_question_rows,
                    grouped_question_rows,
                    prediction_batches,
                    grouping_column,
                    provider,
                    model,
                    prediction_reasoning_mode,
                    api_key,
                    profile_provider,
                    profile_model,
                    profile_api_key,
                    profile_llm_name,
                    None,
                    shared_profile_state,
                    print_lock,
                ): agent_mode
                for agent_mode in phase_modes
            }
            for future in as_completed(future_to_mode):
                agent_mode = future_to_mode[future]
                try:
                    run_results.append(future.result())
                except Exception as exc:
                    print(f"[{mode_name(agent_mode)}] mode_failed: {exc}")
                    run_results.append(
                        {
                            "agent_mode": agent_mode,
                            "output_path": None,
                            "predictions_generated": 0,
                            "rows_with_pred_answer": 0,
                            "failures": [str(exc)],
                        }
                    )

    print("\n--- Run Summary ---")
    for result in run_results:
        output_path = result.get("output_path")
        output_text = str(output_path) if output_path is not None else "<not written>"
        print(
            f"{mode_name(str(result['agent_mode']))}: "
            f"output={output_text}, "
            f"predictions_generated={result['predictions_generated']}, "
            f"rows_with_pred_answer={result['rows_with_pred_answer']}, "
            f"failures={len(result['failures'])}"
        )


if __name__ == "__main__":
    main()
