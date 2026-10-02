from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EXTERNAL_EMBEDDING_OUTPUT_DIR = Path(
    os.environ.get("EMBEDDING_CACHE_DIR", "embedding_cache")
)
METADATA_DIRS = {
    "standalone": PROJECT_ROOT / "standalone",
    "standalone_categories_only": PROJECT_ROOT / "standalone_categories_only",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: Dict[str, List[float]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True)


def get_row_value(row: Dict[str, str], candidates: Sequence[str]) -> str:
    for key in candidates:
        if key in row:
            value = row.get(key, "")
            if value is not None:
                return str(value).strip()

    lowered = {key.lower(): key for key in row.keys()}
    for key in candidates:
        found = lowered.get(key.lower())
        if found:
            value = row.get(found, "")
            if value is not None:
                return str(value).strip()
    return ""


def get_embedding_output_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_embeddings.json")


def get_external_embedding_output_path(csv_path: Path) -> Path:
    return EXTERNAL_EMBEDDING_OUTPUT_DIR / f"{csv_path.stem}_embeddings.json"


def build_question_embedding_text(row: Dict[str, str]) -> str:
    parts = [
        ("variable_name", get_row_value(row, ["variable_name", "Variable Name"])),
        ("question_text", get_row_value(row, ["variable_label", "Variable Label"])),
        ("representation_type", get_row_value(row, ["representation_type", "Representation Type"])),
        ("categories", get_row_value(row, ["categories", "Categories"])),
        ("question_type", get_row_value(row, ["question type", "Question Type"])),
        ("variable_type", get_row_value(row, ["variable_type", "Variable Type"])),
        ("year", get_row_value(row, ["year", "Year"])),
    ]
    return "\n".join(f"{label}: {value}" for label, value in parts if value)


def build_embedding_lookup(
    client: OpenAI,
    source_rows: List[Dict[str, str]],
    model: str,
    batch_size: int,
) -> Dict[str, List[float]]:
    if not source_rows:
        return {}

    texts = [build_question_embedding_text(row) for row in source_rows]
    embedding_lookup: Dict[str, List[float]] = {}
    for start in range(0, len(source_rows), batch_size):
        batch_rows = source_rows[start:start + batch_size]
        batch_texts = texts[start:start + batch_size]
        response = client.embeddings.create(model=model, input=batch_texts)
        for row, item in zip(batch_rows, response.data):
            variable_name = get_row_value(row, ["variable_name", "Variable Name"])
            if not variable_name:
                raise ValueError("Every row must include variable_name or Variable Name.")
            embedding = [float(value) for value in item.embedding]
            existing = embedding_lookup.get(variable_name)
            if existing is not None and existing != embedding:
                raise ValueError(
                    f"Duplicate variable_name with different embeddings: {variable_name}"
                )
            embedding_lookup[variable_name] = embedding
    return embedding_lookup


def get_metadata_csv_paths(directory_scope: str, file_kind: str) -> List[Path]:
    selected_dirs: List[Path]
    if directory_scope == "all":
        selected_dirs = list(METADATA_DIRS.values())
    else:
        selected_dirs = [METADATA_DIRS[directory_scope]]

    paths: List[Path] = []
    for directory in selected_dirs:
        if file_kind in {"all", "before"}:
            paths.extend(sorted(directory.glob("*_before.csv")))
        if file_kind in {"all", "target"}:
            paths.extend(sorted(directory.glob("*_target.csv")))
    return paths


def normalize_csv_path(csv_path: str) -> Path:
    path = Path(csv_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def get_explicit_csv_paths(csv_paths: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    seen = set()
    for csv_path in csv_paths:
        path = normalize_csv_path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"CSV path is not a file: {path}")
        if path.suffix.lower() != ".csv":
            raise ValueError(f"CSV path must end with .csv: {path}")
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def prepare_question_rows(path: Path) -> List[Dict[str, str]]:
    return read_csv(path)


def process_csv(
    client: OpenAI,
    source_path: Path,
    rows: List[Dict[str, str]],
    model: str,
    batch_size: int,
    overwrite: bool,
) -> bool:
    output_path = get_embedding_output_path(source_path)
    external_output_path = get_external_embedding_output_path(source_path)
    if output_path.exists() and external_output_path.exists() and not overwrite:
        print(f"skip_existing: {output_path}")
        return False
    if not rows:
        print(f"skip_empty: {source_path}")
        return False

    print(f"embedding_source: {source_path}")
    print(f"embedding_rows: {len(rows)}")
    embedding_lookup = build_embedding_lookup(client, rows, model, batch_size)
    write_json(output_path, embedding_lookup)
    external_output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(external_output_path, embedding_lookup)
    print(f"embedding_output: {output_path}")
    print(f"embedding_output_external: {external_output_path}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute question embeddings for metadata *_before.csv and *_target.csv files "
            "in standalone/ and standalone_categories_only/, or for explicitly provided CSV files."
        )
    )
    parser.add_argument(
        "--csv",
        action="append",
        default=[],
        help=(
            "Explicit CSV path to embed. Repeat --csv for multiple files. "
            "Relative paths are resolved from the repository root."
        ),
    )
    parser.add_argument(
        "--directory-scope",
        choices=["all", "standalone", "standalone_categories_only"],
        default="all",
        help="Which metadata directories to scan when --csv is not provided.",
    )
    parser.add_argument(
        "--file-kind",
        choices=["all", "before", "target"],
        default="all",
        help="Whether to embed *_before.csv, *_target.csv, or both when --csv is not provided.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model id. Default: text-embedding-3-small",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Embedding batch size for API calls.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_embeddings.json files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required to precompute embeddings.")
    client = OpenAI(api_key=api_key)

    processed_count = 0

    if args.csv:
        paths = get_explicit_csv_paths(args.csv)
    else:
        paths = get_metadata_csv_paths(args.directory_scope, args.file_kind)
    if not paths:
        raise FileNotFoundError(
            f"No matching CSV files found in scope={args.directory_scope} for file_kind={args.file_kind}."
        )

    for path in paths:
        if process_csv(
            client=client,
            source_path=path,
            rows=prepare_question_rows(path),
            model=args.model,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        ):
            processed_count += 1

    print(f"embedding_files_written: {processed_count}")


if __name__ == "__main__":
    main()
