from pathlib import Path
import pandas as pd
import numpy as np
import re
import json

# =========================================================
# 1. SET PATHS
# =========================================================
source_dir = Path("datasets_categories_only/nomem_encr_categories_only_cutoff_2023")
output_dir = Path("sample_500_2023_2024_categories_only")
output_dir.mkdir(parents=True, exist_ok=True)

json_path = Path("datasets_categories_only/nomem_encr_categories_only_cutoff_2023/all_user_ids.json")

target_n = 500
min_stratum_size = 5
random_state = 42

# Use ["all"] to run every sample, or list specific keys:
#   sw_to_cs, sw_to_sw, cssw_to_sw, cs_to_sw
samples_to_run = ["sw_to_cs", "cs_to_sw"]


# =========================================================
# 2. HELPERS
# =========================================================
def clean_text(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    return x if x != "" else None


def read_csv_auto(file_path):
    for sep in [",", None, ";", "\t", "|"]:
        try:
            if sep is None:
                df = pd.read_csv(file_path, sep=None, engine="python", dtype=str)
            else:
                df = pd.read_csv(file_path, sep=sep, dtype=str)
            if df.shape[1] > 0:
                return df
        except Exception:
            continue
    raise ValueError(f"Could not read {file_path}")


def standardize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def coerce_bg_to_long(df):
    df = standardize_columns(df)

    if "variable_name" in df.columns and "answer" in df.columns:
        out = df[["variable_name", "answer"]].copy()
        out["variable_name"] = out["variable_name"].astype(str).str.strip().str.lower()
        return out

    q_candidates = ["question_id", "questionid", "variable", "var", "key", "name"]
    a_candidates = ["answer", "response", "value", "label", "text"]

    q_col = None
    a_col = None

    for c in q_candidates:
        if c in df.columns:
            q_col = c
            break

    for c in a_candidates:
        if c in df.columns:
            a_col = c
            break

    if q_col is not None and a_col is not None:
        out = df[[q_col, a_col]].copy()
        out.columns = ["variable_name", "answer"]
        out["variable_name"] = out["variable_name"].astype(str).str.strip().str.lower()
        return out

    if df.shape[0] == 1:
        out = df.T.reset_index()
        out.columns = ["variable_name", "answer"]
        out["variable_name"] = out["variable_name"].astype(str).str.strip().str.lower()
        return out

    out = df.melt(var_name="variable_name", value_name="answer")
    out["variable_name"] = out["variable_name"].astype(str).str.strip().str.lower()
    return out


def get_bg_answer(bg_long, variable_names):
    for var in variable_names:
        var = var.lower()
        match = bg_long.loc[
            bg_long["variable_name"].astype(str).str.strip().str.lower() == var,
            "answer"
        ]
        if not match.empty:
            val = clean_text(match.iloc[0])
            if val is not None:
                return val
    return None


def parse_numeric(x):
    if x is None:
        return np.nan
    x = str(x).replace(",", ".").strip()
    m = re.search(r"-?\d+(\.\d+)?", x)
    if m:
        try:
            return float(m.group())
        except Exception:
            return np.nan
    return np.nan


def normalize_gender_3cat(x):
    if x is None:
        return "Other"

    s = str(x).strip().lower()

    if s in ["male", "man", "m", "mannelijk", "boy"]:
        return "Male"
    if s in ["female", "woman", "f", "vrouw", "vrouwelijk", "girl"]:
        return "Female"

    return "Other"


def normalize_age_group(age, age_cat):
    if not pd.isna(age):
        if age < 35:
            return "18-34"
        elif age < 50:
            return "35-49"
        elif age < 65:
            return "50-64"
        else:
            return "65+"

    if age_cat is None:
        return "Unknown"

    s = str(age_cat).lower()

    if "25 - 34" in s or "18" in s or "24" in s or "34" in s:
        return "18-34"
    elif "35 - 44" in s or "35" in s or "44" in s:
        return "35-49"
    elif "45 - 54" in s or "55 - 64" in s or "50" in s or "64" in s:
        return "50-64"
    elif "65" in s or "older" in s or "ouder" in s:
        return "65+"

    return "Unknown"


def parse_children_count(x):
    if x is None:
        return np.nan

    s = str(x).strip().lower()

    mapping = {
        "none": 0,
        "geen": 0,
        "geen kinderen": 0,
        "no children": 0,
        "one child": 1,
        "two children": 2,
        "three children": 3,
        "four children": 4,
        "five children": 5
    }
    if s in mapping:
        return mapping[s]

    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))

    return np.nan


def normalize_partner(x):
    if x is None:
        return "Unknown"

    s = str(x).strip().lower()
    if s in ["yes", "ja", "1", "true"]:
        return "Yes"
    if s in ["no", "nee", "0", "false"]:
        return "No"

    return "Unknown"


def derive_household_stage(partner, children_count, woonvorm):
    w = str(woonvorm).lower() if woonvorm is not None else ""

    if not pd.isna(children_count) and children_count > 0:
        return "Family_with_children"

    if partner == "Yes":
        return "Couple_no_children"

    if "single" in w or "alleen" in w:
        return "Single_no_children"

    if partner == "No":
        return "Single_no_children"

    if woonvorm is not None:
        return "Other_multi_person"

    return "Unknown"


def count_answered_questions(df):
    """
    Count answered questions in one file.
    If an answer/response/value column exists, count non-missing there.
    Otherwise count all non-missing cells.
    """
    df = standardize_columns(df)

    for col in ["answer", "response", "value"]:
        if col in df.columns:
            return int(df[col].notna().sum())

    return int(df.notna().sum().sum())


def proportional_allocation(counts, target_n):
    alloc = counts / counts.sum() * target_n
    alloc_floor = np.floor(alloc).astype(int)
    remainder = alloc - alloc_floor

    leftover = target_n - alloc_floor.sum()
    alloc_final = alloc_floor.copy()

    if leftover > 0:
        add_order = remainder.sort_values(ascending=False).index[:leftover]
        alloc_final.loc[add_order] += 1

    alloc_final = pd.Series(
        np.minimum(alloc_final.values, counts.values),
        index=counts.index
    )

    current_total = int(alloc_final.sum())
    if current_total < target_n:
        deficit = target_n - current_total
        room = counts - alloc_final
        for grp in room.sort_values(ascending=False).index:
            if deficit == 0:
                break
            if room.loc[grp] > 0:
                add = min(int(room.loc[grp]), deficit)
                alloc_final.loc[grp] += add
                deficit -= add

    return alloc_final.astype(int)


def add_baseline_stratum(meta):
    meta = meta.copy()

    meta["stratum_primary"] = (
        meta["age_group"].fillna("Unknown") + " | " +
        meta["gender"].fillna("Other") + " | " +
        meta["household_stage"].fillna("Unknown")
    )

    meta["stratum_fallback"] = (
        meta["age_group"].fillna("Unknown") + " | " +
        meta["gender"].fillna("Other")
    )

    counts = meta["stratum_primary"].value_counts()

    meta["stratum_final"] = meta["stratum_primary"]
    small_mask = meta["stratum_primary"].map(counts) < min_stratum_size
    meta.loc[small_mask, "stratum_final"] = meta.loc[small_mask, "stratum_fallback"]

    return meta


def safe_zscore(s):
    s = pd.Series(s).astype(float)
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / sd


def sample_with_prev_target_matching(meta, stratum_col, prev_col, target_col, target_n, random_state=42):
    """
    1) Proportional allocation across strata
    2) Within each stratum, prioritize respondents high on BOTH:
       - previous answered count
       - target answered count

    Ranking:
      1. z(previous) + z(target)
      2. previous * target
      3. target descending
      4. previous descending
    """
    counts = meta[stratum_col].value_counts().sort_index()
    alloc_final = proportional_allocation(counts, target_n)

    sampled_parts = []

    for grp, n_take in alloc_final.items():
        subset = meta[meta[stratum_col] == grp].copy()

        subset = subset.sample(frac=1, random_state=random_state).reset_index(drop=True)

        subset["prev_z"] = safe_zscore(subset[prev_col])
        subset["target_z"] = safe_zscore(subset[target_col])
        subset["match_score"] = subset["prev_z"] + subset["target_z"]
        subset["pair_score"] = subset[prev_col].astype(float) * subset[target_col].astype(float)

        subset = subset.sort_values(
            by=["match_score", "pair_score", target_col, prev_col],
            ascending=[False, False, False, False]
        )

        sampled_grp = subset.head(int(n_take))
        sampled_parts.append(sampled_grp)

    sampled = pd.concat(sampled_parts, ignore_index=True)

    allocation_table = pd.DataFrame({
        "Stratum": counts.index,
        "Available": counts.values,
        "Allocated": alloc_final.reindex(counts.index).values
    })

    return sampled, allocation_table


def population_vs_sample_table(pop_df, sample_df, group_cols):
    pop_counts = pop_df.groupby(group_cols).size().reset_index(name="Population_N")
    samp_counts = sample_df.groupby(group_cols).size().reset_index(name="Sample_N")

    out = pop_counts.merge(samp_counts, on=group_cols, how="left").fillna(0)
    out["Sample_N"] = out["Sample_N"].astype(int)
    out["Expected_Sample_N"] = out["Population_N"] / out["Population_N"].sum() * len(sample_df)
    out["Abs_Diff"] = (out["Sample_N"] - out["Expected_Sample_N"]).abs()

    mad = out["Abs_Diff"].mean()
    maxad = out["Abs_Diff"].max()
    return out, mad, maxad


def folder_id_from_name(name):
    m = re.search(r"(\d+)", str(name))
    return m.group(1) if m else str(name)


def find_bg_file(csv_files):
    for f in csv_files:
        if "_bg_" in f.name.lower() or "bg_" in f.name.lower():
            return f
    for f in csv_files:
        if "bg" in f.name.lower():
            return f
    return None


def classify_non_bg_file(file_name):
    """
    Example 800001:
      800001_core_cutoff_2023_before.csv
      800001_core_cutoff_2023_latest_before.csv
      800001_core_cutoff_2023_target.csv
      800001_single_cutoff_2023_before.csv
      800001_single_cutoff_2023_target.csv
      800001_single_cutoff_2023_2024_target.csv
    """
    s = file_name.lower()

    is_core = "core" in s
    is_single = "single" in s
    is_latest_before = "latest_before" in s
    is_2023_2024_target = "2023_2024_target" in s
    is_before = "before" in s
    is_target = "target" in s

    if is_core and is_latest_before:
        return "core_latest_before"
    if is_core and is_before:
        return "core_before"
    if is_single and is_2023_2024_target:
        return "single_2023_2024_target"
    if is_core and is_target:
        return "core_target"
    if is_single and is_before:
        return "single_before"
    if is_single and is_target:
        return "single_target"

    return "other"


# =========================================================
# 3. LOAD ELIGIBLE IDS FROM JSON
# =========================================================
with open(json_path, "r", encoding="utf-8") as f:
    eligible_json = json.load(f)

if isinstance(eligible_json, dict):
    eligible_ids_raw = eligible_json["user_ids"]
elif isinstance(eligible_json, list):
    eligible_ids_raw = eligible_json
else:
    raise TypeError(f"Expected JSON list or dict, got {type(eligible_json).__name__}")

eligible_ids = set(map(str, eligible_ids_raw))
print(f"Eligible IDs from JSON: {len(eligible_ids)}")


# =========================================================
# 4. BUILD RESPONDENT-LEVEL METADATA
#    KEEP ONLY IDS IN JSON
# =========================================================
rows = []

folders = sorted([p for p in source_dir.iterdir() if p.is_dir()])
print(f"Found total folders: {len(folders)}")

for folder in folders:
    try:
        respondent_id = folder_id_from_name(folder.name)

        if respondent_id not in eligible_ids:
            continue

        csv_files = sorted(folder.glob("*.csv"))
        if len(csv_files) == 0:
            continue

        bg_file = find_bg_file(csv_files)

        gender = None
        age = np.nan
        age_cat = None
        partner = None
        aantalki = None
        woonvorm = None
        aantalhh = None
        sted = None

        if bg_file is not None:
            bg_df = read_csv_auto(bg_file)
            bg_long = coerce_bg_to_long(bg_df)

            gender = get_bg_answer(bg_long, ["geslacht", "gender", "d1"])
            age_raw = get_bg_answer(bg_long, ["age", "leeftijd", "d2"])
            age = parse_numeric(age_raw)
            age_cat = get_bg_answer(bg_long, ["age_cat", "age_category", "leeftijd_cat", "d2_cat"])
            partner = get_bg_answer(bg_long, ["partner", "partner_yesno", "heeft_partner"])
            aantalki = get_bg_answer(bg_long, ["aantalki", "children", "num_children"])
            woonvorm = get_bg_answer(bg_long, ["woonvorm", "living_situation"])
            aantalhh = get_bg_answer(bg_long, ["aantalhh", "household_size"])
            sted = get_bg_answer(bg_long, ["sted", "urbanicity"])

        core_before_answered = 0
        core_latest_before_answered = 0
        core_target_answered = 0
        single_before_answered = 0
        single_target_answered = 0
        single_2023_2024_target_answered = 0
        other_answered = 0

        core_target_exists = False
        single_target_exists = False
        single_2023_2024_target_exists = False

        for f in csv_files:
            if bg_file is not None and f == bg_file:
                continue

            try:
                df = read_csv_auto(f)
                n_answered = count_answered_questions(df)
                file_type = classify_non_bg_file(f.name)

                if file_type == "core_before":
                    core_before_answered += n_answered
                elif file_type == "core_latest_before":
                    core_latest_before_answered += n_answered
                elif file_type == "core_target":
                    core_target_answered += n_answered
                    core_target_exists = True
                elif file_type == "single_before":
                    single_before_answered += n_answered
                elif file_type == "single_target":
                    single_target_answered += n_answered
                    single_target_exists = True
                elif file_type == "single_2023_2024_target":
                    single_2023_2024_target_answered += n_answered
                    single_2023_2024_target_exists = True
                else:
                    other_answered += n_answered

            except Exception as e:
                print(f"Could not process {f}: {e}")

        gender_clean = normalize_gender_3cat(gender)
        age_group = normalize_age_group(age, age_cat)
        partner_clean = normalize_partner(partner)
        children_count = parse_children_count(aantalki)
        household_stage = derive_household_stage(partner_clean, children_count, woonvorm)

        total_answered = (
            core_before_answered +
            core_latest_before_answered +
            core_target_answered +
            single_before_answered +
            single_target_answered +
            single_2023_2024_target_answered +
            other_answered
        )

        rows.append({
            "respondent_id": respondent_id,
            "folder_name": folder.name,
            "folder_path": str(folder),
            "bg_file_name": bg_file.name if bg_file is not None else None,
            "gender": gender_clean,
            "age": age,
            "age_group": age_group,
            "partner": partner_clean,
            "children_count": children_count,
            "household_stage": household_stage,
            "woonvorm": woonvorm,
            "aantalhh": aantalhh,
            "sted": sted,
            "core_before_answered": core_before_answered,
            "core_latest_before_answered": core_latest_before_answered,
            "core_target_answered": core_target_answered,
            "single_before_answered": single_before_answered,
            "single_target_answered": single_target_answered,
            "single_2023_2024_target_answered": single_2023_2024_target_answered,
            "other_answered": other_answered,
            "total_answered": total_answered,
            "core_target_exists": core_target_exists,
            "single_target_exists": single_target_exists,
            "single_2023_2024_target_exists": single_2023_2024_target_exists
        })

    except Exception as e:
        print(f"Skipping folder {folder.name}: {e}")

meta = pd.DataFrame(rows)

if meta.empty:
    raise ValueError("No eligible respondents found after JSON filtering.")

print("\nEligible respondents after JSON filtering:", len(meta))
print(meta.head())


# =========================================================
# 5. BASELINE STRATIFICATION
# =========================================================
meta = add_baseline_stratum(meta)


# =========================================================
# 6. BUILD SELECTED SAMPLE POOLS
#    LEFT  = before count(s)
#    RIGHT = target count
# =========================================================
valid_sample_keys = {"sw_to_cs", "sw_to_sw", "cssw_to_sw", "cs_to_sw"}
selected_samples = set(samples_to_run)

if "all" in selected_samples:
    selected_samples = valid_sample_keys.copy()

unknown_samples = selected_samples - valid_sample_keys
if unknown_samples:
    raise ValueError(
        "Unknown sample key(s): "
        f"{sorted(unknown_samples)}. Valid keys: {sorted(valid_sample_keys)} or ['all']."
    )

print("\nSamples selected:", ", ".join(sorted(selected_samples)))

sample_outputs = {}


def add_sample_output(
    key,
    sample_number,
    name,
    population_df,
    sample_df,
    allocation_df,
    compare_df,
    mad,
    maxad,
    exists_col,
    ranked_columns
):
    sample_outputs[key] = {
        "sample_number": sample_number,
        "name": name,
        "population_df": population_df,
        "sample_df": sample_df,
        "allocation_df": allocation_df,
        "compare_df": compare_df,
        "mad": mad,
        "maxad": maxad,
        "exists_col": exists_col,
        "ranked_columns": ranked_columns
    }


if "sw_to_cs" in selected_samples:
    # Sample 1: SW -> CS
    # left  = single_before_answered
    # right = core_target_answered
    # require core target file to exist
    meta_sw_to_cs = meta[meta["core_target_exists"]].copy()
    meta_sw_to_cs["prev_count"] = meta_sw_to_cs["single_before_answered"]
    meta_sw_to_cs["target_count"] = meta_sw_to_cs["core_target_answered"]

    sample_sw_to_cs, alloc_sw_to_cs = sample_with_prev_target_matching(
        meta=meta_sw_to_cs,
        stratum_col="stratum_final",
        prev_col="prev_count",
        target_col="target_count",
        target_n=min(target_n, len(meta_sw_to_cs)),
        random_state=random_state
    )

    compare_sw_to_cs, mad_sw_to_cs, maxad_sw_to_cs = population_vs_sample_table(
        meta_sw_to_cs,
        sample_sw_to_cs,
        ["age_group", "gender", "household_stage"]
    )

    add_sample_output(
        key="sw_to_cs",
        sample_number=1,
        name="SW -> CS",
        population_df=meta_sw_to_cs,
        sample_df=sample_sw_to_cs,
        allocation_df=alloc_sw_to_cs,
        compare_df=compare_sw_to_cs,
        mad=mad_sw_to_cs,
        maxad=maxad_sw_to_cs,
        exists_col="core_target_exists",
        ranked_columns=[
            "respondent_id", "folder_name", "stratum_final",
            "single_before_answered", "core_target_answered",
            "prev_count", "target_count",
            "core_target_exists", "has_target_file", "target_answered_positive",
            "gender", "age_group", "household_stage"
        ]
    )

if "sw_to_sw" in selected_samples:
    # Sample 2: SW -> SW
    # left  = single_before_answered
    # right = single_target_answered
    # require single target file to exist
    meta_sw_to_sw = meta[meta["single_target_exists"]].copy()
    meta_sw_to_sw["prev_count"] = meta_sw_to_sw["single_before_answered"]
    meta_sw_to_sw["target_count"] = meta_sw_to_sw["single_target_answered"]

    sample_sw_to_sw, alloc_sw_to_sw = sample_with_prev_target_matching(
        meta=meta_sw_to_sw,
        stratum_col="stratum_final",
        prev_col="prev_count",
        target_col="target_count",
        target_n=min(target_n, len(meta_sw_to_sw)),
        random_state=random_state
    )

    compare_sw_to_sw, mad_sw_to_sw, maxad_sw_to_sw = population_vs_sample_table(
        meta_sw_to_sw,
        sample_sw_to_sw,
        ["age_group", "gender", "household_stage"]
    )

    add_sample_output(
        key="sw_to_sw",
        sample_number=2,
        name="SW -> SW",
        population_df=meta_sw_to_sw,
        sample_df=sample_sw_to_sw,
        allocation_df=alloc_sw_to_sw,
        compare_df=compare_sw_to_sw,
        mad=mad_sw_to_sw,
        maxad=maxad_sw_to_sw,
        exists_col="single_target_exists",
        ranked_columns=[
            "respondent_id", "folder_name", "stratum_final",
            "single_before_answered", "single_target_answered",
            "prev_count", "target_count",
            "single_target_exists", "has_target_file", "target_answered_positive",
            "gender", "age_group", "household_stage"
        ]
    )

if "cssw_to_sw" in selected_samples:
    # Sample 3: CS+SW -> SW
    # left  = core_before_answered + single_before_answered
    # right = single_target_answered
    # require single target file to exist
    meta_cssw_to_sw = meta[meta["single_target_exists"]].copy()
    meta_cssw_to_sw["prev_count"] = (
        meta_cssw_to_sw["core_before_answered"] +
        meta_cssw_to_sw["single_before_answered"]
    )
    meta_cssw_to_sw["target_count"] = meta_cssw_to_sw["single_target_answered"]

    sample_cssw_to_sw, alloc_cssw_to_sw = sample_with_prev_target_matching(
        meta=meta_cssw_to_sw,
        stratum_col="stratum_final",
        prev_col="prev_count",
        target_col="target_count",
        target_n=min(target_n, len(meta_cssw_to_sw)),
        random_state=random_state
    )

    compare_cssw_to_sw, mad_cssw_to_sw, maxad_cssw_to_sw = population_vs_sample_table(
        meta_cssw_to_sw,
        sample_cssw_to_sw,
        ["age_group", "gender", "household_stage"]
    )

    add_sample_output(
        key="cssw_to_sw",
        sample_number=3,
        name="CS+SW -> SW",
        population_df=meta_cssw_to_sw,
        sample_df=sample_cssw_to_sw,
        allocation_df=alloc_cssw_to_sw,
        compare_df=compare_cssw_to_sw,
        mad=mad_cssw_to_sw,
        maxad=maxad_cssw_to_sw,
        exists_col="single_target_exists",
        ranked_columns=[
            "respondent_id", "folder_name", "stratum_final",
            "core_before_answered", "single_before_answered", "single_target_answered",
            "prev_count", "target_count",
            "single_target_exists", "has_target_file", "target_answered_positive",
            "gender", "age_group", "household_stage"
        ]
    )

if "cs_to_sw" in selected_samples:
    # Sample 4: CS -> SW
    # left  = core_latest_before_answered
    # right = single_2023_2024_target_answered
    # require single 2023-2024 target file to exist
    meta_cs_to_sw = meta[meta["single_2023_2024_target_exists"]].copy()
    meta_cs_to_sw["prev_count"] = meta_cs_to_sw["core_latest_before_answered"]
    meta_cs_to_sw["target_count"] = meta_cs_to_sw["single_2023_2024_target_answered"]

    sample_cs_to_sw, alloc_cs_to_sw = sample_with_prev_target_matching(
        meta=meta_cs_to_sw,
        stratum_col="stratum_final",
        prev_col="prev_count",
        target_col="target_count",
        target_n=min(target_n, len(meta_cs_to_sw)),
        random_state=random_state
    )

    compare_cs_to_sw, mad_cs_to_sw, maxad_cs_to_sw = population_vs_sample_table(
        meta_cs_to_sw,
        sample_cs_to_sw,
        ["age_group", "gender", "household_stage"]
    )

    add_sample_output(
        key="cs_to_sw",
        sample_number=4,
        name="CS -> SW",
        population_df=meta_cs_to_sw,
        sample_df=sample_cs_to_sw,
        allocation_df=alloc_cs_to_sw,
        compare_df=compare_cs_to_sw,
        mad=mad_cs_to_sw,
        maxad=maxad_cs_to_sw,
        exists_col="single_2023_2024_target_exists",
        ranked_columns=[
            "respondent_id", "folder_name", "stratum_final",
            "core_latest_before_answered", "single_2023_2024_target_answered",
            "prev_count", "target_count",
            "single_2023_2024_target_exists", "has_target_file", "target_answered_positive",
            "gender", "age_group", "household_stage"
        ]
    )


for output in sorted(sample_outputs.values(), key=lambda x: x["sample_number"]):
    print(f"\n================ SAMPLE {output['sample_number']}: {output['name']} ================")
    print("Population size:", len(output["population_df"]))
    print("Sample size:", len(output["sample_df"]))
    print("Mean abs diff:", round(output["mad"], 4))
    print("Max abs diff:", round(output["maxad"], 4))


# =========================================================
# 7. CHECK THAT SELECTED SAMPLES HAVE TARGET FILES
# =========================================================
def check_target_existence(sample_df, exists_col, target_col, name):
    out = sample_df.copy()
    out["has_target_file"] = out[exists_col]
    out["target_answered_positive"] = out[target_col] > 0

    print(f"\n----- {name} target check -----")
    print("Has target file:")
    print(out["has_target_file"].value_counts(dropna=False))
    print("Target answered > 0:")
    print(out["target_answered_positive"].value_counts(dropna=False))

    return out


for key, output in sample_outputs.items():
    output["sample_df"] = check_target_existence(
        output["sample_df"], output["exists_col"], "target_count", output["name"]
    )


# =========================================================
# 8. SAVE OUTPUTS
# =========================================================
meta.to_csv(output_dir / "eligible_metadata_from_json.csv", index=False)

saved_files = ["eligible_metadata_from_json.csv"]

for key, output in sample_outputs.items():
    sample_output_dir = output_dir / key
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    output["allocation_df"].to_csv(sample_output_dir / f"allocation_{key}.csv", index=False)
    output["compare_df"].to_csv(sample_output_dir / f"strata_compare_{key}.csv", index=False)
    output["sample_df"].to_csv(sample_output_dir / f"sample_{key}.csv", index=False)

    ranked_df = output["sample_df"][output["ranked_columns"]].sort_values(
        ["stratum_final", "prev_count", "target_count"],
        ascending=[True, False, False]
    )
    ranked_df.to_csv(sample_output_dir / f"sample_{key}_ranked_check.csv", index=False)

    saved_files.extend([
        f"{key}/allocation_{key}.csv",
        f"{key}/strata_compare_{key}.csv",
        f"{key}/sample_{key}.csv",
        f"{key}/sample_{key}_ranked_check.csv"
    ])

print("\nSaved files to:", output_dir)
for file_name in saved_files:
    print(f" - {file_name}")
