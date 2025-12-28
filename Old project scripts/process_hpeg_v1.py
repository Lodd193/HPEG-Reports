import sys
import argparse
from pathlib import Path, PureWindowsPath
import re
import pandas as pd
import numpy as np
from datetime import datetime

# ------------------------------
# Configuration: columns & maps
# ------------------------------
DROP_COLS = [
    "Complaint Response Provided (Date Complaint Closed)",
    "Complaint - Patient Relations Logging (Patient details)",
    "Complaint - Patient Relations Team Management (PRM Handler)",
    "Complaint - Patient Relations Team Management (Third party interest)",
    "Complaint - Patient Relations Team Management (Deadline missed reason)",
    "Complaint - Patient Relations Team Management (Current position (Complaint))",
    "Complaint - Patient Relations Team Management (Management Plan)",
    "Complaint - Patient Relations Team Management (Site (Subject Complaint))",
]

RENAME_MAP = {
    "Reference": "ID",
    "Complaint - Patient Relations Team Management (First Received)": "First Received",
    "Complaint Response Provided (--Due date)": "Due Date",
    "Complaint Response Provided (--Completion Date)": "Completed Date",
    "Complaint - Patient Relations Team Management (Responsible Hospital Exec Team (Main))": "Exec Team",
    "Complaint - Patient Relations Team Management (Specialty (Main))": "Specialty",
    "Complaint - Patient Relations Team Management (Clinical Decision Group (CDG) (Main))": "CDG",
    "Complaint - Patient Relations Team Management (Site of event (Main))": "Site",
    "Complaint - Patient Relations Team Management (Location (Main))": "Location",
    "Complaint - Patient Relations Team Management (Deadline Met)": "Deadline Met",
    "Complaint - Patient Relations Team Management (Current stage)": "Current Stage",
    "Complaint - Patient Relations Team Management (Complaint Complexity)": "Complexity 1",
    "Complaint - Patient Relations Team Management (Type of Complaint/PALS)": "Type",
    "Complaint - Patient Relations Team Management (Subjects (Complaint))": "Subjects",
    "Complaint - Patient Relations Team Management (Sub-Subjects (Complaint))": "Sub-subjects",
    "Complaint - Patient Relations Logging (Description of Complaint)": "Description",
    "Complaint - Patient Relations Team Management (Outcome code (Complaint))": "Outcome Code",
    "Complaint Triage (Complaint Complexity)": "Complexity 2",
}

DATE_COLS_RENAMED = ["First Received", "Due Date", "Completed Date"]
NA_VALUES = ["", "NA", "N/A", "NULL", "null", "na", None]

# Canonical Subjects
SUBJECTS_CANON = [
    "Access to Treatment or Drugs",
    "Admissions, Discharges, and Transfers (excluding delayed discharge due to absence of care package)",
    "Appointments",
    "Clinical Treatment",
    "Communications",
    "Consent",
    "End of Life Care",
    "Facilities Services",
    "Other",
    "Patient Care including Nutrition/Hydration",
    "Prescribing Errors",
    "Privacy, Dignity, and Wellbeing",
    "Restraint",
    "Staff",
    "Trust Administration",
    "Waiting Times",
]

# Canonical Sub-subjects
SUBSUBJECTS_CANON = [
    "Attitude of staff",
    "Nursing care inadequate",
    "Inappropriate procedure/treatment",
    "Communication with relatives/carers",
    "Diagnosis - Incorrect/missed",
    "Delay or failure in observations",
    "Transfer - Internal issues",
    "Delay or failure to undertake scan/x-ray",
    "Emotional/Psychological Abuse by Staff",
    "Delay or failure in treatment/procedure/clinical assessment",
    "Failure to act in a professional manner",
    "Diagnosis - Delay/Failure",
    "Diagnosis -Incorrect",
    "Delay or failure to give information/results/letter/report",
    "Diagnosis - Delay in acting on test results",
    "Privacy dignity and wellbeing",
    "Communication with patient",
    "Discharge - Planned - Failed",
    "OP - Waiting list (Time on)",
    "Delay or failure in ordering tests",
    "Medication - delay receiving",
    "Injury sustained during treatment or operation",
    "Pain management issues",
    "Diagnosis - Dispute",
    "IP - Waiting list (Not on)",
    "EOL - Death certificate issues",
    "Discharge - Delay in planned discharge",
    "Loss of/damage to personal property including compensation issues",
    "Conflicting information",
    "Medication side effects",
    "Discharge - Premature",
    "IP - Appointment Cancellations",
    "Food and Hydration - Failure to provide appropriate foods linked to clinical need (e.g.",
    "Prescribing error",
    "Availability/non-availability of records (e.g.",
    "OP - Appointment delayed",
    "Diagnosis - Failure",
    "Cannula issues",
    "Post-treatment complications",
    "Discharge - Planned - Inadequate planning",
    "Nursing staff issues",
    "Communication re: appointments",
    "Discharge - Delay with medication",
    "Admission - Waiting on trolley",
    "Maternity/Childbirth - Mismanagement of labour",
    "IP - Appointment availability (including urgent)",
    "Food and Hydration - food/drink left out of reach",
    "EOL - Overall patient care at EOL",
    "Failure to provide privacy for dying patients",
    "Failure in communication to community or other organisation",
    "All aspects of restraint issues",
    "Trust administration issues",
    "OP - Appointment availability (including urgent)",
    "Dementia - Treatment of patients with",
    "Delay or failure in acting on reports",
    "Verbal Abuse by Staff (including Alleged)",
    "OP - Appointment not kept by Staff",
    "Maternity/Childbirth - Delay in induction of labour",
    "Discharge - Medication issue(s)(not delayed medication)",
    "Failure to obtain appropriate informed consent",
    "Refusal to prescribe",
    "Other Privacy and Dignity issues",
    "Failure to prescribe",
    "OP - Failure to provide follow-up appointment",
    "Dispensing error",
    "Admission - Bed not available",
    "Catheter care/issues",
    "Food and Hydration - Failure to provide appropriate foods linked to personal/cultural needs (e.g.",
    "Deteriorating patient - failure to recognise",
    "OP - Appointment Cancellations",
    "Facilities - Other",
    "Physical Abuse/Assault by Staff (including Alleged)",
    "IP - Waiting list (time on)",
    "Patient undergoes procedure without consent",
    "Facilities - Equipment availability (non-clinical)",
    "OP - Appointment Error",
    "Facilities - Cleanliness Non Clinical (All)",
    "Infection - Delay or failure in treatment for",
    "EOL - Pain management",
    "Food and Hydration - Failure to provide adequate fluids during the period of admission",
    "Method/style of communication (e.g.",
    "Adverse drug reactions",
    "Maternity/Childbirth - Neonatal death",
    "ED/CDU/AMU waiting times (including waiting time on a trolley)",
    "Public Concern",
    "Handling of requests for information (including FoI)",
    "Facilities - Portering services",
    "Doctor/Patient relationship broken down",
    "Discharge - Inappropriate time",
    "Failure to follow procedures",
    "Breach Of Confidentiality By Staff",
    "Infection - Other issues",
    "Financial Procedures/Patient finance",
    "Facilities - Cleanliness Clinical (All)",
    "Failure to provide privacy for patients",
    "Anaesthetic - Awareness whilst under",
    "EOL - Attitude of staff at EOL",
    "Maternity/Childbirth - Stillbirth issues",
    "OP - Appointment letter not issued/not received",
    "Facilities - Front of House",
    "Referral - Failure",
    "Insufficient information provided prior to consent",
    "Damage to personal property",
    "Facilities - Equipment condition (non-clinical)",
    "OP - Appointment booking system (including Choose and Book)",
    "Referral - Delay",
    "Mental health issue - Treatment of patient with",
    "Pressure sore - issues",
    "Referral - Refusal/Non",
]

# ------------------------------
# Helpers
# ------------------------------
def find_single_csv(folder: Path) -> Path:
    csvs = list(folder.glob("*.csv"))
    if len(csvs) == 0:
        raise FileNotFoundError(f"No CSV files found in: {folder}")
    if len(csvs) > 1:
        raise FileExistsError(f"Expected 1 CSV file but found {len(csvs)} in: {folder}")
    return csvs[0]

def normalise_header(s: str) -> str:
    if not isinstance(s, str):
        return s
    return re.sub(r"\s+", " ", s.strip())

def parse_args():
    p = argparse.ArgumentParser(description="Process HPEG CSV to cleaned monthly dataset.")
    p.add_argument("folder", type=str, nargs="?", help="Path to folder containing the single CSV (Windows path allowed).")
    p.add_argument("--report-date", type=str, default=None, help="YYYY-MM-DD for reproducible 'today'. Defaults to system today.")
    p.add_argument("--outdir", type=str, default=None, help="Optional output directory. Defaults to <folder>/processed.")
    return p.parse_args()

def load_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
        keep_default_na=True,
        na_values=NA_VALUES,
        dtype={"Reference": "string"},
        engine="python",
        on_bad_lines="warn",
    )
    df.columns = [normalise_header(c) for c in df.columns]
    return df

# ---------- Early drops ----------
def early_drop_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Early removals:
      1) Type contains 'Complaint - Opened in error'
      2) Test IDs COM-005/COM-008
      3) 'Deadline Met' contains 'Complaint Withdrawn'
      4) 'Current Stage' contains 'Complaint Withdrawn'
    """
    # 1) Opened in error
    type_candidates = [
        "Complaint - Patient Relations Team Management (Type of Complaint/PALS)",
        "Type",
    ]
    if not any(c in df.columns for c in type_candidates):
        for col in df.columns:
            if "type" in col.lower() and "complaint" in col.lower():
                type_candidates.append(col)
    type_col = next((c for c in type_candidates if c in df.columns), None)
    if type_col is not None:
        mask_err = df[type_col].astype(str).str.contains(r"complaint\s*-\s*opened\s*in\s*error", case=False, na=False)
        before = len(df)
        df = df.loc[~mask_err].copy()
        dropped = before - len(df)
        if dropped:
            print(f"[INFO] Early-dropped {dropped} 'Opened in error' row(s).")
    else:
        print("[WARN] Type column not found for early drop of 'Opened in error'.")

    # 2) Test IDs
    test_ids = {"COM-005", "COM-008"}
    id_candidates = ["Reference", "ID"]
    id_candidates = [c for c in id_candidates if c in df.columns]
    if id_candidates:
        prev_len = len(df)
        mask_test = pd.Series(False, index=df.index)
        for c in id_candidates:
            mask_test = mask_test | df[c].isin(test_ids)
        df = df.loc[~mask_test].copy()
        removed = prev_len - len(df)
        if removed:
            print(f"[INFO] Early-dropped {removed} test case row(s) {sorted(test_ids)}.")
    else:
        print("[WARN] No ID/Reference column for early drop of test cases.")

    # 3) Withdrawn via Deadline Met
    dm_candidates = [
        "Complaint - Patient Relations Team Management (Deadline Met)",
        "Deadline Met",
    ]
    if not any(c in df.columns for c in dm_candidates):
        for col in df.columns:
            low = col.lower()
            if "deadline" in low and "met" in low:
                dm_candidates.append(col)
    dm_col = next((c for c in dm_candidates if c in df.columns), None)
    if dm_col is not None:
        mask_withdrawn_dm = df[dm_col].astype(str).str.contains(r"\bcomplaint\s*withdrawn\b", case=False, na=False)
        before = len(df)
        df = df.loc[~mask_withdrawn_dm].copy()
        dropped = before - len(df)
        if dropped:
            print(f"[INFO] Early-dropped {dropped} 'Complaint Withdrawn' via Deadline Met.")
    else:
        print("[WARN] 'Deadline Met' column not found for early drop of 'Complaint Withdrawn'.")

    # 4) Withdrawn via Current Stage
    cs_candidates = [
        "Complaint - Patient Relations Team Management (Current stage)",
        "Current Stage",
    ]
    if not any(c in df.columns for c in cs_candidates):
        for col in df.columns:
            low = col.lower()
            if "current" in low and "stage" in low:
                cs_candidates.append(col)
    cs_col = next((c for c in cs_candidates if c in df.columns), None)
    if cs_col is not None:
        mask_withdrawn_cs = df[cs_col].astype(str).str.contains(r"\bcomplaint\s*withdrawn\b", case=False, na=False)
        before = len(df)
        df = df.loc[~mask_withdrawn_cs].copy()
        dropped = before - len(df)
        if dropped:
            print(f"[INFO] Early-dropped {dropped} 'Complaint Withdrawn' via Current Stage.")
    else:
        print("[WARN] 'Current Stage' column not found for early drop of 'Complaint Withdrawn'.")

    return df

# ---------- Standard transforms ----------
def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    present_to_drop = [c for c in DROP_COLS if c in df.columns]
    missing = [c for c in DROP_COLS if c not in df.columns]
    if missing:
        print(f"[WARN] Expected drop columns missing and ignored: {missing}")
    return df.drop(columns=present_to_drop, errors="ignore")

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [old for old in RENAME_MAP.keys() if old not in df.columns]
    if missing:
        print(f"[WARN] Expected rename columns missing: {missing}")
    df = df.rename(columns=RENAME_MAP)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def to_datetime_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in DATE_COLS_RENAMED:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce").dt.normalize()
        else:
            print(f"[WARN] Date column missing after rename: {col}")
    return df

def clean_type_value(raw: str) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return raw
    s = str(raw).strip()
    if s == "":
        return s
    parts = [p.strip() for p in s.split(",")]
    followup_pat = re.compile(r"\b(FU\d|FLR\d|FL\d)\b", flags=re.IGNORECASE)
    for p in parts:
        if followup_pat.search(p):
            return "Follow Up"
    for p in parts:
        if "complaint" in p.lower():
            return "Complaint"
    return s

def normalise_type(df: pd.DataFrame) -> pd.DataFrame:
    if "Type" not in df.columns:
        print("[WARN] 'Type' column not found; skipping normalisation.")
        return df
    df["Type"] = df["Type"].apply(clean_type_value)
    return df

def add_closed_and_deadline(df: pd.DataFrame, report_date: pd.Timestamp) -> pd.DataFrame:
    if "Completed Date" in df.columns:
        df["Closed?"] = df["Completed Date"].notna().map({True: "Yes", False: "Ongoing"})
    else:
        df["Closed?"] = "Ongoing"

    def compute_deadline_status(row):
        due = row.get("Due Date", pd.NaT)
        comp = row.get("Completed Date", pd.NaT)
        closed = pd.notna(comp)

        dm = row.get("Deadline Met", pd.NA)
        if isinstance(dm, str) and "deadline met" in dm.lower():
            return "Deadline Met"

        if pd.isna(due):
            return "Unknown"

        if closed:
            return "Deadline Met" if comp <= due else "Deadline Missed"
        else:
            return "Still in Progress - In Time" if due >= report_date else "Still in Progress - Out of Time"

    df["Deadline Status"] = df.apply(compute_deadline_status, axis=1)
    return df

def add_six_months_flag(df: pd.DataFrame, report_date: pd.Timestamp) -> pd.DataFrame:
    colname = "6 months"
    df[colname] = "No"
    if "First Received" not in df.columns:
        return df
    mask_open = df.get("Closed?", pd.Series(index=df.index, dtype=object)).eq("Ongoing")
    mask_valid = mask_open & df["First Received"].notna()
    if mask_valid.any():
        starts = df.loc[mask_valid, "First Received"].values.astype("datetime64[D]")
        end = np.datetime64((report_date + pd.Timedelta(days=1)).date(), "D")
        counts = np.busday_count(starts, end)
        df.loc[mask_valid, colname] = np.where(counts >= 131, "Yes", "No")
    return df

def clean_location_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Location" not in df.columns:
        return df
    df["Location"] = (
        df["Location"]
        .astype("string")
        .str.replace(r"\s*\([^)]*\)", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df

def override_exec_team_wc(df: pd.DataFrame) -> pd.DataFrame:
    if "Specialty" not in df.columns:
        return df

    def _norm(x: str) -> str:
        s = str(x) if x is not None else ""
        s = re.sub(r"\s+", " ", s).strip().lower()
        s = re.sub(r"[.\s]+$", "", s)
        return s

    targets = {
        _norm("Maternity UHB"),
        _norm("Gynaecology UHB"),
        _norm("Paediatrics ED GHH"),
        _norm("Paediatrics UHB"),
        _norm("Paediatrics ED BHH"),
        _norm("Neonates UHB"),
        _norm("Obstetrics UHB"),
        _norm("Community Paediatrics SH"),
        _norm("Community Paediatrics SH."),
    }

    spec_norm = df["Specialty"].astype("string").map(_norm)
    mask = spec_norm.isin(targets)
    if "Exec Team" not in df.columns:
        df["Exec Team"] = pd.NA
    df.loc[mask, "Exec Team"] = "W&C Exec Team"
    return df

# ---------- Subjects/Sub-subjects cleaning ----------
def _norm_token(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)      # drop punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _first_canonical_match(cell: str, canon_norms: dict) -> tuple:
    text = _norm_token(cell)
    best_label, best_idx = None, 10**9
    for original, normed in canon_norms.items():
        idx = text.find(normed)
        if idx != -1 and idx < best_idx:
            best_label, best_idx = original, idx
    return (best_label, best_idx) if best_label is not None else (None, -1)

def keep_first_subjects_and_subsubjects(
    df: pd.DataFrame, subjects_list: list[str], subsubjects_list: list[str]
):
    unmatched = {"Subjects": [], "Sub-subjects": []}

    subj_norms = {s: _norm_token(s) for s in subjects_list}
    subsub_norms = {s: _norm_token(s) for s in subsubjects_list}

    # Subjects
    if "Subjects" in df.columns:
        chosen = []
        col = df["Subjects"].astype("string")
        for val in col:
            if pd.isna(val) or (isinstance(val, str) and val.strip() == ""):
                chosen.append(val); continue
            match, _ = _first_canonical_match(val, subj_norms)
            if match is None:
                unmatched["Subjects"].append(val); chosen.append(val)
            else:
                chosen.append(match)
        df["Subjects"] = pd.Series(chosen, index=df.index, dtype="string")

    # Sub-subjects
    if "Sub-subjects" in df.columns:
        chosen = []
        col = df["Sub-subjects"].astype("string")
        for val in col:
            if pd.isna(val) or (isinstance(val, str) and val.strip() == ""):
                chosen.append(val); continue
            match, _ = _first_canonical_match(val, subsub_norms)
            if match is None:
                unmatched["Sub-subjects"].append(val); chosen.append(val)
            else:
                chosen.append(match)
        df["Sub-subjects"] = pd.Series(chosen, index=df.index, dtype="string")

    return df, unmatched

# ---------- Potential Safeguarding (highly conservative) ----------
_NEGATIONS = [
    "no evidence of", "no indication of", "does not", "did not", "not", "denies",
    "denied", "ruled out", "unfounded", "unsubstantiated", "allegation withdrawn"
]

def _compile_patterns():
    # Strong, explicit phrases only; avoid generic terms to limit false positives
    patt = {
        "Sexual abuse": [
            r"\bsexual abuse\b", r"\bsexual assault\b", r"\brape(d)?\b",
        ],
        "Domestic violence or abuse": [
            r"\bdomestic (violence|abuse)\b",
        ],
        "Modern slavery": [
            r"\bmodern slavery\b", r"\bhuman traffick\w*\b",
        ],
        "Physical Abuse": [
            r"\bphysical abuse\b", r"\bphysical assault\b",
            r"\bassault(?:ed)? by (?:staff|nurse|doctor|carer|family|relative|partner|patient|another patient)\b",
        ],
        "Financial or material abuse": [
            r"\bfinancial (abuse|exploitation)\b", r"\bmaterial abuse\b",
        ],
        "Discriminatory abuse": [
            r"\bdiscriminatory abuse\b", r"\bracial abuse\b", r"\bracist abuse\b", r"\bhomophobic abuse\b",
        ],
        "Psychological or emotional abuse": [
            r"\bpsychological abuse\b", r"\bemotional abuse\b",
        ],
        "Organisational or institutional abuse": [
            r"\borganis(?:ational|ational) abuse\b", r"\binstitutional abuse\b",
        ],
        "Neglect and acts of omission": [
            r"\bacts? of omission\b", r"\bneglect(?:ed)?\b",
        ],
        "Self-neglect": [
            r"\bself-?neglect\b",
        ],
    }
    # Compile with case-insensitive
    return {k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in patt.items()}

_SG_PATTERNS = _compile_patterns()

# Priority: rare/serious first, then broader
_SG_PRIORITY = [
    "Sexual abuse",
    "Domestic violence or abuse",
    "Modern slavery",
    "Physical Abuse",
    "Financial or material abuse",
    "Discriminatory abuse",
    "Psychological or emotional abuse",
    "Organisational or institutional abuse",
    "Neglect and acts of omission",
    "Self-neglect",
]

def _has_negation(text_lower: str, start_idx: int) -> bool:
    # Check a window before the match for any negation phrase
    window_start = max(0, start_idx - 80)  # ~10-15 words back
    pre = text_lower[window_start:start_idx]
    return any(neg in pre for neg in _NEGATIONS)

def classify_safeguarding(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "None"
    t = text.strip()
    t_lower = t.lower()
    for cat in _SG_PRIORITY:
        for rx in _SG_PATTERNS[cat]:
            for m in rx.finditer(t):
                if not _has_negation(t_lower, m.start()):
                    return cat
    return "None"

def add_potential_safeguarding(df: pd.DataFrame) -> pd.DataFrame:
    col = "Potential Safeguarding"
    if "Description" not in df.columns:
        df[col] = "None"
        return df
    df[col] = df["Description"].apply(classify_safeguarding).astype("string")
    return df

# ---------- Column positioning ----------
def relocate_closed_deadline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Place 'Closed?', 'Deadline Status', '6 months', and 'Potential Safeguarding'
    immediately after 'Completed Date' (before 'Exec Team' when present).
    """
    if "Completed Date" not in df.columns:
        return df
    cols = list(df.columns)
    to_insert = [c for c in ["Closed?", "Deadline Status", "6 months", "Potential Safeguarding"] if c in cols]
    if not to_insert:
        return df
    for c in to_insert:
        if c in cols:
            cols.remove(c)
    insert_pos = cols.index("Completed Date") + 1
    new_cols = cols[:insert_pos] + to_insert + cols[insert_pos:]
    new_cols = [c for c in new_cols if c in df.columns]
    return df.loc[:, new_cols]

def merge_complexity(df: pd.DataFrame) -> pd.DataFrame:
    c2 = "Complexity 2" in df.columns
    c1 = "Complexity 1" in df.columns
    if c2 or c1:
        if c2:
            df["Complexity"] = df["Complexity 2"]
        else:
            df["Complexity"] = pd.Series(pd.NA, index=df.index, dtype="object")
        if c1:
            df["Complexity"] = df["Complexity"].fillna(df["Complexity 1"])
        df["Complexity"] = df["Complexity"].astype("string").str.strip()
        drop_cols = [c for c in ["Complexity 1", "Complexity 2"] if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
    else:
        print("[WARN] No complexity columns found.")
    return df

# ---------- QA & I/O ----------
def quality_checks(df: pd.DataFrame):
    anomalies = []
    if all(col in df.columns for col in ["First Received", "Due Date"]):
        bad = df[(df["First Received"].notna()) & (df["Due Date"].notna()) & (df["First Received"] > df["Due Date"])]
        if not bad.empty:
            anomalies.append(("FirstReceived_after_DueDate", bad["ID"].tolist() if "ID" in bad.columns else bad.index.tolist()))
    if all(col in df.columns for col in ["First Received", "Completed Date"]):
        bad2 = df[(df["First Received"].notna()) & (df["Completed Date"].notna()) & (df["Completed Date"] < df["First Received"])]
        if not bad2.empty:
            anomalies.append(("Completed_before_FirstReceived", bad2["ID"].tolist() if "ID" in bad.columns else bad.index.tolist()))
    return anomalies

def write_outputs(df: pd.DataFrame, outdir: Path, source_csv: Path, report_date: pd.Timestamp, anomalies, unmatched=None) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_path = outdir / f"processed_{report_date.date()}_{stamp}.csv"
    df.to_csv(processed_path, index=False, encoding="utf-8-sig")

    qa_lines = []
    qa_lines.append(f"Source CSV: {source_csv}")
    qa_lines.append(f"Rows: {len(df)}")

    def add_count_series(title, series):
        qa_lines.append(f"\n{title}:")
        try:
            counts = series.value_counts(dropna=False).sort_index()
            for idx, val in counts.items():
                qa_lines.append(f"  {idx}: {val}")
        except Exception as e:
            qa_lines.append(f"  [Error computing {title}: {e}]")

    if "Type" in df.columns:
        add_count_series("Type counts", df["Type"])
    if "Closed?" in df.columns:
        add_count_series("Closed? counts", df["Closed?"])
    if "Deadline Status" in df.columns:
        add_count_series("Deadline Status counts", df["Deadline Status"])
    if "6 months" in df.columns:
        add_count_series('"6 months" counts', df["6 months"])
    if "Potential Safeguarding" in df.columns:
        add_count_series("Potential Safeguarding counts", df["Potential Safeguarding"])
    if "Due Date" in df.columns:
        qa_lines.append(f"\nMissing Due Date: {df['Due Date'].isna().sum()}")

    # Unmatched subjects/sub-subjects summary
    if unmatched:
        for key in ["Subjects", "Sub-subjects"]:
            items = unmatched.get(key, [])
            if items:
                qa_lines.append(f"\nUnmatched {key}: {len(items)} occurrences")
                try:
                    vc = pd.Series(items, dtype="string").value_counts().head(20)
                    for v, c in vc.items():
                        qa_lines.append(f"  {v}: {c}")
                except Exception as e:
                    qa_lines.append(f"  [Error summarising unmatched {key}: {e}]")

    if anomalies:
        qa_lines.append("\nAnomalies detected:")
        for name, ids in anomalies:
            qa_lines.append(f"  {name}: {len(ids)} rows -> IDs: {ids[:20]}{'...' if len(ids) > 20 else ''}")
    else:
        qa_lines.append("\nAnomalies detected: none")

    qa_path = outdir / f"qa_summary_{report_date.date()}_{stamp}.txt"
    with open(qa_path, "w", encoding="utf-8") as f:
        f.write("\n".join(qa_lines))

    print(f"[INFO] Wrote processed CSV: {processed_path}")
    print(f"[INFO] Wrote QA summary:   {qa_path}")
    return processed_path

# ---------- Main ----------
def main():
    pd.options.mode.copy_on_write = True
    args = parse_args()

    if not args.folder:
        args.folder = input("Enter path to folder containing the single CSV: ").strip()

    folder = Path(args.folder)
    if not folder.exists():
        folder = Path(PureWindowsPath(args.folder))
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {args.folder}")

    outdir = Path(args.outdir) if args.outdir else (folder / "processed")
    report_date = pd.to_datetime(args.report_date, errors="coerce").normalize() if args.report_date else pd.Timestamp.today().normalize()

    csv_path = find_single_csv(folder)
    print(f"[INFO] Using source CSV: {csv_path}")

    df = load_dataframe(csv_path)

    # 1) Early drops
    df = early_drop_preprocessing(df)

    # 2) Drop only specified columns
    df = drop_columns(df)

    # 3) Rename columns
    df = rename_columns(df)

    # 3a) Clean 'Location'
    df = clean_location_column(df)

    # 3b) Exec Team override (W&C) by Specialty
    df = override_exec_team_wc(df)

    # 4) Ensure ID is string
    if "ID" in df.columns:
        df["ID"] = df["ID"].astype("string")

    # 5) Dates
    df = to_datetime_cols(df)

    # 6) Type normalisation
    df = normalise_type(df)

    # 7) Closed? + Deadline Status (with 'Deadline Met' override)
    df = add_closed_and_deadline(df, report_date=report_date)

    # 8) "6 months" flag
    df = add_six_months_flag(df, report_date=report_date)

    # 9) Subjects/Sub-subjects cleaner (keep first valid); capture unmatched for QA
    df, unmatched = keep_first_subjects_and_subsubjects(df, SUBJECTS_CANON, SUBSUBJECTS_CANON)

    # 10) Potential Safeguarding (conservative)
    df = add_potential_safeguarding(df)

    # 11) Relocate Closed?/Deadline Status/"6 months"/"Potential Safeguarding"
    df = relocate_closed_deadline(df)

    # 12) Merge complexity and drop source columns
    df = merge_complexity(df)

    # Optional categoricals (keep 'Deadline Met' textual)
    for cat_col in [
        "Exec Team", "CDG", "Site", "Location", "Current Stage", "Type",
        "Closed?", "Deadline Status", "6 months", "Potential Safeguarding", "Complexity"
    ]:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype("category")

    anomalies = quality_checks(df)
    write_outputs(df, outdir, csv_path, report_date, anomalies, unmatched=unmatched)
    print("[DONE] Processing complete.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
