#!/usr/bin/env python3
"""
HPEG Data Processor - Script 1 of 2
=====================================
Processes NHS complaints data for HPEG reporting.

This script:
1. Loads and cleans Radar CSV export
2. Calculates all required metrics
3. Performs NMF topic modeling (trust-wide + HPEG-specific)
4. Runs demographic statistical analysis
5. Saves processed data for report generation

Author: Patient Relations Quality & Performance Manager
Version: 1.0
Date: December 2024
"""

import sys
import re
import pickle
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Fix Windows console encoding for Unicode characters
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# NLP and topic modeling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import spacy

# ============================================================================
# USER CONFIGURATION - UPDATE THESE PATHS FOR YOUR WORK ENVIRONMENT
# ============================================================================

INPUT_CSV = r"C:/Users/lod19/HPEGs/Real data/28.12.25 report.csv"
PROCESSED_DATA_PATH = r"C:/Users/lod19/HPEGs/processed_data.pkl"

# ============================================================================
# CONFIGURATION - NHS Standards and Data Mappings
# ============================================================================

# NHS Color Palette (for reference in report generation)
NHS_COLORS = {
    'nhs_blue': '#005EB8',
    'nhs_dark_blue': '#003087',
    'nhs_bright_blue': '#0072CE',
    'nhs_light_blue': '#41B6E6',
    'nhs_aqua_blue': '#00A9CE',
    'nhs_black': '#231F20',
    'nhs_dark_grey': '#425563',
    'nhs_mid_grey': '#768692',
    'nhs_pale_grey': '#E8EDEE',
    'white': '#FFFFFF',
    'nhs_dark_green': '#006747',
    'nhs_green': '#009639',
    'nhs_light_green': '#78BE20',
    'nhs_aqua_green': '#00A499',
    'nhs_purple': '#330072',
    'dark_pink': '#7C2855',
    'nhs_pink': '#AE2573',
    'nhs_dark_red': '#8A1538',
    'emergency_red': '#DA291C',
    'nhs_orange': '#ED8B00',
    'nhs_warm_yellow': '#FFB81C',
    'nhs_yellow': '#FAE100'
}

# Column mappings
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
    "Complaint - Patient Relations Logging (Date of Birth (Complaint))": "Date of Birth",
    "Complaint - Patient Relations Team Management (Gender (KO41))": "Gender",
    "Complaint - Patient Relations Team Management (Patient Ethnicity (KO41(A)))": "Ethnicity",
}

# Exec Team mappings
EXEC_TEAMS = {
    "BHH": "BHH Exec Team",
    "QEH": "QEH Exec Team",
    "GHH": "GHH Exec Team",
    "SH": "SH Exec Team",
    "W&C": "W&C Exec Team",
    "WC": "W&C Exec Team",
    "CSS": "CSS Exec Team",
    "CORP": "Corporate",
    "CORPORATE": "Corporate",
}

# CDG to HPEG mapping (based on CDG number prefix)
# BHH: CDG1, CDG2
# W&C: CDG3
# GHH: CDG4, CDG5
# SH: CDG6, CDG7
# QEHB: CDG8, CDG9, CDG10
# CSS: CDG11
CDG_TO_HPEG_MAP = {
    "1": "BHH Exec Team",
    "2": "BHH Exec Team",
    "3": "W&C Exec Team",
    "4": "GHH Exec Team",
    "5": "GHH Exec Team",
    "6": "SH Exec Team",
    "7": "SH Exec Team",
    "8": "QEH Exec Team",
    "9": "QEH Exec Team",
    "10": "QEH Exec Team",
    "11": "CSS Exec Team",
}

# Canonical Subjects (KO41a aligned)
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

# Domain-specific stopwords for NMF topic modeling
NMF_STOPWORDS = [
    # NHS/Hospital terms
    'nhs', 'hospital', 'patient', 'complaint', 'trust', 'staff', 'uhb',
    'birmingham', 'heartlands', 'queen', 'elizabeth', 'solihull', 'good',
    'hope', 'service', 'care', 'team', 'ward', 'department',
    # Generic time words
    'week', 'day', 'time', 'hour', 'minute', 'month', 'year', 'today', 'yesterday',
    # Generic action verbs
    'told', 'said', 'asked', 'called', 'went', 'came', 'give', 'given', 'take', 'taken',
    'make', 'made', 'get', 'got', 'see', 'saw', 'seen', 'speak', 'spoke', 'go', 'come',
    # Generic wanting/needing words
    'want', 'wanted', 'need', 'needed', 'require', 'required', 'wish', 'wished',
    # Generic ability words
    'able', 'unable', 'could', 'would', 'should', 'can', 'cannot',
    # Generic relatives
    'mother', 'father', 'daughter', 'son', 'family', 'relative', 'husband', 'wife',
    # Generic medical terms (too broad)
    'doctor', 'nurse', 'appointment', 'bed', 'room',
    # Age-related (generic)
    'old', 'age', 'aged', 'year old', 'years old',
    # Generic reporting words
    'received', 'informed', 'contacted', 'phoned', 'emailed', 'letter', 'wrote',
    # Generic issues (use more specific complaint subjects)
    'issue', 'problem', 'concern', 'matter',
]

# ============================================================================
# HELPER FUNCTIONS - Data Cleaning
# ============================================================================

def normalise_header(s: str) -> str:
    """Normalize column headers by removing extra whitespace."""
    if not isinstance(s, str):
        return s
    return re.sub(r"\s+", " ", s.strip())

def early_drop_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove invalid complaints:
    - Opened in error
    - Test cases (COM-005, COM-008)
    - Withdrawn complaints
    """
    print("  Applying early drop preprocessing...")
    initial_count = len(df)

    # Find Type column
    type_col = next((c for c in ["Complaint - Patient Relations Team Management (Type of Complaint/PALS)", "Type"]
                     if c in df.columns), None)

    if type_col:
        mask_err = df[type_col].astype(str).str.contains(r"complaint\s*-\s*opened\s*in\s*error", case=False, na=False)
        df = df.loc[~mask_err].copy()

    # Remove test cases
    test_ids = {"COM-005", "COM-008"}
    id_col = next((c for c in ["Reference", "ID"] if c in df.columns), None)
    if id_col:
        df = df.loc[~df[id_col].isin(test_ids)].copy()

    # Remove withdrawn complaints
    for col_name in ["Deadline Met", "Current Stage"]:
        col_candidates = [c for c in df.columns if col_name.lower() in c.lower()]
        for col in col_candidates:
            if col in df.columns:
                mask_withdrawn = df[col].astype(str).str.contains(r"\bcomplaint\s*withdrawn\b", case=False, na=False)
                df = df.loc[~mask_withdrawn].copy()

    dropped = initial_count - len(df)
    print(f"    ✓ Dropped {dropped} invalid complaints")
    return df

def clean_type_value(raw: str) -> str:
    """Normalize complaint type field."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return raw
    s = str(raw).strip()
    if s == "":
        return s

    # Check for follow-up patterns
    followup_pat = re.compile(r"\b(FU\d|FLR\d|FL\d)\b", flags=re.IGNORECASE)
    parts = [p.strip() for p in s.split(",")]
    for p in parts:
        if followup_pat.search(p):
            return "Follow Up"
    for p in parts:
        if "complaint" in p.lower():
            return "Complaint"
    return s

def override_exec_team_wc(df: pd.DataFrame) -> pd.DataFrame:
    """Override Exec Team to W&C for maternity/gynae/paeds specialties."""
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

def override_exec_team_by_cdg(df: pd.DataFrame) -> pd.DataFrame:
    """Override Exec Team based on CDG number prefix."""
    if "CDG" not in df.columns:
        return df

    if "Exec Team" not in df.columns:
        df["Exec Team"] = pd.NA

    def extract_cdg_number(cdg_value):
        """Extract the numeric prefix from CDG (e.g., CDG11RAD -> 11, CDG1A -> 1)."""
        if pd.isna(cdg_value):
            return None
        cdg_str = str(cdg_value).strip().upper()
        # Match CDG followed by numbers
        match = re.match(r'CDG(\d+)', cdg_str)
        if match:
            return match.group(1)
        return None

    # Extract CDG numbers
    cdg_numbers = df["CDG"].apply(extract_cdg_number)

    # Map to HPEGs
    for cdg_num, hpeg in CDG_TO_HPEG_MAP.items():
        mask = cdg_numbers == cdg_num
        df.loc[mask, "Exec Team"] = hpeg

    return df

def add_closed_and_deadline(df: pd.DataFrame, report_date: pd.Timestamp) -> pd.DataFrame:
    """Add Closed? and Deadline Status fields."""
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

def calculate_business_days(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    """
    Calculate business days between two dates (excludes weekends only).
    Returns 0 if dates are invalid.
    """
    if pd.isna(start_date) or pd.isna(end_date):
        return 0

    try:
        # Convert to numpy datetime64
        start = np.datetime64(start_date.date(), 'D')
        end = np.datetime64(end_date.date(), 'D')

        # Calculate business days (excludes weekends)
        bdays = np.busday_count(start, end)
        return int(bdays)
    except:
        return 0

def add_resolution_time_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add resolution time metrics in business days.
    Calculates days from First Received to Completed Date.
    """
    df['Resolution Days'] = 0

    # Only calculate for closed cases
    mask_closed = df.get('Closed?', pd.Series(index=df.index, dtype=object)).eq('Yes')
    mask_valid = mask_closed & df['First Received'].notna() & df['Completed Date'].notna()

    if mask_valid.any():
        for idx in df[mask_valid].index:
            start = df.loc[idx, 'First Received']
            end = df.loc[idx, 'Completed Date']
            df.loc[idx, 'Resolution Days'] = calculate_business_days(start, end)

    # Add target met flag based on complexity
    df['Met Target'] = 'N/A'

    # Define targets
    targets = {
        'Basic': 25,
        'Regular': 40,
        'Complex': 65
    }

    for complexity, target in targets.items():
        mask = mask_valid & (df['Complexity'] == complexity)
        if mask.any():
            df.loc[mask, 'Met Target'] = (df.loc[mask, 'Resolution Days'] <= target).map({True: 'Yes', False: 'No'})

    return df

def add_six_months_flag(df: pd.DataFrame, report_date: pd.Timestamp) -> pd.DataFrame:
    """
    Flag cases open for 6+ months (>=131 business days) and 12+ months (>=262 business days).
    Uses TODAY's date, not report end date, to calculate age of ongoing cases.
    """
    df["6 months"] = "No"
    df["12 months"] = "No"

    if "First Received" not in df.columns:
        return df

    mask_open = df.get("Closed?", pd.Series(index=df.index, dtype=object)).eq("Ongoing")
    mask_valid = mask_open & df["First Received"].notna()

    if mask_valid.any():
        starts = df.loc[mask_valid, "First Received"].values.astype("datetime64[D]")
        # CRITICAL: Use TODAY's date, not report_date, for accurate age calculation
        today = np.datetime64(pd.Timestamp.now().date(), "D")
        counts = np.busday_count(starts, today)

        # Flag 6+ months (>=131 business days)
        df.loc[mask_valid, "6 months"] = np.where(counts >= 131, "Yes", "No")

        # Flag 12+ months (>=262 business days)
        df.loc[mask_valid, "12 months"] = np.where(counts >= 262, "Yes", "No")

    return df

# ============================================================================
# NMF TOPIC MODELING
# ============================================================================

def preprocess_text_for_nmf(texts, nlp):
    """
    Preprocess complaint descriptions for NMF topic modeling.

    Args:
        texts: Series of complaint descriptions
        nlp: spaCy language model

    Returns:
        List of preprocessed text strings
    """
    print("  Preprocessing text for topic modeling...")
    processed = []

    for doc in nlp.pipe(texts, batch_size=50, n_process=1):
        # Lemmatize, remove stopwords, punctuation, keep only nouns/verbs/adjectives
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and token.is_alpha
            and len(token.text) > 2
            and token.pos_ in ['NOUN', 'VERB', 'ADJ']
            and token.lemma_.lower() not in NMF_STOPWORDS
        ]
        processed.append(' '.join(tokens))

    return processed

def fit_nmf_model(texts, n_topics=10, max_features=1000):
    """
    Fit NMF topic model.

    Args:
        texts: Preprocessed text documents
        n_topics: Number of topics to extract
        max_features: Maximum vocabulary size

    Returns:
        tuple: (model, vectorizer, document_topic_matrix, feature_names)
    """
    print(f"  Fitting NMF model with {n_topics} topics...")

    # TF-IDF vectorization with aggressive filtering for meaningful topics
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        max_df=0.6,  # Ignore terms in >60% of documents (very common words)
        min_df=5,    # Ignore terms in <5 documents (very rare words)
        ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams for specificity
        token_pattern=r'\b[a-z]{3,}\b'  # At least 3 characters
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # NMF decomposition
    nmf_model = NMF(
        n_components=n_topics,
        random_state=42,
        max_iter=500,
        alpha_W=0.1,
        alpha_H=0.1,
        l1_ratio=0.5
    )

    document_topic_matrix = nmf_model.fit_transform(tfidf_matrix)

    return nmf_model, vectorizer, document_topic_matrix, feature_names

def extract_topic_keywords(model, feature_names, n_keywords=5):
    """Extract top keywords for each topic."""
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-n_keywords:][::-1]
        top_keywords = [feature_names[i] for i in top_indices]
        topics.append({
            'id': topic_idx + 1,
            'keywords': top_keywords,
            'label': ' / '.join(top_keywords[:3]).title()
        })
    return topics

def calculate_topic_distributions(document_topic_matrix, df, group_col, group_value=None):
    """
    Calculate topic distribution for a group of complaints.

    Args:
        document_topic_matrix: NMF output matrix
        df: DataFrame with complaints
        group_col: Column to group by
        group_value: Specific value to filter (if None, uses all data)

    Returns:
        Array of topic proportions
    """
    if group_value is not None:
        mask = df[group_col] == group_value
        group_matrix = document_topic_matrix[mask]
    else:
        group_matrix = document_topic_matrix

    if len(group_matrix) == 0:
        return np.zeros(document_topic_matrix.shape[1])

    # Average topic weights across documents
    topic_dist = group_matrix.mean(axis=0)
    # Normalize to sum to 1 (handle case where sum is 0)
    total = topic_dist.sum()
    if total > 0:
        topic_dist = topic_dist / total
    return topic_dist

# ============================================================================
# DEMOGRAPHIC ANALYSIS
# ============================================================================

def run_demographic_analysis(df, topics_df):
    """
    Run chi-square tests to identify demographic associations with topics.

    Args:
        df: Main DataFrame with demographic info
        topics_df: DataFrame with dominant topic per complaint

    Returns:
        List of significant findings (p < 0.05)
    """
    print("  Running demographic statistical analysis...")

    findings = []

    # Merge topics with demographics
    merged = df.merge(topics_df, left_index=True, right_index=True, how='left')

    # Test Gender associations
    if 'Gender' in merged.columns and 'dominant_topic' in merged.columns:
        gender_clean = merged['Gender'].fillna('Unknown').astype(str)
        gender_clean = gender_clean[gender_clean.isin(['Male', 'Female'])]  # Only test binary gender

        if len(gender_clean) > 30:  # Minimum sample size
            for topic_id in merged['dominant_topic'].dropna().unique():
                # Create contingency table
                gender_subset = merged.loc[gender_clean.index]
                topic_mask = gender_subset['dominant_topic'] == topic_id

                contingency = pd.crosstab(
                    gender_subset['Gender'],
                    topic_mask,
                    margins=False
                )

                if contingency.shape == (2, 2) and contingency.min().min() >= 5:  # Valid for chi-square
                    chi2, p_value, dof, expected = chi2_contingency(contingency)

                    if p_value < 0.05:
                        findings.append({
                            'demographic': 'Gender',
                            'topic_id': topic_id,
                            'p_value': p_value,
                            'interpretation': f"Topic {topic_id} shows association with gender (p={p_value:.3f})"
                        })

    # Test Ethnicity associations (simplified - compare White British vs Others)
    if 'Ethnicity' in merged.columns and 'dominant_topic' in merged.columns:
        ethnicity_clean = merged['Ethnicity'].fillna('Unknown').astype(str)
        ethnicity_binary = ethnicity_clean.apply(
            lambda x: 'White British' if 'white' in x.lower() and 'british' in x.lower() else 'Other'
        )
        ethnicity_binary = ethnicity_binary[ethnicity_binary != 'Unknown']

        if len(ethnicity_binary) > 30:
            for topic_id in merged['dominant_topic'].dropna().unique():
                ethnicity_subset = merged.loc[ethnicity_binary.index]
                topic_mask = ethnicity_subset['dominant_topic'] == topic_id

                contingency = pd.crosstab(
                    ethnicity_binary,
                    topic_mask,
                    margins=False
                )

                if contingency.shape == (2, 2) and contingency.min().min() >= 5:
                    chi2, p_value, dof, expected = chi2_contingency(contingency)

                    if p_value < 0.05:
                        findings.append({
                            'demographic': 'Ethnicity',
                            'topic_id': topic_id,
                            'p_value': p_value,
                            'interpretation': f"Topic {topic_id} shows association with ethnicity (p={p_value:.3f})"
                        })

    print(f"    ✓ Found {len(findings)} statistically significant associations")
    return findings

# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def load_and_clean_data(csv_path: Path) -> pd.DataFrame:
    """Load CSV and apply all cleaning transformations."""
    print(f"\n{'='*70}")
    print("STEP 1: LOADING AND CLEANING DATA")
    print(f"{'='*70}")

    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
        keep_default_na=True,
        dtype={"Reference": "string"},
        engine="python",
        on_bad_lines="warn",
    )

    print(f"  ✓ Loaded {len(df):,} rows")

    # Normalize headers
    df.columns = [normalise_header(c) for c in df.columns]

    # Early preprocessing
    df = early_drop_preprocessing(df)

    # Drop and rename columns
    present_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=present_to_drop, errors="ignore")
    df = df.rename(columns=RENAME_MAP)
    df = df.loc[:, ~df.columns.duplicated()]

    # Parse dates
    for col in ["First Received", "Due Date", "Completed Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce").dt.normalize()

    # Clean Location
    if "Location" in df.columns:
        df["Location"] = (
            df["Location"]
            .astype("string")
            .str.replace(r"\s*\([^)]*\)", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # Override W&C Exec Team
    df = override_exec_team_wc(df)

    # Override Exec Team based on CDG mapping
    df = override_exec_team_by_cdg(df)

    # Normalize Type
    if "Type" in df.columns:
        df["Type"] = df["Type"].apply(clean_type_value)

    # Merge Complexity columns
    if "Complexity 2" in df.columns or "Complexity 1" in df.columns:
        df["Complexity"] = df.get("Complexity 2", pd.Series(pd.NA, index=df.index))
        if "Complexity 1" in df.columns:
            df["Complexity"] = df["Complexity"].fillna(df["Complexity 1"])
        df["Complexity"] = df["Complexity"].astype("string").str.strip()
        df = df.drop(columns=[c for c in ["Complexity 1", "Complexity 2"] if c in df.columns])

    # Ensure ID is string
    if "ID" in df.columns:
        df["ID"] = df["ID"].astype("string")

    print(f"  ✓ Cleaned data: {len(df):,} rows remaining")
    return df

def get_reporting_period(df):
    """Prompt user for reporting period."""
    print(f"\n{'='*70}")
    print("REPORTING PERIOD SELECTION")
    print(f"{'='*70}")

    # Show available data range
    dates = df['First Received'].dropna()
    if len(dates) > 0:
        min_date = dates.min()
        max_date = dates.max()
        years = sorted(dates.dt.year.unique())
        print(f"\nData available: {min_date.strftime('%B %Y')} to {max_date.strftime('%B %Y')}")
        print(f"Years in dataset: {', '.join(map(str, years))}")

        # Count by year
        print("\nComplaints by year:")
        for year in years:
            count = (dates.dt.year == year).sum()
            print(f"  {year}: {count:,} complaints")

    print("\nThis report analyzes 3 complete months of complaints.")
    print("Please enter the END MONTH of your reporting period.")
    print("Example: For Sept-Oct-Nov 2025, enter 'November 2025'")

    while True:
        end_month_str = input("\nEnter reporting end month (e.g., 'November 2024'): ").strip()

        if not end_month_str:
            print("  ✗ Please enter a date")
            continue

        # Try parsing the input
        for fmt in ["%B %Y", "%b %Y", "%m/%Y", "%m-%Y"]:
            try:
                end_month = pd.to_datetime(end_month_str, format=fmt)

                # Validate that we got a valid date
                if pd.isna(end_month):
                    continue

                # Get last day of that month
                end_date = (end_month + pd.offsets.MonthEnd(0)).normalize()

                # Calculate start date (3 months back)
                start_date = (end_date - pd.DateOffset(months=2)).replace(day=1).normalize()

                # Calculate previous period for comparison (6 months total needed)
                prev_end_date = (start_date - pd.Timedelta(days=1)).normalize()
                prev_start_date = (prev_end_date - pd.DateOffset(months=2)).replace(day=1).normalize()

                print(f"\n✓ Reporting period confirmed:")
                print(f"  Current period:  {start_date.strftime('%B %Y')} - {end_date.strftime('%B %Y')}")
                print(f"  Previous period: {prev_start_date.strftime('%B %Y')} - {prev_end_date.strftime('%B %Y')}")

                return {
                    'current_start': start_date,
                    'current_end': end_date,
                    'previous_start': prev_start_date,
                    'previous_end': prev_end_date,
                    'label_current': f"{start_date.strftime('%b %Y')}-{end_date.strftime('%b %Y')}",
                    'label_previous': f"{prev_start_date.strftime('%b %Y')}-{prev_end_date.strftime('%b %Y')}"
                }
            except (ValueError, TypeError, AttributeError) as e:
                continue

        print("  ✗ Could not parse date. Try format like: November 2025")

def filter_and_prepare_data(df: pd.DataFrame, periods: dict) -> tuple:
    """Filter data to reporting periods and add calculated fields."""
    print(f"\n{'='*70}")
    print("STEP 2: FILTERING TO REPORTING PERIODS")
    print(f"{'='*70}")

    # Filter to 6-month window
    total_start = periods['previous_start']
    total_end = periods['current_end']

    df_filtered = df[
        (df['First Received'] >= total_start) &
        (df['First Received'] <= total_end)
    ].copy()

    print(f"  ✓ Filtered to {len(df_filtered):,} complaints in 6-month window")

    # Warn if suspiciously low
    percentage_captured = len(df_filtered) / len(df) * 100 if len(df) > 0 else 0
    if percentage_captured < 10 and len(df) > 500:
        print(f"  ⚠ WARNING: Only captured {percentage_captured:.1f}% of total data ({len(df_filtered):,} / {len(df):,})")
        print(f"  ⚠ Check you entered the correct year (e.g., 'November 2025' not 'November 2024')")

    # Add derived fields using the end of current period as "today"
    report_date = periods['current_end']
    df_filtered = add_closed_and_deadline(df_filtered, report_date)
    df_filtered = add_six_months_flag(df_filtered, report_date)

    # Add period labels
    df_filtered['Period'] = 'Previous'
    current_mask = (df_filtered['First Received'] >= periods['current_start'])
    df_filtered.loc[current_mask, 'Period'] = 'Current'

    # Add month column
    df_filtered['Month'] = df_filtered['First Received'].dt.to_period('M')

    # Split into current and previous
    df_current = df_filtered[df_filtered['Period'] == 'Current'].copy()
    df_previous = df_filtered[df_filtered['Period'] == 'Previous'].copy()

    print(f"  ✓ Current period:  {len(df_current):,} complaints")
    print(f"  ✓ Previous period: {len(df_previous):,} complaints")

    return df_filtered, df_current, df_previous

def perform_topic_modeling(df_current: pd.DataFrame) -> dict:
    """
    Perform NMF topic modeling - both trust-wide and HPEG-specific.

    Returns:
        Dictionary with topic models and distributions
    """
    print(f"\n{'='*70}")
    print("STEP 3: TOPIC MODELING (NMF)")
    print(f"{'='*70}")

    # Load spaCy model (try multiple models in order of preference)
    print("  Loading spaCy English model...")
    nlp = None
    for model_name in ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]:
        try:
            nlp = spacy.load(model_name)
            print(f"  ✓ Loaded spaCy model: {model_name}")
            break
        except OSError:
            continue

    if nlp is None:
        print("\n  ✗ ERROR: No spaCy English model found!")
        print("  Please run one of:")
        print("    python -m spacy download en_core_web_sm")
        print("    python -m spacy download en_core_web_lg")
        sys.exit(1)

    # Preprocess text
    descriptions = df_current['Description'].fillna('').astype(str)
    processed_texts = preprocess_text_for_nmf(descriptions, nlp)

    # TRUST-WIDE MODEL
    print("\n  Fitting trust-wide topic model...")
    trust_model, trust_vectorizer, trust_doc_topics, trust_features = fit_nmf_model(
        processed_texts, n_topics=10
    )
    trust_topics = extract_topic_keywords(trust_model, trust_features, n_keywords=5)

    print("  ✓ Trust-wide topics identified:")
    for topic in trust_topics:
        print(f"    Topic {topic['id']}: {topic['label']}")

    # Calculate HPEG distributions within trust-wide topics
    hpeg_distributions = {}
    for hpeg in ['BHH Exec Team', 'QEH Exec Team', 'GHH Exec Team', 'SH Exec Team', 'W&C Exec Team', 'CSS Exec Team']:
        dist = calculate_topic_distributions(trust_doc_topics, df_current, 'Exec Team', hpeg)
        hpeg_distributions[hpeg] = dist

    # HPEG-SPECIFIC MODELS
    print("\n  Fitting HPEG-specific topic models...")
    hpeg_models = {}

    for hpeg in ['BHH Exec Team', 'QEH Exec Team', 'GHH Exec Team', 'SH Exec Team', 'W&C Exec Team', 'CSS Exec Team']:
        hpeg_df = df_current[df_current['Exec Team'] == hpeg]

        if len(hpeg_df) < 30:  # Need minimum complaints for meaningful topics
            print(f"    ⚠ {hpeg}: Only {len(hpeg_df)} complaints - skipping HPEG-specific model")
            continue

        hpeg_desc = hpeg_df['Description'].fillna('').astype(str)
        hpeg_processed = preprocess_text_for_nmf(hpeg_desc, nlp)

        # Fit model (fewer topics for smaller datasets)
        n_topics_hpeg = min(8, len(hpeg_df) // 10)
        model, vectorizer, doc_topics, features = fit_nmf_model(
            hpeg_processed, n_topics=n_topics_hpeg
        )
        topics = extract_topic_keywords(model, features, n_keywords=5)

        hpeg_models[hpeg] = {
            'model': model,
            'vectorizer': vectorizer,
            'topics': topics,
            'doc_topics': doc_topics
        }

        print(f"    ✓ {hpeg}: {n_topics_hpeg} topics identified")

    return {
        'trust_wide': {
            'model': trust_model,
            'vectorizer': trust_vectorizer,
            'topics': trust_topics,
            'doc_topics': trust_doc_topics,
            'features': trust_features,
            'hpeg_distributions': hpeg_distributions
        },
        'hpeg_specific': hpeg_models
    }

def analyze_topic_performance(df_current: pd.DataFrame, doc_topics: np.ndarray, topics: list) -> dict:
    """
    Analyze performance metrics by topic for actionable insights.

    Links topics to:
    - Resolution times (median days to close)
    - Complexity distribution
    - CDG/Specialty associations

    Returns priority-ranked topics for service improvement.
    """
    print("\n  Analyzing topic performance metrics...")

    # Assign dominant topic to each complaint
    dominant_topics = doc_topics.argmax(axis=1)
    df_with_topics = df_current.copy()
    df_with_topics['dominant_topic'] = dominant_topics

    topic_performance = []

    for topic_idx, topic in enumerate(topics):
        topic_complaints = df_with_topics[df_with_topics['dominant_topic'] == topic_idx]

        if len(topic_complaints) == 0:
            continue

        # Resolution time analysis (only for closed cases)
        closed_complaints = topic_complaints[topic_complaints['Closed?'] == 'Yes']
        if len(closed_complaints) > 0 and 'Resolution Days' in closed_complaints.columns:
            median_resolution = closed_complaints['Resolution Days'].median()
            resolution_count = len(closed_complaints)
        else:
            median_resolution = None
            resolution_count = 0

        # Complexity distribution
        complexity_dist = {}
        if 'Complexity' in topic_complaints.columns:
            complexity_counts = topic_complaints['Complexity'].value_counts()
            total = len(topic_complaints)
            for comp in ['Basic', 'Regular', 'Complex']:
                complexity_dist[comp] = (complexity_counts.get(comp, 0) / total * 100) if total > 0 else 0

        # Top CDGs for this topic
        top_cdgs = []
        if 'CDG' in topic_complaints.columns:
            cdg_counts = topic_complaints['CDG'].value_counts().head(3)
            top_cdgs = [{'cdg': cdg, 'count': count} for cdg, count in cdg_counts.items()]

        # Top specialties for this topic
        top_specialties = []
        if 'Specialty' in topic_complaints.columns:
            spec_counts = topic_complaints['Specialty'].value_counts().head(3)
            top_specialties = [{'specialty': spec, 'count': count} for spec, count in spec_counts.items()]

        topic_performance.append({
            'topic_id': topic['id'],
            'topic_label': topic['label'],
            'keywords': topic['keywords'],
            'complaint_count': len(topic_complaints),
            'prevalence_pct': len(topic_complaints) / len(df_current) * 100,
            'median_resolution_days': median_resolution,
            'resolution_count': resolution_count,
            'complexity_dist': complexity_dist,
            'top_cdgs': top_cdgs,
            'top_specialties': top_specialties
        })

    print(f"  ✓ Performance analysis complete for {len(topic_performance)} topics")

    return topic_performance

def calculate_topic_priorities(hpeg_performance: dict, trust_avg_dist: np.ndarray) -> list:
    """
    Calculate priority scores for topics based on deviation and performance.

    Priority = (Deviation from trust * 0.5) + (Resolution time percentile * 0.5)

    Returns:
        list: Priority-ranked topics (descending by score) with actionable recommendations.
              Each topic dict includes priority_score, priority_level, priority_color, and recommendation.
    """
    print("\n  Calculating topic priorities...")

    # Get HPEG distribution
    hpeg_dist = hpeg_performance.get('topic_distribution', trust_avg_dist)
    topic_perf = hpeg_performance.get('topic_performance', [])

    priorities = []

    # Calculate resolution time percentiles
    resolution_times = [t['median_resolution_days'] for t in topic_perf
                       if t['median_resolution_days'] is not None]

    for topic in topic_perf:
        topic_idx = topic['topic_id'] - 1  # Convert to 0-indexed

        # Deviation from trust average
        hpeg_weight = hpeg_dist[topic_idx] * 100 if topic_idx < len(hpeg_dist) else 0
        trust_weight = trust_avg_dist[topic_idx] * 100 if topic_idx < len(trust_avg_dist) else 0
        deviation = hpeg_weight - trust_weight
        deviation_score = min(abs(deviation) / 10, 1.0)  # Normalize to 0-1

        # Resolution time score
        resolution_score = 0.0
        if topic['median_resolution_days'] is not None and len(resolution_times) > 0:
            percentile = sum(1 for rt in resolution_times if rt <= topic['median_resolution_days']) / len(resolution_times)
            resolution_score = percentile

        # Combined priority score
        priority_score = (deviation_score * 0.5) + (resolution_score * 0.5)

        # Determine priority level
        if priority_score > 0.7:
            priority_level = "CRITICAL"
            priority_color = "#AE2573"  # NHS Pink
        elif priority_score > 0.4:
            priority_level = "MONITOR"
            priority_color = "#005EB8"  # NHS Blue
        else:
            priority_level = "MAINTAIN"
            priority_color = "#009639"  # NHS Green

        # Generate recommendation
        if priority_level == "CRITICAL":
            recommendation = f"Immediate review of {topic['topic_label']} processes"
        elif priority_level == "MONITOR":
            recommendation = f"Monitor {topic['topic_label']} trends and consider preventive action"
        else:
            recommendation = f"Continue current approach for {topic['topic_label']}"

        priorities.append({
            **topic,
            'hpeg_prevalence': hpeg_weight,
            'trust_prevalence': trust_weight,
            'deviation': deviation,
            'deviation_direction': '↑' if deviation > 0 else '↓' if deviation < 0 else '→',
            'priority_score': priority_score,
            'priority_level': priority_level,
            'priority_color': priority_color,
            'recommendation': recommendation
        })

    # Sort by priority score descending
    priorities.sort(key=lambda x: x['priority_score'], reverse=True)

    print(f"  ✓ Priority analysis complete")
    print(f"    CRITICAL: {sum(1 for p in priorities if p['priority_level'] == 'CRITICAL')}")
    print(f"    MONITOR: {sum(1 for p in priorities if p['priority_level'] == 'MONITOR')}")
    print(f"    MAINTAIN: {sum(1 for p in priorities if p['priority_level'] == 'MAINTAIN')}")

    return priorities

def generate_narrative_insights(df_current: pd.DataFrame, hpeg_name: str) -> list:
    """
    Generate narrative insights about significant trends.

    Identifies:    - CDGs with significant changes month-over-month
    - Top subjects driving those changes
    - Generates natural language statements

    Returns list of insight dictionaries with 'text' and 'priority' (high/medium/low)
    """
    insights = []

    if len(df_current) == 0 or 'CDG' not in df_current.columns:
        return insights

    # Get monthly breakdown by CDG
    cdg_monthly = df_current.groupby(['CDG', 'Month']).size().unstack(fill_value=0)

    if cdg_monthly.empty or len(cdg_monthly.columns) < 2:
        return insights

    # Calculate for each CDG
    for cdg in cdg_monthly.index:
        if pd.isna(cdg) or cdg == '':
            continue

        # Get monthly values
        monthly_values = cdg_monthly.loc[cdg].sort_index()

        if len(monthly_values) < 2:
            continue

        # Latest month vs previous months average
        latest_month = monthly_values.index[-1]
        latest_value = monthly_values.iloc[-1]
        previous_months = monthly_values.iloc[:-1]
        previous_avg = previous_months.mean()

        # Skip if no meaningful data
        if previous_avg == 0 and latest_value == 0:
            continue

        # Calculate change
        if previous_avg > 0:
            pct_change = ((latest_value - previous_avg) / previous_avg) * 100
            abs_change = latest_value - previous_avg
        else:
            pct_change = 100 if latest_value > 0 else 0
            abs_change = latest_value

        # Threshold: significant if meets one of these criteria:
        # - High volume (>10 cases in latest month) with >15% change
        # - Medium volume (5-10 cases) with >25% change
        # - Any volume with >5 case absolute change
        # - Low volume (<5 cases) with >50% change AND >3 case change

        is_high_volume = latest_value > 10 and abs(pct_change) > 15
        is_medium_volume = 5 <= latest_value <= 10 and abs(pct_change) > 25
        is_large_absolute = abs(abs_change) >= 5
        is_dramatic_small = latest_value < 5 and abs(pct_change) > 50 and abs(abs_change) >= 3

        is_significant = is_high_volume or is_medium_volume or is_large_absolute or is_dramatic_small

        if not is_significant:
            continue

        # Find top subjects for this CDG in latest month
        cdg_latest_data = df_current[
            (df_current['CDG'] == cdg) &
            (df_current['Month'] == latest_month)
        ]

        if len(cdg_latest_data) == 0 or 'Subjects' not in cdg_latest_data.columns:
            continue

        top_subjects = cdg_latest_data['Subjects'].value_counts().head(3)

        if len(top_subjects) == 0:
            continue

        # Generate narrative
        direction = "increase" if abs_change > 0 else "decrease"
        month_name = latest_month.strftime('%B %Y')

        # Build subject text - capitalize first letter of first subject
        subjects_list = []
        for idx, (subj, count) in enumerate(top_subjects.head(3).items()):
            if idx == 0:
                # Capitalize first subject
                subjects_list.append(subj)
            else:
                subjects_list.append(subj.lower())

        if len(subjects_list) == 1:
            subject_text = f"relating to {subjects_list[0]}"
        elif len(subjects_list) == 2:
            subject_text = f"relating to {subjects_list[0]} and {subjects_list[1]}"
        else:
            subject_text = f"relating to {subjects_list[0]}, {subjects_list[1]}, and {subjects_list[2]}"

        # Enhanced narrative with month name
        narrative = (
            f"{cdg} saw a {direction} in complaints in {month_name} "
            f"({int(latest_value)} cases vs {previous_avg:.1f} average in previous months, "
            f"{'+' if abs_change > 0 else ''}{int(abs_change)} cases), "
            f"{subject_text}."
        )

        # Determine priority
        if abs(abs_change) >= 10:
            priority = 'high'
        elif abs(abs_change) >= 5:
            priority = 'medium'
        else:
            priority = 'low'

        # Generate actionable recommendation
        if abs_change > 0:
            # Increase - recommend investigation
            if priority == 'high':
                action = "Recommend immediate deep-dive review with clinical leads to identify root causes"
            elif priority == 'medium':
                action = "Monitor closely and schedule review if trend continues next month"
            else:
                action = "Continue monitoring - early trend indication"
        else:
            # Decrease - recommend sharing learnings
            if priority == 'high':
                action = "Significant improvement - recommend documenting interventions for replication"
            else:
                action = "Positive trend - continue current approach"

        insights.append({
            'text': narrative,
            'priority': priority,
            'cdg': cdg,
            'change': abs_change,
            'pct_change': pct_change,
            'action': action,
            'latest_value': int(latest_value),
            'previous_avg': previous_avg
        })

    # Sort by absolute change (most significant first)
    insights.sort(key=lambda x: abs(x['change']), reverse=True)

    return insights

def calculate_metrics_by_hpeg(df_current: pd.DataFrame, df_previous: pd.DataFrame, df_all: pd.DataFrame) -> dict:
    """
    Calculate all required metrics for each HPEG.

    Args:
        df_current: Cases received in current 3-month period
        df_previous: Cases received in previous 3-month period
        df_all: ALL cases (for calculating ongoing 6+ month cases regardless of received date)
    """
    print(f"\n{'='*70}")
    print("STEP 4: CALCULATING METRICS BY HPEG")
    print(f"{'='*70}")

    hpegs = ['BHH Exec Team', 'QEH Exec Team', 'GHH Exec Team', 'SH Exec Team', 'W&C Exec Team', 'CSS Exec Team']
    metrics = {}

    for hpeg in hpegs:
        print(f"\n  Processing {hpeg}...")

        # Filter data for this HPEG
        current = df_current[df_current['Exec Team'] == hpeg].copy()
        previous = df_previous[df_previous['Exec Team'] == hpeg].copy()
        all_hpeg = df_all[df_all['Exec Team'] == hpeg].copy()

        # Basic counts from current period
        total_current = len(current)
        total_previous = len(previous)

        # CRITICAL: Ongoing and 6-month cases should be ALL ongoing cases for this HPEG
        # Not just cases received in current period
        all_ongoing = all_hpeg[all_hpeg['Closed?'] == 'Ongoing']
        ongoing = len(all_ongoing)  # ALL ongoing cases for this HPEG
        six_month_cases = (all_ongoing['6 months'] == 'Yes').sum()
        twelve_month_cases = (all_ongoing['12 months'] == 'Yes').sum()

        # Calculate changes
        change_absolute = total_current - total_previous
        change_percent = (change_absolute / total_previous * 100) if total_previous > 0 else 0

        # Resolution rate
        closed_current = (current['Closed?'] == 'Yes').sum()
        resolution_rate = (closed_current / total_current * 100) if total_current > 0 else 0

        # Deadline compliance
        deadline_met = (current['Deadline Status'] == 'Deadline Met').sum()
        deadline_compliance = (deadline_met / total_current * 100) if total_current > 0 else 0

        # Monthly breakdown for current period
        monthly_counts = current.groupby('Month').size().to_dict()

        # Monthly breakdown by status (for Slide 1 stacked chart)
        monthly_by_status = current.groupby(['Month', 'Closed?']).size().unstack(fill_value=0).to_dict('index')

        # Top 5 locations
        top_locations = current['Location'].value_counts().head(5).to_dict()

        # Top 5 specialties
        top_specialties = current['Specialty'].value_counts().head(5).to_dict()

        # Top 5 CDGs
        top_cdgs = current['CDG'].value_counts().head(5).to_dict() if 'CDG' in current.columns else {}

        # Subject analysis - get all subjects that were top 5 in ANY of the 3 months
        all_top_subjects = set()
        for month in current['Month'].unique():
            month_data = current[current['Month'] == month]
            top_5_month = month_data['Subjects'].value_counts().head(5).index.tolist()
            all_top_subjects.update(top_5_month)

        # Build subject monthly counts
        subject_monthly = {}
        for subject in all_top_subjects:
            subject_monthly[subject] = current[current['Subjects'] == subject].groupby('Month').size().to_dict()

        # Safeguarding (clean "None" values)
        if 'Potential Safeguarding' in current.columns:
            sg_clean = current['Potential Safeguarding'].astype('string').str.strip().str.lower()
            sg_clean = sg_clean.replace({'none': pd.NA, '': pd.NA})
            safeguarding_cases = sg_clean.notna().sum()
        else:
            safeguarding_cases = 0

        # Complexity breakdown - show ALL ongoing cases, not just current period
        complexity_dist = all_ongoing['Complexity'].value_counts().to_dict() if 'Complexity' in all_ongoing.columns else {}

        # Generate narrative insights
        narrative_insights = generate_narrative_insights(current, hpeg)

        # Calculate resolution time metrics for CLOSED cases in current period
        closed_current = current[current['Closed?'] == 'Yes'].copy()
        if len(closed_current) > 0 and 'Resolution Days' in closed_current.columns:
            resolution_mean = closed_current['Resolution Days'].mean()
            resolution_median = closed_current['Resolution Days'].median()

            # By complexity
            resolution_by_complexity = {}
            targets_met_by_complexity = {}
            for complexity in ['Basic', 'Regular', 'Complex']:
                comp_data = closed_current[closed_current['Complexity'] == complexity]
                if len(comp_data) > 0:
                    resolution_by_complexity[complexity] = {
                        'mean': comp_data['Resolution Days'].mean(),
                        'median': comp_data['Resolution Days'].median(),
                        'count': len(comp_data)
                    }
                    # Calculate % meeting target
                    met_target = (comp_data['Met Target'] == 'Yes').sum()
                    targets_met_by_complexity[complexity] = (met_target / len(comp_data) * 100) if len(comp_data) > 0 else 0
        else:
            resolution_mean = 0
            resolution_median = 0
            resolution_by_complexity = {}
            targets_met_by_complexity = {}

        metrics[hpeg] = {
            'total_current': total_current,
            'total_previous': total_previous,
            'change_absolute': change_absolute,
            'change_percent': change_percent,
            'ongoing': ongoing,
            'resolution_rate': resolution_rate,
            'deadline_compliance': deadline_compliance,
            'monthly_counts': monthly_counts,
            'monthly_by_status': monthly_by_status,
            'top_locations': top_locations,
            'top_specialties': top_specialties,
            'top_cdgs': top_cdgs,
            'subject_monthly': subject_monthly,
            'six_month_cases': six_month_cases,
            'twelve_month_cases': twelve_month_cases,
            'safeguarding_cases': safeguarding_cases,
            'complexity_dist': complexity_dist,
            'narrative_insights': narrative_insights,
            'resolution_mean': resolution_mean,
            'resolution_median': resolution_median,
            'resolution_by_complexity': resolution_by_complexity,
            'targets_met_by_complexity': targets_met_by_complexity,
            'dataframe_current': current,
            'dataframe_previous': previous
        }

        print(f"    ✓ Total: {total_current} | Ongoing: {ongoing} | Resolution: {resolution_rate:.1f}%")

    return metrics

def generate_12month_trends(df: pd.DataFrame) -> dict:
    """
    Generate 12-month trend data for resolution times and volumes.

    Returns dict with:
    - resolution_by_cdg_monthly: median days by CDG per month
    - resolution_by_specialty_monthly: median days by top specialties per month
    - volume_monthly: total complaints per month
    """
    print(f"\n{'='*70}")
    print("GENERATING 12-MONTH TRENDS")
    print(f"{'='*70}")

    # Filter to closed cases with resolution days
    closed = df[(df['Closed?'] == 'Yes') & (df['Resolution Days'] > 0)].copy()

    if len(closed) == 0:
        print("  ⚠ No closed cases with resolution times")
        return {
            'resolution_by_cdg_monthly': {},
            'resolution_by_specialty_monthly': {},
            'volume_monthly': {}
        }

    # Extract month from Completed Date
    closed['Completion Month'] = closed['Completed Date'].dt.to_period('M')

    # CDG monthly median resolution times
    cdg_monthly = closed.groupby(['CDG', 'Completion Month'])['Resolution Days'].median().unstack(fill_value=0)

    # Specialty monthly median - get top specialties first
    top_specialties = closed['Specialty'].value_counts().head(10).index.tolist()
    specialty_monthly = closed[closed['Specialty'].isin(top_specialties)].groupby(['Specialty', 'Completion Month'])['Resolution Days'].median().unstack(fill_value=0)

    # Volume trends (all complaints, not just closed)
    if 'Month' in df.columns:
        volume_monthly = df.groupby('Month').size().to_dict()
    else:
        # Use First Received month
        df_copy = df.copy()
        df_copy['Received Month'] = df_copy['First Received'].dt.to_period('M')
        volume_monthly = df_copy.groupby('Received Month').size().to_dict()

    print(f"  ✓ Generated trends for {len(cdg_monthly)} CDGs and {len(specialty_monthly)} specialties")

    return {
        'resolution_by_cdg_monthly': cdg_monthly.to_dict('index'),
        'resolution_by_specialty_monthly': specialty_monthly.to_dict('index'),
        'volume_monthly': volume_monthly,
        'top_specialties': top_specialties
    }

def main():
    """Main execution pipeline."""
    print("\n" + "="*70)
    print(" HPEG DATA PROCESSOR")
    print(" NHS Complaints Analysis - Script 1 of 2")
    print("="*70)

    # Check input file exists
    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        print(f"\n✗ ERROR: Input CSV not found at: {INPUT_CSV}")
        print("Please update the INPUT_CSV path at the top of this script.")
        sys.exit(1)

    # Load and clean data
    df = load_and_clean_data(input_path)

    # Get reporting period (show user what years are available)
    periods = get_reporting_period(df)

    # Add Closed? and 6 months columns to FULL dataset (needed for ALL ongoing cases check)
    report_date = periods['current_end']
    df = add_closed_and_deadline(df, report_date)
    df = add_six_months_flag(df, report_date)

    # Add resolution time metrics (business days)
    df = add_resolution_time_metrics(df)

    # Filter and prepare (df_filtered = 6 months, df = full dataset)
    df_filtered, df_current, df_previous = filter_and_prepare_data(df, periods)

    # Topic modeling
    topic_models = perform_topic_modeling(df_current)

    # Topic performance analysis (links topics to actionable insights)
    print(f"\n{'='*70}")
    print("STEP 3B: TOPIC PERFORMANCE & PRIORITY ANALYSIS")
    print(f"{'='*70}")

    trust_doc_topics = topic_models['trust_wide']['doc_topics']
    trust_topics = topic_models['trust_wide']['topics']
    trust_avg_dist = np.mean(list(topic_models['trust_wide']['hpeg_distributions'].values()), axis=0)

    hpeg_topic_analysis = {}
    for hpeg in ['BHH Exec Team', 'QEH Exec Team', 'GHH Exec Team', 'SH Exec Team', 'W&C Exec Team', 'CSS Exec Team']:
        hpeg_df = df_current[df_current['Exec Team'] == hpeg].copy()

        if len(hpeg_df) < 10:
            print(f"  ⚠ {hpeg}: Insufficient data ({len(hpeg_df)} complaints), skipping topic analysis")
            continue

        # Get HPEG-specific topic distribution from original indices
        hpeg_indices = hpeg_df.index

        # Get corresponding rows from trust_doc_topics
        # Need to map DataFrame indices to doc_topics array indices
        original_index_map = {idx: i for i, idx in enumerate(df_current.index)}
        hpeg_array_indices = [original_index_map[idx] for idx in hpeg_indices if idx in original_index_map]
        hpeg_doc_topics = trust_doc_topics[hpeg_array_indices]

        # Analyze performance
        topic_perf = analyze_topic_performance(hpeg_df, hpeg_doc_topics, trust_topics)

        # Calculate priorities
        hpeg_dist = topic_models['trust_wide']['hpeg_distributions'][hpeg]
        priorities = calculate_topic_priorities(
            {'topic_distribution': hpeg_dist, 'topic_performance': topic_perf},
            trust_avg_dist
        )

        hpeg_topic_analysis[hpeg] = priorities

        # Display top 3 priorities
        critical_count = sum(1 for p in priorities if p['priority_level'] == 'CRITICAL')
        monitor_count = sum(1 for p in priorities if p['priority_level'] == 'MONITOR')
        print(f"    ✓ {hpeg}: {critical_count} CRITICAL, {monitor_count} MONITOR priorities identified")

    print(f"  ✓ Topic performance analysis complete for {len(hpeg_topic_analysis)} HPEGs")

    # Calculate metrics (pass full df for 6+ month calculation)
    hpeg_metrics = calculate_metrics_by_hpeg(df_current, df_previous, df)

    # Demographic analysis (using trust-wide topics)
    print(f"\n{'='*70}")
    print("STEP 5: DEMOGRAPHIC ANALYSIS")
    print(f"{'='*70}")

    # Create topic assignments
    trust_doc_topics = topic_models['trust_wide']['doc_topics']
    dominant_topics = trust_doc_topics.argmax(axis=1) + 1  # +1 for 1-indexed
    topics_df = pd.DataFrame({
        'dominant_topic': dominant_topics
    }, index=df_current.index)

    demographic_findings = run_demographic_analysis(df_current, topics_df)

    # Generate 12-month trends from full dataset
    trends_12month = generate_12month_trends(df)

    # Package everything for saving
    print(f"\n{'='*70}")
    print("STEP 6: SAVING PROCESSED DATA")
    print(f"{'='*70}")

    output_data = {
        'periods': periods,
        'topic_models': topic_models,
        'topic_analysis': hpeg_topic_analysis,
        'hpeg_metrics': hpeg_metrics,
        'demographic_findings': demographic_findings,
        'trends_12month': trends_12month,
        'df_current': df_current,
        'df_previous': df_previous,
        'colors': NHS_COLORS,
        'metadata': {
            'processed_date': datetime.now().isoformat(),
            'total_complaints': len(df_current),
            'hpegs': list(hpeg_metrics.keys())
        }
    }

    output_path = Path(PROCESSED_DATA_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"\n  ✓ Processed data saved to: {output_path}")
    print(f"  ✓ File size: {output_path.stat().st_size / (1024*1024):.1f} MB")

    print(f"\n{'='*70}")
    print(" DATA PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"\nSummary:")
    print(f"  • Current period:  {periods['label_current']}")
    print(f"  • Previous period: {periods['label_previous']}")
    print(f"  • Total complaints analyzed: {len(df_current):,}")
    print(f"  • Trust-wide topics identified: {len(topic_models['trust_wide']['topics'])}")
    print(f"  • HPEG-specific models: {len(topic_models['hpeg_specific'])}")
    print(f"  • Topic priority analysis: {len(hpeg_topic_analysis)} HPEGs")
    print(f"  • Demographic findings: {len(demographic_findings)}")

    print(f"\nNext step: Run hpeg_report_generator.py to create PowerPoint reports")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Process cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
