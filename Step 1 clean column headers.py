### Step 1 - Import libraries and load data

import re
import pandas as pd

FILE = r"C:\Users\lod19\OneDrive\Desktop\The BIG project\report PII redacted.csv"
df = pd.read_csv(FILE)

print("Original columns:")
print(df.columns)


### Step 2 - Rename columns by keeping only the text inside the OUTERMOST brackets
# Works with (), [], {}, <> and handles nested brackets like "(Responsible Hospital Exec Team (Main))"

def extract_outer_bracket_text(s: str) -> str:
    pairs = [('(', ')'), ('[', ']'), ('{', '}'), ('<', '>')]
    for left, right in pairs:
        i = s.find(left)
        j = s.rfind(right)
        if i != -1 and j != -1 and j > i:
            out = s[i+1:j]
            # Tidy: drop leading punctuation like "--", collapse spaces, trim
            out = re.sub(r'^[^\w]+', '', out)
            out = re.sub(r'\s+', ' ', out).strip()
            return out if out else s
    # No bracket pair found → keep original
    return s

new_cols = [extract_outer_bracket_text(c) for c in df.columns]

# OPTIONAL: remove a trailing " (Main)" token if present (comment out to keep it)
new_cols = [re.sub(r'\s*\(main\)\s*$', '', c, flags=re.IGNORECASE) for c in new_cols]

df.columns = new_cols

print("\nRenamed columns (after bracket extraction):")
print(df.columns)


### Step 3 - Analyse data types and missing values

print("\nDTypes:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isna().sum())
