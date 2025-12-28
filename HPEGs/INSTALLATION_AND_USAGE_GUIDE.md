# HPEG Automated Reporting System
## Installation and Usage Guide

**Version:** 1.0
**Author:** Patient Relations Quality & Performance Manager
**Date:** December 2024

---

## Overview

This system automates HPEG complaints reporting using two Python scripts:

1. **`hpeg_data_processor.py`** - Processes data, performs topic modeling, calculates metrics
2. **`hpeg_report_generator.py`** - Generates PowerPoint presentations (one per HPEG)

---

## ONE-TIME INSTALLATION (At Work Computer)

### Step 1: Install Required Python Packages

Open Command Prompt or Terminal and run:

```bash
pip install pandas numpy scipy scikit-learn spacy matplotlib python-pptx
```

### Step 2: Download SpaCy Language Model

This is needed for topic modeling (runs 100% offline):

```bash
python -m spacy download en_core_web_sm
```

**Security Note:** SpaCy is completely offline. No data leaves your computer.

---

## CONFIGURING FILE PATHS

Before first use, you MUST update the file paths in both scripts:

### In `hpeg_data_processor.py` (Lines 50-51):

```python
# === USER CONFIGURATION - EDIT THESE PATHS ===
INPUT_CSV = r"C:/path/to/your/radar_export.csv"
PROCESSED_DATA_PATH = r"C:/path/to/save/processed_data.pkl"
```

### In `hpeg_report_generator.py` (Lines 41-42):

```python
# === USER CONFIGURATION - EDIT THESE PATHS ===
PROCESSED_DATA_PATH = r"C:/path/to/save/processed_data.pkl"  # Same as above!
OUTPUT_FOLDER = r"C:/path/to/save/reports"
```

**Important:** The `PROCESSED_DATA_PATH` must be the SAME in both scripts!

---

## MONTHLY WORKFLOW

### Step 1: Export Data from Radar

1. Export 12-month rolling complaints data from Radar
2. Save as CSV (e.g., `radar_export_dec2024.csv`)
3. Place in your designated input folder

### Step 2: Run Data Processor

Open Command Prompt, navigate to the folder containing the scripts, and run:

```bash
python hpeg_data_processor.py
```

**What happens:**
- Loads and cleans the CSV
- **Prompts you for reporting period** (e.g., "November 2024")
  - Enter the END MONTH of your 3-month reporting period
  - Example: If you want Sept-Oct-Nov 2024, enter "November 2024"
- Performs NMF topic modeling (takes 2-5 minutes)
- Calculates all metrics
- Saves processed data to pickle file

**Expected output:**
```
======================================================================
 DATA PROCESSING COMPLETE
======================================================================
Summary:
  • Current period:  Sep 2024-Nov 2024
  • Previous period: Jun 2024-Aug 2024
  • Total complaints analyzed: 1,247
  • Trust-wide topics identified: 10
  • HPEG-specific models: 6
  • Demographic findings: 3

Next step: Run hpeg_report_generator.py to create PowerPoint reports
======================================================================
```

### Step 3: Generate PowerPoint Reports

```bash
python hpeg_report_generator.py
```

**What happens:**
- Loads processed data
- Generates all 6 HPEG PowerPoint presentations automatically
- Saves to your designated output folder

**Expected output:**
```
======================================================================
 REPORT GENERATION COMPLETE
======================================================================
Generated 6 reports:
  ✓ BHH_Exec_Team_Complaints_Sep-Nov_2024.pptx
  ✓ QEH_Exec_Team_Complaints_Sep-Nov_2024.pptx
  ✓ GHH_Exec_Team_Complaints_Sep-Nov_2024.pptx
  ✓ SH_Exec_Team_Complaints_Sep-Nov_2024.pptx
  ✓ W&C_Exec_Team_Complaints_Sep-Nov_2024.pptx
  ✓ CSS_Exec_Team_Complaints_Sep-Nov_2024.pptx

All reports saved to: C:/path/to/save/reports
======================================================================
```

---

## REPORT STRUCTURE

Each PowerPoint contains 10 slides:

| Slide | Title | Content |
|-------|-------|---------|
| 1 | Executive Dashboard | KPIs + 6-month trend + alerts |
| 2 | Where Are Complaints Coming From? | Top 5 Locations + Top 5 Specialties |
| 3 | What's Changing? - Subject Trends | Rolling 3-month subject analysis |
| 4 | Topic Intelligence - Hidden Patterns | NMF topics with deviation indicators |
| 5 | Performance Metrics | Deadline compliance + complexity |
| 6 | Risk Dashboard | Long-standing cases + safeguarding |
| 7 | Demographic Insights | Statistical findings (conditional) |
| 8 | 10 Oldest Cases | **BLANK - Manual completion required (PII)** |
| 9 | Actions Status | **BLANK - Manual completion required** |
| 10 | Current Performance & Key Achievements | **BLANK - Manual completion required** |

**Note:** Slides 8-10 are intentionally blank templates for HPEG leads to complete manually.

---

## TROUBLESHOOTING

### Error: "No module named 'sklearn'"
**Solution:** Run `pip install scikit-learn`

### Error: "spaCy model 'en_core_web_sm' not found"
**Solution:** Run `python -m spacy download en_core_web_sm`

### Error: "Input CSV not found"
**Solution:** Check the `INPUT_CSV` path in `hpeg_data_processor.py` is correct

### Error: "Processed data not found"
**Solution:**
1. Check `PROCESSED_DATA_PATH` is the same in both scripts
2. Ensure you ran `hpeg_data_processor.py` first

### Script runs but no output
**Solution:** Check your file paths are correct and you have write permissions

### Topic modeling takes too long
**Expected:** 2-5 minutes for ~2,000 complaints. If longer:
1. Check CPU isn't overloaded with other tasks
2. Close unnecessary applications

### Charts look wrong in PowerPoint
**Solution:** This shouldn't happen, but if it does:
1. Check matplotlib version: `pip install --upgrade matplotlib`
2. Ensure python-pptx is installed: `pip install python-pptx`

---

## DATA SECURITY & PRIVACY

**All processing happens locally on your computer:**
- ✓ No cloud services used
- ✓ No API calls to external servers
- ✓ No data transmission
- ✓ SpaCy runs 100% offline
- ✓ All packages are open-source and widely used in NHS

**PII Handling:**
- Scripts process complaint descriptions for topic modeling
- NO patient names, addresses, or identifiable info are included in outputs
- Slide 8 ("10 Oldest Cases") is intentionally blank - you add PII manually only if needed

---

## FILE MANAGEMENT

### What to Keep
- The two Python scripts: `hpeg_data_processor.py` and `hpeg_report_generator.py`
- This guide

### What to Delete (Monthly)
- Old `processed_data.pkl` files (after reports generated)
- Temporary chart images (auto-deleted)
- Old CSV exports (after processing)

### What to Archive
- Generated PowerPoint reports (for historical record)
- Monthly CSV exports (if required for audit)

---

## CUSTOMIZATION

If you need to change settings, look for these sections in the scripts:

### NHS Colors
Defined in both scripts - all use official NHS brand colors.

### Subject Lists
In `hpeg_data_processor.py`, line ~118: `SUBJECTS_CANON` list

### Topic Model Settings
In `hpeg_data_processor.py`, line ~413: `fit_nmf_model()` function
- Default: 10 topics trust-wide, 8 topics per HPEG
- Can adjust `n_topics` parameter

### Chart Styles
In `hpeg_report_generator.py`, functions starting `create_*_chart()`
- All use NHS branding automatically
- Modify only if required by Communications team

---

## GETTING HELP

If you encounter issues:

1. **Check error message** - Most errors tell you exactly what's wrong
2. **Check file paths** - 90% of issues are incorrect paths
3. **Verify packages installed** - Run `pip list` to see installed packages
4. **Check data quality** - Ensure CSV has required columns

---

## TECHNICAL NOTES

### Why Two Scripts?
- **Separation of concerns:** Process once, generate reports multiple times if needed
- **Efficiency:** Topic modeling is slow; chart generation is fast
- **Debugging:** Easier to fix report formatting without reprocessing data

### CSS HPEG Filtering
CSS is trust-wide and filtered by `CDG` starting with "11", not by `Exec Team`.

### W&C HPEG Filtering
W&C includes all CDG 3 cases plus specialty-based overrides (Maternity, Gynaecology, Paediatrics).

### Topic Modeling
- **Trust-wide model:** Fits on ALL complaints, then calculates HPEG distributions
- **HPEG-specific models:** Separate models per HPEG for local patterns
- **Both included:** Trust-wide for comparability, HPEG-specific for detail

---

## VERSION HISTORY

**v1.0 (December 2024)**
- Initial release
- 10-slide format
- Trust-wide + HPEG-specific topic modeling
- Automated 6-month comparison
- NHS-branded PowerPoint output

---

## QUICK REFERENCE

```bash
# One-time setup
pip install pandas numpy scipy scikit-learn spacy matplotlib python-pptx
python -m spacy download en_core_web_sm

# Monthly workflow
python hpeg_data_processor.py      # Enter reporting period when prompted
python hpeg_report_generator.py    # Generates all 6 reports automatically
```

---

**End of Guide**
