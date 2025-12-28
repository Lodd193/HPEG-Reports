# HPEG Automated Reporting System

## Project Brief

**Version:** 1.0  
**Date:** 28 December 2025  
**Author:** Richard (Patient Relations Quality & Performance Manager)  
**Purpose:** Comprehensive specification for automated monthly HPEG reporting with topic modelling and demographic analysis

---

## 1. Executive Summary

This project delivers an automated monthly reporting system for Healthcare Patient Experience Group (HPEG) meetings at University Hospitals Birmingham NHS Foundation Trust. The system will process 12 months of rolling complaints data, perform topic modelling to identify emerging themes, and generate six standardised presentation-ready reports—one for each HPEG group.

The core value proposition is transforming a manual process (currently taking hours per report) into an automated pipeline that surfaces actionable insights, flags statistically significant deviations, and enables rapid response to emerging complaint patterns.

---

## 2. Business Context

### 2.1 Current State
- Manual report creation taking hours per HPEG
- Limited ability to detect emerging themes or deviations
- No systematic demographic cross-tabulation
- Inconsistent report structure across groups

### 2.2 Target State
- Single-command report generation for all 6 HPEGs
- Automated topic discovery with deviation flagging
- Demographic analysis identifying potential inequities
- Standardised, presentation-ready outputs

### 2.3 Strategic Alignment
This project sits alongside (but separate from) the broader Patient Feedback Intelligence Platform using DeBERTa for HCAT classification. The HCAT project faces longer operationalisation timelines due to NHS governance requirements; this HPEG reporting system can deliver immediate value while that work progresses.

---

## 3. Technical Requirements

### 3.1 Constraints
| Constraint | Detail |
|------------|--------|
| **Environment** | Fully offline—no cloud services, no API calls |
| **Platform** | Python, VSCode, local execution |
| **Dependencies** | All packages must work offline (pip installable) |
| **Data sensitivity** | PII-free dataset available for development; production data stays local |

### 3.2 Source Data

**System:** Radar (complaints management system)  
**Format:** Single CSV export  
**Scope:** 12-month rolling window, trust-wide  
**Volume:** ~200 complaints/month (~2,400 rows total)  
**Frequency:** Monthly extract

### 3.3 Key Data Fields

| Field | Purpose |
|-------|---------|
| `First Received` | Date field for filtering current month and trends |
| `Due Date` | Deadline calculation |
| `Completed Date` | Closure status |
| `Exec Team` | HPEG assignment (BHH, QEH, GHH, SH, W&C, CSS, Corporate) |
| `CDG` | Clinical Delivery Group breakdown |
| `Specialty` | Service-level granularity |
| `Description` | Free-text complaint narrative (topic modelling input) |
| `Gender` | Demographic field (reliable coverage) |
| `Ethnicity` | Demographic field (reliable coverage) |
| `Subjects` | KO41a-aligned category |
| `Sub-subjects` | KO41a-aligned subcategory |
| `Deadline Met` / `Deadline Status` | Performance tracking |
| `Potential Safeguarding` | Keyword-flagged concerns |

**Note:** Age field has ~0.05% coverage and should be excluded from analysis.

---

## 4. Scope

### 4.1 In Scope
- 6 HPEG reports: BHH, QEH, GHH, SH, W&C, CSS
- Monthly automated generation
- Topic modelling with deviation detection
- Demographic cross-tabulation
- Three output formats for testing: PowerPoint, PDF (via HTML), interactive HTML

### 4.2 Out of Scope
- Corporate complaints (excluded from HPEG reporting)
- Real-time processing
- Web interface
- Integration with other systems (DeBERTa/HCAT platform)

---

## 5. Report Specification

### 5.1 Report Sections (Each HPEG)

Each of the 6 reports follows an identical structure:

| # | Section | Content |
|---|---------|---------|
| 1 | **Executive Summary** | Key numbers: received, closed, ongoing, deadline performance % |
| 2 | **Volume Trends** | 12-month line chart with current month highlighted |
| 3 | **Deadline Performance** | Met/Missed/In Progress breakdown with trend |
| 4 | **CDG Breakdown** | Bar chart of volumes by CDG within this HPEG |
| 5 | **Specialty Breakdown** | Table or chart of top specialties by volume |
| 6 | **Subject & Sub-subject Analysis** | Category distribution |
| 7 | **Demographic Profile** | Gender/ethnicity distribution; cross-tab with top themes |
| 8 | **Topic Analysis** | This month's topics vs previous month & annual average; deviations flagged |
| 9 | **Long-standing Cases** | List of complaints open 6+ months (131+ business days) |

### 5.2 Visual Standards

- **Colour scheme:** NHS branding
  - Primary: `#005EB8` (NHS Blue)
  - Accent: `#7C2855` (Burgundy)
  - Supporting palette from NHS identity guidelines
- **Charts:** Strategic purpose only—no decorative visualisations
- **Narrative:** Minimal auto-generated text; presenter will talk to the data

---

## 6. Topic Modelling Specification

### 6.1 Algorithm

**NMF (Non-negative Matrix Factorisation)** selected over LDA for:
- Better interpretability with short texts
- More coherent topic clusters
- Fully offline operation

### 6.2 Preprocessing Pipeline

1. Lowercase normalisation
2. PII removal (Presidio patterns if needed, though dataset is PII-free)
3. Lemmatisation (spaCy)
4. Domain stopword removal: NHS, hospital, patient, complaint, trust, staff, etc.
5. TF-IDF vectorisation

### 6.3 Model Architecture

| Component | Specification |
|-----------|---------------|
| **Trust baseline** | Fit on full 12-month corpus; extract 8–12 topics |
| **Topic count** | Determined empirically; coherence score optimisation |
| **HPEG comparison** | Calculate topic distribution per HPEG; compare to trust average |
| **Temporal deviation** | Compare this month to: (a) previous month, (b) 12-month rolling average |
| **Deviation threshold** | **1 standard deviation** flags as significant |

### 6.4 Topic Labelling

**Auto-generate from keywords**—no manual labelling required.

Output format per topic:
- Topic ID
- Top 5 keywords
- Representative example complaint (redacted if needed)
- Deviation indicator (↑ above threshold / ↓ below threshold / — stable)

### 6.5 Demographic Cross-tabulation

Statistical tests to identify whether certain topics are over-represented in specific demographic groups:
- Chi-square or proportion test
- Flag significant associations (p < 0.05)
- Present as cross-tab table with highlighting

---

## 7. Derived Fields & Calculations

### 7.1 Status Fields

| Field | Logic |
|-------|-------|
| `Closed?` | `Yes` if `Completed Date` is populated |
| `Deadline Status` | `Met` / `Missed` / `In Progress` based on `Due Date` vs `Completed Date` |
| `6 months` | Flag if open > 131 business days |

### 7.2 Exec Team Normalisation

Map variations to canonical names:

```python
EXEC_TEAMS = {
    "BHH": "BHH Exec Team",
    "QEH": "QEHB Exec Team",
    "GHH": "GHH Exec Team",
    "SH": "SH Exec Team",
    "W&C": "W&C Exec Team",
    "WC": "W&C Exec Team",
    "CSS": "CSS Exec Team",
    "CORP": "Corporate",
    "CORPORATE": "Corporate",
}
```

### 7.3 Safeguarding Detection

Keyword-based flagging in `Description` field. Carry forward existing patterns from previous scripts.

---

## 8. Project Structure

```
hpeg-reports/
│
├── config/
│   ├── settings.yaml           # Paths, date ranges, thresholds
│   ├── exec_team_mapping.yaml  # Specialty → CDG → Exec Team
│   ├── subjects.yaml           # Canonical subject/sub-subject lists
│   └── stopwords.yaml          # Domain-specific stopwords for NMF
│
├── src/
│   ├── __init__.py
│   ├── ingest.py               # Load CSV, validate columns, filter date range
│   ├── transform.py            # Normalise fields, calculated columns, QA checks
│   ├── metrics.py              # Volume, deadline, demographic aggregations
│   ├── topics.py               # NMF modelling, deviation detection
│   ├── demographics.py         # Cross-tabulation and statistical tests
│   └── report.py               # Generate outputs (one per HPEG)
│
├── templates/
│   ├── hpeg_report.pptx        # PowerPoint master template
│   ├── hpeg_report.html        # Jinja2 HTML template
│   └── styles.css              # NHS-branded styles
│
├── outputs/
│   └── YYYY-MM/                # Monthly output folders
│       ├── BHH_HPEG_MonYYYY.pptx
│       ├── BHH_HPEG_MonYYYY.pdf
│       ├── BHH_HPEG_MonYYYY.html
│       └── ...
│
├── data/
│   └── raw/                    # Drop monthly CSV here
│
├── tests/
│   ├── test_ingest.py
│   ├── test_transform.py
│   ├── test_topics.py
│   └── sample_data.csv         # PII-free test dataset
│
├── run_report.py               # Single entry point
├── requirements.txt
└── README.md
```

---

## 9. Pipeline Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  ingest.py  │ ──▶ │ transform.py│ ──▶ │  metrics.py │
│  Load CSV   │     │  Normalise  │     │  Aggregate  │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
       ┌───────────────────────────────────────┤
       │                                       │
       ▼                                       ▼
┌─────────────┐                       ┌──────────────┐
│  topics.py  │                       │demographics.py│
│  NMF model  │                       │  Cross-tabs  │
└─────────────┘                       └──────────────┘
       │                                       │
       └───────────────┬───────────────────────┘
                       ▼
               ┌─────────────┐
               │  report.py  │
               │ Generate x6 │
               └─────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
     .pptx           .pdf          .html
```

---

## 10. Output Formats

All three formats will be generated for evaluation:

| Format | Library | Pros | Cons |
|--------|---------|------|------|
| **PowerPoint** | python-pptx | Editable, familiar, presenter notes | Less flexible styling |
| **PDF** | WeasyPrint (HTML→PDF) | Clean rendering, consistent | Not editable |
| **HTML** | Jinja2 + Plotly | Interactive charts, flexible | Requires browser |

---

## 11. Dependencies

```
# Core
pandas>=2.0
numpy>=1.24

# NLP / Topic Modelling
scikit-learn>=1.3       # NMF, TF-IDF
spacy>=3.6              # Lemmatisation
# Download: python -m spacy download en_core_web_sm

# Visualisation
matplotlib>=3.7
plotly>=5.15

# Report Generation
python-pptx>=0.6.21     # PowerPoint
jinja2>=3.1             # HTML templating
weasyprint>=59          # HTML → PDF

# Statistical Testing
scipy>=1.11             # Chi-square tests

# Configuration
pyyaml>=6.0
```

---

## 12. Usage

### 12.1 Single Command Execution

```bash
python run_report.py --input data/raw/complaints_dec2025.csv --month 2025-12
```

### 12.2 Expected Output

```
Processing: complaints_dec2025.csv
Loaded 2,412 complaints (12-month window)
Current month: December 2025 (198 new complaints)

Fitting trust-wide topic model...
  Optimal topics: 10
  Coherence score: 0.42

Generating HPEG reports...
  ✓ BHH_HPEG_Dec2025 (3 formats)
  ✓ QEH_HPEG_Dec2025 (3 formats)
  ✓ GHH_HPEG_Dec2025 (3 formats)
  ✓ SH_HPEG_Dec2025 (3 formats)
  ✓ W&C_HPEG_Dec2025 (3 formats)
  ✓ CSS_HPEG_Dec2025 (3 formats)

Reports saved to: outputs/2025-12/
```

---

## 13. Development Approach

### 13.1 Phased Build

| Phase | Deliverable | Validation |
|-------|-------------|------------|
| 1 | `config/` + `ingest.py` | CSV loads correctly, columns validated |
| 2 | `transform.py` | Derived fields calculate correctly |
| 3 | `metrics.py` | Aggregations match manual spot-checks |
| 4 | `topics.py` | Topics are coherent and interpretable |
| 5 | `demographics.py` | Statistical tests produce sensible results |
| 6 | `report.py` | All 3 output formats render correctly |
| 7 | Integration | Full pipeline runs end-to-end |

### 13.2 Test Data

A full 12-month PII-free dataset is available for development and testing.

---

## 14. Success Criteria

1. **Functional:** Single command generates all 6 reports in all 3 formats
2. **Accurate:** Metrics match manual calculations within 1%
3. **Interpretable:** Topics are coherent; auto-labels are meaningful
4. **Actionable:** Deviations correctly flagged at 1 SD threshold
5. **Efficient:** Full run completes in < 5 minutes on standard hardware

---

## 15. Future Considerations

These items are explicitly **out of scope** for v1 but noted for potential future development:

- Integration with DeBERTa/HCAT classification platform
- Web interface for ad-hoc queries
- Automated email distribution of reports
- Historical trend database
- Predictive modelling (complaints as leading indicator for incidents)

---

## 16. Reference: Existing Code Assets

Previous scripts contain useful patterns that can be adapted:

| Script | Useful Elements |
|--------|-----------------|
| HPEG Data Processing | Column normalisation, Exec Team mapping, Subject validation, Safeguarding keywords |
| HPEG Visualization | NHS colour scheme, Matplotlib styling, Date parsing patterns |

---

## 17. Questions Resolved

| Question | Decision |
|----------|----------|
| Output format | Test all 3 (PowerPoint, PDF, HTML) |
| Topic labelling | Auto-generate from keywords |
| Deviation threshold | 1 standard deviation |
| Run frequency | Monthly, all groups |
| Safeguarding | Carry forward keyword detection |
| Demographics | Gender + ethnicity; exclude age (poor coverage) |

---

## Appendix A: NHS Colour Palette

```yaml
primary:
  nhs_blue: "#005EB8"
  
secondary:
  dark_blue: "#003087"
  bright_blue: "#0072CE"
  light_blue: "#41B6E6"
  aqua_green: "#00A499"
  
accent:
  purple: "#330072"
  pink: "#AE2573"
  burgundy: "#7C2855"
  
neutral:
  black: "#231F20"
  dark_grey: "#425563"
  mid_grey: "#768692"
  pale_grey: "#E8EDEE"
```

---

## Appendix B: Sample Report Section - Topic Analysis

```
TOPIC ANALYSIS - BHH HPEG - December 2025

Trust-wide topics (fitted on 2,412 complaints):

  Topic 1: "Communication"
  Keywords: information, told, explained, understand, contact
  This month: 23% ↑ (vs 18% annual avg) — DEVIATION FLAGGED
  
  Topic 2: "Waiting Times"  
  Keywords: wait, hours, delayed, appointment, cancelled
  This month: 15% — (stable)
  
  Topic 3: "Staff Attitude"
  Keywords: rude, dismissive, uncaring, manner, spoke
  This month: 12% — (stable)

Demographic cross-tabulation:
  "Communication" topic over-represented in:
    - Female complainants (p=0.03)
    - White British ethnicity (p=0.04)
```

---

*End of Project Brief*
