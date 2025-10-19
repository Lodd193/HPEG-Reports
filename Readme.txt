Hospital Patient Experience Group (HPEG) Report Automator – UHB

Automates the monthly Hospital Patient Experience Group (HPEG) reports for University Hospitals Birmingham (UHB).
The script ingests exported data, cleans and formats it, handles nulls, analyses metadata and key indicators, generates standardised NHS-branded charts, and uses an LLM API to draft concise narratives. It then assembles charts, tables, and narrative into a PDF ready for circulation.


✨ Key Features

Data prep: column selection, type casting, date normalisation, null handling, value standardisation.

Metadata analysis: coverage, recency, completeness, schema checks.

KPIs & analysis: complaints/PALS volume, subjects/sub-subjects, trends, top issues, missed deadlines reasons, breakdowns by Site, Executive Team, and CDG.

NHS-branded charts: consistent palette, typography, and labelling (applied across all figures).

Narrative generation: LLM-powered 2–4 sentence executive summary per chart/table.

PDF output: collated charts, tables, and narratives with front page and contents.

Reproducible runs: parameterised by month and file paths; deterministic config.


🗂️ Inputs & Outputs

Inputs

Primary CSV/Excel export(s) from Radar/BI (see data/).

Optional lookups (Site, Exec Team, CDG) in reference/ as CSV/XLSX.

Config file (config.yaml) controlling report period, filters, and branding.

Outputs

/output/HPEG_YYYY-MM_Report.pdf – final report.

/output/artifacts/ – PNG charts, CSV tables, and run logs.


🏗️ Project Structure
.
├─ src/
│  ├─ main.py                 # entry point
│  ├─ io_loader.py            # data load + schema checks
│  ├─ clean.py                # cleaning, typing, nulls
│  ├─ analyse.py              # KPIs, aggregations
│  ├─ charts.py               # NHS-styled charting
│  ├─ narrative.py            # LLM summaries
│  ├─ assemble_pdf.py         # PDF generation
│  └─ utils.py                # helpers (dates, logging, config)
├─ config.yaml                # period, paths, branding, kpis
├─ .env.example               # API keys & runtime settings
├─ requirements.txt
├─ README.md
├─ data/                      # input files (excluded in .gitignore)
└─ output/                    # reports and artifacts (gitignored)


🎨 NHS Branding (standardised)

NHS Blue: #005EB8 (primary)

NHS Dark Blue: #003087 (titles/accents)

Support greys: #425563 (dark), #768692 (mid), #E8EDEE (light)

Accessible highlight set (optional for categorical series):

#005EB8, #003087, #0072CE, #4C8EDA, #7FB2E5, #9BCAEB

Typography: Segoe UI / Arial (fallback), consistent font sizes, clear axis labels, value annotations where helpful.

Rules: Title-case chart titles; sentence-case axis labels; include units; consistent legend placement; data source + period noted.

The charting module enforces this palette and layout for all figures to ensure visual consistency.


⚙️ Configuration

config.yaml (example)

report:
  month: "2025-09"           # reporting month (YYYY-MM)
  timezone: "Europe/London"
  include_sections:
    - volume_over_time
    - by_site
    - by_exec_team
    - by_cdg
    - subjects_top10
    - subsubjects_top10
    - deadlines_status
    - reopen_reasons
paths:
  input: "data/input.csv"
  lookups:
    exec_team: "reference/exec_team_lookup.csv"
    cdg: "reference/cdg_lookup.csv"
  output_dir: "output"
branding:
  palette:
    primary: "#005EB8"
    dark: "#003087"
    grey_dark: "#425563"
    grey_mid: "#768692"
    grey_light: "#E8EDEE"
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  max_tokens: 200
  temperature: 0.2
security:
  persist_raw_text: false
  redact_pii: true


.env.example

OPENAI_API_KEY=your_key_here
# or other provider keys as required


🔍 What the Pipeline Does

Load: reads CSV/XLSX, validates schema, coerces types, normalises dates.

Clean: drops unused cols, renames to canonical labels, fills nulls (rule-based).

Analyse:

Volumes by month/week/day.

Breakdown by Site / Exec Team / CDG.

Subjects & sub-subjects (Top-N).

Deadline status & reasons for missed deadlines.

Reopened complaints counts and proportions.

Visualise: renders NHS-branded charts (line, bar, stacked, Pareto as relevant).

Narrate: calls LLM to produce short, executive-friendly blurbs per figure/table.

Assemble: compiles PDF with front page, contents, sections, and appendix.


🔒 Data Protection & Governance

No raw text retention: configurable persist_raw_text=false enforces that only derived aggregates and charts are stored.

PII handling: optional pre-processing to redact identifiers before analysis.

Secrets: API keys via .env; never commit secrets to git.

Auditability: each run logs config, hash of inputs, and generated artifacts.

Standards: align with PHSO Complaint Standards, PSIRF learning ethos, and internal UHB governance; where relevant, ensure compliance activities under DCB0129/DCB0160 are tracked separately (hazard log, clinical safety case, change control).


🧪 Testing

Unit tests for cleaning rules and aggregations (pytest).

Snapshot tests for chart objects (image diff optional).

Schema tests: verify expected columns and types before analysis.

Run:

pytest -q


🛠️ Tech Stack

Python: pandas, numpy, matplotlib / plotnine / seaborn (locked palette), pydantic (config).

PDF: reportlab / weasyprint / matplotlib backends.

LLM: OpenAI API (pluggable provider via narrative.py).

CLI: argparse / Typer.

CI (optional): GitHub Actions to lint, test, and build sample report.


🗺️ Roadmap

Parameterised section toggles and site-specific report variants.

Drill-down tables and appendix by CDG/sub-subject.

Automatic trend flags (improving/worsening) and RAG thresholds.

Accessibility checks for chart contrast and alt text in PDF.

Caching for large datasets and reproducible seeds for LLM calls.


🤝 Contributing

Create an issue describing the change.

Branch from main using feature/<short-description>.

Add tests for behaviour changes.

Submit a Pull Request linking the issue.


📄 License

Copyright © University Hospitals Birmingham.
Internal use only unless otherwise agreed.


📬 Contact

Richard Lodder, Patient Relations Quality & Performance Manager (UHB)