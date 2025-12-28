# QA Template Analyzer

Analyzes multiple Excel QA templates to identify patterns, consistency, and data quality issues.

## Quick Start

1. **Run the analyzer:**
   - Double-click `run_qa_analyzer.bat`
   - OR manually: `python qa_template_analyzer.py`

2. **View results:**
   - Console output shows summary
   - `qa_analysis_report.json` contains detailed analysis

## Files

- `qa_template_analyzer.py` - Main analysis script
- `usage_example.py` - Custom usage examples
- `requirements.txt` - Python dependencies
- `run_qa_analyzer.bat` - Windows batch runner

## Features

- **Multi-file processing**: Analyzes all Excel files in directory
- **Sheet analysis**: Examines structure across all worksheets
- **Column mapping**: Identifies common/unique columns
- **Data quality**: Reports errors and inconsistencies
- **Flexible output**: Console + JSON reports

## Customization

Edit the `main()` function in `qa_template_analyzer.py` to:
- Change target directory
- Filter specific file patterns
- Modify analysis parameters

## Output

The script generates:
- Console summary with key metrics
- `qa_analysis_report.json` with detailed findings
- Error reporting for problematic files