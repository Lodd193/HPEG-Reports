@echo off
echo Installing required packages...
pip install pandas openpyxl xlrd

echo.
echo Running QA Template Analyzer...
python qa_template_analyzer.py

echo.
echo Analysis complete. Check qa_analysis_report.json for detailed results.
pause