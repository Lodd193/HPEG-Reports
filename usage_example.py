from qa_template_analyzer import QATemplateAnalyzer

# Example 1: Analyze templates in your QA1 Test directory
analyzer = QATemplateAnalyzer(r"C:\Users\lod19\OneDrive\Desktop\QA1 Test")
analyzer.ingest_templates()
analyzer.print_summary()

# Example 2: Analyze specific files
specific_files = [
    r"C:\Users\lod19\OneDrive\Desktop\QA1 Test\QA1 Quality Check Template v1.0.xlsx"
]
analyzer2 = QATemplateAnalyzer()
analyzer2.ingest_templates(specific_files)
analyzer2.print_summary()

# Example 3: Generate detailed JSON report
analyzer.generate_report("detailed_qa_analysis.json")

# Example 4: Access raw data for custom analysis
for template in analyzer.templates_data:
    print(f"File: {template['file_name']}")
    if 'sheets' in template:
        for sheet_name, sheet_data in template['sheets'].items():
            if 'columns' in sheet_data:
                print(f"  Sheet '{sheet_name}' has {len(sheet_data['columns'])} columns")
                print(f"  Columns: {', '.join(sheet_data['columns'][:5])}...")
    print()