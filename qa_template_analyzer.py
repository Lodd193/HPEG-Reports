import pandas as pd
import os
import glob
from pathlib import Path
import json
from datetime import datetime

class QATemplateAnalyzer:
    def __init__(self, template_directory=None):
        self.template_directory = template_directory or "."
        self.templates_data = []
        self.analysis_results = {}
    
    def find_excel_files(self, pattern="*.xlsx"):
        """Find all Excel files matching the pattern in the directory"""
        search_path = os.path.join(self.template_directory, "**", pattern)
        excel_files = glob.glob(search_path, recursive=True)
        return excel_files
    
    def read_template(self, file_path):
        """Read a single Excel template and extract all sheets"""
        try:
            template_data = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'sheets': {},
                'metadata': {
                    'file_size': os.path.getsize(file_path),
                    'modified_date': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                }
            }
            
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    template_data['sheets'][sheet_name] = {
                        'shape': df.shape,
                        'columns': list(df.columns),
                        'non_null_counts': df.count().to_dict(),
                        'data_types': df.dtypes.astype(str).to_dict(),
                        'sample_data': df.head(3).to_dict('records') if not df.empty else []
                    }
                except Exception as e:
                    template_data['sheets'][sheet_name] = {'error': str(e)}
            
            return template_data
        except Exception as e:
            return {'file_path': file_path, 'error': str(e)}
    
    def ingest_templates(self, file_paths=None):
        """Ingest multiple templates"""
        if file_paths is None:
            file_paths = self.find_excel_files()
        
        print(f"Found {len(file_paths)} Excel files to process...")
        
        for file_path in file_paths:
            print(f"Processing: {os.path.basename(file_path)}")
            template_data = self.read_template(file_path)
            self.templates_data.append(template_data)
        
        return self.templates_data
    
    def analyze_templates(self):
        """Perform analysis on all ingested templates"""
        if not self.templates_data:
            print("No templates loaded. Run ingest_templates() first.")
            return
        
        analysis = {
            'summary': {
                'total_files': len(self.templates_data),
                'successful_reads': sum(1 for t in self.templates_data if 'error' not in t),
                'failed_reads': sum(1 for t in self.templates_data if 'error' in t)
            },
            'sheet_analysis': {},
            'column_analysis': {},
            'data_quality': {}
        }
        
        # Analyze sheets across all templates
        all_sheet_names = set()
        sheet_frequency = {}
        
        for template in self.templates_data:
            if 'sheets' in template:
                for sheet_name in template['sheets'].keys():
                    all_sheet_names.add(sheet_name)
                    sheet_frequency[sheet_name] = sheet_frequency.get(sheet_name, 0) + 1
        
        analysis['sheet_analysis'] = {
            'unique_sheet_names': list(all_sheet_names),
            'sheet_frequency': sheet_frequency,
            'common_sheets': [name for name, freq in sheet_frequency.items() if freq > len(self.templates_data) * 0.5]
        }
        
        # Analyze columns across templates
        all_columns = set()
        column_frequency = {}
        
        for template in self.templates_data:
            if 'sheets' in template:
                for sheet_name, sheet_data in template['sheets'].items():
                    if 'columns' in sheet_data:
                        for col in sheet_data['columns']:
                            col_key = f"{sheet_name}::{col}"
                            all_columns.add(col_key)
                            column_frequency[col_key] = column_frequency.get(col_key, 0) + 1
        
        analysis['column_analysis'] = {
            'total_unique_columns': len(all_columns),
            'column_frequency': column_frequency,
            'common_columns': [col for col, freq in column_frequency.items() if freq > len(self.templates_data) * 0.5]
        }
        
        # Data quality analysis
        quality_issues = []
        for template in self.templates_data:
            if 'error' in template:
                quality_issues.append({
                    'file': template['file_path'],
                    'issue': 'Failed to read file',
                    'details': template['error']
                })
            elif 'sheets' in template:
                for sheet_name, sheet_data in template['sheets'].items():
                    if 'error' in sheet_data:
                        quality_issues.append({
                            'file': template['file_path'],
                            'sheet': sheet_name,
                            'issue': 'Failed to read sheet',
                            'details': sheet_data['error']
                        })
        
        analysis['data_quality'] = {
            'total_issues': len(quality_issues),
            'issues': quality_issues
        }
        
        self.analysis_results = analysis
        return analysis
    
    def generate_report(self, output_file=None):
        """Generate a comprehensive report"""
        if not self.analysis_results:
            self.analyze_templates()
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_results': self.analysis_results,
            'detailed_templates': self.templates_data
        }
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"Report saved to: {output_file}")
        
        return report
    
    def print_summary(self):
        """Print a summary of the analysis"""
        if not self.analysis_results:
            self.analyze_templates()
        
        analysis = self.analysis_results
        
        print("\n" + "="*50)
        print("QA TEMPLATE ANALYSIS SUMMARY")
        print("="*50)
        
        print(f"\nFiles Processed:")
        print(f"  Total files: {analysis['summary']['total_files']}")
        print(f"  Successfully read: {analysis['summary']['successful_reads']}")
        print(f"  Failed to read: {analysis['summary']['failed_reads']}")
        
        print(f"\nSheet Analysis:")
        print(f"  Unique sheet names: {len(analysis['sheet_analysis']['unique_sheet_names'])}")
        print(f"  Common sheets: {analysis['sheet_analysis']['common_sheets']}")
        
        print(f"\nColumn Analysis:")
        print(f"  Total unique columns: {analysis['column_analysis']['total_unique_columns']}")
        print(f"  Most common columns:")
        sorted_cols = sorted(analysis['column_analysis']['column_frequency'].items(), 
                           key=lambda x: x[1], reverse=True)
        for col, freq in sorted_cols[:5]:
            print(f"    {col}: {freq} files")
        
        if analysis['data_quality']['total_issues'] > 0:
            print(f"\nData Quality Issues: {analysis['data_quality']['total_issues']}")
            for issue in analysis['data_quality']['issues'][:3]:
                print(f"  - {issue['file']}: {issue['issue']}")
        else:
            print("\nNo data quality issues found!")

def main():
    # Example usage
    template_dir = r"C:\Users\lod19\OneDrive\Desktop\QA1 Test"
    
    analyzer = QATemplateAnalyzer(template_dir)
    
    # Ingest all Excel templates
    analyzer.ingest_templates()
    
    # Perform analysis
    analyzer.analyze_templates()
    
    # Print summary
    analyzer.print_summary()
    
    # Generate detailed report
    report_file = "qa_analysis_report.json"
    analyzer.generate_report(report_file)

if __name__ == "__main__":
    main()