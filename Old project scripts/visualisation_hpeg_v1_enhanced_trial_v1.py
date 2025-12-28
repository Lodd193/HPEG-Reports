def chart9_deadline_performance(df, exec_team, output_dir):
    """Chart 9: Deadline Performance Dashboard - Meeting vs Missing targets."""
    print("\nCreating Chart 9: Deadline Performance Analysis...")
    
    if 'Deadline Status' not in df.columns:
        print(f"  ⚠ Warning: 'Deadline Status' column not found. Skipping.")
        return
    
    # Create figure with consistent size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    fig.patch.set_facecolor('white')
    
    # Left plot: Overall deadline status distribution
    deadline_counts = df['Deadline Status'].value_counts()
    
    # Define order and colors for deadline statuses
    status_order = ['Deadline Met', 'Deadline Missed', 'Still in Progress - In Time', 
                   'Still in Progress - Out of Time', 'Unknown']
    status_colors = ['#009639', '#8A1538', '#41B6E6', '#ED8B00', '#768692']
    
    # Prepare data in order
    ordered_counts = []
    ordered_colors = []
    ordered_labels = []
    for status, color in zip(status_order, status_colors):
        if status in deadline_counts.index:
            ordered_counts.append(deadline_counts[status])
            ordered_colors.append(color)
            ordered_labels.append(status)
    
    # Create donut chart
    wedges, texts, autotexts = ax1.pie(ordered_counts, 
                                        labels=ordered_labels,
                                        colors=ordered_colors,
                                        autopct='%1.1f%%',
                                        startangle=90,
                                        pctdistance=0.85,
                                        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
                                        shadow=True)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    # Add center metrics
    total = sum(ordered_counts)
    met_count = deadline_counts.get('Deadline Met', 0)
    performance_rate = (met_count / total * 100) if total > 0 else 0
    
    ax1.text(0, 0, f'{performance_rate:.1f}%\nCompliance', ha='center', va='center', 
            fontsize=16, fontweight='bold', color='#003087')
    
    ax1.set_title('Overall Deadline Performance', fontsize=12, fontweight='bold', color='#003087')
    
    # Right plot: Monthly deadline performance trend
    months = sorted(df['month'].dropna().unique())
    
    monthly_met = []
    monthly_missed = []
    monthly_in_progress = []
    
    for month in months:
        month_data = df[df['month'] == month]
        met = (month_data['Deadline Status'] == 'Deadline Met').sum()
        missed = (month_data['Deadline Status'] == 'Deadline Missed').sum()
        in_progress = month_data['Deadline Status'].str.contains('Still in Progress', na=False).sum()
        
        monthly_met.append(met)
        monthly_missed.append(missed)
        monthly_in_progress.append(in_progress)
    
    x_pos = np.arange(len(months))
    width = 0.25
    
    bars1 = ax2.bar(x_pos - width, monthly_met, width, label='Met', 
                   color='#009639', edgecolor='white', linewidth=1, alpha=0.9)
    bars2 = ax2.bar(x_pos, monthly_missed, width, label='Missed',
                   color='#8A1538', edgecolor='white', linewidth=1, alpha=0.9)
    bars3 = ax2.bar(x_pos + width, monthly_in_progress, width, label='In Progress',
                   color='#41B6E6', edgecolor='white', linewidth=1, alpha=0.9)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                        f'{int(height)}', ha='center', va='center',
                        color='white', fontweight='bold', fontsize=9)
    
    ax2.set_xlabel('Month', fontsize=12, fontweight='bold', color='#425563')
    ax2.set_ylabel('Number of Cases', fontsize=12, fontweight='bold', color='#425563')
    ax2.set_title('Monthly Deadline Performance Trend', fontsize=12, fontweight='bold', color='#003087')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([m.strftime('%b %Y') for m in months], rotation=45, ha='right')
    ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True,
              facecolor='white', edgecolor='#E8EDEE')
    ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    # Main title
    fig.suptitle(f'Deadline Performance Dashboard - {exec_team}',
                fontsize=16, fontweight='bold', color='#003087', y=0.98)
    
    plt.tight_layout()
    output_path = output_dir / f"chart9_deadline_performance_{exec_team.lower().replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")

def chart10_outcome_analysis(df, exec_team, output_dir):
    """Chart 10: Outcome Code Analysis - Understanding resolution patterns."""
    print("\nCreating Chart 10: Outcome Analysis...")
    
    if 'Outcome Code' not in df.columns:
        print(f"  ⚠ Warning: 'Outcome Code' column not found. Skipping.")
        return
    
    # Create figure with consistent size
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor('white')
    
    # Get top 10 outcome codes
    outcome_counts = df['Outcome Code'].value_counts().head(10)
    
    # Create horizontal bar chart for better label visibility
    y_pos = np.arange(len(outcome_counts))
    
    # Use gradient colors
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(outcome_counts)))[::-1]
    
    bars = ax.barh(y_pos, outcome_counts.values, color=colors, 
                   edgecolor='white', linewidth=2, alpha=0.9)
    
    # Add value labels
    for bar, val in zip(bars, outcome_counts.values):
        width = bar.get_width()
        ax.text(width + ax.get_xlim()[1] * 0.01, bar.get_y() + bar.get_height()/2,
               f'{val} ({val/len(df)*100:.1f}%)', ha='left', va='center',
               fontweight='bold', fontsize=10, color='#425563')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(outcome_counts.index, fontsize=11)
    ax.set_xlabel('Number of Cases', fontsize=12, fontweight='bold', color='#425563')
    ax.invert_yaxis()
    
    apply_professional_style(ax, 
                           title=f'Top 10 Outcome Codes - {exec_team}',
                           subtitle='Understanding how cases are resolved')
    
    # Add insight box
    if len(outcome_counts) > 0:
        top_outcome = outcome_counts.index[0]
        top_percentage = outcome_counts.values[0] / len(df) * 100
        insight_text = f'Most Common Outcome:\n{top_outcome}\n({top_percentage:.1f}% of cases)'
        ax.text(0.98, 0.15, insight_text, transform=ax.transAxes,
               ha='right', va='bottom',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8EDEE', 
                        edgecolor='#768692', linewidth=2),
               fontsize=11, fontweight='bold', color='#425563')
    
    plt.tight_layout()
    output_path = output_dir / f"chart10_outcome_analysis_{exec_team.lower().replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")

def chart11_site_comparison(df, exec_team, output_dir):
    """Chart 11: Site Comparison - Performance across different sites."""
    print("\nCreating Chart 11: Site Comparison Analysis...")
    
    if 'Site' not in df.columns:
        print(f"  ⚠ Warning: 'Site' column not found. Skipping.")
        return
    
    # Create figure with consistent size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    fig.patch.set_facecolor('white')
    
    # Get site data
    site_counts = df['Site'].value_counts()
    
    # Left plot: Site volume distribution
    sites = site_counts.index[:8]  # Top 8 sites for clarity
    values = site_counts.values[:8]
    
    colors = CONFIG['colors']['months_palette'][:len(sites)]
    
    bars = ax1.bar(range(len(sites)), values, color=colors,
                   edgecolor='white', linewidth=2, alpha=0.9)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + ax1.get_ylim()[1] * 0.01,
                f'{val}\n({val/len(df)*100:.1f}%)', ha='center', va='bottom',
                fontweight='bold', fontsize=10, color='#425563')
    
    ax1.set_xticks(range(len(sites)))
    ax1.set_xticklabels(sites, rotation=45, ha='right')
    ax1.set_ylabel('Number of Cases', fontsize=12, fontweight='bold', color='#425563')
    apply_professional_style(ax1, title='Case Volume by Site')
    
    # Right plot: Site performance metrics
    if 'Deadline Status' in df.columns and CONFIG['six_months_column'] in df.columns:
        site_metrics = []
        for site in sites[:5]:  # Top 5 sites for detailed metrics
            site_data = df[df['Site'] == site]
            total = len(site_data)
            if total > 0:
                met_rate = (site_data['Deadline Status'] == 'Deadline Met').sum() / total * 100
                six_months_rate = (site_data[CONFIG['six_months_column']] == 'Yes').sum() / total * 100
                site_metrics.append({
                    'Site': site,
                    'Deadline Met %': met_rate,
                    '6 Months %': six_months_rate
                })
        
        if site_metrics:
            metrics_df = pd.DataFrame(site_metrics)
            
            x = np.arange(len(metrics_df))
            width = 0.35
            
            bars1 = ax2.bar(x - width/2, metrics_df['Deadline Met %'], width,
                           label='Deadline Met %', color='#009639',
                           edgecolor='white', linewidth=2, alpha=0.9)
            bars2 = ax2.bar(x + width/2, metrics_df['6 Months %'], width,
                           label='Cases >6 Months %', color='#ED8B00',
                           edgecolor='white', linewidth=2, alpha=0.9)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                            f'{height:.1f}%', ha='center', va='center',
                            color='white', fontweight='bold', fontsize=10)
            
            ax2.set_xticks(x)
            ax2.set_xticklabels(metrics_df['Site'], rotation=45, ha='right')
            ax2.set_ylabel('Percentage', fontsize=12, fontweight='bold', color='#425563')
            ax2.set_ylim(0, max(metrics_df['Deadline Met %'].max(), 
                               metrics_df['6 Months %'].max()) * 1.2)
            apply_professional_style(ax2, title='Site Performance Metrics')
            ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
                      facecolor='white', edgecolor='#E8EDEE')
    else:
        # If performance metrics not available, show ongoing vs closed
        site_ongoing = []
        for site in sites[:5]:
            site_data = df[df['Site'] == site]
            ongoing_rate = site_data['is_ongoing'].sum() / len(site_data) * 100 if len(site_data) > 0 else 0
            site_ongoing.append(ongoing_rate)
        
        bars = ax2.bar(range(len(sites[:5])), site_ongoing, 
                      color='#7C2855', edgecolor='white', linewidth=2, alpha=0.9)
        
        for bar, val in zip(bars, site_ongoing):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
                    f'{val:.1f}%', ha='center', va='center',
                    color='white', fontweight='bold', fontsize=10)
        
        ax2.set_xticks(range(len(sites[:5])))
        ax2.set_xticklabels(sites[:5], rotation=45, ha='right')
        ax2.set_ylabel('% Cases Ongoing', fontsize=12, fontweight='bold', color='#425563')
        apply_professional_style(ax2, title='Ongoing Cases by Site')
    
    # Main title
    fig.suptitle(f'Site Performance Comparison - {exec_team}',
                fontsize=16, fontweight='bold', color='#003087', y=0.98)
    
    plt.tight_layout()
    output_path = output_dir / f"chart11_site_comparison_{exec_team.lower().replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")

def chart12_response_time_analysis(df, exec_team, output_dir):
    """Chart 12: Response Time Analysis - Days to completion."""
    print("\nCreating Chart 12: Response Time Analysis...")
    
    if 'First Received' not in df.columns or 'Completed Date' not in df.columns:
        print(f"  ⚠ Warning: Required date columns not found. Skipping.")
        return
    
    # Calculate response times for completed cases
    completed_df = df[df['Completed Date'].notna()].copy()
    if len(completed_df) == 0:
        print("  No completed cases found. Skipping chart.")
        return
    
    completed_df['Response Days'] = (completed_df['Completed Date'] - completed_df['First Received']).dt.days
    completed_df = completed_df[completed_df['Response Days'] >= 0]  # Remove invalid values
    
    # Create figure with consistent size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    fig.patch.set_facecolor('white')
    
    # Left plot: Response time distribution
    bins = [0, 7, 14, 21, 30, 60, 90, 180, 365, 1000]
    labels = ['0-7', '8-14', '15-21', '22-30', '31-60', '61-90', '91-180', '181-365', '365+']
    
    counts, _ = np.histogram(completed_df['Response Days'], bins=bins)
    
    colors = ['#009639' if i < 3 else '#FFB81C' if i < 5 else '#8A1538' 
             for i in range(len(counts))]
    
    bars = ax1.bar(range(len(counts)), counts, color=colors,
                   edgecolor='white', linewidth=2, alpha=0.9)
    
    # Add value labels
    for bar, val in zip(bars, counts):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{val}\n({val/len(completed_df)*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold', 
                    fontsize=9, color='#425563')
    
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_xlabel('Days to Completion', fontsize=12, fontweight='bold', color='#425563')
    ax1.set_ylabel('Number of Cases', fontsize=12, fontweight='bold', color='#425563')
    apply_professional_style(ax1, title='Response Time Distribution')
    
    # Right plot: Monthly average response time trend
    months = sorted(completed_df['month'].dropna().unique())
    monthly_avg = []
    monthly_median = []
    
    for month in months:
        month_data = completed_df[completed_df['month'] == month]['Response Days']
        if len(month_data) > 0:
            monthly_avg.append(month_data.mean())
            monthly_median.append(month_data.median())
        else:
            monthly_avg.append(0)
            monthly_median.append(0)
    
    x_pos = np.arange(len(months))
    
    line1 = ax2.plot(x_pos, monthly_avg, marker='o', linewidth=2.5, 
                    markersize=8, color='#005EB8', label='Average',
                    markeredgecolor='white', markeredgewidth=2)
    line2 = ax2.plot(x_pos, monthly_median, marker='s', linewidth=2.5,
                    markersize=8, color='#7C2855', label='Median',
                    markeredgecolor='white', markeredgewidth=2)
    
    # Add target line at 30 days
    ax2.axhline(y=30, color='#009639', linestyle='--', linewidth=2, 
               alpha=0.7, label='30-Day Target')
    
    # Add value labels
    for i, (avg, med) in enumerate(zip(monthly_avg, monthly_median)):
        if avg > 0:
            ax2.annotate(f'{avg:.0f}', (i, avg), textcoords="offset points",
                        xytext=(0,5), ha='center', fontsize=8, fontweight='bold')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([m.strftime('%b %Y') for m in months], rotation=45, ha='right')
    ax2.set_xlabel('Month', fontsize=12, fontweight='bold', color='#425563')
    ax2.set_ylabel('Days to Completion', fontsize=12, fontweight='bold', color='#425563')
    apply_professional_style(ax2, title='Monthly Response Time Trend')
    ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True,
              facecolor='white', edgecolor='#E8EDEE')
    
    # Calculate and display KPIs
    avg_response = completed_df['Response Days'].mean()
    median_response = completed_df['Response Days'].median()
    within_30 = (completed_df['Response Days'] <= 30).sum() / len(completed_df) * 100
    
    # Main title with KPIs
    fig.suptitle(f'Response Time Analysis - {exec_team}',
                fontsize=16, fontweight='bold', color='#003087', y=0.98)
    
    kpi_text = (f'Average: {avg_response:.1f} days | Median: {median_response:.1f} days | '
               f'Within 30 days: {within_30:.1f}%')
    fig.text(0.5, 0.94, kpi_text, ha='center', fontsize=11,
            color='#425563', fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / f"chart12_response_time_{exec_team.lower().replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")#!/usr/bin/env python3
"""
HPEG Executive Dashboard - Multi-Chart Visualization Suite
Professional charts for NHS senior executive presentations
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# Set professional font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# NHS Brand Colors and Professional Theme
CONFIG = {
    'csv_encoding': 'utf-8-sig',
    'date_column': 'First Received',
    'exec_team_column': 'Exec Team',
    'location_column': 'Location',
    'specialty_column': 'Specialty',
    'subjects_column': 'Subjects',
    'subsubjects_column': 'Sub-subjects',
    'safeguarding_column': 'Potential Safeguarding',
    'six_months_column': '6 months',
    'complexity_column': 'Complexity',
    'status_columns': ['Deadline Status', 'Deadline Met'],
    'outputs_dir': Path(r"C:\Users\lod19\OneDrive\Desktop\Work Related\HPEGs\Master\2025 - 26\1. August\processed\outputs"),
    'colors': {
        # NHS Brand Colors
        'nhs_blue': '#005EB8',
        'nhs_dark_blue': '#003087',
        'nhs_bright_blue': '#0072CE',
        'nhs_light_blue': '#41B6E6',
        'nhs_aqua': '#00A9CE',
        'nhs_dark_grey': '#425563',
        'nhs_mid_grey': '#768692',
        'nhs_pale_grey': '#E8EDEE',
        'nhs_dark_green': '#006747',
        'nhs_green': '#009639',
        'nhs_light_green': '#78BE20',
        'nhs_purple': '#330072',
        'nhs_dark_pink': '#7C2855',
        'nhs_pink': '#AE2573',
        'nhs_dark_red': '#8A1538',
        'nhs_orange': '#ED8B00',
        'nhs_warm_yellow': '#FFB81C',
        'nhs_yellow': '#FAE100',
        # Chart specific
        'received': '#005EB8',
        'ongoing': '#7C2855',
        'months_palette': ['#005EB8', '#0072CE', '#41B6E6', '#00A9CE', '#009639', 
                          '#78BE20', '#FFB81C', '#ED8B00', '#AE2573', '#7C2855'],
        'pie_palette': ['#005EB8', '#009639', '#ED8B00', '#7C2855', '#00A9CE'],
        'safeguarding': '#8A1538',
        'six_months': '#ED8B00',
        'complexity': ['#009639', '#FFB81C', '#8A1538']  # Green, Amber, Red
    }
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

def apply_professional_style(ax, title="", subtitle="", spine_color='#E8EDEE'):
    """Apply consistent professional styling to axes."""
    # Title formatting
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', color='#003087', pad=20)
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha='center', 
               fontsize=10, color='#425563', style='italic')
    
    # Spine and grid styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(spine_color)
    ax.spines['bottom'].set_color(spine_color)
    ax.tick_params(colors='#425563')
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='#768692')
    ax.set_axisbelow(True)
    
    # Label styling
    ax.xaxis.label.set_color('#425563')
    ax.yaxis.label.set_color('#425563')

def add_value_labels(ax, bars, format_str="{:.0f}", rotation=0, color='white', fontsize=10):
    """Add professional value labels to bars."""
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            label = format_str.format(height)
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   label, ha='center', va='center', rotation=rotation,
                   color=color, fontweight='bold', fontsize=fontsize)

def add_executive_footer(fig, exec_team, date_range_str):
    """Add professional footer with metadata."""
    footer_text = f"Data: {exec_team} | Period: {date_range_str} | Generated: {datetime.now().strftime('%d %B %Y')}"
    fig.text(0.5, 0.01, footer_text, ha='center', va='bottom', 
            fontsize=8, color='#768692', style='italic')

def format_percentage(value, total):
    """Format percentage for display."""
    if total == 0:
        return "0%"
    return f"{value/total*100:.1f}%"

def get_csv_path():
    """Prompt user for CSV file path."""
    while True:
        path_input = input("Enter CSV path (or folder): ").strip().strip('"')
        if not path_input:
            print("Please enter a path.")
            continue
            
        path = Path(path_input)
        if not path.exists():
            print(f"Path not found: {path_input}")
            continue
            
        if path.is_file() and path.suffix.lower() == '.csv':
            return path
            
        if path.is_dir():
            csvs = sorted(path.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
            if not csvs:
                print(f"No CSV files in: {path}")
                continue
                
            if len(csvs) == 1:
                print(f"Using: {csvs[0].name}")
                return csvs[0]
                
            print("\nCSV files found:")
            for i, f in enumerate(csvs[:10], 1):
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {i}. {f.name} ({mtime:%Y-%m-%d %H:%M}, {size_mb:.1f} MB)")
                
            while True:
                try:
                    choice = int(input("Select file number: "))
                    if 1 <= choice <= len(csvs):
                        return csvs[choice - 1]
                    print(f"Enter number 1-{len(csvs)}")
                except ValueError:
                    print("Enter a valid number")

def get_exec_team(df=None):
    """Prompt for exec team selection."""
    if df is not None and CONFIG['exec_team_column'] in df.columns:
        actual_teams = df[CONFIG['exec_team_column']].dropna().unique()
        print("\nTeams found in data:")
        for team in sorted(actual_teams):
            count = (df[CONFIG['exec_team_column']] == team).sum()
            print(f"  - {team}: {count} cases")
    
    options = list(EXEC_TEAMS.keys())
    print(f"\nExec Team shortcuts: {' / '.join(options)}")
    print("Or type the full team name exactly as it appears in the data")
    
    while True:
        team_input = input("\nSelect Exec Team: ").strip()
        
        team_upper = team_input.upper().replace(' ', '')
        if team_upper == 'W&C':
            team_upper = 'WC'
        
        if team_upper in EXEC_TEAMS:
            return EXEC_TEAMS[team_upper]
        
        if df is not None and CONFIG['exec_team_column'] in df.columns:
            if team_input in df[CONFIG['exec_team_column']].values:
                return team_input
        
        print("Invalid team. Try again.")

def get_date_range(df=None):
    """Prompt for date range."""
    if df is not None and CONFIG['date_column'] in df.columns:
        dates = pd.to_datetime(df[CONFIG['date_column']], errors='coerce')
        valid_dates = dates.dropna()
        if len(valid_dates) > 0:
            print(f"\nData date range: {valid_dates.min().date()} to {valid_dates.max().date()}")
    
    def parse_date(date_str):
        formats = ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d"]
        for fmt in formats:
            try:
                return pd.Timestamp(datetime.strptime(date_str, fmt).date())
            except ValueError:
                continue
        return None
    
    while True:
        start_str = input("\nStart date (dd/mm/yyyy or yyyy-mm-dd): ").strip()
        end_str = input("End date (dd/mm/yyyy or yyyy-mm-dd): ").strip()
        
        start_date = parse_date(start_str)
        end_date = parse_date(end_str)
        
        if not start_date:
            print("Invalid start date format")
            continue
        if not end_date:
            print("Invalid end date format") 
            continue
        if start_date > end_date:
            print("Start date must be before end date")
            continue
            
        print(f"Date range selected: {start_date.date()} to {end_date.date()}")
        return start_date, end_date

def load_and_prepare_data(csv_path, exec_team, start_date, end_date):
    """Load CSV and prepare data with all necessary columns."""
    print("\nLoading and preparing data...")
    
    df = pd.read_csv(csv_path, encoding=CONFIG['csv_encoding'], low_memory=False)
    print(f"Loaded {len(df)} rows")
    
    # Parse dates
    date_col = df[CONFIG['date_column']].copy()
    parsed_dates = pd.to_datetime(date_col, errors='coerce', dayfirst=True)
    parsed_dates_2 = pd.to_datetime(date_col, errors='coerce', dayfirst=False)
    
    if parsed_dates_2.notna().sum() > parsed_dates.notna().sum():
        parsed_dates = parsed_dates_2
    
    df[CONFIG['date_column']] = parsed_dates.dt.normalize()
    
    # Filter by team and date
    team_mask = df[CONFIG['exec_team_column']] == exec_team
    if team_mask.sum() == 0:
        team_mask = df[CONFIG['exec_team_column']].str.strip().str.lower() == exec_team.lower()
    
    date_mask = df[CONFIG['date_column']].between(start_date, end_date, inclusive='both')
    filtered_df = df[team_mask & date_mask].copy()
    
    # Add month column
    filtered_df['month'] = filtered_df[CONFIG['date_column']].dt.to_period('M')
    
    # Identify ongoing cases
    status_col = None
    for col in CONFIG['status_columns']:
        if col in filtered_df.columns:
            status_col = col
            break
    
    if status_col:
        def is_ongoing(status):
            if pd.isna(status):
                return False
            status_lower = str(status).strip().lower()
            ongoing_patterns = ['still in progress', 'in progress', 'ongoing', 'out of time', 'in time']
            return any(pattern in status_lower for pattern in ongoing_patterns)
        
        filtered_df['is_ongoing'] = filtered_df[status_col].apply(is_ongoing)
    else:
        filtered_df['is_ongoing'] = False
    
    print(f"Filtered to {len(filtered_df)} rows for {exec_team}")
    return filtered_df

def chart1_received_vs_ongoing_monthly(df, exec_team, start_date, end_date, output_dir):
    """Chart 1: Executive monthly overview - received vs ongoing cases."""
    print("\nCreating Chart 1: Monthly Overview...")
    
    start_month = start_date.to_period('M')
    end_month = end_date.to_period('M')
    all_months = pd.period_range(start=start_month, end=end_month, freq='M')
    
    received = df.groupby('month').size().reindex(all_months, fill_value=0)
    ongoing = df[df['is_ongoing']].groupby('month').size().reindex(all_months, fill_value=0)
    
    # Create figure with consistent size
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor('white')
    
    x = np.arange(len(all_months))
    width = 0.35
    
    # Create bars with subtle gradient effect
    bars1 = ax.bar(x - width/2, received.values, width, 
                   color=CONFIG['colors']['nhs_blue'], label='Cases Received', 
                   edgecolor='white', linewidth=2, alpha=0.9)
    bars2 = ax.bar(x + width/2, ongoing.values, width,
                   color=CONFIG['colors']['nhs_dark_pink'], label='Cases Ongoing',
                   edgecolor='white', linewidth=2, alpha=0.9)
    
    # Add value labels
    add_value_labels(ax, bars1)
    add_value_labels(ax, bars2)
    
    # Styling
    ax.set_xlabel('Month', fontsize=12, fontweight='bold', color='#425563')
    ax.set_ylabel('Number of Cases', fontsize=12, fontweight='bold', color='#425563')
    apply_professional_style(ax, 
                           title=f'Monthly Case Overview - {exec_team}',
                           subtitle=f'Tracking received and ongoing cases')
    
    ax.set_xticks(x)
    ax.set_xticklabels([p.strftime('%b %Y') for p in all_months], rotation=45, ha='right')
    ax.set_ylim(0, max(received.max(), ongoing.max(), 5) * 1.15)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Professional legend
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True,
             facecolor='white', edgecolor='#E8EDEE')
    
    # Add KPI box
    total_received = received.sum()
    total_ongoing = ongoing.sum()
    resolution_rate = ((total_received - total_ongoing) / total_received * 100) if total_received > 0 else 0
    
    kpi_text = f'Total Received: {total_received:,}\nTotal Ongoing: {total_ongoing:,}\nResolution Rate: {resolution_rate:.1f}%'
    ax.text(0.98, 0.97, kpi_text, transform=ax.transAxes, ha='right', va='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8EDEE', edgecolor='#768692', linewidth=2),
           fontsize=11, fontweight='bold', color='#425563')
    
    # Add footer
    add_executive_footer(fig, exec_team, f"{start_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')}")
    
    plt.tight_layout()
    output_path = output_dir / f"chart1_monthly_overview_{exec_team.lower().replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")

def chart2_top_locations(df, exec_team, output_dir):
    """Chart 2: Top locations analysis with executive presentation quality."""
    print("\nCreating Chart 2: Location Analysis...")
    
    if CONFIG['location_column'] not in df.columns:
        print(f"  ⚠ Warning: '{CONFIG['location_column']}' column not found. Skipping.")
        return
    
    # Get top 10 locations
    location_counts = df[CONFIG['location_column']].value_counts().head(10)
    top_locations = location_counts.index.tolist()
    
    # Prepare data
    data = []
    for loc in reversed(top_locations):
        loc_data = df[df[CONFIG['location_column']] == loc]
        received = len(loc_data)
        ongoing = loc_data['is_ongoing'].sum()
        data.append({'Location': loc, 'Received': received, 'Ongoing': ongoing})
    
    plot_df = pd.DataFrame(data)
    
    # Create figure with consistent size
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor('white')
    
    y_pos = np.arange(len(plot_df))
    bar_height = 0.35
    
    # Create horizontal bars
    bars1 = ax.barh(y_pos - bar_height/2, plot_df['Received'], bar_height,
                    color=CONFIG['colors']['nhs_blue'], label='Cases Received',
                    edgecolor='white', linewidth=2, alpha=0.9)
    bars2 = ax.barh(y_pos + bar_height/2, plot_df['Ongoing'], bar_height,
                    color=CONFIG['colors']['nhs_dark_pink'], label='Cases Ongoing',
                    edgecolor='white', linewidth=2, alpha=0.9)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            width = bar.get_width()
            if width > 0:
                ax.text(width/2, bar.get_y() + bar.get_height()/2, f'{int(width)}',
                       ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['Location'], fontsize=11)
    ax.set_xlabel('Number of Cases', fontsize=12, fontweight='bold', color='#425563')
    apply_professional_style(ax, 
                           title=f'Top 10 Locations by Case Volume - {exec_team}',
                           subtitle='Identifying high-demand areas')
    
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True,
             facecolor='white', edgecolor='#E8EDEE')
    
    # Add percentage annotations on the right
    total = plot_df['Received'].sum()
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        percentage = row['Received'] / total * 100
        ax.text(ax.get_xlim()[1] * 0.98, i, f'{percentage:.1f}%',
               ha='right', va='center', fontsize=9, color='#768692', style='italic')
    
    plt.tight_layout()
    output_path = output_dir / f"chart2_location_analysis_{exec_team.lower().replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")

def chart3_top_specialties_stacked(df, exec_team, output_dir):
    """Chart 3: Specialty analysis with monthly breakdown."""
    print("\nCreating Chart 3: Specialty Analysis...")
    
    if CONFIG['specialty_column'] not in df.columns:
        print(f"  ⚠ Warning: '{CONFIG['specialty_column']}' column not found. Skipping.")
        return
    
    # Get top 10 specialties
    specialty_counts = df[CONFIG['specialty_column']].value_counts().head(10)
    top_specialties = specialty_counts.index.tolist()
    
    # Get all months
    months = sorted(df['month'].dropna().unique())
    month_colors = CONFIG['colors']['months_palette'][:len(months)]
    
    # Create data matrix
    data_matrix = []
    for spec in reversed(top_specialties):
        spec_data = df[df[CONFIG['specialty_column']] == spec]
        month_counts = []
        for month in months:
            count = len(spec_data[spec_data['month'] == month])
            month_counts.append(count)
        data_matrix.append(month_counts)
    
    # Create figure with consistent size
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor('white')
    
    y_pos = np.arange(len(top_specialties))
    
    # Create stacked bars
    left = np.zeros(len(top_specialties))
    for i, month in enumerate(months):
        values = [row[i] for row in data_matrix]
        bars = ax.barh(y_pos, values, left=left, 
                      color=month_colors[i % len(month_colors)],
                      label=month.strftime('%b %Y'), 
                      edgecolor='white', linewidth=0.5, alpha=0.9)
        
        # Add data labels for segments > 0
        for j, (bar, val) in enumerate(zip(bars, values)):
            if val > 0:
                x_pos = bar.get_x() + bar.get_width() / 2
                ax.text(x_pos, bar.get_y() + bar.get_height() / 2, str(val),
                       ha='center', va='center', color='white', 
                       fontweight='bold', fontsize=9)
        
        left += values
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(reversed(top_specialties), fontsize=11)
    ax.set_xlabel('Number of Cases', fontsize=12, fontweight='bold', color='#425563')
    apply_professional_style(ax, 
                           title=f'Top 10 Specialties - Monthly Distribution - {exec_team}',
                           subtitle='Understanding case distribution across clinical areas')
    
    # Professional legend
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1,
             frameon=True, fancybox=True, shadow=True,
             facecolor='white', edgecolor='#E8EDEE')
    
    # Add total labels on the right
    for i, total in enumerate(left):
        if total > 0:
            ax.text(total + ax.get_xlim()[1] * 0.01, i, f'Total: {int(total)}',
                   ha='left', va='center', fontweight='bold', 
                   fontsize=9, color='#425563')
    
    plt.tight_layout()
    output_path = output_dir / f"chart3_specialty_analysis_{exec_team.lower().replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")

def chart4_top_subjects_pie(df, exec_team, output_dir):
    """Chart 4: Subject distribution - executive pie chart."""
    print("\nCreating Chart 4: Subject Analysis...")
    
    if CONFIG['subjects_column'] not in df.columns:
        print(f"  ⚠ Warning: '{CONFIG['subjects_column']}' column not found. Skipping.")
        return
    
    # Get top 5 subjects
    subject_counts = df[CONFIG['subjects_column']].value_counts().head(5)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    
    # Create pie chart with professional styling
    colors = CONFIG['colors']['pie_palette']
    explode = [0.02] * len(subject_counts)  # Slight separation
    
    wedges, texts, autotexts = ax.pie(subject_counts.values, 
                                       labels=None,  # We'll add custom labels
                                       colors=colors,
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       explode=explode,
                                       shadow=True,
                                       pctdistance=0.85)
    
    # Enhance text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    # Title
    ax.set_title(f'Top 5 Complaint Subjects - {exec_team}', 
                fontsize=14, fontweight='bold', color='#003087', pad=20)
    
    # Create professional legend with counts and percentages
    total = subject_counts.sum()
    legend_labels = []
    for subj, count in subject_counts.items():
        percentage = count / total * 100
        legend_labels.append(f'{subj}\n{count:,} cases ({percentage:.1f}%)')
    
    ax.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5),
             frameon=True, fancybox=True, shadow=True,
             facecolor='white', edgecolor='#E8EDEE', fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / f"chart4_subject_analysis_{exec_team.lower().replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")

def chart5_top_subsubjects_pie(df, exec_team, output_dir):
    """Chart 5: Sub-subject distribution - executive pie chart."""
    print("\nCreating Chart 5: Sub-subject Analysis...")
    
    if CONFIG['subsubjects_column'] not in df.columns:
        print(f"  ⚠ Warning: '{CONFIG['subsubjects_column']}' column not found. Skipping.")
        return
    
    # Get top 5 sub-subjects
    subsubject_counts = df[CONFIG['subsubjects_column']].value_counts().head(5)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    
    # Create pie chart
    colors = CONFIG['colors']['pie_palette']
    explode = [0.02] * len(subsubject_counts)
    
    wedges, texts, autotexts = ax.pie(subsubject_counts.values, 
                                       labels=None,
                                       colors=colors,
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       explode=explode,
                                       shadow=True,
                                       pctdistance=0.85)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    # Title
    ax.set_title(f'Top 5 Complaint Sub-subjects - {exec_team}',
                fontsize=14, fontweight='bold', color='#003087', pad=20)
    
    # Legend with wrapped text
    total = subsubject_counts.sum()
    legend_labels = []
    for subj, count in subsubject_counts.items():
        # Wrap long text
        if len(subj) > 40:
            subj_display = subj[:40] + '...'
        else:
            subj_display = subj
        percentage = count / total * 100
        legend_labels.append(f'{subj_display}\n{count:,} cases ({percentage:.1f}%)')
    
    ax.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5),
             frameon=True, fancybox=True, shadow=True,
             facecolor='white', edgecolor='#E8EDEE', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / f"chart5_subsubject_analysis_{exec_team.lower().replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")

def chart6_safeguarding_analysis(df, exec_team, output_dir):
    """Chart 6: Safeguarding analysis - CDG and Location breakdown (robust, 'None' treated as null)."""
    print("\nCreating Chart 6: Safeguarding Analysis...")

    # --- config shorthands ---
    sg_col = CONFIG['safeguarding_column']
    loc_col = CONFIG['location_column']

    # --- column existence checks ---
    if sg_col not in df.columns:
        print(f"  ⚠ Warning: '{sg_col}' column not found. Skipping.")
        return

    has_cdg = 'CDG' in df.columns
    has_location = loc_col in df.columns
    if not has_cdg and not has_location:
        print("  ⚠ Warning: Neither CDG nor Location columns found. Skipping.")
        return

    # --- CLEAN SAFEGUARDING COLUMN ---
    # Treat "None" (any case/whitespace) and "" as null; keep true NaNs as nulls.
    # Use pandas 'string' dtype so NaNs become <NA>, not the literal "nan".
    sg_clean = (
        df[sg_col]
        .astype('string')          # preserves <NA>
        .str.strip()
        .str.lower()
        .replace({'none': pd.NA, '': pd.NA})
    )

    # rows with an actual safeguarding concern (non-null after cleaning)
    mask_sg = sg_clean.notna()
    if not mask_sg.any():
        print("  No safeguarding concerns found. Skipping chart.")
        return

    safeguarding_df = df.loc[mask_sg].copy()
    # keep the cleaned safeguarding value for type counts
    safeguarding_df['_sg_value'] = sg_clean.loc[mask_sg].astype('string')

    # --- normalise strings used for grouping to avoid odd whitespace etc. ---
    if has_cdg:
        safeguarding_df['CDG'] = safeguarding_df['CDG'].astype('string').str.strip()
    if has_location:
        safeguarding_df[loc_col] = safeguarding_df[loc_col].astype('string').str.strip()

    # --- create figure layout ---
    if has_cdg and has_location:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        ax2 = None

    fig.patch.set_facecolor('white')

    # -------- Left: CDG breakdown (horizontal) --------
    if has_cdg:
        cdg_counts = (
            safeguarding_df['CDG']
            .dropna()
            .value_counts()
            .head(10)
        )

        if len(cdg_counts) > 0:
            y_pos = np.arange(len(cdg_counts))
            bars1 = ax1.barh(
                y_pos,
                cdg_counts.values.astype(float),
                color=CONFIG['colors']['safeguarding'],
                edgecolor='white',
                linewidth=2,
                alpha=0.9
            )

            # value labels
            for bar in bars1:
                width = float(bar.get_width())
                if width > 0:
                    ax1.text(
                        width / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f'{int(round(width))}',
                        ha='center',
                        va='center',
                        color='white',
                        fontweight='bold',
                        fontsize=10
                    )

            # axes styling
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(list(cdg_counts.index), fontsize=10)
            ax1.set_xlabel('Number of safeguarding cases',
                           fontsize=11, fontweight='bold', color='#425563')
            ax1.invert_yaxis()  # highest at top
            apply_professional_style(ax1, title='Safeguarding cases by Clinical Delivery Group')

            # percentage annotations (share of all safeguarding)
            total_sg = float(mask_sg.sum())
            xlim_right = ax1.get_xlim()[1] if ax1.get_xlim()[1] > 0 else max(cdg_counts.values) * 1.1
            for i, count in enumerate(cdg_counts.values.astype(float)):
                pct = (count / total_sg * 100.0) if total_sg > 0 else 0.0
                ax1.text(xlim_right * 0.98, i, f'{pct:.1f}%',
                         ha='right', va='center', fontsize=9,
                         color='#768692', style='italic')

    # -------- Right: Location breakdown (vertical) --------
    def _plot_location(ax, counts, color_key, title):
        """Internal helper to draw locations and safe trendline."""
        x_pos = np.arange(len(counts), dtype=float)
        y_vals = pd.to_numeric(counts.values, errors='coerce').astype(float)

        bars = ax.bar(
            x_pos,
            y_vals,
            color=CONFIG['colors'][color_key],
            edgecolor='white',
            linewidth=2,
            alpha=0.9
        )

        # value labels
        for bar in bars:
            height = float(bar.get_height())
            if np.isfinite(height) and height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.2,
                    f'{int(round(height))}',
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    fontsize=10,
                    color='#425563'
                )

        # styling
        ax.set_xticks(x_pos)
        ax.set_xticklabels(list(counts.index), rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Number of safeguarding cases',
                      fontsize=11, fontweight='bold', color='#425563')
        apply_professional_style(ax, title=title)

        # --- safe trendline (prevents SVD crashes) ---
        import warnings
        from numpy.linalg import LinAlgError
        from numpy.polynomial.polyutils import RankWarning

        mask = np.isfinite(x_pos) & np.isfinite(y_vals)
        x_fit, y_fit = x_pos[mask], y_vals[mask]

        trend_drawn = False
        if x_fit.size >= 2 and np.any(y_fit != y_fit.mean()):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RankWarning)
                    z = np.polyfit(x_fit, y_fit, 1)  # linear
                p = np.poly1d(z)
                ax.plot(x_pos, p(x_pos), linestyle="--", linewidth=1.5, label="Trend")
                trend_drawn = True
            except LinAlgError:
                pass

        if trend_drawn:
            ax.legend()

    if has_location and ax2 is not None:
        loc_counts = (
            safeguarding_df[loc_col]
            .dropna()
            .value_counts()
            .head(10)
        )
        if len(loc_counts) > 0:
            _plot_location(ax2, loc_counts, color_key='nhs_dark_red',
                           title='Safeguarding cases by Location')

    elif has_location and ax2 is None:
        # single-panel layout: use the left axis for locations
        loc_counts = (
            safeguarding_df[loc_col]
            .dropna()
            .value_counts()
            .head(10)
        )
        if len(loc_counts) > 0:
            _plot_location(ax1, loc_counts, color_key='safeguarding',
                           title='Safeguarding cases by Location')

    # -------- KPIs and title --------
    total_with_sg = int(mask_sg.sum())
    total_cases = int(len(df))
    pct = (total_with_sg / total_cases * 100.0) if total_cases > 0 else 0.0

    # Use the cleaned safeguarding values for “most common”
    sg_types = safeguarding_df['_sg_value'].value_counts()
    top_concern = str(sg_types.index[0]) if len(sg_types) > 0 else "N/A"
    top_concern_count = int(sg_types.values[0]) if len(sg_types) > 0 else 0

    fig.suptitle(
        f'Safeguarding analysis - {exec_team}',
        fontsize=16, fontweight='bold', color='#003087', y=1.02
    )

    kpi_text = (f'{total_with_sg} safeguarding cases identified '
                f'({pct:.1f}% of total) | '
                f'Most common: {top_concern} ({top_concern_count} cases)')
    fig.text(0.5, 0.96, kpi_text, ha='center', fontsize=11,
             color='#8A1538', fontweight='bold')

    # alert box for elevated shares
    alert_ax = ax2 if (has_cdg and has_location and ax2 is not None) else ax1
    if pct > 5:
        alert_color = '#8A1538' if pct > 10 else '#ED8B00'
        alert_text = 'HIGH ALERT' if pct > 10 else 'ELEVATED CONCERN'
        alert_ax.text(
            0.98, 0.97, alert_text, transform=alert_ax.transAxes,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE5E5',
                      edgecolor=alert_color, linewidth=2),
            fontsize=10, fontweight='bold', color=alert_color
        )

    # -------- save --------
    import re
    safe_team = re.sub(r'[^a-z0-9]+', '_', str(exec_team).lower()).strip('_')
    plt.tight_layout()
    output_path = output_dir / f"chart6_safeguarding_analysis_{safe_team}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")



def chart7_six_months_analysis(df, exec_team, output_dir):
    """Chart 7: Six months analysis - executive performance metric."""
    print("\nCreating Chart 7: Six Months Performance Analysis...")
    
    if CONFIG['six_months_column'] not in df.columns:
        print(f"  ⚠ Warning: '{CONFIG['six_months_column']}' column not found. Skipping.")
        return
    
    # Get months
    months = sorted(df['month'].dropna().unique())
    
    # Monthly and cumulative data
    six_months_monthly = []
    cumulative = []
    total_over_six = 0
    
    for month in months:
        month_data = df[df['month'] == month]
        count = (month_data[CONFIG['six_months_column']] == 'Yes').sum()
        six_months_monthly.append(count)
        total_over_six += count
        cumulative.append(total_over_six)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor('white')
    
    x_pos = np.arange(len(months))
    
    # Bar chart for monthly counts
    bars = ax.bar(x_pos, six_months_monthly, 
                  color=CONFIG['colors']['six_months'], 
                  edgecolor='white', linewidth=2, alpha=0.9,
                  label='Monthly Cases >6 Months')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   f'{int(height)}', ha='center', va='center',
                   color='white', fontweight='bold', fontsize=10)
    
    # Add cumulative line with second y-axis
    ax2 = ax.twinx()
    line = ax2.plot(x_pos, cumulative, color='#8A1538', marker='o', 
                    linewidth=3, markersize=8, label='Cumulative Total',
                    markeredgecolor='white', markeredgewidth=2)
    
    # Add value labels on the line
    for i, val in enumerate(cumulative):
        ax2.annotate(str(val), (i, val), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontweight='bold',
                    fontsize=10, color='#8A1538')
    
    # Styling
    ax.set_xlabel('Month', fontsize=12, fontweight='bold', color='#425563')
    ax.set_ylabel('Monthly Cases Open >6 Months', fontsize=12, fontweight='bold', color=CONFIG['colors']['six_months'])
    ax2.set_ylabel('Cumulative Total', fontsize=12, fontweight='bold', color='#8A1538')
    
    apply_professional_style(ax, title=f'Long-Standing Cases Analysis - {exec_team}',
                           subtitle='Cases exceeding 6-month threshold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.strftime('%b %Y') for m in months], rotation=45, ha='right')
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0, top=max(six_months_monthly) * 1.3 if six_months_monthly else 1)
    ax2.set_ylim(bottom=0, top=max(cumulative) * 1.1 if cumulative else 1)
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
             frameon=True, fancybox=True, shadow=True,
             facecolor='white', edgecolor='#E8EDEE')
    
    # Add KPI box
    total_cases = len(df)
    total_six_months = (df[CONFIG['six_months_column']] == 'Yes').sum()
    percentage = (total_six_months / total_cases * 100) if total_cases > 0 else 0
    
    kpi_text = f'Performance Alert\n{total_six_months} of {total_cases} cases\nexceed 6 months\n({percentage:.1f}%)'
    
    # Color-code the KPI box based on percentage
    if percentage > 20:
        box_color = '#FFE5E5'  # Light red
        text_color = '#8A1538'  # Dark red
    elif percentage > 10:
        box_color = '#FFF4E5'  # Light amber
        text_color = '#ED8B00'  # Amber
    else:
        box_color = '#E5F5E5'  # Light green
        text_color = '#006747'  # Dark green
    
    ax.text(0.98, 0.97, kpi_text, transform=ax.transAxes, 
           ha='right', va='top', 
           bbox=dict(boxstyle='round,pad=0.5', facecolor=box_color, 
                    edgecolor=text_color, linewidth=2),
           fontsize=11, fontweight='bold', color=text_color)
    
    plt.tight_layout()
    output_path = output_dir / f"chart7_six_months_analysis_{exec_team.lower().replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")

def chart8_complexity_analysis(df, exec_team, output_dir):
    """Chart 8: Complexity analysis - executive workload assessment."""
    print("\nCreating Chart 8: Complexity Analysis...")
    
    if CONFIG['complexity_column'] not in df.columns:
        print(f"  ⚠ Warning: '{CONFIG['complexity_column']}' column not found. Skipping.")
        return
    
    # Define categories
    categories = ['Basic', 'Regular', 'Complex']
    category_colors = ['#009639', '#FFB81C', '#8A1538']  # Green, Amber, Red
    
    # Get months
    months = sorted(df['month'].dropna().unique())
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('white')
    
    # Left plot: Overall distribution
    complexity_counts = df[CONFIG['complexity_column']].value_counts()
    ordered_counts = [complexity_counts.get(cat, 0) for cat in categories]
    
    # Create donut chart instead of bar chart for executive appeal
    wedges, texts, autotexts = ax1.pie(ordered_counts, 
                                        labels=categories,
                                        colors=category_colors,
                                        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(ordered_counts))})',
                                        startangle=90,
                                        pctdistance=0.85,
                                        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2))
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    ax1.set_title('Overall Complexity Distribution', fontsize=12, fontweight='bold', color='#003087')
    
    # Add center text
    total = sum(ordered_counts)
    ax1.text(0, 0, f'Total\n{total} cases', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='#425563')
    
    # Right plot: Monthly stacked area chart
    monthly_data = {cat: [] for cat in categories}
    
    for month in months:
        month_data = df[df['month'] == month]
        month_complexity = month_data[CONFIG['complexity_column']].value_counts()
        for cat in categories:
            monthly_data[cat].append(month_complexity.get(cat, 0))
    
    x_pos = np.arange(len(months))
    
    # Create stacked area chart
    ax2.stackplot(x_pos, 
                  monthly_data['Basic'],
                  monthly_data['Regular'],
                  monthly_data['Complex'],
                  labels=categories,
                  colors=category_colors,
                  alpha=0.8)
    
    # Add total line on top
    totals = [sum(monthly_data[cat][i] for cat in categories) for i in range(len(months))]
    ax2.plot(x_pos, totals, color='#003087', linewidth=2, marker='o', 
            markersize=6, markeredgecolor='white', markeredgewidth=2)
    
    # Add value labels on the line
    for i, val in enumerate(totals):
        if val > 0:
            ax2.annotate(str(val), (i, val), textcoords="offset points", 
                        xytext=(0,5), ha='center', fontweight='bold',
                        fontsize=9, color='#003087')
    
    ax2.set_xlabel('Month', fontsize=12, fontweight='bold', color='#425563')
    ax2.set_ylabel('Number of Cases', fontsize=12, fontweight='bold', color='#425563')
    ax2.set_title('Monthly Complexity Trend', fontsize=12, fontweight='bold', color='#003087')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([m.strftime('%b %Y') for m in months], rotation=45, ha='right')
    ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True,
              facecolor='white', edgecolor='#E8EDEE')
    ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    # Main title
    fig.suptitle(f'Case Complexity Analysis - {exec_team}',
                fontsize=16, fontweight='bold', color='#003087', y=1.02)
    
    # Add insight box
    complex_percentage = (ordered_counts[2] / total * 100) if total > 0 else 0
    insight_text = f'Complex cases: {complex_percentage:.1f}%'
    
    if complex_percentage > 30:
        insight_color = '#8A1538'
        insight_message = 'High complexity load'
    elif complex_percentage > 20:
        insight_color = '#ED8B00'
        insight_message = 'Moderate complexity'
    else:
        insight_color = '#009639'
        insight_message = 'Manageable complexity'
    
    fig.text(0.5, 0.94, f'{insight_text} - {insight_message}',
            ha='center', fontsize=12, color=insight_color, 
            fontweight='bold', style='italic')
    
    plt.tight_layout()
    output_path = output_dir / f"chart8_complexity_analysis_{exec_team.lower().replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")

def generate_executive_summary(df, exec_team, output_dir):
    """Generate executive summary document with key metrics."""
    print("\nGenerating Executive Summary...")
    
    # Calculate key metrics
    total_cases = len(df)
    ongoing_cases = df['is_ongoing'].sum()
    closed_cases = total_cases - ongoing_cases
    resolution_rate = (closed_cases / total_cases * 100) if total_cases > 0 else 0
    
    # Additional metrics
    metrics = {
        'Total Cases': f'{total_cases:,}',
        'Ongoing Cases': f'{ongoing_cases:,}',
        'Closed Cases': f'{closed_cases:,}',
        'Resolution Rate': f'{resolution_rate:.1f}%',
        'Date Range': f"{df[CONFIG['date_column']].min().strftime('%d %b %Y')} to {df[CONFIG['date_column']].max().strftime('%d %b %Y')}"
    }
    
    if CONFIG['safeguarding_column'] in df.columns:
        safeguarding = (df[CONFIG['safeguarding_column']] != 'None').sum()
        metrics['Safeguarding Concerns'] = f'{safeguarding:,} ({safeguarding/total_cases*100:.1f}%)'
    
    if CONFIG['six_months_column'] in df.columns:
        six_months = (df[CONFIG['six_months_column']] == 'Yes').sum()
        metrics['Cases >6 Months'] = f'{six_months:,} ({six_months/total_cases*100:.1f}%)'
    
    if CONFIG['complexity_column'] in df.columns:
        complex_cases = (df[CONFIG['complexity_column']] == 'Complex').sum()
        metrics['Complex Cases'] = f'{complex_cases:,} ({complex_cases/total_cases*100:.1f}%)'
    
    # Create summary document
    summary = []
    summary.append("=" * 60)
    summary.append(f"EXECUTIVE SUMMARY - {exec_team}")
    summary.append("=" * 60)
    summary.append(f"Report Generated: {datetime.now().strftime('%d %B %Y at %H:%M')}")
    summary.append("")
    summary.append("KEY PERFORMANCE INDICATORS")
    summary.append("-" * 30)
    
    for key, value in metrics.items():
        summary.append(f"{key:<25} {value}")
    
    # Top locations
    if CONFIG['location_column'] in df.columns:
        summary.append("\nTOP 5 LOCATIONS BY VOLUME")
        summary.append("-" * 30)
        for loc, count in df[CONFIG['location_column']].value_counts().head(5).items():
            percentage = count/total_cases*100
            summary.append(f"• {loc}: {count:,} ({percentage:.1f}%)")
    
    # Top specialties
    if CONFIG['specialty_column'] in df.columns:
        summary.append("\nTOP 5 SPECIALTIES BY VOLUME")
        summary.append("-" * 30)
        for spec, count in df[CONFIG['specialty_column']].value_counts().head(5).items():
            percentage = count/total_cases*100
            summary.append(f"• {spec}: {count:,} ({percentage:.1f}%)")
    
    # Risk indicators
    summary.append("\nRISK INDICATORS")
    summary.append("-" * 30)
    
    risk_level = "LOW"
    if CONFIG['six_months_column'] in df.columns:
        six_months_pct = (df[CONFIG['six_months_column']] == 'Yes').sum() / total_cases * 100
        if six_months_pct > 20:
            risk_level = "HIGH"
        elif six_months_pct > 10:
            risk_level = "MEDIUM"
    
    summary.append(f"Overall Risk Level: {risk_level}")
    
    # Save summary
    summary_path = output_dir / f"executive_summary_{exec_team.lower().replace(' ', '_')}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))
    
    print(f"  ✓ Saved: {summary_path.name}")
    return '\n'.join(summary)

def main():
    """Main execution function."""
    try:
        print("\n" + "="*60)
        print(" HPEG EXECUTIVE DASHBOARD")
        print(" Multi-Chart Visualization Suite")
        print("="*60)
        print("\nGenerating professional charts for executive presentation:")
        print("  1. Monthly Overview (Received vs Ongoing)")
        print("  2. Location Analysis (Top 10)")
        print("  3. Specialty Distribution (Monthly breakdown)")
        print("  4. Subject Analysis (Top 5)")
        print("  5. Sub-subject Analysis (Top 5)")
        print("  6. Safeguarding Dashboard")
        print("  7. Six Months Performance Metric")
        print("  8. Complexity Assessment")
        
        # Get inputs
        csv_path = get_csv_path()
        print(f"\n✓ Selected file: {csv_path.name}")
        
        # Load CSV for preview
        print("\nLoading data for analysis...")
        df_preview = pd.read_csv(csv_path, encoding=CONFIG['csv_encoding'], low_memory=False)
        print(f"✓ Loaded {len(df_preview):,} rows")
        
        # Check columns
        print("\nValidating data structure...")
        required_cols = [CONFIG['date_column'], CONFIG['exec_team_column']]
        optional_cols = [CONFIG['location_column'], CONFIG['specialty_column'], 
                        CONFIG['subjects_column'], CONFIG['subsubjects_column'],
                        CONFIG['safeguarding_column'], CONFIG['six_months_column'], 
                        CONFIG['complexity_column']]
        
        missing_required = [c for c in required_cols if c not in df_preview.columns]
        if missing_required:
            print(f"\n✗ Error: Required columns missing: {missing_required}")
            return
        
        available_optional = [c for c in optional_cols if c in df_preview.columns]
        missing_optional = [c for c in optional_cols if c not in df_preview.columns]
        
        print(f"✓ All required columns present")
        if missing_optional:
            print(f"⚠ Some optional columns missing: {missing_optional}")
        
        # Get selections
        exec_team = get_exec_team(df_preview)
        start_date, end_date = get_date_range(df_preview)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = CONFIG['outputs_dir'] / f"executive_dashboard_{exec_team.lower().replace(' ', '_')}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n✓ Output directory created")
        
        # Load and prepare data
        df = load_and_prepare_data(csv_path, exec_team, start_date, end_date)
        
        if len(df) == 0:
            print("\n⚠ No data found for the selected filters!")
            return
        
        print(f"\n{'='*60}")
        print(" GENERATING EXECUTIVE CHARTS")
        print(f"{'='*60}")
        
        # Generate all charts
        date_range_str = f"{start_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')}"
        
        chart1_received_vs_ongoing_monthly(df, exec_team, start_date, end_date, output_dir)
        chart2_top_locations(df, exec_team, output_dir)
        chart3_top_specialties_stacked(df, exec_team, output_dir)
        chart4_top_subjects_pie(df, exec_team, output_dir)
        chart5_top_subsubjects_pie(df, exec_team, output_dir)
        chart6_safeguarding_analysis(df, exec_team, output_dir)
        chart7_six_months_analysis(df, exec_team, output_dir)
        chart8_complexity_analysis(df, exec_team, output_dir)
        
        # Generate executive summary
        summary = generate_executive_summary(df, exec_team, output_dir)
        
        print(f"\n{'='*60}")
        print(" EXECUTIVE SUMMARY")
        print(f"{'='*60}")
        print(summary)
        
        print(f"\n{'='*60}")
        print(" ✓ DASHBOARD GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"\nAll files saved to: {output_dir}")
        print(f"Ready for executive presentation")
        
        # Open output folder (Windows)
        import os
        if os.name == 'nt':
            os.startfile(output_dir)
        
    except KeyboardInterrupt:
        print("\n\n✗ Process cancelled")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        print("\nError details:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()