#!/usr/bin/env python3
"""
AI Prophet Final Daily Learning Report Generator
Combines multiple analysis reports into a single, comprehensive daily summary.
"""

import os
from datetime import datetime
from pathlib import Path

def main():
    """Main function to generate the final report."""
    today = datetime.now().strftime('%Y%m%d')
    data_dir = Path('data/learning')
    output_dir = Path('/home/ubuntu/ai-prophet/')

    reflection_report_path = data_dir / f'daily_reflection_{today}.md'
    weight_report_path = data_dir / f'weight_update_{today}.md'
    final_report_path = output_dir / f'AI_PROPHET_DAILY_LEARNING_REPORT_{today}.md'

    print("=" * 60)
    print("Generating Final AI Prophet Daily Learning Report")
    print("=" * 60)

    if not reflection_report_path.exists() or not weight_report_path.exists():
        print(f"âï¸  Missing required reports for {today}. Cannot generate final report.")
        return 1

    print(f"Reading content from {reflection_report_path}")
    with open(reflection_report_path, 'r') as f:
        reflection_content = f.read()

    print(f"Reading content from {weight_report_path}")
    with open(weight_report_path, 'r') as f:
        weight_content = f.read()

    # Combine the reports
    final_report_content = (
        f"# AI Prophet: Daily Learning & Optimization Report\n\n"
        f"**Report Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"This report summarizes AI Prophet's self-reflection process, including performance analysis, prediction evaluation, and model weight adjustments. The system continuously learns and evolves to maximize prediction accuracy.\n\n"
        f"---\n\n"
        f"{reflection_content.split('---', 1)[1].strip()}\n\n"
        f"---\n\n"
        f"{weight_content.split('---', 1)[1].strip()}\n"
    )

    # Save the final report
    with open(final_report_path, 'w') as f:
        f.write(final_report_content)

    print(f"\nð Final report saved to: {final_report_path}")
    print("\n" + "=" * 60)
    print("â Final Report Generation Complete!")
    print("=" * 60)

    return 0

if __name__ == '__main__':
    exit(main())
