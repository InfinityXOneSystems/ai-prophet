#!/usr/bin/env python3
"""
Reporting Module
Generates comprehensive reports and visualizations from reflection results.

Author: Manus AI
Version: 1.0.0
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class ReportingModule:
    """
    Main reporting module that generates reports in multiple formats
    """
    
    def __init__(self, agent, config: Optional[Dict] = None):
        """
        Initialize reporting module
        
        Args:
            agent: Agent instance
            config: Configuration dictionary
        """
        self.agent = agent
        self.config = config or {}
        self.root_dir = Path(self.config.get('root_dir', '.'))
        
        self.formats = self.config.get('formats', ['json', 'markdown'])
        self.include_visualizations = self.config.get('include_visualizations', True)
        self.save_to = Path(self.config.get('save_to', 'data/reflection'))
        self.save_to.mkdir(parents=True, exist_ok=True)
        
        self.results = None
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute report generation
        
        Returns:
            Dictionary with reporting results
        """
        print("   📊 Generating reports...")
        
        # This module is typically called by the reflection engine
        # with complete results, so we just return a placeholder here
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'formats_enabled': self.formats,
            'save_directory': str(self.save_to)
        }
        
        return self.results
    
    def generate_reports(self, reflection_result) -> Dict[str, Any]:
        """
        Generate reports from reflection results
        
        Args:
            reflection_result: ReflectionResult object
        
        Returns:
            Dictionary with paths to generated reports
        """
        generated_reports = {
            'json': None,
            'markdown': None,
            'html': None
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate JSON report
        if 'json' in self.formats:
            json_path = self._generate_json_report(reflection_result, timestamp)
            generated_reports['json'] = str(json_path)
            print(f"      ✅ JSON report: {json_path.name}")
        
        # Generate Markdown report
        if 'markdown' in self.formats:
            md_path = self._generate_markdown_report(reflection_result, timestamp)
            generated_reports['markdown'] = str(md_path)
            print(f"      ✅ Markdown report: {md_path.name}")
        
        return generated_reports
    
    def _generate_json_report(self, result, timestamp: str) -> Path:
        """Generate JSON report"""
        json_path = self.save_to / f"reflection_{timestamp}.json"
        
        with open(json_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        return json_path
    
    def _generate_markdown_report(self, result, timestamp: str) -> Path:
        """Generate Markdown report"""
        md_path = self.save_to / f"reflection_{timestamp}.md"
        
        content = []
        
        # Header
        content.append(f"# Self-Reflection Report")
        content.append(f"")
        content.append(f"**Agent:** {result.agent_name}  ")
        content.append(f"**Timestamp:** {result.timestamp}  ")
        content.append(f"**Modules Executed:** {len(result.modules_executed)}  ")
        content.append(f"")
        
        # Executive Summary
        content.append(f"## 📋 Executive Summary")
        content.append(f"")
        content.append(self._generate_executive_summary(result))
        content.append(f"")
        
        # Inventory Results
        if result.inventory:
            content.append(f"## 📦 Inventory Analysis")
            content.append(f"")
            content.append(self._format_inventory_section(result.inventory))
            content.append(f"")
        
        # Taxonomy Results
        if result.taxonomy:
            content.append(f"## 🏷️ Taxonomy & Classification")
            content.append(f"")
            content.append(self._format_taxonomy_section(result.taxonomy))
            content.append(f"")
        
        # Documentation Results
        if result.documentation:
            content.append(f"## 📚 Documentation Status")
            content.append(f"")
            content.append(self._format_documentation_section(result.documentation))
            content.append(f"")
        
        # Evolution Results
        if result.evolution:
            content.append(f"## 🧬 Evolution & Optimization")
            content.append(f"")
            content.append(self._format_evolution_section(result.evolution))
            content.append(f"")
        
        # Errors
        if result.errors:
            content.append(f"## ⚠️ Errors & Warnings")
            content.append(f"")
            for error in result.errors:
                content.append(f"- {error}")
            content.append(f"")
        
        # Footer
        content.append(f"---")
        content.append(f"")
        content.append(f"*Generated by Universal Self-Reflection System v1.0.0*")
        
        # Write report
        with open(md_path, 'w') as f:
            f.write('\n'.join(content))
        
        return md_path
    
    def _generate_executive_summary(self, result) -> str:
        """Generate executive summary"""
        summary_parts = []
        
        summary_parts.append(f"This report summarizes the self-reflection analysis for **{result.agent_name}**.")
        summary_parts.append(f"")
        
        # Count findings
        total_findings = 0
        if result.inventory:
            total_findings += result.inventory.get('statistics', {}).get('total_assets', 0)
        
        summary_parts.append(f"**Key Highlights:**")
        summary_parts.append(f"")
        
        if result.inventory:
            stats = result.inventory.get('statistics', {})
            summary_parts.append(f"- **Assets:** {stats.get('total_assets', 0)} files tracked")
        
        if result.taxonomy:
            violations = len(result.taxonomy.get('naming_violations', []))
            if violations > 0:
                summary_parts.append(f"- **Naming:** {violations} convention violations found")
            else:
                summary_parts.append(f"- **Naming:** All conventions compliant ✅")
        
        if result.evolution:
            score = result.evolution.get('evolution_score', 0)
            summary_parts.append(f"- **Evolution Score:** {score:.1f}/100")
        
        return '\n'.join(summary_parts)
    
    def _format_inventory_section(self, inventory: Dict[str, Any]) -> str:
        """Format inventory section"""
        lines = []
        
        stats = inventory.get('statistics', {})
        
        lines.append(f"### Summary")
        lines.append(f"")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Assets | {stats.get('total_assets', 0)} |")
        lines.append(f"| Total Size | {stats.get('total_size_bytes', 0) / (1024*1024):.2f} MB |")
        lines.append(f"")
        
        # By type
        by_type = stats.get('by_type', {})
        if by_type:
            lines.append(f"### Assets by Type")
            lines.append(f"")
            lines.append(f"| Type | Count |")
            lines.append(f"|------|-------|")
            for asset_type, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"| {asset_type} | {count} |")
            lines.append(f"")
        
        return '\n'.join(lines)
    
    def _format_taxonomy_section(self, taxonomy: Dict[str, Any]) -> str:
        """Format taxonomy section"""
        lines = []
        
        classification = taxonomy.get('agent_classification', {})
        
        lines.append(f"### Agent Classification")
        lines.append(f"")
        lines.append(f"- **Type:** {classification.get('type', 'unknown')}")
        lines.append(f"- **Domain:** {classification.get('domain', 'unknown')}")
        lines.append(f"- **Maturity:** {classification.get('maturity', 'unknown')}")
        lines.append(f"- **Complexity:** {classification.get('complexity', 'unknown')}")
        lines.append(f"")
        
        # Naming violations
        violations = taxonomy.get('naming_violations', [])
        if violations:
            lines.append(f"### Naming Convention Violations")
            lines.append(f"")
            lines.append(f"Found {len(violations)} violations:")
            lines.append(f"")
            for violation in violations[:10]:  # Show first 10
                lines.append(f"- `{violation['file']}`: {violation['issue']}")
            if len(violations) > 10:
                lines.append(f"- ... and {len(violations) - 10} more")
            lines.append(f"")
        
        # Recommendations
        recommendations = taxonomy.get('recommendations', [])
        if recommendations:
            lines.append(f"### Recommendations")
            lines.append(f"")
            for rec in recommendations:
                lines.append(f"- {rec}")
            lines.append(f"")
        
        return '\n'.join(lines)
    
    def _format_documentation_section(self, documentation: Dict[str, Any]) -> str:
        """Format documentation section"""
        lines = []
        
        lines.append(f"### Generated Documentation")
        lines.append(f"")
        
        if documentation.get('readme_path'):
            lines.append(f"- ✅ README.md generated")
        
        if documentation.get('changelog_path'):
            lines.append(f"- ✅ CHANGELOG.md generated")
        
        # API documentation
        api_docs = documentation.get('api_documentation', {})
        if api_docs:
            lines.append(f"- ✅ API documentation extracted:")
            lines.append(f"  - {len(api_docs.get('classes', []))} classes")
            lines.append(f"  - {len(api_docs.get('functions', []))} functions")
        
        # TODO items
        todo_list = documentation.get('todo_list', [])
        if todo_list:
            lines.append(f"- ⚠️  {len(todo_list)} TODO items found")
        
        lines.append(f"")
        
        return '\n'.join(lines)
    
    def _format_evolution_section(self, evolution: Dict[str, Any]) -> str:
        """Format evolution section"""
        lines = []
        
        score = evolution.get('evolution_score', 0)
        lines.append(f"### Evolution Score: {score:.1f}/100")
        lines.append(f"")
        
        # Trends
        trends = evolution.get('trends', {})
        if trends:
            improving = trends.get('improving', [])
            declining = trends.get('declining', [])
            
            if improving:
                lines.append(f"**Improving Metrics:**")
                for metric in improving:
                    lines.append(f"- {metric['metric']}: {metric['current_value']:.4f}")
                lines.append(f"")
            
            if declining:
                lines.append(f"**Declining Metrics:**")
                for metric in declining:
                    lines.append(f"- {metric['metric']}: {metric['current_value']:.4f}")
                lines.append(f"")
        
        # Optimizations
        optimizations = evolution.get('optimizations', [])
        if optimizations:
            lines.append(f"### Recommended Optimizations")
            lines.append(f"")
            for opt in optimizations:
                lines.append(f"#### {opt['type'].replace('_', ' ').title()}")
                lines.append(f"")
                lines.append(f"- **Priority:** {opt['priority']}")
                lines.append(f"- **Description:** {opt['description']}")
                lines.append(f"- **Expected Improvement:** {opt['expected_improvement']}")
                lines.append(f"- **Confidence:** {opt['confidence']:.0%}")
                lines.append(f"")
        
        return '\n'.join(lines)


__all__ = ['ReportingModule']
