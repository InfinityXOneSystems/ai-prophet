#!/usr/bin/env python3
"""
Documentation Module
Auto-generates and maintains comprehensive documentation for AI agents.

Author: Manus AI
Version: 1.0.0
"""

import ast
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict


class DocumentationModule:
    """
    Main documentation module that auto-generates README, changelogs, and API docs
    """
    
    def __init__(self, agent, config: Optional[Dict] = None):
        """
        Initialize documentation module
        
        Args:
            agent: Agent instance
            config: Configuration dictionary
        """
        self.agent = agent
        self.config = config or {}
        self.root_dir = Path(self.config.get('root_dir', '.'))
        
        self.auto_generate_readme = self.config.get('auto_generate_readme', True)
        self.auto_generate_changelog = self.config.get('auto_generate_changelog', True)
        self.generate_diagrams = self.config.get('generate_diagrams', True)
        self.update_on_change = self.config.get('update_on_change', True)
        
        self.results = None
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute complete documentation generation
        
        Returns:
            Dictionary with documentation results
        """
        print("   📚 Generating documentation...")
        
        # Get agent metadata
        metadata = self.agent.get_metadata()
        
        # Generate README
        readme_path = None
        if self.auto_generate_readme:
            print("      Generating README...")
            readme_path = self._generate_readme(metadata)
        
        # Generate changelog
        changelog_path = None
        if self.auto_generate_changelog:
            print("      Generating CHANGELOG...")
            changelog_path = self._generate_changelog()
        
        # Extract and document API
        api_docs = None
        print("      Extracting API documentation...")
        api_docs = self._extract_api_documentation()
        
        # Generate TODO list
        print("      Extracting TODO items...")
        todo_list = self._extract_todo_items()
        
        # Generate index/table of contents
        print("      Generating INDEX...")
        index = self._generate_index()
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'readme_path': str(readme_path) if readme_path else None,
            'changelog_path': str(changelog_path) if changelog_path else None,
            'api_documentation': api_docs,
            'todo_list': todo_list,
            'index': index
        }
        
        return self.results
    
    def _generate_readme(self, metadata: Dict[str, Any]) -> Path:
        """Generate comprehensive README.md"""
        
        readme_path = self.root_dir / 'README.md'
        
        # Get performance metrics
        metrics = self.agent.get_performance_metrics()
        
        # Build README content
        content = []
        
        # Header
        content.append(f"# {metadata.get('name', 'AI Agent')}")
        content.append(f"")
        content.append(f"**Version:** {metadata.get('version', '1.0.0')}  ")
        content.append(f"**Type:** {metadata.get('type', 'AI Agent')}  ")
        content.append(f"**Author:** {metadata.get('author', 'Manus AI')}  ")
        content.append(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d')}  ")
        content.append(f"")
        
        # Description
        content.append(f"## 🎯 Purpose")
        content.append(f"")
        content.append(f"{metadata.get('description', 'AI agent for automated tasks')}")
        content.append(f"")
        
        # Capabilities
        if metadata.get('capabilities'):
            content.append(f"## ✨ Capabilities")
            content.append(f"")
            for capability in metadata['capabilities']:
                content.append(f"- **{capability.replace('_', ' ').title()}**")
            content.append(f"")
        
        # Architecture
        content.append(f"## 🏗️ Architecture")
        content.append(f"")
        content.append(f"```")
        content.append(self._generate_architecture_diagram())
        content.append(f"```")
        content.append(f"")
        
        # Components
        content.append(f"## 📦 Components")
        content.append(f"")
        components = self._discover_components()
        for component in components:
            content.append(f"### {component['name']}")
            content.append(f"")
            content.append(f"{component['description']}")
            content.append(f"")
        
        # Quick Start
        content.append(f"## 🚀 Quick Start")
        content.append(f"")
        content.append(f"### Installation")
        content.append(f"")
        content.append(f"```bash")
        content.append(f"# Clone the repository")
        content.append(f"git clone {metadata.get('repository', 'https://github.com/...')}")
        content.append(f"")
        content.append(f"# Install dependencies")
        if (self.root_dir / 'requirements.txt').exists():
            content.append(f"pip install -r requirements.txt")
        elif (self.root_dir / 'package.json').exists():
            content.append(f"npm install")
        content.append(f"```")
        content.append(f"")
        
        # Usage
        content.append(f"### Usage")
        content.append(f"")
        content.append(f"```python")
        content.append(f"# Example usage")
        content.append(f"from {metadata.get('name', 'agent').lower().replace(' ', '_').replace('-', '_')} import Agent")
        content.append(f"")
        content.append(f"agent = Agent()")
        content.append(f"result = agent.execute()")
        content.append(f"```")
        content.append(f"")
        
        # Performance
        if metrics:
            content.append(f"## 📊 Performance")
            content.append(f"")
            content.append(f"| Metric | Value |")
            content.append(f"|--------|-------|")
            for key, value in metrics.items():
                if value is not None and key != 'custom_metrics':
                    if isinstance(value, float):
                        if 0 <= value <= 1:
                            content.append(f"| {key.replace('_', ' ').title()} | {value:.2%} |")
                        else:
                            content.append(f"| {key.replace('_', ' ').title()} | {value:.2f} |")
                    else:
                        content.append(f"| {key.replace('_', ' ').title()} | {value} |")
            content.append(f"")
        
        # Configuration
        content.append(f"## ⚙️ Configuration")
        content.append(f"")
        content.append(f"Configuration files:")
        config_files = list(self.root_dir.glob('*.json')) + list(self.root_dir.glob('*.yaml')) + list(self.root_dir.glob('*.yml'))
        for config_file in config_files[:5]:
            content.append(f"- `{config_file.name}`")
        content.append(f"")
        
        # Documentation
        content.append(f"## 📚 Documentation")
        content.append(f"")
        content.append(f"- [API Documentation](docs/API.md)")
        content.append(f"- [Changelog](CHANGELOG.md)")
        content.append(f"- [TODO List](TODO.md)")
        content.append(f"")
        
        # Contributing
        content.append(f"## 🤝 Contributing")
        content.append(f"")
        content.append(f"This is an autonomous AI agent system. Contributions are managed through the self-reflection and evolution system.")
        content.append(f"")
        
        # License
        content.append(f"## 📄 License")
        content.append(f"")
        content.append(f"Proprietary - InfinityXOne Systems")
        content.append(f"")
        
        # Footer
        content.append(f"---")
        content.append(f"")
        content.append(f"*Auto-generated by Universal Self-Reflection System*  ")
        content.append(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        # Write README
        with open(readme_path, 'w') as f:
            f.write('\n'.join(content))
        
        return readme_path
    
    def _generate_architecture_diagram(self) -> str:
        """Generate ASCII architecture diagram"""
        metadata = self.agent.get_metadata()
        name = metadata.get('name', 'Agent')
        
        diagram = f"""
┌─────────────────────────────────────────────────────────────────┐
│                        {name.upper()}                            │
│                     AI Agent System                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Core       │  │   Modules    │  │   Data       │          │
│  │   Engine     │──│   System     │──│   Layer      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                 │                 │                    │
│         ▼                 ▼                 ▼                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Self-Reflection System                      │   │
│  │         (Inventory, Taxonomy, Evolution)                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
"""
        return diagram.strip()
    
    def _discover_components(self) -> List[Dict[str, str]]:
        """Discover main components of the agent"""
        components = []
        
        # Check for common directories
        common_dirs = ['src', 'core', 'modules', 'lib', 'components']
        
        for dir_name in common_dirs:
            dir_path = self.root_dir / dir_name
            if dir_path.exists() and dir_path.is_dir():
                components.append({
                    'name': dir_name.title(),
                    'description': f"Contains {dir_name} functionality and components"
                })
        
        return components
    
    def _generate_changelog(self) -> Optional[Path]:
        """Generate CHANGELOG.md from Git history"""
        changelog_path = self.root_dir / 'CHANGELOG.md'
        
        try:
            # Get Git log
            result = subprocess.run(
                ['git', 'log', '--pretty=format:%H|%an|%ae|%at|%s', '--no-merges'],
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return None
            
            # Parse commits
            commits = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                parts = line.split('|')
                if len(parts) == 5:
                    commits.append({
                        'hash': parts[0][:7],
                        'author': parts[1],
                        'email': parts[2],
                        'timestamp': int(parts[3]),
                        'message': parts[4]
                    })
            
            # Group by date
            commits_by_date = defaultdict(list)
            for commit in commits:
                date = datetime.fromtimestamp(commit['timestamp']).strftime('%Y-%m-%d')
                commits_by_date[date].append(commit)
            
            # Generate changelog content
            content = []
            content.append("# Changelog")
            content.append("")
            content.append("All notable changes to this project will be documented in this file.")
            content.append("")
            
            for date in sorted(commits_by_date.keys(), reverse=True)[:30]:  # Last 30 days
                content.append(f"## {date}")
                content.append("")
                for commit in commits_by_date[date]:
                    content.append(f"- {commit['message']} (`{commit['hash']}`)")
                content.append("")
            
            content.append("---")
            content.append("")
            content.append(f"*Auto-generated by Universal Self-Reflection System*")
            
            # Write changelog
            with open(changelog_path, 'w') as f:
                f.write('\n'.join(content))
            
            return changelog_path
        
        except Exception as e:
            print(f"      Warning: Could not generate changelog: {e}")
            return None
    
    def _extract_api_documentation(self) -> Dict[str, Any]:
        """Extract API documentation from docstrings"""
        api_docs = {
            'classes': [],
            'functions': [],
            'modules': []
        }
        
        # Find all Python files
        for py_file in self.root_dir.rglob('*.py'):
            if '__pycache__' in str(py_file) or 'venv' in str(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read(), filename=str(py_file))
                
                # Extract classes and functions
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        api_docs['classes'].append({
                            'name': node.name,
                            'file': str(py_file.relative_to(self.root_dir)),
                            'docstring': ast.get_docstring(node) or "No documentation"
                        })
                    
                    elif isinstance(node, ast.FunctionDef):
                        # Only top-level functions
                        if node.col_offset == 0:
                            api_docs['functions'].append({
                                'name': node.name,
                                'file': str(py_file.relative_to(self.root_dir)),
                                'docstring': ast.get_docstring(node) or "No documentation"
                            })
            
            except Exception:
                pass
        
        return api_docs
    
    def _extract_todo_items(self) -> List[Dict[str, Any]]:
        """Extract TODO items from code comments"""
        todo_items = []
        
        # Search for TODO, FIXME, HACK, XXX comments
        patterns = ['TODO', 'FIXME', 'HACK', 'XXX', 'NOTE']
        
        for py_file in self.root_dir.rglob('*.py'):
            if '__pycache__' in str(py_file) or 'venv' in str(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        for pattern in patterns:
                            if f'# {pattern}' in line or f'#{pattern}' in line:
                                todo_items.append({
                                    'type': pattern,
                                    'file': str(py_file.relative_to(self.root_dir)),
                                    'line': line_num,
                                    'text': line.strip()
                                })
            except Exception:
                pass
        
        return todo_items
    
    def _generate_index(self) -> Dict[str, Any]:
        """Generate project index/table of contents"""
        index = {
            'directories': [],
            'key_files': [],
            'documentation_files': []
        }
        
        # Index directories
        for directory in sorted(self.root_dir.iterdir()):
            if directory.is_dir() and not directory.name.startswith('.'):
                if directory.name not in ['__pycache__', 'venv', 'env', 'node_modules']:
                    index['directories'].append({
                        'name': directory.name,
                        'path': str(directory.relative_to(self.root_dir))
                    })
        
        # Index key files
        key_patterns = ['main.py', 'app.py', 'server.py', 'index.js', 'package.json', 'requirements.txt']
        for pattern in key_patterns:
            if (self.root_dir / pattern).exists():
                index['key_files'].append(pattern)
        
        # Index documentation files
        doc_patterns = ['README.md', 'CHANGELOG.md', 'TODO.md', 'LICENSE', 'CONTRIBUTING.md']
        for pattern in doc_patterns:
            if (self.root_dir / pattern).exists():
                index['documentation_files'].append(pattern)
        
        return index


__all__ = ['DocumentationModule']
