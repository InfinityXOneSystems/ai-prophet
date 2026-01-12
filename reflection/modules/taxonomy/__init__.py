#!/usr/bin/env python3
"""
Taxonomy Module
Classifies, categorizes, and enforces naming conventions across agents.

Author: Manus AI
Version: 1.0.0
"""

import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict


class TaxonomyModule:
    """
    Main taxonomy module for classification and naming convention enforcement
    """
    
    # Naming conventions
    NAMING_CONVENTIONS = {
        'snake_case': r'^[a-z][a-z0-9_]*$',
        'kebab-case': r'^[a-z][a-z0-9-]*$',
        'camelCase': r'^[a-z][a-zA-Z0-9]*$',
        'PascalCase': r'^[A-Z][a-zA-Z0-9]*$',
        'UPPER_SNAKE_CASE': r'^[A-Z][A-Z0-9_]*$'
    }
    
    # Repository type suffixes
    REPO_SUFFIXES = {
        'system': '-system',      # Independently deployable systems
        'library': '-lib',        # Reusable libraries
        'tool': '-tool',          # Standalone tools
        'template': '-template',  # Project templates
        'config': '-config',      # Configuration repositories
        'docs': '-docs'           # Documentation repositories
    }
    
    # File type classifications
    FILE_CLASSIFICATIONS = {
        'core': ['core', 'engine', 'main', 'base'],
        'module': ['module', 'plugin', 'extension'],
        'utility': ['util', 'utils', 'helper', 'helpers'],
        'config': ['config', 'configuration', 'settings'],
        'test': ['test', 'tests', 'spec', 'specs'],
        'documentation': ['doc', 'docs', 'readme'],
        'data': ['data', 'dataset', 'assets'],
        'model': ['model', 'models', 'weights']
    }
    
    def __init__(self, agent, config: Optional[Dict] = None):
        """
        Initialize taxonomy module
        
        Args:
            agent: Agent instance
            config: Configuration dictionary
        """
        self.agent = agent
        self.config = config or {}
        self.root_dir = Path(self.config.get('root_dir', '.'))
        
        self.naming_convention = self.config.get('naming_convention', 'snake_case')
        self.enforce_standards = self.config.get('enforce_standards', True)
        self.auto_rename = self.config.get('auto_rename', False)
        self.system_suffix = self.config.get('system_suffix', '-system')
        
        self.results = None
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute complete taxonomy analysis
        
        Returns:
            Dictionary with taxonomy results
        """
        print("   🏷️  Classifying components...")
        
        # Get agent metadata
        metadata = self.agent.get_metadata()
        
        # Classify agent
        agent_classification = self._classify_agent(metadata)
        print(f"      Agent type: {agent_classification['type']}")
        
        # Analyze repository naming
        repo_analysis = self._analyze_repository_naming(metadata)
        
        # Analyze file naming conventions
        print("   📝 Checking naming conventions...")
        naming_violations = self._check_naming_conventions()
        print(f"      Found {len(naming_violations)} naming violations")
        
        # Build taxonomy hierarchy
        taxonomy_tree = self._build_taxonomy_tree()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            repo_analysis,
            naming_violations
        )
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'agent_classification': agent_classification,
            'repository_analysis': repo_analysis,
            'naming_violations': naming_violations,
            'taxonomy_tree': taxonomy_tree,
            'recommendations': recommendations
        }
        
        return self.results
    
    def _classify_agent(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Classify the agent based on its metadata and capabilities"""
        
        agent_type = metadata.get('type', 'generic')
        capabilities = metadata.get('capabilities', [])
        
        # Determine domain
        domain = self._determine_domain(agent_type, capabilities)
        
        # Determine tier (production, staging, development)
        tier = self._determine_tier()
        
        # Determine maturity (experimental, beta, stable, mature)
        maturity = self._determine_maturity()
        
        # Determine complexity (low, medium, high)
        complexity = self._determine_complexity()
        
        return {
            'type': agent_type,
            'domain': domain,
            'tier': tier,
            'maturity': maturity,
            'complexity': complexity,
            'capabilities': capabilities
        }
    
    def _determine_domain(self, agent_type: str, capabilities: List[str]) -> str:
        """Determine the domain of the agent"""
        domain_keywords = {
            'financial': ['trading', 'prediction', 'finance', 'market', 'portfolio'],
            'data_processing': ['scraper', 'etl', 'pipeline', 'data'],
            'ml_ai': ['model', 'training', 'inference', 'ml', 'ai'],
            'automation': ['workflow', 'automation', 'orchestration'],
            'analytics': ['analysis', 'analytics', 'reporting', 'insights']
        }
        
        # Check agent type and capabilities
        all_terms = [agent_type.lower()] + [c.lower() for c in capabilities]
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in term for term in all_terms for keyword in keywords):
                return domain
        
        return 'general'
    
    def _determine_tier(self) -> str:
        """Determine deployment tier"""
        # Check for environment indicators
        if (self.root_dir / '.env.production').exists():
            return 'production'
        elif (self.root_dir / '.env.staging').exists():
            return 'staging'
        else:
            return 'development'
    
    def _determine_maturity(self) -> str:
        """Determine maturity level"""
        # Check version from metadata
        metadata = self.agent.get_metadata()
        version = metadata.get('version', '0.1.0')
        
        major, minor, patch = version.split('.')[:3]
        major = int(major)
        
        if major >= 2:
            return 'mature'
        elif major >= 1:
            return 'stable'
        elif minor >= 5:
            return 'beta'
        else:
            return 'experimental'
    
    def _determine_complexity(self) -> str:
        """Determine complexity level based on codebase size"""
        # Count Python files and total lines
        py_files = list(self.root_dir.rglob('*.py'))
        total_lines = 0
        
        for py_file in py_files[:100]:  # Limit to avoid long processing
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    total_lines += len(f.readlines())
            except:
                pass
        
        if total_lines > 10000:
            return 'high'
        elif total_lines > 3000:
            return 'medium'
        else:
            return 'low'
    
    def _analyze_repository_naming(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze repository naming conventions"""
        
        repo_url = metadata.get('repository', '')
        if not repo_url:
            return {
                'has_repository': False,
                'compliant': False,
                'issues': ['No repository URL provided']
            }
        
        # Extract repo name from URL
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        
        # Check if it's a system
        agent_type = metadata.get('type', '')
        is_system = 'system' in agent_type or 'agent' in agent_type
        
        issues = []
        suggestions = []
        
        # Check for system suffix
        if is_system and not repo_name.endswith(self.system_suffix):
            issues.append(f"System repository should end with '{self.system_suffix}'")
            suggestions.append(f"Rename to: {repo_name}{self.system_suffix}")
        
        # Check naming convention
        if not re.match(self.NAMING_CONVENTIONS['kebab-case'], repo_name.replace(self.system_suffix, '')):
            issues.append("Repository name should use kebab-case")
        
        return {
            'has_repository': True,
            'repository_name': repo_name,
            'is_system': is_system,
            'compliant': len(issues) == 0,
            'issues': issues,
            'suggestions': suggestions
        }
    
    def _check_naming_conventions(self) -> List[Dict[str, Any]]:
        """Check naming conventions for all files"""
        violations = []
        
        # Check Python files
        for py_file in self.root_dir.rglob('*.py'):
            if self._should_skip_file(py_file):
                continue
            
            filename = py_file.stem
            
            # Skip special files
            if filename.startswith('__'):
                continue
            
            # Check if filename follows convention
            pattern = self.NAMING_CONVENTIONS[self.naming_convention]
            if not re.match(pattern, filename):
                violations.append({
                    'file': str(py_file.relative_to(self.root_dir)),
                    'current_name': filename,
                    'issue': f"Does not follow {self.naming_convention}",
                    'suggestion': self._suggest_name_fix(filename, self.naming_convention)
                })
        
        # Check directory names
        for directory in self.root_dir.rglob('*'):
            if not directory.is_dir() or self._should_skip_file(directory):
                continue
            
            dir_name = directory.name
            
            # Skip special directories
            if dir_name.startswith('.') or dir_name.startswith('__'):
                continue
            
            pattern = self.NAMING_CONVENTIONS[self.naming_convention]
            if not re.match(pattern, dir_name):
                violations.append({
                    'file': str(directory.relative_to(self.root_dir)),
                    'current_name': dir_name,
                    'issue': f"Directory does not follow {self.naming_convention}",
                    'suggestion': self._suggest_name_fix(dir_name, self.naming_convention)
                })
        
        return violations
    
    def _should_skip_file(self, filepath: Path) -> bool:
        """Check if file should be skipped"""
        skip_patterns = ['__pycache__', 'venv', 'env', '.git', 'node_modules', '.pytest_cache']
        return any(pattern in str(filepath) for pattern in skip_patterns)
    
    def _suggest_name_fix(self, name: str, convention: str) -> str:
        """Suggest a corrected name based on convention"""
        if convention == 'snake_case':
            # Convert to snake_case
            name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
            name = name.replace('-', '_').replace(' ', '_')
            return name.lower()
        
        elif convention == 'kebab-case':
            # Convert to kebab-case
            name = re.sub('(.)([A-Z][a-z]+)', r'\1-\2', name)
            name = re.sub('([a-z0-9])([A-Z])', r'\1-\2', name)
            name = name.replace('_', '-').replace(' ', '-')
            return name.lower()
        
        elif convention == 'PascalCase':
            # Convert to PascalCase
            parts = re.split(r'[-_\s]', name)
            return ''.join(word.capitalize() for word in parts)
        
        return name
    
    def _build_taxonomy_tree(self) -> Dict[str, Any]:
        """Build hierarchical taxonomy of components"""
        tree = {
            'root': str(self.root_dir),
            'children': []
        }
        
        # Classify directories
        for directory in sorted(self.root_dir.iterdir()):
            if not directory.is_dir() or self._should_skip_file(directory):
                continue
            
            classification = self._classify_directory(directory.name)
            
            tree['children'].append({
                'name': directory.name,
                'type': 'directory',
                'classification': classification,
                'path': str(directory.relative_to(self.root_dir))
            })
        
        return tree
    
    def _classify_directory(self, dir_name: str) -> str:
        """Classify a directory based on its name"""
        dir_lower = dir_name.lower()
        
        for classification, keywords in self.FILE_CLASSIFICATIONS.items():
            if any(keyword in dir_lower for keyword in keywords):
                return classification
        
        return 'other'
    
    def _generate_recommendations(self, repo_analysis: Dict, violations: List[Dict]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Repository naming recommendations
        if not repo_analysis.get('compliant', True):
            for issue in repo_analysis.get('issues', []):
                recommendations.append(f"🏷️  Repository: {issue}")
            for suggestion in repo_analysis.get('suggestions', []):
                recommendations.append(f"   💡 {suggestion}")
        
        # Naming convention recommendations
        if violations:
            recommendations.append(f"📝 Found {len(violations)} naming convention violations")
            if self.auto_rename:
                recommendations.append("   ✅ Auto-rename is enabled - violations will be fixed automatically")
            else:
                recommendations.append("   ⚠️  Enable auto_rename in config to fix automatically")
                recommendations.append(f"   💡 Or manually rename files to follow {self.naming_convention}")
        
        # General recommendations
        if len(violations) == 0 and repo_analysis.get('compliant', True):
            recommendations.append("✅ All naming conventions are compliant!")
        
        return recommendations


__all__ = ['TaxonomyModule']
