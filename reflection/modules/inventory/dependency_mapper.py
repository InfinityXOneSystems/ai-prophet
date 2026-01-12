#!/usr/bin/env python3
"""
Dependency Mapper Module
Maps and tracks internal and external dependencies for agents.

Author: Manus AI
Version: 1.0.0
"""

import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict


class DependencyMapper:
    """
    Maps dependencies for Python projects
    Tracks both internal (project) and external (package) dependencies
    """
    
    def __init__(self, root_dir: str, config: Optional[Dict] = None):
        """
        Initialize dependency mapper
        
        Args:
            root_dir: Root directory of the project
            config: Configuration dictionary
        """
        self.root_dir = Path(root_dir)
        self.config = config or {}
        self.internal_deps = defaultdict(set)  # file -> set of internal imports
        self.external_deps = set()  # set of external package names
        self.dependency_graph = {}
    
    def analyze_dependencies(self) -> Dict[str, Any]:
        """
        Analyze all dependencies in the project
        
        Returns:
            Dictionary with dependency analysis results
        """
        # Find all Python files
        python_files = list(self.root_dir.rglob('*.py'))
        
        for py_file in python_files:
            if self._should_skip(py_file):
                continue
            
            try:
                self._analyze_file(py_file)
            except Exception as e:
                print(f"Error analyzing {py_file}: {e}")
        
        # Build dependency graph
        self._build_dependency_graph()
        
        # Generate report
        return self._generate_report()
    
    def _should_skip(self, filepath: Path) -> bool:
        """Check if file should be skipped"""
        skip_patterns = ['__pycache__', 'venv', 'env', '.git', 'node_modules']
        return any(pattern in str(filepath) for pattern in skip_patterns)
    
    def _analyze_file(self, filepath: Path):
        """Analyze a single Python file for imports"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content, filename=str(filepath))
            
            # Get relative path from root
            try:
                relative_path = filepath.relative_to(self.root_dir)
            except ValueError:
                relative_path = filepath
            
            file_key = str(relative_path)
            
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self._process_import(file_key, alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self._process_import(file_key, node.module)
        
        except SyntaxError:
            # Skip files with syntax errors
            pass
    
    def _process_import(self, file_key: str, import_name: str):
        """Process an import statement"""
        # Get the top-level package name
        top_level = import_name.split('.')[0]
        
        # Check if it's an internal import
        if self._is_internal_import(top_level):
            self.internal_deps[file_key].add(import_name)
        else:
            self.external_deps.add(top_level)
    
    def _is_internal_import(self, package_name: str) -> bool:
        """Check if an import is internal to the project"""
        # Check if there's a directory or file with this name in the project
        potential_paths = [
            self.root_dir / package_name,
            self.root_dir / f"{package_name}.py",
            self.root_dir / 'src' / package_name,
            self.root_dir / 'src' / f"{package_name}.py"
        ]
        
        return any(p.exists() for p in potential_paths)
    
    def _build_dependency_graph(self):
        """Build a dependency graph showing relationships"""
        self.dependency_graph = {
            'nodes': [],
            'edges': []
        }
        
        # Add nodes for each file
        for file_key in self.internal_deps.keys():
            self.dependency_graph['nodes'].append({
                'id': file_key,
                'type': 'internal_file'
            })
        
        # Add edges for dependencies
        for file_key, deps in self.internal_deps.items():
            for dep in deps:
                self.dependency_graph['edges'].append({
                    'source': file_key,
                    'target': dep,
                    'type': 'imports'
                })
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate dependency analysis report"""
        # Convert sets to lists for JSON serialization
        internal_deps_list = {
            k: list(v) for k, v in self.internal_deps.items()
        }
        
        # Find most imported modules
        import_counts = defaultdict(int)
        for deps in self.internal_deps.values():
            for dep in deps:
                import_counts[dep] += 1
        
        most_imported = sorted(
            import_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Find files with most dependencies
        most_dependent = sorted(
            [(k, len(v)) for k, v in self.internal_deps.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            'internal_dependencies': internal_deps_list,
            'external_dependencies': sorted(list(self.external_deps)),
            'statistics': {
                'total_internal_files': len(self.internal_deps),
                'total_external_packages': len(self.external_deps),
                'total_internal_imports': sum(len(v) for v in self.internal_deps.values())
            },
            'most_imported_modules': [
                {'module': module, 'import_count': count}
                for module, count in most_imported
            ],
            'most_dependent_files': [
                {'file': file, 'dependency_count': count}
                for file, count in most_dependent
            ],
            'dependency_graph': self.dependency_graph
        }
    
    def get_external_dependencies(self) -> List[str]:
        """Get list of external package dependencies"""
        return sorted(list(self.external_deps))
    
    def get_internal_dependencies(self, file_path: str) -> List[str]:
        """Get internal dependencies for a specific file"""
        return list(self.internal_deps.get(file_path, set()))
    
    def find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies in the project"""
        # Simple cycle detection using DFS
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for dep in self.internal_deps.get(node, []):
                if dep not in visited:
                    dfs(dep, path.copy())
                elif dep in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(dep)
                    cycle = path[cycle_start:] + [dep]
                    cycles.append(cycle)
            
            rec_stack.remove(node)
        
        for file in self.internal_deps.keys():
            if file not in visited:
                dfs(file, [])
        
        return cycles
    
    def analyze_requirements_file(self, requirements_path: str = 'requirements.txt') -> Dict[str, Any]:
        """
        Analyze requirements.txt file and compare with actual usage
        
        Returns:
            Dictionary with requirements analysis
        """
        req_file = self.root_dir / requirements_path
        
        if not req_file.exists():
            return {
                'exists': False,
                'declared_packages': [],
                'used_packages': sorted(list(self.external_deps)),
                'unused_packages': [],
                'missing_packages': sorted(list(self.external_deps))
            }
        
        # Parse requirements.txt
        declared_packages = set()
        with open(req_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name (before ==, >=, etc.)
                    package = re.split(r'[=<>!]', line)[0].strip()
                    declared_packages.add(package.lower())
        
        # Compare with actual usage
        used_packages = {pkg.lower() for pkg in self.external_deps}
        
        unused = declared_packages - used_packages
        missing = used_packages - declared_packages
        
        return {
            'exists': True,
            'declared_packages': sorted(list(declared_packages)),
            'used_packages': sorted(list(used_packages)),
            'unused_packages': sorted(list(unused)),
            'missing_packages': sorted(list(missing)),
            'is_synchronized': len(unused) == 0 and len(missing) == 0
        }
    
    def save_report(self, filepath: str, report: Dict[str, Any]):
        """Save dependency report to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)


if __name__ == '__main__':
    # Example usage
    mapper = DependencyMapper('.')
    report = mapper.analyze_dependencies()
    print(f"External dependencies: {len(report['external_dependencies'])}")
    print(f"Internal files: {report['statistics']['total_internal_files']}")
