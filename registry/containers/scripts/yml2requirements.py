#!/usr/bin/env python3
"""
Convert conda environment.yml to separate conda and pip requirements files.

Usage:
    python yml2requirements.py <environment.yml>
    
This will generate:
    - conda_requirements.txt (conda packages)
    - pip_requirements.txt (pip packages)
"""

import sys
import yaml
from pathlib import Path


def parse_environment_yml(yml_path):
    """Parse environment.yml and split into conda and pip requirements."""
    with open(yml_path, 'r') as f:
        env_data = yaml.safe_load(f)
    
    conda_packages = []
    pip_packages = []
    
    # Extract conda dependencies
    if 'dependencies' in env_data:
        for dep in env_data['dependencies']:
            if isinstance(dep, str):
                # Regular conda package
                conda_packages.append(dep)
            elif isinstance(dep, dict) and 'pip' in dep:
                # Pip packages listed under conda dependencies
                pip_packages.extend(dep['pip'])
    
    return conda_packages, pip_packages


def write_requirements(packages, output_path, prefix=''):
    """Write packages to a requirements file."""
    if not packages:
        print(f"  ⚠️  No packages found for {output_path.name}")
        return
    
    with open(output_path, 'w') as f:
        if prefix:
            f.write(f"# {prefix}\n")
        for pkg in packages:
            f.write(f"{pkg}\n")
    
    print(f"  ✓ Created {output_path.name} ({len(packages)} packages)")


def main():
    if len(sys.argv) != 2:
        print("Usage: python yml2requirements.py <environment.yml>")
        sys.exit(1)
    
    yml_path = Path(sys.argv[1])
    
    if not yml_path.exists():
        print(f"Error: File not found: {yml_path}")
        sys.exit(1)
    
    print(f"Processing {yml_path}...")
    
    # Parse environment.yml
    conda_packages, pip_packages = parse_environment_yml(yml_path)
    
    # Output directory (same as environment.yml)
    output_dir = yml_path.parent
    
    # Write conda requirements
    conda_req_path = output_dir / "conda_requirements.txt"
    write_requirements(
        conda_packages,
        conda_req_path,
        prefix="Conda packages (install with: conda install --file conda_requirements.txt)"
    )
    
    # Write pip requirements
    pip_req_path = output_dir / "pip_requirements.txt"
    write_requirements(
        pip_packages,
        pip_req_path,
        prefix="Pip packages (install with: pip install -r pip_requirements.txt)"
    )
    
    print("\n✅ Conversion complete!")
    print(f"   Conda packages: {len(conda_packages)}")
    print(f"   Pip packages: {len(pip_packages)}")


if __name__ == "__main__":
    main()
