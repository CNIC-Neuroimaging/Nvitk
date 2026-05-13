#!/usr/bin/env python3
"""
List all CLI commands defined in pyproject.toml organized by sub-modules
"""

import re
from pathlib import Path

import sys
project_root = Path(__file__).resolve().parents[3]
src_dir = project_root / 'src' / 'nvitk'
sys.path.insert(0, str(src_dir))

from nvitk.util.colors import bcolors as Colors


def _get_log():
    # Lazy import avoids triggering logger-related import cycles at module import time.
    from nvitk.core.logger import Logger
    return Logger()

def categorize_command(cmd, module):
    """Categorize command based on its name and module path"""
    
    # General/Conversion commands (top priority)
    if any(conv in cmd.lower() for conv in ['dcm2nii', 'stl2nifti', 'nikon2nifti', 'phase2volume']):
        return "Image Conversion"
    
    # Segmentation commands
    if any(seg in cmd for seg in ['nvitk-totalseg', 'nvitk-eicab']):
        return "Segmentation"

    # Registration commands
    if 'nvitk-flirt' in cmd:
        return "Registration"
    
    # PESA Fat commands
    if cmd.startswith('nvitk-pesa-fat') or any(pesa in cmd for pesa in []):
        return "PESA-Fat Analysis"
    
    # PESA Brain commands
    if cmd.startswith('nvitk-qvtpy') or any(qvt in cmd for qvt in []):
        return "PESA-Brain Analysis"
    
    # Default category
    return "General"


def get_command_color(category):
    """Get the appropriate color for each command category"""
    color_map = {
        "Image Conversion": Colors.OKCYAN,
        "Segmentation": Colors.OKGREEN,
        "Registration": Colors.HEADER,
        "PESA-Fat Analysis": Colors.WARNING,
        "PESA-Brain Analysis": Colors.OKBLUE,
        "General": Colors.WHITE
    }
    return color_map.get(category, Colors.WHITE)


def find_pyproject_toml():
    """Find pyproject.toml file by searching from current directory up to project root"""
    current = Path.cwd()
    
    # Try current directory first
    pyproject_path = current / "pyproject.toml"
    if pyproject_path.exists():
        return pyproject_path
    
    # Walk up the directory tree to find pyproject.toml
    for parent in current.parents:
        pyproject_path = parent / "pyproject.toml"
        if pyproject_path.exists():
            return pyproject_path
    
    # If not found, try relative to this file's location
    file_dir = Path(__file__).parent
    project_root = file_dir.parent.parent  # Go up from src/util/ to project root
    pyproject_path = project_root / "pyproject.toml"
    if pyproject_path.exists():
        return pyproject_path
    
    return None


def list_cli_commands():
    """Parse pyproject.toml and list all defined CLI commands organized by category"""
    log = _get_log()
    log.info(f"Adding {src_dir} to sys.path")

    pyproject_path = find_pyproject_toml()
    if not pyproject_path:
        log.error("Error: pyproject.toml not found")
        return

    with open(pyproject_path, 'r') as f:
        content = f.read()

    # Find the [project.scripts] section
    scripts_match = re.search(r'\[project\.scripts\](.*?)(?=\[|$)', content, re.DOTALL)

    if not scripts_match:
        log.warning("No [project.scripts] section found in pyproject.toml")
        return

    scripts_content = scripts_match.group(1)

    # Parse each script entry
    commands = []
    for line in scripts_content.strip().split('\n'):
        if '=' in line and not line.strip().startswith('#'):
            parts = line.split('=', 1)
            if len(parts) == 2:
                cmd = parts[0].strip()
                module = parts[1].strip().strip('"')
                commands.append((cmd, module))

    if not commands:
        log.warning("No CLI commands found")
        return

    # Categorize commands
    categorized = {}
    for cmd, module in commands:
        category = categorize_command(cmd, module)
        if category not in categorized:
            categorized[category] = []
        categorized[category].append((cmd, module))

    # Define display order (general commands first)
    display_order = [
        "Image Conversion",
        "Segmentation", 
        "Registration",
        "PESA-Fat Analysis",
        "PESA-Brain Analysis",
        "General"
    ]

    # Display header
    log.info("=" * 80)
    log.info(f"{Colors.BOLD}{Colors.OKBLUE}Nvitk CLI Commands{Colors.ENDC}")
    log.info("=" * 80)
    log.info(f"{Colors.WHITE}Available command-line interfaces organized by functionality{Colors.ENDC}")
    log.info("=" * 80)

    total_commands = 0
    
    # Display commands by category
    for category in display_order:
        if category in categorized:
            commands_in_category = categorized[category]
            total_commands += len(commands_in_category)
            
            log.info(f"{Colors.BOLD}{category}{Colors.ENDC}")
            log.info("─" * len(category))
            
            for cmd, module in commands_in_category:
                # Format module path for better readability
                module_parts = module.split('.')
                if len(module_parts) >= 2:
                    module_display = f"{'.'.join(module_parts[:-1])}.{module_parts[-1]}"
                else:
                    module_display = module
                
                # Get color for this command category
                cmd_color = get_command_color(category)
                colored_cmd = f"{cmd_color}{Colors.BOLD}{cmd}{Colors.ENDC}"
                
                # Check if the line would be too long
                line = f"  {colored_cmd:<20} → {Colors.GRAY}{module_display}{Colors.ENDC}"
                if len(f"  {cmd:<20} → {module_display}") > 100:
                    log.info(f"  {colored_cmd:<20} →")
                    log.info(f"    {Colors.GRAY}{module_display}{Colors.ENDC}")
                else:
                    log.info(line)
            
            log.info("")  # Empty line between categories

    # Display footer
    log.info("=" * 80)
    log.info(f"{Colors.BOLD}Total commands: {Colors.OKGREEN}{total_commands}{Colors.ENDC}")
    log.info("=" * 80)
    log.info(f"{Colors.BOLD}Backend Options:{Colors.ENDC}")
    log.info(f"   {Colors.WHITE}• --backend numpy  (CPU processing){Colors.ENDC}")
    log.info(f"   {Colors.WHITE}• --backend cupy   (GPU processing){Colors.ENDC}")
    log.info(f"   {Colors.WHITE}[• --backend gpu    (GPU processing alias for some commands){Colors.ENDC}]")
    log.info("=" * 80)


def main():
    """Entry point for the pyhelp CLI command"""
    list_cli_commands()


if __name__ == "__main__":
    main()

