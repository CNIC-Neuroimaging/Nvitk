"""Prove that edited Python files kept their code: compare each one's AST with a base revision.

Documentation strings are blanked before the comparison, so a file passes when only comments,
docstrings or formatting changed. Files whose code was changed on purpose are listed with
``--allow`` and reported separately.

Usage::

    python audit/scripts/check_code_unchanged.py ec706c1 --allow src/nvitk/filters/sliding_threshold.py ...
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from deslop import code_fingerprint  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("base", help="git revision to compare against")
ap.add_argument("--allow", nargs="*", default=[], help="files whose code changed on purpose")
args = ap.parse_args()
root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
changed = subprocess.run(["git", "diff", "--name-only", args.base], cwd=root, capture_output=True, text=True,
                         check=True).stdout.split()
same, unexpected, allowed = 0, [], []
for path in (p for p in changed if p.endswith(".py")):
    full = os.path.join(root, path)
    if not os.path.exists(full):
        continue
    old = subprocess.run(["git", "show", f"{args.base}:{path}"], cwd=root, capture_output=True, text=True).stdout
    if not old:
        continue  # new file
    new = open(full, encoding="utf-8").read()
    if code_fingerprint(old) == code_fingerprint(new):
        same += 1
    elif path in args.allow:
        allowed.append(path)
    else:
        unexpected.append(path)
print(f"Python files changed since {args.base}: {same + len(allowed) + len(unexpected)} | code identical: {same}"
      f" | intended code edits: {len(allowed)} | unexpected: {len(unexpected)}")
for path in allowed:
    print("  intended  ", path)
for path in unexpected:
    print("  UNEXPECTED", path)
sys.exit(1 if unexpected else 0)
