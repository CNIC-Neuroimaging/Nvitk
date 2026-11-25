"""The application shell: a left activity bar switching module panels.

Instead of fourteen tabs side by side, nvitk's panels are grouped into a few *modules* (Data &
QC, Tools, Labeling, Mesh, Analysis, Export & record, Inspect). An icon bar down the left edge
picks the module; its panels show in the left dock area (tabbed when there are several), with
Napari's layer list below them. Clicking the active module's icon again folds the panel away
and gives the viewport the room. The centre is the viewport; context panels (results tables,
cross-sections, CPR, lesion inspector...) open on the right.

The panels are still the docks they were (``nvitk:<key>``), so popping one out, the Panels menu
and :func:`~nvitk.gui.core.workspace.raise_panel` keep working; raising a panel switches to its
module first.
"""

from __future__ import annotations

from nvitk.gui.shell.activity_bar import MODULES, Module, Shell, install_shell

__all__ = ["MODULES", "Module", "Shell", "install_shell"]
