"""Tables and plots shared by Results, the database explorer, QC and Statmodels.

* :mod:`.plot_spec`: :class:`PlotSpec`, a plot as data.
* :mod:`.render_mpl`: draws a spec with seaborn / Matplotlib.
* :mod:`.plot_builder`: :class:`PlotBuilder`, the interactive widget.
"""

from __future__ import annotations

from nvitk.gui.analysis.plot_spec import PLOT_KINDS, PlotSpec

__all__ = ["PLOT_KINDS", "PlotSpec"]
