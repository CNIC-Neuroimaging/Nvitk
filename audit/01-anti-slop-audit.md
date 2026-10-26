# Anti-slop audit

Scope: every nvitk-authored file in `src/`, `slicer/`, `scripts/` and the packaging files
(`pyproject.toml`, `install.sh`, `recipe/`), at commit `ec706c1`. The vendored nnU-Net and nnssl
engines under `src/nvitk/pipes/_engines/` were left alone, except for the trainers nvitk wrote
inside them.

Three rule sets were applied:

| Rule set | What it asks for | How it was applied here |
|---|---|---|
| **Anti-slop** (MCP Market `anti-slop` skill) | Remove generic AI patterns: decorative section banners, filler words, meta-commentary, obvious comments, commented-out code, generic error handling | Comments and docstrings rewritten (section 1); code patterns measured and reported (section 3) |
| **Thermos** (`thermo-nuclear-code-quality-review`, `thermo-nuclear-review`) | Structural quality: files over 1000 lines, spaghetti branching, thin wrappers, magic mechanisms, duplicated helpers, correctness bugs | Measured with radon, ruff, vulture and an AST duplicate finder; findings in section 3; correctness bugs in the methodological audit |
| **Karpathy guidelines** | Surgical changes, no speculative abstraction, surface assumptions, verifiable success criteria | Every edit is either comment/docstring-only (proved by AST comparison) or a code change with an equivalence test; no refactor was done without a test to hold it |

The constraint for this step was that outputs must not change. Every applied change is in one of
two classes, each with its own proof:

1. **Comment and docstring edits.** The AST of each edited file, with docstrings blanked, is
   identical to the original: no statement, literal, name or default value changed.
2. **Three performance rewrites.** Bit-identical outputs, checked against the previous
   implementation on thousands of random inputs (section 2), plus a permanent test file.

Check it yourself:

```bash
python audit/scripts/check_code_unchanged.py ec706c1 \
    --allow src/nvitk/filters/sliding_threshold.py src/nvitk/measure/segmentation_metrics.py
# Python files changed since ec706c1: 556 | code identical: 554 | intended code edits: 2 | unexpected: 0

pytest tests/test_output_preservation.py      # 90 tests, previous loops vs current code
```

---

## 1. Comments and docstrings

### What the code looked like

The comments are mostly good engineering comments: they explain *why* (a vendor quirk, a
failure seen in a real subject, a numerical trap) rather than restating the code. The
docstrings document parameters, units and conventions. The AI signature was in the form, not the
content:

* decorative section banners, three lines each (`# ─────` / `# Title` / `# ─────`);
* em dashes as the default punctuation, often in pairs;
* typographic arrows and symbols (`→`, `≥`, `…`) where a person types `->`, `>=`, `...`;
* filler adverbs and rhetorical framing: "actually", "genuinely", "simply", "quietly",
  "the honest answer", "which is the whole point", "the worst kind of silent error";
* commented-out code left in place of a decision.

### Counts (comments and docstrings of nvitk-authored Python, Slicer vendored copies excluded)

| Pattern | Before | After |
|---|---|---|
| Rule-only comment lines (banners) | 1,487 | 0 |
| Box-drawing characters in comments | 74,866 | 65 (inside the ASCII diagrams, kept) |
| Em dashes | 2,810 | 1 (a literal dash placeholder that the GUI displays) |
| Typographic arrows | 647 | 0 |
| Ellipsis character | 301 | 0 |
| "actually" / "genuinely" / "simply" / "essentially" | 173 / 34 / 31 / 3 | 3 / 3 / 1 / 0 (left where the word carries a contrast) |
| "quietly" | 20 | 0 ("silently", the term used everywhere else) |
| "honest" / "the whole point" / "the point is" and similar | ~40 | 0 in comments and docstrings |
| Comment lines | 10,965 | 9,417 |

User-facing strings (GUI labels, log messages, CLI help passed as `help=`, HTML templates) were
not touched: they are program output.

### How the em dashes were replaced

Mechanically, but not blindly. `audit/scripts/deslop.py` reads each comment block or docstring as
one text so a pair can span lines, and picks the punctuation from the context:

| Pattern | Becomes | Example |
|---|---|---|
| list label or term before the dash | colon | ``* :func:`dice` — volumetric overlap`` -> ``* :func:`dice`: volumetric overlap`` |
| two dashes in one sentence | parentheses (commas when the aside already has parentheses) | `a row — or a block of rows — per layer` -> `a row (or a block of rows) per layer` |
| dash before `which`, `so`, `but`, `and`, `not`, ... | comma | `an alias — so a shared column` -> `an alias, so a shared column` |
| dash before `see` | semicolon | ``— see :mod:`x` `` -> ``; see :mod:`x` `` |
| explanation after a clause | colon (semicolon if the sentence already has one) | `on the host — sending a frame costs more` -> `on the host: sending a frame costs more` |
| `# noqa: CODE — reason` | `# noqa: CODE (reason)` | still parsed by ruff |
| dash quoted as a literal value (inside double backticks or quotes) | kept | it is a value, not punctuation |

About 150 random rewrites and every case that changed parenthesis balance were read by hand, and
a few dozen were rewritten manually.

### Edits made by hand

* **Commented-out code** removed in `db/storage.py` (a debug block writing to `/tmp`),
  `measure/hemodynamics.py`, `stats/mixedlm.py`, `io/conversors/_dicom_conversion.py`. Where the
  block documented a feature that is switched off, it became one line saying so:
  tissue-segmentation export (`_dicom_conversion.py`), the SGE parallel environment
  (`cluster/sge.py`), automatic XNAT download insertion (`pipes/qvtpy/run.py`), centerline export
  for 3D+t (`stage4t_4dflow_t_segmentation.py`), HEAD paravertebral measures
  (`pesa_fat/dixon_v5/config.py`).
* **Comments that were wrong**, corrected to match the code:
  * `io/writers/nifti.py`: "Use code 16 (comment)". Code 16 is `PYPICKLE` in the NIfTI registry;
    6 is `COMMENT`. The comment now says so.
  * `measure/suv.py`: `philips` was documented as "If True (default)"; the default is `False`.
  * `segmentation/hemisphere.py`: the docstrings said the side comes from the sign of
    `affine[0, 0]` for both functions; `split_lr_by_cc` compares RAS world x, and
    `split_lr_by_midline` assumes voxel axis 0 is left-right. Both now say what they do.
  * `restoration/bilateral.py`: the empty `if do_3d: pass` block now states that the CPU path
    filters slice by slice and therefore differs from the CuPy path.
  * `segmentation/losses.py`: reference to a function that does not exist
    (`_foreground_probabilities` -> `_foreground_pairs`).
* **`io/imageio.py`**: the `imshow` section comments (`# 1. Data Prep`, `# BLOCK: FIGURE SETUP`,
  `# CASE: MOSAIC`) replaced with one line per branch saying what it draws.
* **`stats/brain_map.py`**: invalid escape `\|` in a docstring (SyntaxWarning on Python 3.12+);
  written as `\\|`, which keeps the string value identical.
* **Rhetorical phrasing** ("that is the whole point", "the worst kind of silent error", "keep the
  button honest", ...) rewritten in about 40 comments and docstrings.

### Other files

* `pyproject.toml`, `recipe/recipe.yaml`, `recipe/post-link.sh`, `recipe/activate.d/nvitk-gui.sh`,
  `recipe/fastsurfer/install_fastsurfer.sh`: same banner and punctuation rules on full-line `#`
  comments. Parsed data is identical (`tomllib` / `yaml.safe_load`), shell scripts pass
  `bash -n`, and their non-comment lines are unchanged.
* `slicer/MouseTOFMorphometrics/.../nvitk_vendor/`: regenerated with `vendor_sync.py` from the
  cleaned `src/`, so the vendored copy stays an exact image of upstream
  (`vendor_sync.py --check` passes). The generated "VENDORED FROM nvitk" header comes from
  `vendor_sync.py` and was left as it is.

---

## 2. Output-preserving performance changes

Only loops that were both slow and replaceable by an exact computation were changed. Each new
version returns bit-identical results; the equivalence runs compared it with the previous code
on random inputs, including ties, NaN/inf, negative and very large label values and boolean
masks.

| Function | Before | After | Equivalence | Speed |
|---|---|---|---|---|
| `filters/sliding_threshold.py`: `binary_mask_sliding_threshold_3d/_2d` | one full-volume `count_nonzero(x > t)` per threshold (801) | one sort + `searchsorted(side="right")`; falls back to the loop for non-finite maxima | 1,024 cases identical (masks and thresholds) | 3-D 160x160x80: 2.0 s -> 1.05 s; 2-D cross-section (called per station): 2.9 ms -> 0.3 ms |
| `measure/segmentation_metrics.py`: `detection_matches` | two full-volume masks per (reference, prediction) component pair | component sizes with `bincount`, overlaps with one `unique` on label pairs; same greedy order and tie-breaking | 1,800 cases identical | 20 k components: 620 s -> 1.2 s |
| `measure/segmentation_metrics.py`: `neighbouring_labels` | one dilation per label plus one full-volume AND per label pair | 13 shifted comparisons over the non-zero bounding box, compact codes, K x K histogram; the dilation path is kept for 3 labels or fewer, where it is faster | 3,600 cases identical | 36 labels on 256x256x160: 17.4 s -> 0.56 s; 1-2 labels unchanged |

`tests/test_output_preservation.py` keeps the previous loops as reference implementations and
compares them with the current code on seeded random inputs (90 tests, under a second).

Not changed, although slower than necessary, because a faster version would not be
bit-identical: the O(n²) circular cross-correlation in `measure/hemodynamics.py` (an FFT changes
the floating-point sums and could change the argmax on near-ties), and N4 running twice when the
bias field is requested (`restoration/n4_bias.py`).

---

## 3. Code-level findings (reported, not changed)

These would change behaviour or structure, so they are recommendations. Numbers are for
nvitk-authored files outside the Slicer vendored copy.

### 3.1 Size and structure (Thermos)

* **62 files exceed 1000 lines.** The largest are GUI modules: `gui/panels/statmodels/window.py`
  (7,166), `gui/tools/runner.py` (5,108), `gui/viz/ortho_panel.py` (4,550),
  `io/conversors/_dicom_conversion.py` (3,340), `gui/labels/labeling.py` (3,220),
  `viz/flowshow.py` (3,183), `gui/tools/registry.py` (3,125), `measure/voxelwise.py` (2,636),
  `pipes/qvtpy/run.py` (2,364).
* **Branching hot spots** (radon cyclomatic complexity): `gui/tools/runner.py:_run_gui_tool`
  (327), `io/conversors/_dicom_zeiss.py:_extract_zeiss_raw_oct` (171), `pipes/qvtpy/run.py:main`
  (130), `gui/viz/qc_loader.py:load_qvtpy_qc_layers` (99),
  `measure/morpho/orchestration.py:_process_component_tree_vmtk` (88),
  `io/conversors/_dicom_rtstructs.py:_create_masks_from_contours` (86),
  `pipes/qvtpy/util/hemodynamics/vessel_hemodynamics.py:compute_vessel_hemodynamics` (79).
  `_run_gui_tool` is a chain of about 180 `if tool_id == ...` branches over the GUI tools. The
  tool registry already describes each tool; storing the runner callable in the registry entry
  turns the dispatcher into a lookup and lets each tool live next to its parameters.
* **The DICOM converter** (`_convert_ds_list_to_nifti_image`) is a cascade of nested
  `try/except` retries keyed on substrings of dicom2nifti error messages. A table of
  (error code -> strategy) and one loop would make the order of fallbacks explicit and testable;
  see also M-14 in the methodological audit for the strategy that should be dropped.

### 3.2 Magic and hidden coupling

* **`setup(globals())` in 77 modules** injects `np`, `scipy` and `ndi` into module globals and
  rebinds them whenever the backend changes. Static tools cannot see these names (ruff reports
  2,790 undefined `np`/`ndi`/`scipy` references), readers have to know the convention, and the
  rebinding is shared by all threads (M-27). An explicit `xp = get_array_module(x)` at the top of
  each function, or an `xp` argument, gives the same NumPy/CuPy flexibility without the globals.
* **`import nvitk` imports the world.** `nvitk/__init__.py` eagerly imports `db`, `stats`, `io`
  (which pulls in `viz`) and their dependencies, so `from nvitk.types import Image` needs pandas,
  statsmodels and plotting. `lazy-loader` is already a dependency and fits this case.
* **Geometry stored three times** in `Image.metadata` (affine, spacing/`x_res`..., ITK
  origin/direction). This is the root cause of M-10, M-11 and M-12. One source (the affine) with
  derived read-only properties removes the whole class of bugs.
* **Process-wide warning filters at import time**, for example
  `io/conversors/_dicom_conversion.py:84-87` silences scipy and pydicom `UserWarning`s for the
  whole process as soon as the module is imported, including the warning that pointed at M-14.

### 3.3 Error handling (Anti-slop: generic and empty catch blocks)

* 1,396 `except Exception` clauses; 457 of them end in `pass`. Some are deliberate and commented
  (GUI paint code that must not raise), but many sit in numerical paths: `io/writers/nifti.py`
  (`set_zooms` failure ignored), the DICOM orientation fixer, SUV metric loops that turn any error
  into a warning. A failed step should either raise or be recorded in the output provenance.
* `cluster/sshfs.py:605` returns from `finally`, which swallows exceptions from the `with` body.

### 3.4 Defensive code for impossible cases (Karpathy: simplicity first)

* Fallbacks for hard dependencies: `phase2volume.py` defines a no-op `_cli_decorator` in case
  `click` is missing, and several readers wrap `import nibabel` with an install hint, although
  both are required dependencies in `pyproject.toml`.
* `try/except` around plain dictionary access and float conversion in many metadata helpers,
  returning defaults that hide malformed input.

### 3.5 Duplication

Exact duplicate function bodies (AST comparison, names ignored): 27 groups, 453 lines. Examples:
`_experiment_session_uid` (3 copies in `db/xnat_*`), `_neighbors26` (3 copies:
`morphology/centerline.py`, `morphology/polyline_graph.py`, `viz/flowshow.py`),
`_skeleton_graph` (2), `keep_largest_component` (2 in `measure/morpho`),
`_resolve_ssh_credentials` (PESA-Fat and qvtpy), `_stage_resources` (CT-PET and Dixon runners),
`_map_sex_to_numeric` (2 in `stats`), `_plane_basis` (`meshlab/vessels.py`,
`morphology/centering.py`). The importer scripts in `scripts/database/` share several helpers
(`seqn_to_subject`, `publish_replacing`, `_normalize_seqn`).

Near-duplicates matter more than exact ones: there are two Dice functions, two HD95
implementations and two "volume similarity" definitions with different semantics (M-19).

### 3.6 Leftovers

* 201 `print()` calls in `src/` library code (for example `db/storage.py` prints every Parquet
  write); the package has a logger.
* `ruff` F401: 137 unused imports. Not removed here: several modules re-export names that other
  modules import, so each removal needs a cross-module check.
* Trivial docstrings on every dunder and property (`"""Elementwise self + other."""` on
  `Image.__add__` and its eleven siblings). Kept, since they cost nothing at run time and the docs
  site renders them, but they are the "every function has a docstring" pattern the anti-slop
  guide warns about; one class-level note covers them.

### 3.7 Missing guard rails

* **No test suite.** `pyproject.toml` points pytest at `tests/`, which did not exist before this
  audit. The new `tests/test_output_preservation.py` is the first file there.
* **No lint configuration.** Adding a `[tool.ruff]` section with `F`, `E9`, `B`, `BLE`, `ERA`,
  `T20` and `PLE` would have caught M-17 (undefined names passed as keywords), the
  `return`-in-`finally` and the `UnboundLocalError` in `viz/flowshow.py`.
* **CI builds docs and conda packages only.** A job running ruff and pytest on pushes would keep
  the above from coming back. `audit/scripts/deslop.py FILES` (without `--write`) reports
  `files changed: 0` when new comments follow the same style, and can run in the same job.

---

## 4. Writing style for new comments

What the cleaned code now follows, so new code can match it:

* No section banners. A one-line `# Title` comment is enough; a large file that needs
  navigation aids needs splitting.
* Plain ASCII punctuation in comments and docstrings: colon, comma, semicolon, parentheses;
  `->`, `>=`, `...`. Math symbols (σ, ², ×) are fine in formulas.
* Say why, once. Drop "actually", "genuinely", "simply", and framing such as "which is the whole
  point" or "the honest answer"; state the reason directly.
* No commented-out code. If a feature is switched off, one line saying so (and why) is enough;
  git keeps the old code.
* Units and conventions in docstrings (mm, RAS, axis order) are the most useful thing a docstring
  in this package can carry; keep them.
