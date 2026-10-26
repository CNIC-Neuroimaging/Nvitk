"""FSL BASIL: Bayesian Inference for Arterial Spin Labelling MRI: perfusion (CBF), arrival time
(ATT) and arterial blood volume (aCBV) from ASL data.

* :func:`oxford_asl`: the whole pipeline: label-control subtraction, motion correction,
  the BASIL fit, calibration to ml/100g/min, registration to a structural image (and
  through ``fsl_anat`` to MNI), partial-volume correction and region analysis;
* :func:`basil`: the BASIL fit alone, on label-control *differenced* data;
* :func:`asl_file`: subtraction, averaging and splitting of ASL series;
* :func:`asl_calib`: M0 from a calibration image in a reference region (CSF, WM, GM),
  and a perfusion image calibrated with it.

pCASL takes post-labelling delays (*plds*), pASL inversion times (*tis*); the labelling
(bolus) duration is *bolus* (one value, or one per delay). Single-delay data are fitted
with the bolus, arrival time and arterial component fixed, as ``oxford_asl`` does.

:func:`oxford_asl` writes ``<out>/native_space/``: ``perfusion`` (arbitrary units),
``perfusion_calib`` (ml/100g/min, with a calibration image or M0), ``arrival`` (s),
``aCBV`` / ``aCBV_calib`` (%), their variances, ``mask``, and with a structural image the
GM/WM partial-volume maps, ``perfusion_calib_gm_mean.txt``-style summaries and, with
partial-volume correction, ``pvcorr/`` (GM and WM perfusion separately). Its scripts do
not quote paths: inputs and outputs must not contain whitespace (checked here).

FSL packages ``fsl-oxford_asl``, ``fsl-basil``, ``fsl-fabber_core`` and
``fsl-fabber_models_asl`` (``nvitk-fsl install --components asl``). References: Chappell
et al., IEEE TSP 2009 (BASIL); Groves et al., NeuroImage 2009 (spatial priors); Chappell
et al., MRM 2011 (partial-volume correction); Alsop et al., MRM 2015 (white paper).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

from .run import FslCommand, nii

LABELLINGS: tuple[str, ...] = ("pcasl", "pasl")
#: ``--iaf``: differenced data, tag-control or control-tag pairs (``b``: tags / controls grouped).
ASL_FORMATS: tuple[str, ...] = ("tc", "ct", "diff", "tcb", "ctb")
#: ``--ibf``: blocks of all delays repeated (``rpt``), or all repeats of one delay together (``tis``).
BLOCK_FORMATS: tuple[str, ...] = ("rpt", "tis")
#: ``--cmethod``: voxelwise M0, one M0 from a reference region, or both (two output sets).
CALIB_METHODS: tuple[str, ...] = ("voxel", "single", "both")
TISSUE_REFS: tuple[str, ...] = ("csf", "wm", "gm", "none")
CALIB_MODES: tuple[str, ...] = ("longtr", "satrecov")
PE_DIRECTIONS: tuple[str, ...] = ("x", "y", "z", "-x", "-y", "-z")

#: BASIL parameter -> oxford_asl output name (the ``mean_<param>`` images of a BASIL step).
BASIL_PARAMS: dict[str, str] = {
    "ftiss": "perfusion", "delttiss": "arrival", "fblood": "aCBV", "fwm": "perfusion_wm", "deltwm": "arrival_wm",
}

#: The ASL commands of each kind of call (what ``requires`` checks).
_REQUIRES = {
    "oxford_asl": ("oxford_asl", "basil", "fabber_asl", "asl_file", "fslmaths", "bet"),
    "basil": ("basil", "fabber_asl"),
    "asl_file": ("asl_file",),
    "asl_calib": ("asl_calib", "fslmaths"),
}


# small helpers


def parse_times(value: str | float | Iterable[float] | None) -> list[float]:
    """Delays / durations from ``"0.25, 0.5,0.75"``, a number or a sequence (empty -> ``[]``)."""
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        return [float(t) for t in value.replace(";", ",").replace(" ", ",").split(",") if t.strip()]
    return [float(v) for v in value]


def _times(values: Sequence[float]) -> str:
    return ",".join(f"{float(v):g}" for v in values)


def _no_spaces(**paths: Path | str | None) -> None:
    """FSL's ASL scripts word-split every path: refuse whitespace up front with a clear message."""
    bad = [f"{name} ({p})" for name, p in paths.items() if p is not None and any(c.isspace() for c in str(p))]
    if bad:
        raise ValueError(
            "FSL's ASL scripts cannot handle spaces in paths; move or rename: " + ", ".join(bad)
        )


def _delays(labelling: str, plds: Sequence[float], tis: Sequence[float]) -> tuple[str, list[float]]:
    """``("plds" | "tis", values)``, checking the labelling's kind of delay was given."""
    if labelling not in LABELLINGS:
        raise ValueError(f"labelling must be one of {LABELLINGS}, not {labelling!r}")
    if plds and tis:
        raise ValueError("Give post-labelling delays (pCASL) or inversion times (pASL), not both.")
    if not plds and not tis:
        raise ValueError("Give the post-labelling delay(s) (pCASL) or inversion time(s) (pASL).")
    if plds:
        return "plds", list(plds)
    return "tis", list(tis)


# oxford_asl


def oxford_asl_outputs(out_dir: Path, *, cmethod: str | None = None, calibrated: bool = False, pvcorr: bool = False,
                       structural: bool = False, pv_maps: bool = False, standard: bool = False) -> dict[str, Path]:
    """The images :func:`oxford_asl` writes, by role (``perfusion_calib``, ``arrival`` ...).

    With ``cmethod="both"`` the voxelwise results are the main roles and the reference-region
    ones carry a ``refregion_`` prefix. *pv_maps*: GM/WM partial-volume maps were available (a
    structural image, or given), so their ASL-space copies and the GM/WM ROIs are written.
    *standard*: a structural -> MNI registration was given (``fsl_anat``), so ``std_space/``
    holds the maps in MNI space (``std_`` roles).
    """
    native = Path(out_dir) / "native_space"
    sets = [("", native / "voxelwise"), ("refregion_", native / "refregion")] if cmethod == "both" else [("", native)]
    out: dict[str, Path] = {"dir": Path(out_dir), "native_dir": native, "mask": native / "mask.nii.gz",
                            "log": Path(out_dir) / "logfile"}
    for lead, folder in sets:
        names = ["perfusion", "perfusion_var", "perfusion_norm", "arrival", "arrival_var", "aCBV"]
        if calibrated:
            names += ["perfusion_calib", "perfusion_var_calib", "aCBV_calib"]
        for name in names:
            out[f"{lead}{name}"] = folder / f"{name}.nii.gz"
        if pvcorr:
            pv = folder / "pvcorr"
            for name in ("perfusion", "perfusion_wm", "arrival", "arrival_wm"):
                out[f"{lead}pvcorr_{name}"] = pv / f"{name}.nii.gz"
            if calibrated:
                for name in ("perfusion_calib", "perfusion_wm_calib"):
                    out[f"{lead}pvcorr_{name}"] = pv / f"{name}.nii.gz"
    if structural or pv_maps:
        for name in ("pvgm_inasl", "pvwm_inasl", "gm_roi", "wm_roi"):
            out[name] = native / f"{name}.nii.gz"
    if structural:
        out["asl2struct"] = native / "asl2struct.mat"
        cbf = "perfusion_calib" if calibrated else "perfusion"
        for space, flag in (("struct", True), ("std", standard)):
            if not flag:
                continue
            folder = Path(out_dir) / f"{space}_space"
            out[f"{space}_{cbf}"] = folder / f"{cbf}.nii.gz"
            out[f"{space}_arrival"] = folder / "arrival.nii.gz"
            if pvcorr:
                out[f"{space}_pvcorr_{cbf}"] = folder / "pvcorr" / f"{cbf}.nii.gz"
    if calibrated and cmethod in ("single", "both"):
        out["m0_value"] = Path(out_dir) / "calib" / "M0.txt"
    if calibrated and cmethod in (None, "voxel", "both"):
        out["m0"] = Path(out_dir) / "calib" / "M0.nii.gz"
    return out


def oxford_asl(
    asl: Path,
    out_dir: Path,
    *,
    labelling: str = "pcasl",
    plds: str | Sequence[float] = (),
    tis: str | Sequence[float] = (),
    bolus: str | float | Sequence[float] = 1.8,
    iaf: str = "tc",
    ibf: str = "rpt",
    rpts: str | Sequence[int] = (),
    slicedt: float = 0.0,
    sliceband: int | None = None,
    mask: Path | None = None,
    calib: Path | None = None,
    cmethod: str | None = None,
    tr: float | None = None,
    cgain: float | None = None,
    alpha: float | None = None,
    tissref: str | None = None,
    csf_mask: Path | None = None,
    te: float | None = None,
    m0: float | None = None,
    structural: Path | None = None,
    structural_brain: Path | None = None,
    fslanat: Path | None = None,
    white_paper: bool = False,
    spatial: bool = True,
    motion_correction: bool = False,
    pvcorr: bool = False,
    pvgm: Path | None = None,
    pvwm: Path | None = None,
    infert1: bool = False,
    fixbolus: bool | None = None,
    fixbat: bool = False,
    artsupp: bool = False,
    bat: float | None = None,
    t1: float | None = None,
    t1b: float | None = None,
    region_analysis: bool = False,
    cblip: Path | None = None,
    echospacing: float | None = None,
    pedir: str | None = None,
    extra: Sequence[str] = (),
) -> FslCommand:
    """The full BASIL pipeline (``oxford_asl``) on an ASL series.

    *asl* is the 4D series as acquired (*iaf* ``tc``/``ct`` pairs, or ``diff`` already
    subtracted; *ibf* how the delays are ordered). *calib* (an M0 / proton-density image)
    or *m0* (a value) give perfusion in ml/100g/min; *cmethod* ``voxel`` divides by the M0
    image voxel by voxel, ``single`` by one M0 measured in a reference region (*tissref*,
    CSF by default: needs a structural image or *csf_mask*). A *structural* (whole head;
    *structural_brain* if already brain-extracted) or an *fslanat* folder adds registration,
    GM/WM summaries, structural-space outputs and (with *fslanat*'s MNI warp) standard
    space and *region_analysis*. *pvcorr* separates GM and WM perfusion, with partial-volume
    maps from the structural segmentation or given as *pvgm* / *pvwm* (in ASL space; they also
    give the GM / WM summaries without a structural image).
    """
    kind, delays = _delays(labelling, parse_times(plds), parse_times(tis))
    taus = parse_times(bolus)
    if not taus:
        raise ValueError("Give the labelling (bolus) duration.")
    if len(taus) not in (1, len(delays)):
        raise ValueError(f"{len(taus)} bolus durations for {len(delays)} delays: give one, or one per delay.")
    for name, value, allowed in (("iaf", iaf, ASL_FORMATS), ("ibf", ibf, BLOCK_FORMATS),
                                 ("cmethod", cmethod, CALIB_METHODS), ("tissref", tissref, TISSUE_REFS),
                                 ("pedir", pedir, PE_DIRECTIONS)):
        if value is not None and value not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, not {value!r}")
    if (pvgm is None) != (pvwm is None):
        raise ValueError("Give both the GM and the WM partial-volume maps, or neither.")
    if pvcorr and structural is None and fslanat is None and pvgm is None:
        raise ValueError("Partial-volume correction needs a structural image, an fsl_anat folder or GM/WM "
                         "partial-volume maps.")
    if region_analysis and fslanat is None:
        raise ValueError("Region analysis needs an fsl_anat folder (its MNI registration places the atlas).")
    if region_analysis and calib is None and m0 is None:
        raise ValueError("Region analysis reports calibrated CBF: give a calibration image or an M0 value.")
    if cmethod in ("single", "both") and calib is not None and structural is None and fslanat is None \
            and csf_mask is None:
        raise ValueError("Reference-region calibration needs a structural image, an fsl_anat folder or a CSF mask.")
    if cblip is not None and (echospacing is None or pedir is None):
        raise ValueError("Distortion correction with a reversed-blip calibration needs echospacing and pedir.")
    out_dir = Path(out_dir)
    _no_spaces(asl=asl, out_dir=out_dir, mask=mask, calib=calib, csf_mask=csf_mask, structural=structural,
               structural_brain=structural_brain, fslanat=fslanat, cblip=cblip, pvgm=pvgm, pvwm=pvwm)

    argv = ["oxford_asl", f"-i={asl}", f"-o={out_dir}", f"--iaf={iaf}", f"--ibf={ibf}",
            f"--{kind}={_times(delays)}", f"--bolus={_times(taus)}"]
    if labelling == "pcasl":
        argv.append("--casl")
    if parse_times(rpts):
        argv.append(f"--rpts={','.join(str(int(r)) for r in parse_times(rpts))}")
    if slicedt:
        argv.append(f"--slicedt={slicedt:g}")
    if sliceband:
        argv.append(f"--sliceband={int(sliceband)}")
    if mask is not None:
        argv.append(f"-m={mask}")
    if fslanat is not None:
        argv.append(f"--fslanat={fslanat}")
    elif structural is not None:
        argv.append(f"-s={structural}")
        if structural_brain is not None:
            argv.append(f"--sbrain={structural_brain}")
    if calib is not None:
        argv.append(f"-c={calib}")
        for flag, value in (("--cmethod", cmethod), ("--tr", tr), ("--cgain", cgain), ("--tissref", tissref),
                            ("--te", te)):
            if value is not None:
                argv.append(f"{flag}={value:g}" if isinstance(value, float) else f"{flag}={value}")
        if csf_mask is not None:
            argv.append(f"--csf={csf_mask}")
    elif m0 is not None:
        argv.append(f"--M0={m0:g}")
    if pvgm is not None and pvwm is not None:
        argv += [f"--pvgm={pvgm}", f"--pvwm={pvwm}"]
    if alpha is not None:
        argv.append(f"--alpha={alpha:g}")
    for flag, value in (("--bat", bat), ("--t1", t1), ("--t1b", t1b)):
        if value is not None:
            argv.append(f"{flag}={value:g}")
    for flag, on in (("--wp", white_paper), ("--mc", motion_correction), ("--pvcorr", pvcorr),
                     ("--infert1", infert1), ("--fixbat", fixbat), ("--artsupp", artsupp),
                     ("--region-analysis", region_analysis)):
        if on:
            argv.append(flag)
    if not spatial:
        argv.append("--spatial=off")
    if fixbolus is True:
        argv.append("--fixbolus")
    if cblip is not None:
        argv += [f"--cblip={cblip}", f"--echospacing={echospacing:g}", f"--pedir={pedir}"]
    argv += [str(a) for a in extra]

    calibrated = calib is not None or m0 is not None
    method = cmethod if calib is not None else ("single" if m0 is not None else None)
    if calib is not None and method is None:
        method = "single" if (structural is not None or fslanat is not None or csf_mask is not None) else "voxel"
    has_struct = structural is not None or fslanat is not None
    outputs = oxford_asl_outputs(out_dir, cmethod=method, calibrated=calibrated, pvcorr=pvcorr,
                                 structural=has_struct, pv_maps=has_struct or pvgm is not None,
                                 standard=fslanat is not None)
    if calib is None and m0 is not None:
        outputs.pop("m0_value", None)
    if region_analysis:
        folder = outputs["native_dir"] / ("voxelwise" if method == "both" else "") / "region_analysis"
        outputs["region_analysis"] = folder / "region_analysis.csv"
        outputs["region_analysis_dir"] = folder
    return FslCommand(argv, "BASIL (oxford_asl)", outputs, long=True, requires=_REQUIRES["oxford_asl"])


# basil


def basil_options(
    *,
    labelling: str = "pcasl",
    plds: Sequence[float] = (),
    tis: Sequence[float] = (),
    bolus: str | float | Sequence[float] = 1.8,
    repeats: int | None = None,
    t1: float = 1.3,
    t1b: float = 1.65,
    bat: float | None = None,
    batsd: float | None = None,
    slicedt: float = 0.0,
) -> list[str]:
    """The lines of a BASIL options file (``--casl``, ``--ti<n>``, ``--tau``, ``--t1`` ...).

    BASIL takes inversion times: for pCASL, TI = PLD + labelling duration (as oxford_asl
    computes them). The arrival-time prior is oxford_asl's: mean *bat* (1.3 s pCASL, 0.7 s
    pASL), standard deviation *batsd* (1 s with several delays, where the data determine it).
    """
    kind, delays = _delays(labelling, parse_times(plds), parse_times(tis))
    taus = parse_times(bolus)
    if not taus or len(taus) not in (1, len(delays)):
        raise ValueError("Give one labelling duration, or one per delay.")
    lines = ["--casl"] if labelling == "pcasl" else []
    for k, delay in enumerate(delays, start=1):
        ti = delay + (taus[0] if len(taus) == 1 else taus[k - 1]) if kind == "plds" else delay
        lines.append(f"--ti{k}={ti:g}")
    if len(taus) == 1:
        lines.append(f"--tau={taus[0]:g}")
    else:
        lines += [f"--tau{k}={tau:g}" for k, tau in enumerate(taus, start=1)]
    lines += [f"--t1={t1:g}", f"--t1b={t1b:g}"]
    lines.append(f"--bat={(bat if bat is not None else (1.3 if labelling == 'pcasl' else 0.7)):g}")
    if batsd is None and len(delays) > 1:
        batsd = 1.0
    if batsd is not None:
        lines.append(f"--batsd={batsd:g}")
    if repeats:
        lines.append(f"--repeats={int(repeats)}")
    if slicedt:
        lines.append(f"--slicedt={slicedt:g}")
    return lines


def basil(
    diff_data: Path,
    mask: Path,
    out_dir: Path,
    *,
    labelling: str = "pcasl",
    plds: str | Sequence[float] = (),
    tis: str | Sequence[float] = (),
    bolus: str | float | Sequence[float] = 1.8,
    repeats: int | None = None,
    ibf: str = "tis",
    t1: float = 1.3,
    t1b: float = 1.65,
    bat: float | None = None,
    batsd: float | None = None,
    slicedt: float = 0.0,
    spatial: bool = True,
    infertau: bool = False,
    inferart: bool = False,
    infert1: bool = False,
    pvgm: Path | None = None,
    pvwm: Path | None = None,
) -> FslCommand:
    """The BASIL fit on label-control *differenced* data (:func:`asl_file` ``diff``).

    BASIL reads the volumes grouped by delay: every repeat of the first delay, then of the
    second ... (*ibf* ``tis``, :func:`asl_file` with ``obf="tis"``); data in acquisition order
    (*ibf* ``rpt``: all delays, repeated) are reordered with ``asl_file`` first.
    BASIL writes ``<out>/basil/step<n>/`` per inference step (``basil_2`` ... on a rerun); the last step's mean images are
    copied to ``<out>/final/`` (``mean_ftiss`` perfusion, ``mean_delttiss`` arrival,
    ``mean_fblood`` arterial component; ``mean_fwm`` / ``mean_deltwm`` with *pvgm* / *pvwm*).
    Perfusion is in the data's units: divide by M0·α and multiply by 6000 for ml/100g/min
    (:func:`asl_calib`).
    """
    if (pvgm is None) != (pvwm is None):
        raise ValueError("Partial-volume correction needs both the GM and the WM partial-volume maps.")
    out_dir = Path(out_dir)
    _no_spaces(diff_data=diff_data, mask=mask, out_dir=out_dir, pvgm=pvgm, pvwm=pvwm)
    if ibf not in BLOCK_FORMATS:
        raise ValueError(f"ibf must be one of {BLOCK_FORMATS}, not {ibf!r}")
    options = basil_options(labelling=labelling, plds=parse_times(plds), tis=parse_times(tis), bolus=bolus,
                            repeats=repeats, t1=t1, t1b=t1b, bat=bat, batsd=batsd, slicedt=slicedt)
    ntis = len(parse_times(plds) or parse_times(tis))
    if inferart:
        options.append("--inferart")
    if infertau:
        options.append("--infertau")
    optfile = out_dir / "basil_options.txt"
    optfile.parent.mkdir(parents=True, exist_ok=True)
    optfile.write_text("\n".join(options) + "\n", encoding="utf-8")
    # BASIL writes to "<dir>+" when its folder exists: give it a fresh one (a rerun in the same
    # folder keeps the earlier steps, in basil_2 ...).
    steps = out_dir / "basil"
    k = 2
    while steps.exists():
        steps, k = out_dir / f"basil_{k}", k + 1
    reorder = ""
    data = Path(diff_data)
    if ibf == "rpt" and ntis > 1 and (repeats or 0) > 1:
        data = out_dir / "diff_by_delay.nii.gz"
        reorder = " ".join(_quote(a) for a in ["asl_file", f"--data={diff_data}", f"--ntis={ntis}", "--iaf=diff",
                                               "--ibf=rpt", "--obf=tis", f"--out={data}"]) + " && "
    call = ["basil", "-i", str(data), "-m", str(mask), "-o", str(steps), "--optfile", str(optfile)]
    if spatial:
        call.append("--spatial")
    if infert1:
        call.append("--infert1")
    if pvgm is not None and pvwm is not None:
        call += ["--pgm", str(pvgm), "--pwm", str(pvwm)]
    final = out_dir / "final"
    params = ("ftiss", "delttiss", "fblood", "fwm", "deltwm")
    copy = " ".join(f'if [ -e "$s/mean_{p}.nii.gz" ]; then cp "$s/mean_{p}.nii.gz" "{final}/"; fi;' for p in params)
    script = (
        reorder + " ".join(_quote(a) for a in call)
        + f' && s=$(ls -d "{steps}"/step* 2>/dev/null | sort -V | tail -n 1)'
        + ' && if [ -z "$s" ]; then echo "BASIL wrote no step folders" >&2; exit 1; fi'
        + f' && mkdir -p "{final}" && {copy} echo "BASIL result: $s -> {final}"'
    )
    outputs = {"dir": out_dir, "steps": steps, "options": optfile,
               **{f"mean_{p}": final / f"mean_{p}.nii.gz" for p in params}}
    return FslCommand([script], "BASIL fit", outputs, long=True, shell=True,
                      requires=_REQUIRES["basil"] + (("asl_file",) if reorder else ()))


def _quote(value: str) -> str:
    import shlex

    return shlex.quote(str(value))


# asl_file

#: What :func:`asl_file` does: subtract the pairs, split them into tag and control, or only average.
ASL_FILE_OPERATIONS: tuple[str, ...] = ("diff", "split-pairs", "mean")


def asl_file(
    data: Path,
    out_base: Path,
    *,
    ntis: int,
    operation: str = "diff",
    iaf: str = "tc",
    ibf: str = "rpt",
    rpts: str | Sequence[int] = (),
    obf: str | None = None,
    mask: Path | None = None,
) -> FslCommand:
    """Label-control subtraction (``diff``: the difference series and its mean per delay),
    tag / control separation (``split-pairs``) or averaging per delay (``mean``).

    *obf* orders the output series: ``tis`` groups the repeats of each delay (what
    :func:`basil` reads), ``rpt`` keeps acquisition order; default, the input's (*ibf*).
    """
    if operation not in ASL_FILE_OPERATIONS:
        raise ValueError(f"operation must be one of {ASL_FILE_OPERATIONS}, not {operation!r}")
    if iaf not in ASL_FORMATS or ibf not in BLOCK_FORMATS:
        raise ValueError(f"iaf must be one of {ASL_FORMATS} and ibf one of {BLOCK_FORMATS}")
    if operation in ("diff", "split-pairs") and iaf == "diff":
        raise ValueError(f"{operation} needs tag-control pairs (iaf tc/ct/tcb/ctb), not differenced data.")
    out = Path(out_base)
    argv = ["asl_file", f"--data={data}", f"--ntis={int(ntis)}", f"--iaf={iaf}", f"--ibf={ibf}"]
    if parse_times(rpts):
        argv.append(f"--rpts={','.join(str(int(r)) for r in parse_times(rpts))}")
    if obf is not None:
        if obf not in BLOCK_FORMATS:
            raise ValueError(f"obf must be one of {BLOCK_FORMATS}, not {obf!r}")
        argv.append(f"--obf={obf}")
    if mask is not None:
        argv.append(f"--mask={mask}")
    outputs: dict[str, Path] = {}
    if operation == "diff":
        argv += ["--diff", f"--out={nii(f'{out}_diff')}", f"--mean={nii(f'{out}_diff_mean')}"]
        outputs = {"diff": nii(f"{out}_diff"), "mean": nii(f"{out}_diff_mean")}
    elif operation == "split-pairs":
        argv += ["--spairs", f"--out={out}"]
        # asl_file names the two halves <out>_even / <out>_odd (tags first for tc data).
        first, second = ("tag", "control") if iaf.startswith("tc") else ("control", "tag")
        outputs = {first: nii(f"{out}_odd"), second: nii(f"{out}_even")}
    else:
        argv += [f"--out={nii(out)}", f"--mean={nii(f'{out}_mean')}"]
        outputs = {"mean": nii(f"{out}_mean")}
    return FslCommand(argv, f"asl_file ({operation})", outputs, requires=_REQUIRES["asl_file"])


# asl_calib


def asl_calib(
    calib: Path,
    out_dir: Path,
    *,
    perfusion: Path | None = None,
    ref_mask: Path | None = None,
    structural_brain: Path | None = None,
    asl2struct: Path | None = None,
    tissref: str = "csf",
    mode: str = "longtr",
    tr: float = 3.2,
    cgain: float = 1.0,
    te: float = 0.0,
    alpha: float | None = None,
    brain_mask: Path | None = None,
) -> FslCommand:
    """M0 of arterial blood from *calib* in a reference region, and *perfusion* calibrated with it.

    The region is *ref_mask* (in the calibration image's space), or *tissref* segmented from
    *structural_brain* placed by *asl2struct*. ``<out>/M0.txt`` holds the value; with
    *perfusion*, ``<out>/perfusion_calib`` is *perfusion* / (M0·α) · 6000 (ml/100g/min).
    """
    if mode not in CALIB_MODES or tissref not in TISSUE_REFS:
        raise ValueError(f"mode must be one of {CALIB_MODES} and tissref one of {TISSUE_REFS}")
    if ref_mask is None and (structural_brain is None or asl2struct is None):
        raise ValueError("Give a reference-region mask, or a brain-extracted structural and the ASL→structural matrix.")
    out_dir = Path(out_dir)
    _no_spaces(calib=calib, out_dir=out_dir, perfusion=perfusion, ref_mask=ref_mask, structural_brain=structural_brain,
               asl2struct=asl2struct, brain_mask=brain_mask)
    argv = ["asl_calib", "-c", str(calib), "-o", str(out_dir), "--mode", mode, "--tissref", tissref,
            "--tr", f"{tr:g}", "--cgain", f"{cgain:g}"]
    if ref_mask is not None:
        argv += ["-m", str(ref_mask)]
    else:
        argv += ["-s", str(structural_brain), "-t", str(asl2struct)]
    if te:
        argv += ["--te", f"{te:g}"]
    if alpha is not None:
        argv += ["--alpha", f"{alpha:g}"]
    if brain_mask is not None:
        argv += ["--bmask", str(brain_mask)]
    outputs = {"m0_value": out_dir / "M0.txt", "refmask": out_dir / "refmask.nii.gz"}
    if perfusion is not None:
        calibrated = out_dir / "perfusion_calib.nii.gz"
        argv += ["-i", str(perfusion), "--of", str(calibrated)]
        outputs["perfusion_calib"] = calibrated
    return FslCommand(argv, "asl_calib (M0)", outputs, requires=_REQUIRES["asl_calib"])


# reading the results


def read_value(path: Path) -> float | None:
    """The number in a one-value text output (``M0.txt``, ``perfusion_calib_gm_mean.txt``)."""
    try:
        return float(Path(path).read_text(encoding="utf-8").split()[0])
    except (OSError, ValueError, IndexError):
        return None


#: oxford_asl output names as table rows.
_PRETTY = {
    "perfusion_calib": "CBF (ml/100g/min)", "perfusion_wm_calib": "WM CBF (ml/100g/min)",
    "perfusion": "perfusion (a.u.)", "perfusion_wm": "WM perfusion (a.u.)",
    "arrival": "ATT (s)", "arrival_wm": "WM ATT (s)",
}


def summary(native_dir: Path) -> dict[str, dict[str, float]]:
    """``{set: {"perfusion_calib gm": value, ...}}`` from the ``*_mean.txt`` summaries oxford_asl
    writes under *native_dir* (one set per folder: ``native``, ``pvcorr``, ``voxelwise`` ...)."""
    native_dir = Path(native_dir)
    out: dict[str, dict[str, float]] = {}
    for path in sorted(native_dir.rglob("*_mean.txt")):
        if "region_analysis" in path.parts:
            continue
        value = read_value(path)
        if value is None:
            continue
        folder = path.parent.relative_to(native_dir).as_posix()
        key = "native" if folder == "." else folder
        stem = path.name[: -len("_mean.txt")]
        for region in ("cortical_gm", "cerebral_wm", "gm", "wm"):
            if stem.endswith(f"_{region}"):
                name = stem[: -len(region) - 1]
                stem = f"{_PRETTY.get(name, name)} · {region.replace('_', ' ').replace('gm', 'GM').replace('wm', 'WM')}"
                break
        out.setdefault(key, {})[stem] = value
    return out


#: oxford_asl region-analysis CSV suffix -> what its rows measure.
_REGION_TABLES = {
    "": "CBF (ml/100g/min)", "_gm": "GM CBF (ml/100g/min)", "_wm": "WM CBF (ml/100g/min)",
    "_arrival": "ATT (s)", "_arrival_gm": "GM ATT (s)", "_arrival_wm": "WM ATT (s)",
}


def region_table_title(csv_path: Path) -> str:
    """What one of oxford_asl's region-analysis CSVs reports (``region_analysis_gm.csv`` -> GM CBF).

    With partial-volume correction the GM / WM tables are the corrected GM and WM maps; without
    it, the plain maps in voxels above the GM / WM purity thresholds.
    """
    stem = Path(csv_path).name.removesuffix(".csv")
    suffix = stem.removeprefix("region_analysis")
    return _REGION_TABLES.get(suffix, stem)


def region_tables(folder: Path) -> list[tuple[str, dict[str, dict[str, float | str | None]]]]:
    """``[(title, rows)]`` for every region-analysis CSV in *folder*: CBF tables first, then ATT."""
    order = list(_REGION_TABLES.values())
    tables = [(region_table_title(path), read_region_analysis(path)) for path in Path(folder).glob("*.csv")]
    return sorted(tables, key=lambda item: (order.index(item[0]) if item[0] in order else len(order), item[0]))


def read_region_analysis(csv_path: Path) -> dict[str, dict[str, float | str | None]]:
    """``{region: {column: value}}`` from one of oxford_asl's region-analysis CSVs."""
    rows: dict[str, dict[str, float | str | None]] = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = str(row.pop("name", "") or f"region {len(rows) + 1}")
            parsed: dict[str, float | str | None] = {}
            for key, raw in row.items():
                if key is None:
                    continue
                text = (raw or "").strip()
                try:
                    parsed[key] = float(text) if text else None
                except ValueError:
                    parsed[key] = text
            rows[name] = parsed
    return rows


__all__ = [
    "ASL_FILE_OPERATIONS",
    "ASL_FORMATS",
    "BASIL_PARAMS",
    "BLOCK_FORMATS",
    "CALIB_METHODS",
    "CALIB_MODES",
    "LABELLINGS",
    "PE_DIRECTIONS",
    "TISSUE_REFS",
    "asl_calib",
    "asl_file",
    "basil",
    "basil_options",
    "oxford_asl",
    "oxford_asl_outputs",
    "parse_times",
    "read_region_analysis",
    "read_value",
    "region_table_title",
    "region_tables",
    "summary",
]
