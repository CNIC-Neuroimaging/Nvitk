"""
Vendored deep-learning engines shared by the nvitk learning pipelines.

Description
-----------
Two MIC-DKFZ frameworks live here, used by both :mod:`nvitk.pipes.topbrain` and
:mod:`nvitk.pipes.cvit`:

``nnunet/``
    An nnU-Net v2 build that carries nnssl fine-tuning support (``preprocess_like_nnssl``,
    ``PretrainedTrainer``) which the released ``nnunetv2`` lacks, plus the pipelines' own
    trainers (``training/nnUNetTrainer/{topbrain,cvit}/``). It is **never installed** and never
    imported in an nvitk process: the released ``nnunetv2`` stays installed for TotalSegmentator,
    and the in-tree build is put first on ``PYTHONPATH`` for *subprocesses only*
    (:func:`~nvitk.pipes._engines.env.nnunet_env`).

``nnssl/``
    A clone of MIC-DKFZ/nnssl, used off ``sys.path`` in-process after
    :func:`~nvitk.pipes._engines.env.apply_nnssl_env`.

Neither tree is a Python package of ``nvitk`` (no ``__init__`` at their roots), so
``setuptools.packages.find`` leaves them out of wheels; pipelines that need them run from a
source checkout or the SGE container, which binds ``src/``.

Modules
-------
:mod:`~nvitk.pipes._engines.env`
    Locate the trees and build the environment their processes need.
:mod:`~nvitk.pipes._engines.nnunet_run`
    Run in-tree nnU-Net entry points as subprocesses.
"""
