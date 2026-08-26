# ToPBrain / ToPAneu TA36 algorithm container for Grand Challenge.
#
# Built by `nvitk-topbrain --stages stage7`, which copies the trained nnU-Net model directory
# in as `model/` beforehand. Kept deliberately thin: no network access at run time, weights
# baked in, and the same nvitk post-processing the offline pipeline uses.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    nnUNet_raw=/tmp/nnunet_raw \
    nnUNet_preprocessed=/tmp/nnunet_preprocessed \
    nnUNet_results=/tmp/nnunet_results \
    TOPBRAIN_MODEL_DIR=/opt/algorithm/model

# Grand Challenge runs containers as a non-root user.
RUN groupadd -r algorithm && useradd -m -r -g algorithm algorithm

WORKDIR /opt/algorithm

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/algorithm/requirements.txt

# The in-tree nnU-Net build. It carries the nnssl fine-tuning support and the ToPBrain loss
# trainers; the released nnunetv2 is deliberately NOT installed (see requirements.txt), so this
# copy is the only nnunetv2 on PYTHONPATH and inference can resolve the class that trained the
# model.
COPY --chown=algorithm:algorithm nnunetv2 /opt/algorithm/nnunetv2

# nvitk itself, then the trained weights and the entry point.
COPY --chown=algorithm:algorithm nvitk /opt/algorithm/nvitk
COPY --chown=algorithm:algorithm model /opt/algorithm/model
COPY --chown=algorithm:algorithm inference.py /opt/algorithm/inference.py

ENV PYTHONPATH=/opt/algorithm

USER algorithm

ENTRYPOINT ["python", "/opt/algorithm/inference.py"]
