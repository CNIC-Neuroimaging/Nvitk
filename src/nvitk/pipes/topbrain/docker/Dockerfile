# ToPBrain / ToPAneu TA36 algorithm container for Grand Challenge.
# CUDA 12.6: Grand Challenge minimum.
# Overridable with --build-arg for a different torch.
# -----Base Image requested by Grand Challenge-----
# ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
# -----Base Image to test GUP locally-----
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime     

# --platform pinned: Grand Challenge runs amd64. 
FROM --platform=linux/amd64 ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    nnUNet_raw=/tmp/nnunet_raw \
    nnUNet_preprocessed=/tmp/nnunet_preprocessed \
    nnUNet_results=/tmp/nnunet_results

# Grand Challenge runs containers as a non-root user.
RUN groupadd -r algorithm && useradd -m -r -g algorithm algorithm

WORKDIR /opt/algorithm

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/algorithm/requirements.txt

COPY --chown=algorithm:algorithm nnunetv2 /opt/algorithm/nnunetv2

COPY --chown=algorithm:algorithm topbrain_algo /opt/algorithm/topbrain_algo
# model/ holds one subdirectory per modality (model/ct, model/mr); models.json maps each
# socket to one of them along with its channel count and the CT windows applied.

COPY --chown=algorithm:algorithm model /opt/algorithm/model
COPY --chown=algorithm:algorithm models.json /opt/algorithm/models.json
COPY --chown=algorithm:algorithm postprocess.json /opt/algorithm/postprocess.json
COPY --chown=algorithm:algorithm inference.py /opt/algorithm/inference.py

ENV PYTHONPATH=/opt/algorithm

USER algorithm

ENTRYPOINT ["python", "/opt/algorithm/inference.py"]
