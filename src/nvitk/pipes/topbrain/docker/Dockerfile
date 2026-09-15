# ToPBrain / ToPAneu TA36 algorithm container for Grand Challenge.
#
# Built by `nvitk-topbrain --stages stage5`, which lays the trained nnU-Net runs out as
# `model/<modality>/` beforehand and writes `models.json` describing them. Kept deliberately
# thin: no network access at run time, weights baked in, and only the algorithm code the
# offline pipeline uses. One image serves both the CT and the MR socket.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

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

# The in-tree nnU-Net build. It carries the nnssl fine-tuning support and the ToPBrain loss
# trainers; the released nnunetv2 is deliberately NOT installed (see requirements.txt), so this
# copy is the only nnunetv2 on PYTHONPATH and inference can resolve the class that trained the
# model.
COPY --chown=algorithm:algorithm nnunetv2 /opt/algorithm/nnunetv2

# The slim algorithm package (harmonisation + component clean-up). nvitk is deliberately not
# copied in: the image needs three functions from it, and the whole library is 151 MB.
COPY --chown=algorithm:algorithm topbrain_algo /opt/algorithm/topbrain_algo
# model/ holds one subdirectory per modality (model/ct, model/mr); models.json maps each
# socket to one of them along with its channel count and the windows stage 0 applied.
COPY --chown=algorithm:algorithm model /opt/algorithm/model
COPY --chown=algorithm:algorithm models.json /opt/algorithm/models.json
COPY --chown=algorithm:algorithm postprocess.json /opt/algorithm/postprocess.json
COPY --chown=algorithm:algorithm inference.py /opt/algorithm/inference.py

ENV PYTHONPATH=/opt/algorithm

USER algorithm

ENTRYPOINT ["python", "/opt/algorithm/inference.py"]
