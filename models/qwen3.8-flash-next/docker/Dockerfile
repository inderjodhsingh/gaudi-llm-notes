# Qwen3.8-Flash-Next on Gaudi2. Build from models/qwen3.8-flash-next/:
#   docker build -f docker/Dockerfile -t qwen38-flash-next-gaudi2:YYYYMMDD .
#
# Pins: Habana 1.24.1 / PT 2.11, vLLM 98dff2a8, vllm-gaudi 2dd55f97 + patches/0001.
# vLLM itself is stock; only the plugin is patched. No custom TPC / .so.
FROM vault.habana.ai/gaudi-docker/1.24.1/ubuntu24.04/habanalabs/pytorch-installer-2.11.0@sha256:b257eaeffdc6ba5e1deaa4ca3aad8ec9ed0d777d00794a0635050e0160f09fd8

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_COMMIT=98dff2a81d747d1dba01a47f939f48c3526d4206
ARG VLLM_GAUDI_REPO=https://github.com/vllm-project/vllm-gaudi.git
ARG VLLM_GAUDI_COMMIT=2dd55f97eec0bcfcb1f2c52c2d9aee2c4f43f38c

ENV OMPI_MCA_btl_vader_single_copy_mechanism=none \
    HABANA_VISIBLE_DEVICES=all \
    PIP_DEFAULT_TIMEOUT=1000 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=on

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3 /usr/bin/python

COPY patches/0001-qwen4-exp-hpu-port.patch /opt/flash-next/
COPY docker/constraints.txt /opt/flash-next/

RUN mkdir -p /opt/vllm /opt/vllm-gaudi \
 && git -C /opt/vllm init -q && git -C /opt/vllm remote add origin "$VLLM_REPO" \
 && git -C /opt/vllm fetch -q --depth 1 origin "$VLLM_COMMIT" && git -C /opt/vllm checkout -q --detach FETCH_HEAD \
 && git -C /opt/vllm-gaudi init -q && git -C /opt/vllm-gaudi remote add origin "$VLLM_GAUDI_REPO" \
 && git -C /opt/vllm-gaudi fetch -q --depth 1 origin "$VLLM_GAUDI_COMMIT" && git -C /opt/vllm-gaudi checkout -q --detach FETCH_HEAD \
 && git -C /opt/vllm-gaudi -c user.name=flash-next-build -c user.email=build@local \
      am --committer-date-is-author-date -q /opt/flash-next/0001-qwen4-exp-hpu-port.patch

# setuptools is not pinned during the build: vLLM's build requirements need <81; the final pin restores 84.0.0.
RUN cd /opt/vllm \
 && sed '/^torch/d' requirements/build/cuda.txt > /tmp/vllm-build-requirements.txt \
 && pip install -c /opt/flash-next/constraints.txt -r /tmp/vllm-build-requirements.txt \
 && SETUPTOOLS_SCM_PRETEND_VERSION=0.29.0+g98dff2a81 VLLM_TARGET_DEVICE=empty \
    pip install -c /opt/flash-next/constraints.txt --no-build-isolation . \
 && cd /opt/vllm-gaudi \
 && VLLM_TARGET_DEVICE=hpu pip install -c /opt/flash-next/constraints.txt --no-build-isolation . \
 && pip install -c /opt/flash-next/constraints.txt transformers==5.16.1 \
 && pip install setuptools==84.0.0 \
 && pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cpu torchaudio==2.11.0+cpu \
 && pip freeze > /opt/flash-next/pip-freeze.txt

WORKDIR /work
CMD ["sleep", "infinity"]
