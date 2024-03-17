FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04 AS base

# Install NVIDIA CUDA Toolkit and cuDNN
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cuda-cudart-12-2 \
    libcudnn8 \
    libcudnn8-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    wget \
    unzip \
    libopenblas-dev \
    python3.9 \
    python3.9-dev \
    python3-pip \
    nano \
    && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.9 -m pip install --no-cache-dir --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN python3.9 -m pip install torch torchvision torchaudio \
    pip install --no-cache-dir -r /tmp/requirements.txt 

# Configure Git, clone the repository without checking out, then checkout the specific commit
RUN git config --global advice.detachedHead false && \
    git clone --no-checkout https://github.com/MIC-DKFZ/nnUNet.git /opt/algorithm/nnunet/ && \
    cd /opt/algorithm/nnunet/ && \
    git checkout 947eafbb9adb5eb06b9171330b4688e006e6f301


# Install a few dependencies that are not automatically installed
RUN pip install \
    -e /opt/algorithm/nnunet \
    graphviz \
    onnx \
    SimpleITK && \
    rm -rf ~/.cache/pip

### USER
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# RUN chown -R user /opt/algorithm/

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

COPY --chown=user:user process.py /opt/app/
COPY --chown=user:user export2onnx.py /opt/app/

# Copy scripts and data
COPY --chown=user:user process.py export2onnx.py /opt/app/
COPY --chown=user:user architecture/extensions/nnunetv2/ /opt/algorithm/nnunet/nnunetv2/
COPY --chown=user:user Dataset013_ULS/ /opt/algorithm/nnunet/nnUNet_results/Dataset013_ULS/
COPY --chown=user:user input/ /input/

# Set environment variables
ENV nnUNet_raw="/opt/algorithm/nnunet/nnUNet_raw" \
    nnUNet_preprocessed="/opt/algorithm/nnunet/nnUNet_preprocessed" \
    nnUNet_results="/opt/algorithm/nnunet/nnUNet_results"

# Set the entry point
ENTRYPOINT [ "python3.9", "-m", "process" ]
