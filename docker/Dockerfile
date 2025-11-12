# Setting up very basic environment with cuda 11.8 support.
ARG BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
FROM $BASE_IMAGE

# Add mamba functionality
# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean

# install miniforge (conda + mamba)
ENV CONDA_DIR=/opt/conda
RUN wget -O ~/Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
/bin/bash ~/Miniforge3.sh -b -p ${CONDA_DIR}
# Put conda in path so we can use conda/mamba activate and initialise mamba
ENV PATH=$CONDA_DIR/bin:$PATH
SHELL ["/bin/bash", "-i", "-c"]
RUN mamba shell init --shell bash --root-prefix=${CONDA_DIR} 

# Define mamba environments using environment.yamls
RUN mkdir /workspace
COPY benchmarking/ /workspace/
RUN mamba env create -f /workspace/VPR/environment.yaml && \
    mamba env create -f /workspace/DepthAnythingV2/environment.yaml && \
    mamba env create -f /workspace/cross_modal/environment.yaml

# Make all conda environments available to all users (not just root). 
# Also set up relevant folders for RunAI data mounting and operation
RUN mkdir -p /datasets/work && mkdir /workspace && chmod -R a+rw /workspace /datasets && \ 
    chmod -R g+rw ${CONDA_DIR} && \
    chgrp -R root ${CONDA_DIR} && \
    echo "eval \"\$(mamba shell hook --shell bash)\"" >> /etc/bash.bashrc