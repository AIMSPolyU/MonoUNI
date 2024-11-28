FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

WORKDIR /workspace

RUN rm /etc/apt/sources.list.d/cuda.list && \
    rm /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-key del 7fa2af80 && apt-get update

RUN apt-get install wget -y && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt install -y libgl1-mesa-glx

RUN conda install -y \
    numpy \
    numba \
    scikit-image \
    tqdm \
    matplotlib \
    protobuf \
    pyyaml \
    && conda clean -afy

RUN pip install --no-cache-dir \
    opencv-python \
    cuda-python \
    carla==0.9.14

LABEL maintainer="guoxzhang@polyu.edu.hk>"
LABEL description="rope3d with PyTorch and required libraries"

WORKDIR /workspace

# docker run -it --name adas_proj --gpus all -v "C:\Users\Administrator\Desktop\MonoUNI":/workspace rope3d bash
