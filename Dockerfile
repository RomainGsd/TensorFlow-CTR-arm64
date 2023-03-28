#This docker has multi stages, taking previously built images as base to grow into our final image

# ========
# Stage 1:
#   - Base image including OS and key packages
#   - Augment the base image with some essential libs
#   - Install essential python dependencies into a venv
# ========

FROM arm64v8/ubuntu AS tf_base

ARG njobs
ARG tf_id

ENV NP_MAKE="${njobs}" \
    TF_VERSION_ID="${tf_id}"

# Key version numbers
ENV NUMPY_VERSION=1.17.1 \
    SCIPY_VERSION=1.4.1 \
    NPY_DISTUTILS_APPEND_FLAGS=1 \
    PY_VERSION=3.7.0 \
    ARMPL_VERSION=20.2.1

# Package build parameters
ENV PROD_DIR=opt \
    PACKAGE_DIR=packages

# Make directories to hold package source & build directories (PACKAGE_DIR)
# and install build directories (PROD_DIR).
RUN mkdir -p $PACKAGE_DIR && \
    mkdir -p $PROD_DIR

# Common compiler settings
ENV CC=gcc \
    CXX=g++ \
    BASE_CFLAGS="-moutline-atomics"

#Install core OS packages
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get -y update && \
    apt-get -y install software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get -y install \
      autoconf \
      bc \
      build-essential \
      cmake \
      curl \
      gettext-base \
      git \
      iputils-ping \
      libbz2-dev \
      libc++-dev \
      libcgal-dev \
      libffi-dev \
      libfreetype6-dev \
      libhdf5-dev \
      libjpeg-dev \
      liblzma-dev \
      libncurses5-dev \
      libncursesw5-dev \
      libpng-dev \
      libreadline-dev \
      libssl-dev \
      libsqlite3-dev \
      libxml2-dev \
      libxslt-dev \
      locales \
      moreutils \
      openjdk-8-jdk \
      openssl \
      python3 python3-dev python3-pip python3-venv python3-openssl \
      rsync \
      ssh \
      sudo \
      time \
      unzip \
      vim \
      wget \
      xz-utils \
      zip \
      zlib1g-dev \
      g++-7 gcc-7

# DOCKER_USER for the Docker user
ENV DOCKER_USER=root

# Using venv means this can be done in userspace
WORKDIR /home/$DOCKER_USER
USER $DOCKER_USER
ENV PACKAGE_DIR=/home/$DOCKER_USER/$PACKAGE_DIR
RUN mkdir -p $PACKAGE_DIR

# Setup a Python virtual environment
ENV VENV_DIR=/home/$DOCKER_USER/python3-venv
ENV VENV_ACT=$VENV_DIR/bin/activate
RUN python3 -m venv $VENV_DIR
ENV PATH="$VENV_DIR/bin:$PATH"

# Install some basic python packages
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    "setuptools>=41.0.0" six mock wheel \
    cython sh numpy \
    pybind11 pyangbind scipy

# Install some TensorFlow essentials
RUN pip install --no-cache-dir keras_applications --no-deps
RUN pip install --no-cache-dir keras_preprocessing --no-deps

# enum34 is not compatible with Python 3.6+, and not required
# it is installed as a dependency for an earlier package and needs
# to be removed in order for the OpenCV build to complete.
RUN pip uninstall enum34 -y
RUN HDF5_DIR=/usr/lib/aarch64-linux-gnu/hdf5/serial pip install h5py
RUN pip install --no-cache-dir grpcio
RUN pip install --no-cache-dir ck absl-py pycocotools pillow

# Install OpenCV into our venv
RUN pip install --no-cache-dir --no-binary :all: opencv-python-headless

CMD ["bash", "-l"]

# ========
# Stage 2: Build Bazel and add essential python packages
# ========

FROM tf_base AS tf_dev

ARG njobs
ARG bazel_mem
ARG tf_version
ARG tf_id
ARG bazel_version

ENV NP_MAKE="${njobs}" \
    BZL_RAM="${bazel_mem}"

# Key version numbers
ENV BZL_VERSION="${bazel_version}" \
    TF_VERSION="${tf_version}" \
    TF_VERSION_ID="${tf_id}"

# Package build parameters
ENV PROD_DIR=opt \
    PACKAGE_DIR=packages

# Use a PACKAGE_DIR in userspace
WORKDIR /home/$DOCKER_USER
USER $DOCKER_USER
ENV PACKAGE_DIR=/home/$DOCKER_USER/$PACKAGE_DIR
RUN mkdir -p $PACKAGE_DIR

# Copy in the Python virtual environment
ENV VENV_DIR=/home/$DOCKER_USER/python3-venv
COPY --chown=$DOCKER_USER:$DOCKER_USER --from=tf_base $VENV_DIR /home/$DOCKER_USER/python3-venv
ENV PATH="$VENV_DIR/bin:$PATH"

RUN mkdir -p $PACKAGE_DIR
WORKDIR $PACKAGE_DIR

# Compile Bazel from source
RUN DEBIAN_FRONTEND=noninteractive /usr/bin/apt-get install --yes -q --no-install-recommends \
    build-essential openjdk-11-jdk python zip unzip
ENV BAZEL_VERSION=3.7.2
RUN wget https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-dist.zip
RUN unzip bazel-$BAZEL_VERSION-dist.zip
RUN env EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" bash ./compile.sh
RUN cp output/bazel /usr/local/bin/
RUN bazel --version

CMD ["bash", "-l"]

# ========
# Stage 3: Build Tensorflow
# ========
FROM tf_dev AS tf_dev_with_bazel

WORKDIR /home/$DOCKER_USER
USER $DOCKER_USER

# Copy in the Python virtual environment
ENV VENV_DIR=/home/$DOCKER_USER/python3-venv
COPY --chown=$DOCKER_USER:$DOCKER_USER --from=tf_dev $VENV_DIR /home/$DOCKER_USER/python3-venv
ENV PATH="$VENV_DIR/bin:$PATH"
# Copy Bazel binary
COPY --chown=$DOCKER_USER:$DOCKER_USER --from=tf_dev /usr/local/bin/bazel /usr/local/bin/bazel
RUN bazel --version

#Get Cuda (Ubuntu 20.04)
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/sbsa/cuda-ubuntu2004.pin
# RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/sbsa/7fa2af80.pub
# RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/sbsa/ /"
# RUN apt-get update
# RUN DEBIAN_FRONTEND=noninteractive /usr/bin/apt-get install --yes -q --no-install-recommends cuda-minimal-build-11-2

#Get Cuda (Ubuntu 18.04)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/sbsa/cuda-ubuntu1804.pin
RUN mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/sbsa/7fa2af80.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/sbsa/ /"
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive /usr/bin/apt-get install --yes -q --no-install-recommends cuda-11-0
#RUN apt-get -y install cuda-minimal-build-11-0 cuda-nvcc-11-0 cuda-cupti-11-0 cuda-cupti-dev-11-0 libcublas-11-0 libcublas-dev-11-0 libcusolver-11-0 libcusolver-dev-11-0

# #Get CudNN
ENV CUDNN_VERSION=8.1.1
ENV CUDA_VERSION=cuda11.0
RUN DEBIAN_FRONTEND=noninteractive /usr/bin/apt-get install --yes -q --no-install-recommends \
    libcudnn8 libcudnn8-dev
    # libcudnn8=$CUDNN_VERSION-1+$CUDA_VERSION \
    # libcudnn8-dev=$CUDNN_VERSION-1+$CUDA_VERSION

#ENV PATH="/usr/local/cuda-11/targets/sbsa-linux/include:/usr/local/cuda-11/targets/sbsa-linux/lib:/usr/local/cuda-11/bin/:/usr/local/cuda-11.2/bin/:$PATH"
ENV PATH="/usr/local/cuda-11/targets/sbsa-linux/include:/usr/local/cuda-11/targets/sbsa-linux/lib:/usr/local/cuda-11/bin/:/usr/local/cuda-11.2/bin/:$PATH"
ENV PATH="/usr/local/cuda-11.0/targets/sbsa-linux/lib/:$PATH"

#Set ENV variables for Tensorflow build configuration
ENV CUDA_VERSION=11.0
ENV CUDNN_VERSION=8.1
ENV CUDA_TOOLKIT_PATH="/usr/local/cuda/"

# Tensorflow build has problems with gcc/++ 8 and 9 so we install gcc/++7
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes -q --no-install-recommends gcc-7 g++-7
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 1 --slave /usr/bin/g++ g++ /usr/bin/g++-7
RUN update-alternatives --config gcc
RUN gcc --version
RUN g++ --version

# Get TensorFlow sources
RUN git clone https://github.com/tensorflow/tensorflow.git
WORKDIR tensorflow
# The repo defaults to the master development branch so we checkout release 2.4.0
RUN git checkout v2.4.0

# RUN find / -iname "cuparse.h"
# RUN apt-cache search "cuda"

# RUN DEBIAN_FRONTEND=noninteractive /usr/bin/apt-get install --yes -q --no-install-recommends \
#     libcurand-11-0 libcurand-dev-11-0 libcufft-11-0 libcufft-dev-11-0 libcusparse-11-0 libcusparse-dev-11-0

# RUN mkdir -p /usr/local/cuda/ /usr/local/cuda/include /usr/local/cuda/lib /usr/local/cuda/bin/ /usr/local/cuda/nvvm/libdevice/
# RUN mv /usr/local/cuda-11.0/targets/sbsa-linux/include/* /usr/local/cuda/include/
# RUN mv /usr/local/cuda-11.0/targets/sbsa-linux/lib/* /usr/local/cuda/lib/
# RUN mv /usr/local/cuda-11.0/bin/nvcc /usr/local/cuda/bin/
# RUN mv /usr/local/cuda-11.0/nvvm/libdevice/libdevice.10.bc /usr/local/cuda/nvvm/libdevice/

#Install TensorRT
ARG OS="ubuntu1804"
ARG TAG="cuda11.1-trt7.2.1.6-ga-20201007"
COPY nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.1.6-ga-20201007_1-1_arm64.deb .
RUN dpkg -i "nv-tensorrt-repo-$OS-cuda11.1-trt7.2.1.6-ga-20201007_1-1_arm64.deb"
RUN apt-key add /var/nv-tensorrt-repo-cuda11.1-trt7.2.1.6-ga-20201007/7fa2af80.pub
RUN apt-get update
RUN apt-get install -y tensorrt
RUN dpkg -l | grep TensorRT

# /!\ WARNING /!\ Configure inputs changes for each TF version
# Printf args correspond to ./configure inputs :
#   Python path:                            "" (default)
#   Python library path:                    "" (default)
#   ROCm support:                           "" (default is N)
#   CUDA support:                           "y"
#   Build TensorFlow with TensorRT:         "y" (default is N)
#   Download Cland:                         "" (default is N)
#   Interactively configure Android build:  "" (default is N)
RUN (printf "\n\n\ny\ny\n" && cat) | python3 configure.py

# Build Tensorflow Python package
# Optional: MKL is Intel Math Library --config=mkl_opensource_only or --config=mkl
# Option : --config=monolithic
RUN bazel build \
    --config=cuda \
    --config=opt --config=noaws --config=v2 \
    --verbose_failures \
    --jobs=4 --local_ram_resources=HOST_RAM*.5 --local_cpu_resources=HOST_CPUS*.5 \
    //tensorflow/tools/pip_package:build_pip_package

# Build the wheel package
#RUN ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
# Install the package
#RUN pip install $(ls -tr /tmp/tensorflow_pkg/tensorflow-*.whl | tail)

CMD ["bash", "-l"]

# ========
# Stage 4: Install TensorRT
# ========
FROM tf_dev_with_bazel AS tensorflow

WORKDIR /home/$DOCKER_USER
USER $DOCKER_USER

# Copy in the Python virtual environment
ENV VENV_DIR=/home/$DOCKER_USER/python3-venv
COPY --chown=$DOCKER_USER:$DOCKER_USER --from=tf_dev $VENV_DIR /home/$DOCKER_USER/python3-venv
ENV PATH="$VENV_DIR/bin:$PATH"

# Check if TensorRT is installed under /usr/lib/aarch64-linux-gnu/

# RUN git clone -b master https://github.com/nvidia/TensorRT TensorRT
# RUN cd TensorRT
# RUN git submodule update --init --recursive

CMD ["bash", "-l"]
