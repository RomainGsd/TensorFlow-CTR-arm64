#!/bin/bash

set -euo pipefail

function print_usage_and_exit {
    echo "Usage: build.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help                   Display this message."
    echo "      --jobs                   Specify number of jobs to run in parallel during the build."
    echo "      --bazel_memory_limit     Set a memory limit for Bazel build."
    echo "      --clean                  Pull a new base image and build without using any cached images."
    echo ""
    echo "Example:"
    echo "  build.sh --build-type full"
    exit $1
}

tf_version=2
# TensorFlow2 dependency versions
version="v2.3.0"
bazel_version="3.4.0"
extra_args=""

# Enable Buildkit
# Required for advanced multi-stage builds
export DOCKER_BUILDKIT=1

# Default args
extra_args=""
nproc_build=
bazel_mem=
clean_build=

while [ $# -gt 0 ]
do
  case $1 in
    --jobs )
      nproc_build=$2
      shift
      ;;

    --bazel_memory_limit )
      bazel_mem=$2
      shift
      ;;

    --clean )
      clean_build=1
      ;;

    -h | --help )
      print_usage_and_exit 0
      ;;

  esac
  shift
done

if [[ $nproc_build ]]; then
  # Set -j to use for builds, if specified
  extra_args="$extra_args --build-arg njobs=$nproc_build"
fi

if [[ $bazel_mem ]]; then
  # Set -j to use for builds, if specified
  extra_args="$extra_args --build-arg bazel_mem=$bazel_mem"
fi

if [[ $clean_build ]]; then
  # Pull a new base image, and don't use any caches
  extra_args="--pull --no-cache $extra_args"
fi

docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

# Stage 1:
# Base image, Ubuntu with core packages and GCC9
# Add essential maths libs and Python built and installed
# Add Python3 venv with additional Python essentials
docker buildx build --platform linux/arm64/v8 \
    $extra_args \
    --progress=plain \
    --target tf_base \
    -t tensorflow-base-v$tf_version:latest .

# Stage 2 : Adds bazel
docker buildx build --platform linux/arm64/v8 \
    $extra_args \
    --progress=plain \
    --target tf_dev \
    -t tensorflow-dev-v$tf_version:latest .

# Stage 3 : Build TensorFlow from sources and creates a wheel
docker buildx build --platform linux/arm64/v8 \
    $extra_args \
    --progress=plain \
    --target tf_dev_with_bazel \
    -t tensorflow-dev-v$tf_version:latest .

# Stage 4: Clone Python VENV with Tensorflow and install TensorRT
# docker buildx build $extra_args --target tensorflow --platform linux/arm64 -t tensorflow-v$tf_version:latest .

