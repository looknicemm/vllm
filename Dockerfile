# The vLLM Dockerfile is used to construct vLLM image that can be directly used
# to run the OpenAI compatible server.

# Please update any changes made here to
# docs/source/dev/dockerfile/dockerfile.rst and
# docs/source/assets/dev/dockerfile-stages-dependency.png

ARG CUDA_VERSION=12.4.1
#################### BASE BUILD IMAGE ####################
# prepare basic build environment
FROM nvcr.io/nvidia/pytorch:24.06-py3 AS base

ARG CUDA_VERSION=12.4.1
ARG PYTHON_VERSION=3

ENV DEBIAN_FRONTEND=noninteractive

# cuda arch list used by torch
# can be useful for both `dev` and `test`
# explicitly set the list to avoid issues with torch 2.2
# see https://github.com/pytorch/pytorch/pull/123243
ARG torch_cuda_arch_list='7.5 8.0 8.6 8.9 9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv python3-pip \
    && if [ "${PYTHON_VERSION}" != "3" ]; then update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1; fi \
    && python3 --version \
    && python3 -m pip --version

RUN apt-get update -y \
    && apt-get install -y python3-pip git curl sudo

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

WORKDIR /workspace
# max jobs used by Ninja to build extensions
ARG max_jobs=8
ENV MAX_JOBS=${max_jobs}

# install build and runtime dependencies
COPY requirements-common.txt requirements-common.txt
COPY requirements-cuda.txt requirements-cuda.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements-cuda.txt


RUN mkdir vllm-aarch64-whl

RUN --mount=type=cache,target=/root/.cache/pip \
    pip --verbose wheel --use-pep517 --no-deps -w /workspace/vllm-aarch64-whl --no-build-isolation git+https://github.com/vllm-project/flash-attention.git

RUN --mount=type=cache,target=/root/.cache/pip \
    pip --verbose wheel --use-pep517 --no-deps -w /workspace/vllm-aarch64-whl --no-build-isolation xformers==0.0.27

RUN python3 -m pip install packaging

RUN --mount=type=cache,target=/root/.cache/pip \
    git clone https://github.com/openai/triton ; cd triton/python ; git submodule update --init --recursive ;  pip --verbose wheel -w /workspace/vllm-aarch64-whl .

# Needed to ensure causal-conv1d builds from scratch
ENV CAUSAL_CONV1D_FORCE_BUILD=TRUE
ENV CAUSAL_CONV1D_SKIP_CUDA_BUILD=FALSE

RUN --mount=type=cache,target=/root/.cache/pip \
    pip --verbose wheel --use-pep517 --no-deps -w /workspace/vllm-aarch64-whl --no-build-isolation --no-cache-dir git+https://github.com/Dao-AILab/causal-conv1d.git

ENV MAMBA_FORCE_BUILD=TRUE
RUN --mount=type=cache,target=/root/.cache/pip \
    pip --verbose wheel --use-pep517 --no-deps -w /workspace/vllm-aarch64-whl --no-build-isolation --no-cache-dir git+https://github.com/state-spaces/mamba.git

RUN --mount=type=cache,target=/root/.cache/pip \
	git clone https://github.com/flashinfer-ai/flashinfer.git ; cd flashinfer/python ; pip --verbose wheel --use-pep517 --no-deps -w /workspace/vllm-aarch64-whl --no-build-isolation --no-cache-dir .

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install /workspace/vllm-aarch64-whl/*.whl --no-cache-dir --no-deps

#################### BASE BUILD IMAGE ####################

#################### WHEEL BUILD IMAGE ####################
FROM base AS build

ARG PYTHON_VERSION=3

# install build dependencies
COPY requirements-build.txt requirements-build.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements-build.txt

# install compiler cache to speed up compilation leveraging local or remote caching
RUN apt-get update -y && apt-get install -y ccache

# files and directories related to build wheels
COPY csrc csrc
COPY setup.py setup.py
COPY cmake cmake
COPY CMakeLists.txt CMakeLists.txt
COPY requirements-common.txt requirements-common.txt
COPY requirements-cuda.txt requirements-cuda.txt
COPY pyproject.toml pyproject.toml
COPY vllm vllm

# max jobs used by Ninja to build extensions
ARG max_jobs=8
ENV MAX_JOBS=${max_jobs}
# number of threads used by nvcc
ARG nvcc_threads=8
ENV NVCC_THREADS=$nvcc_threads
# make sure punica kernels are built (for LoRA)
ENV VLLM_INSTALL_PUNICA_KERNELS=1

ARG buildkite_commit
ENV BUILDKITE_COMMIT=${buildkite_commit}

ARG USE_SCCACHE
# if USE_SCCACHE is set, use sccache to speed up compilation
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$USE_SCCACHE" = "1" ]; then \
        echo "Installing sccache..." \
        && curl -L -o sccache.tar.gz https://github.com/mozilla/sccache/releases/download/v0.8.1/sccache-v0.8.1-x86_64-unknown-linux-musl.tar.gz \
        && tar -xzf sccache.tar.gz \
        && sudo mv sccache-v0.8.1-x86_64-unknown-linux-musl/sccache /usr/bin/sccache \
        && rm -rf sccache.tar.gz sccache-v0.8.1-x86_64-unknown-linux-musl \
        && export SCCACHE_BUCKET=vllm-build-sccache \
        && export SCCACHE_REGION=us-west-2 \
        && export CMAKE_BUILD_TYPE=Release \
        && sccache --show-stats \
        && python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38 \
        && sccache --show-stats; \
    fi

ENV CCACHE_DIR=/root/.cache/ccache
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/pip \
    if [ "$USE_SCCACHE" != "1" ]; then \
        python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38; \
    fi

# check the size of the wheel, we cannot upload wheels larger than 100MB
COPY .buildkite/check-wheel-size.py check-wheel-size.py
RUN python3 check-wheel-size.py dist

#################### EXTENSION Build IMAGE ####################

#################### DEV IMAGE ####################
FROM base as dev

COPY requirements-lint.txt requirements-lint.txt
COPY requirements-test.txt requirements-test.txt
COPY requirements-dev.txt requirements-dev.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements-dev.txt

#################### DEV IMAGE ####################
#################### MAMBA Build IMAGE ####################
#FROM dev as mamba-builder
# max jobs used for build
#ARG max_jobs=5
#ENV MAX_JOBS=${max_jobs}

#WORKDIR /usr/src/mamba

#COPY requirements-mamba.txt requirements-mamba.txt
#RUN --mount=type=cache,target=/root/.cache/pip \
#    git clone https://github.com/openai/triton ; cd triton/python ; git submodule update --init --recursive ;  python setup.py install

# Download the wheel or build it if a pre-compiled release doesn't exist
#RUN pip --verbose wheel -r requirements-mamba.txt \
#    --no-build-isolation --no-deps --no-cache-dir

#################### MAMBA Build IMAGE ####################

#################### vLLM installation IMAGE ####################
# image with vLLM installed
FROM nvcr.io/nvidia/pytorch:24.06-py3 AS vllm-base
ARG CUDA_VERSION=12.4.1
WORKDIR /vllm-workspace
# max jobs used by Ninja to build extensions
ARG max_jobs=8
ENV MAX_JOBS=${max_jobs}

# cuda arch list used by torch
# can be useful for both `dev` and `test`
# explicitly set the list to avoid issues with torch 2.2
# see https://github.com/pytorch/pytorch/pull/123243
ARG torch_cuda_arch_list='7.5 8.0 8.6 8.9 9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

RUN apt-get update -y \
    && apt-get install -y python3-pip git vim

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

# install build and runtime dependencies
COPY requirements-common.txt requirements-common.txt
COPY requirements-cuda.txt requirements-cuda.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements-cuda.txt
# install vllm wheel first, so that torch etc will be installed
RUN --mount=type=bind,from=build,src=/workspace/dist,target=/vllm-workspace/dist \
    --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install dist/*.whl --verbose

RUN --mount=type=bind,from=base,src=/workspace/vllm-aarch64-whl,target=/workspace/vllm-aarch64-whl \
    --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install /workspace/vllm-aarch64-whl/*.whl --no-deps --no-cache-dir

#################### vLLM installation IMAGE ####################


#################### TEST IMAGE ####################
# image to run unit testing suite
# note that this uses vllm installed by `pip`
FROM vllm-base AS test

ADD . /vllm-workspace/

# install development dependencies (for testing)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements-dev.txt

# doc requires source code
# we hide them inside `test_docs/` , so that this source code
# will not be imported by other tests
RUN mkdir test_docs
RUN mv docs test_docs/
RUN mv vllm test_docs/

#################### TEST IMAGE ####################

#################### OPENAI API SERVER ####################
# openai api server alternative
FROM vllm-base AS vllm-openai

# install additional dependencies for openai api server
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install accelerate hf_transfer 'modelscope!=1.15.0'

ENV VLLM_USAGE_SOURCE production-docker-image

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
#################### OPENAI API SERVER ####################
