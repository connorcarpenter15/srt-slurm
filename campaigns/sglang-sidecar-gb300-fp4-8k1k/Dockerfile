# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=lmsysorg/sglang:v0.5.14-cu130-runtime
ARG RUST_IMAGE=rust:1.96.1-bookworm

FROM ${RUST_IMAGE} AS rust-toolchain

FROM ${RUST_IMAGE} AS sidecar-builder
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        clang \
        cmake \
        libssl-dev \
        pkg-config \
        protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src/dynamo
COPY --from=dynamo-source / /src/dynamo/
RUN cargo build --locked --release --package dynamo-sglang-sidecar

FROM ${BASE_IMAGE} AS sglang-wheel-builder
COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo
COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup
ENV PATH=/usr/local/cargo/bin:${PATH}
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        libssl-dev \
        pkg-config \
        protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src/sglang
COPY --from=sglang-source / /src/sglang/
ARG SGLANG_WHEEL_VERSION=0.0.0+cc7d6659fd
RUN python3 -m pip install --no-cache-dir \
        build \
        'setuptools>=61.0' \
        'setuptools-rust>=1.10' \
        'setuptools-scm>=8.0' \
        wheel \
    && cd python \
    && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=${SGLANG_WHEEL_VERSION} \
        python3 -m build --wheel --no-isolation \
    && mkdir -p /artifacts \
    && cp dist/*.whl /artifacts/

FROM scratch AS campaign-artifacts
COPY --from=sidecar-builder /src/dynamo/target/release/dynamo-sglang-sidecar /dynamo-sglang-sidecar
COPY --from=sglang-wheel-builder /artifacts/ /

FROM ${BASE_IMAGE} AS runtime
ARG BASE_IMAGE
ARG DYNAMO_COMMIT
ARG SGLANG_COMMIT
ARG SIDECAR_SHA256
ARG SGLANG_WHEEL_SHA256
ARG BUILD_TIMESTAMP
ARG CUDA_VERSION=13.0
LABEL org.opencontainers.image.base.name=${BASE_IMAGE} \
      org.opencontainers.image.created=${BUILD_TIMESTAMP} \
      org.opencontainers.image.description="GB300 SGLang native-gRPC Dynamo sidecar benchmark candidate" \
      org.opencontainers.image.revision=${DYNAMO_COMMIT} \
      ai.dynamo.sglang.commit=${SGLANG_COMMIT} \
      ai.dynamo.sidecar.sha256=${SIDECAR_SHA256} \
      ai.dynamo.sglang-wheel.sha256=${SGLANG_WHEEL_SHA256}

COPY --from=sglang-wheel-builder /artifacts/ /tmp/sglang-wheel/
RUN python3 -m pip install --no-cache-dir --no-deps --force-reinstall \
        wheel \
        distro \
        zstandard \
        apache-tvm-ffi==0.1.11 \
        mistral-common==1.11.5 \
        sgl-deep-gemm==0.1.4 \
        tilelang==0.1.11 \
        transformers==5.12.1 \
    && python3 -m pip install --no-cache-dir --no-deps --force-reinstall /tmp/sglang-wheel/*.whl \
    && rm -rf /tmp/sglang-wheel \
    && python3 -c 'from sglang.srt.grpc import _core; assert hasattr(_core, "start_server")'

RUN install -d /usr/local/libexec /usr/local/share/dynamo-sglang-sidecar
COPY --from=sidecar-builder /src/dynamo/target/release/dynamo-sglang-sidecar \
    /usr/local/libexec/dynamo-sglang-sidecar.real
COPY dynamo-sglang-sidecar-wrapper.sh /usr/local/bin/dynamo-sglang-sidecar
RUN chmod 0755 \
        /usr/local/bin/dynamo-sglang-sidecar \
        /usr/local/libexec/dynamo-sglang-sidecar.real \
    && printf '%s\n' \
        "dynamo_commit=${DYNAMO_COMMIT}" \
        "sglang_commit=${SGLANG_COMMIT}" \
        "sidecar_sha256=${SIDECAR_SHA256}" \
        "sglang_wheel_sha256=${SGLANG_WHEEL_SHA256}" \
        "architecture=linux/arm64" \
        "cuda_version=${CUDA_VERSION}" \
        "base_image=${BASE_IMAGE}" \
        "build_timestamp=${BUILD_TIMESTAMP}" \
        > /usr/local/share/dynamo-sglang-sidecar/build-info
