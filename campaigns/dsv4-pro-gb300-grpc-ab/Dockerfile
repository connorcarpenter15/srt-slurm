# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=lmsysorg/sglang:dev-cu13@sha256:4b140bc08eb4782b057109b084b6df94f74c3a66c6984ee383a1d6c3714994d5
ARG RUST_IMAGE=rust:1.96.1-bookworm

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
COPY --from=sglang-source /proto/sglang/runtime/v1/sglang.proto /tmp/sglang-server/sglang.proto
RUN mkdir -p /tmp/sglang-sidecar /artifacts \
    && cp lib/sglang-sidecar/proto/sglang.proto /tmp/sglang-sidecar/sglang.proto \
    && protoc --experimental_allow_proto3_optional -I /tmp/sglang-sidecar --descriptor_set_out=/artifacts/sidecar-descriptor.bin /tmp/sglang-sidecar/sglang.proto \
    && protoc --experimental_allow_proto3_optional -I /tmp/sglang-server --descriptor_set_out=/artifacts/server-descriptor.bin /tmp/sglang-server/sglang.proto \
    && cmp /artifacts/sidecar-descriptor.bin /artifacts/server-descriptor.bin \
    && sha256sum \
        lib/sglang-sidecar/proto/sglang.proto \
        /tmp/sglang-server/sglang.proto \
        /artifacts/server-descriptor.bin \
        > /artifacts/proto-sha256.txt \
    && cargo build --locked --release --package dynamo-sglang-sidecar \
    && cp target/release/dynamo-sglang-sidecar /artifacts/

FROM ${RUST_IMAGE} AS rust-toolchain

FROM ${BASE_IMAGE} AS dynamo-wheel-builder
USER root
COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo
COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup
ENV PATH=/usr/local/cargo/bin:${PATH}
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        clang \
        cmake \
        libclang-dev \
        libssl-dev \
        patchelf \
        pkg-config \
        protobuf-compiler \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir \
        build \
        hatchling \
        'maturin[patchelf]>=1.9,<2' \
        tomlkit
WORKDIR /src/dynamo
COPY --from=dynamo-source / /src/dynamo/
RUN mkdir -p /artifacts \
    && python3 -m build --wheel --no-isolation --outdir /artifacts \
    && cd lib/bindings/python \
    && maturin build --locked --release \
        --features 'kv-indexer,slot-tracker,select-service,mm-routing' \
        --out /artifacts

FROM ${BASE_IMAGE} AS sglang-wheel-builder
USER root
COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo
COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup
ENV PATH=/usr/local/cargo/bin:${PATH}
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        libssl-dev \
        pkg-config \
        protobuf-compiler \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir \
        build \
        'setuptools>=61.0' \
        'setuptools-rust>=1.10' \
        'setuptools-scm>=8.0' \
        wheel
WORKDIR /src/sglang
COPY --from=sglang-source / /src/sglang/
ARG SGLANG_WHEEL_VERSION=0.0.0+e2728ac504c
RUN cd python \
    && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=${SGLANG_WHEEL_VERSION} \
        SGLANG_BUILD_RUST_EXTS=grpc \
        python3 -m build --wheel --no-isolation \
    && mkdir -p /artifacts \
    && cp dist/*.whl /artifacts/

FROM ${BASE_IMAGE} AS runtime
USER root
ARG BASE_IMAGE
ARG DYNAMO_COMMIT
ARG SGLANG_COMMIT
ARG SIDECAR_SHA256
ARG DYNAMO_WHEEL_SHA256
ARG DYNAMO_RUNTIME_WHEEL_SHA256
ARG SGLANG_WHEEL_SHA256
ARG PROTO_DESCRIPTOR_SHA256
ARG BUILD_TIMESTAMP
ARG CUDA_VERSION=13.0
LABEL org.opencontainers.image.base.name=${BASE_IMAGE} \
      org.opencontainers.image.created=${BUILD_TIMESTAMP} \
      org.opencontainers.image.description="GB300 DeepSeek-V4-Pro SGLang gRPC A/B candidate" \
      org.opencontainers.image.revision=${DYNAMO_COMMIT} \
      ai.dynamo.sglang.commit=${SGLANG_COMMIT} \
      ai.dynamo.sidecar.sha256=${SIDECAR_SHA256} \
      ai.dynamo.python-wheel.sha256=${DYNAMO_WHEEL_SHA256} \
      ai.dynamo.runtime-wheel.sha256=${DYNAMO_RUNTIME_WHEEL_SHA256} \
      ai.dynamo.sglang-wheel.sha256=${SGLANG_WHEEL_SHA256} \
      ai.dynamo.proto-descriptor.sha256=${PROTO_DESCRIPTOR_SHA256}

COPY --from=dynamo-wheel-builder /artifacts/ /tmp/dynamo-wheels/
COPY --from=sglang-wheel-builder /artifacts/ /tmp/sglang-wheel/
RUN python3 -m pip install --no-cache-dir --no-deps --force-reinstall \
        /tmp/dynamo-wheels/ai_dynamo_runtime*.whl \
        /tmp/dynamo-wheels/ai_dynamo-*none-any.whl \
        /tmp/sglang-wheel/sglang-*.whl \
    && rm -rf /tmp/dynamo-wheels /tmp/sglang-wheel \
    && python3 -m pip check \
    && python3 -c 'import dynamo._core; import dynamo.frontend; import dynamo.sglang' \
    && python3 -c 'from sglang.srt.grpc import _core; assert hasattr(_core, "start_server")' \
    && python3 -c 'import deep_gemm, flashinfer, mooncake, sgl_kernel; from sglang.srt.layers.attention.dsv4 import metadata; from sglang.srt.layers.quantization import mxfp4; from sglang.srt.mem_cache import swa_memory_pool; from sglang.srt.models import deepseek_v4; from sglang.srt.parser import reasoning_parser' \
    && python3 -m sglang.launch_server --help | grep -F -- '--grpc-port'

RUN install -d /usr/local/libexec /usr/local/share/dynamo-sglang-sidecar
COPY --from=sidecar-builder /artifacts/dynamo-sglang-sidecar /usr/local/libexec/dynamo-sglang-sidecar.real
COPY dynamo-sglang-sidecar-wrapper.sh /usr/local/bin/dynamo-sglang-sidecar
RUN chmod 0755 \
        /usr/local/bin/dynamo-sglang-sidecar \
        /usr/local/libexec/dynamo-sglang-sidecar.real \
    && printf '%s\n' \
        "dynamo_commit=${DYNAMO_COMMIT}" \
        "sglang_commit=${SGLANG_COMMIT}" \
        "sidecar_sha256=${SIDECAR_SHA256}" \
        "dynamo_wheel_sha256=${DYNAMO_WHEEL_SHA256}" \
        "dynamo_runtime_wheel_sha256=${DYNAMO_RUNTIME_WHEEL_SHA256}" \
        "sglang_wheel_sha256=${SGLANG_WHEEL_SHA256}" \
        "proto_descriptor_sha256=${PROTO_DESCRIPTOR_SHA256}" \
        "architecture=linux/arm64" \
        "cuda_version=${CUDA_VERSION}" \
        "base_image=${BASE_IMAGE}" \
        "build_timestamp=${BUILD_TIMESTAMP}" \
        > /usr/local/share/dynamo-sglang-sidecar/build-info \
    && python3 -m pip freeze --all > /usr/local/share/dynamo-sglang-sidecar/package-lock.txt

FROM runtime AS campaign-artifacts
RUN mkdir -p /campaign-artifacts \
    && cp /usr/local/share/dynamo-sglang-sidecar/build-info /campaign-artifacts/ \
    && cp /usr/local/share/dynamo-sglang-sidecar/package-lock.txt /campaign-artifacts/
COPY --from=sidecar-builder /artifacts/ /campaign-artifacts/
COPY --from=dynamo-wheel-builder /artifacts/ /campaign-artifacts/
COPY --from=sglang-wheel-builder /artifacts/ /campaign-artifacts/

FROM runtime AS final
