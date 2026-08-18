# check=skip=InvalidDefaultArgInFrom

# `make release-package` supplies this from .emscripten-version, and CMake
# rejects any SDK version that does not match that pin.
ARG EMSCRIPTEN_VERSION
FROM emscripten/emsdk:${EMSCRIPTEN_VERSION} AS build

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /source
COPY CMakeLists.txt CMakePresets.json .emscripten-version ./
COPY cmake ./cmake
COPY engines ./engines
COPY game ./game
COPY wasm ./wasm

RUN cmake --preset wasm-release \
    && cmake --build --preset wasm-release \
    && ctest --preset wasm-release

RUN mkdir --parents /release \
    && npm pack ./build/wasm-release/package \
      --ignore-scripts \
      --pack-destination /release \
    && cd /release \
    && sha256sum *.tgz > SHA256SUMS

FROM scratch AS release-artifacts
COPY --from=build /release/ /
