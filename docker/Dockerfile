# ==== base ====
# Shared base: system packages, node, conda, rust, formatters, entrypoint.
# A block runs from its marker to the next; derive.sh drops untaken ones.
# OS image; build.sh passes BASE_OS (a CUDA image works too)
ARG OS=ubuntu:24.04
FROM ${OS}
ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

# system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      sudo curl wget git ca-certificates build-essential pkg-config xz-utils \
    && rm -rf /var/lib/apt/lists/*

# git trusts the bind-mounted workspace
RUN git config --system --add safe.directory /workspace

# node from the official tarball; amd64 maps to x64
ARG NODE_VERSION=22.14.0
RUN curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${TARGETARCH/amd64/x64}.tar.xz" \
    | tar -xJ -C /usr/local --strip-components=1 --exclude='*/CHANGELOG.md' --exclude='*/README.md' --exclude='*/LICENSE'

# miniconda pinned to py314, matching environment.yml
ARG MINICONDA_VERSION=py314_26.7.1-1
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN curl -fsSL "https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-$(uname -m).sh" -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p "$CONDA_DIR" \
    && rm /tmp/miniconda.sh \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# rust / cargo, minimal profile plus clippy and rustfmt
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo
ENV PATH=$CARGO_HOME/bin:$PATH
RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.99.0 -c clippy -c rustfmt

# formatters, grouped last so they are easy to edit
RUN CONDA_AUTO_UPDATE_CONDA=false conda install -n base -y conda-forge::ruff=0.17.0 conda-forge::clang-format=23.1.3 conda-forge::go-shfmt=3.14.1 \
    && conda clean -afy \
    && npm install -g prettier@3.9.10

# interactive entrypoint (exec.sh overrides this with bash)
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# ==== project ====
# Project-owned from this marker on; derive.sh --sync keeps it.
# conda env from environment.yml
COPY environment.yml /tmp/environment.yml
RUN conda env update -n base -f /tmp/environment.yml && conda clean -afy

# ---- project toolchain: add the project's own blocks below ----

# nki compiler env; its wheels need python 3.12
RUN conda create -y -n nki python=3.12 libarchive=3.8.8 \
    && /opt/conda/envs/nki/bin/pip install --no-cache-dir \
      --extra-index-url https://pip.repos.neuron.amazonaws.com \
      neuronx-cc==2.27.5334.0 nki==0.6.0 \
    && /opt/conda/envs/nki/bin/pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.13.0+cpu \
    && conda clean -afy

# llvm 18 toolchain, cmake and ninja for C++ builds
RUN apt-get update && apt-get install -y --no-install-recommends \
      clang-18 lld-18 llvm-18-dev zlib1g-dev cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# workspace and interactive entrypoint
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
