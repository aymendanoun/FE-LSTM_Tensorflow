#!/usr/bin/env bash
# Build mmg (https://github.com/MmgTools/mmg) from its official source release and install
# mmg3d_O3 / mmg2d_O3 / mmgs_O3 into the active conda environment (felstm-geometry).
# mmg is needed by the remeshing step of the RVE builders; there is no conda-forge or Homebrew-free
# package for Apple Silicon. Requires a C compiler (Xcode Command Line Tools on macOS).
#   conda activate felstm-geometry
#   bash scripts/install_mmg.sh            # optional: MMG_VERSION=5.8.0
set -euo pipefail
: "${CONDA_PREFIX:?activate the felstm-geometry environment first}"
VERSION="${MMG_VERSION:-5.8.0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
conda install -y -q -c conda-forge cmake make
curl -sSL -o "$WORK/mmg.tar.gz" "https://github.com/MmgTools/mmg/archive/refs/tags/v${VERSION}.tar.gz"
tar xzf "$WORK/mmg.tar.gz" -C "$WORK"
cmake -S "$WORK/mmg-${VERSION}" -B "$WORK/build" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -DUSE_SCOTCH=OFF -DUSE_VTK=OFF -DUSE_ELAS=OFF \
      -DBUILD_TESTING=OFF
cmake --build "$WORK/build" -j "$(getconf _NPROCESSORS_ONLN)"
cmake --install "$WORK/build"
mmg3d_O3 -h | head -2
