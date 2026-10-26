#!/usr/bin/env bash
# Benchmark ISOFIT on the image_cube medium analytical line example.
# Usage: bash benchmark-medium-analytical.sh [workdir] [isofit git ref]
set -euo pipefail

WORKDIR="${1:-$HOME/isofit-benchmark}"
REF="${2:-main}"
REPO="https://github.com/isofit/isofit.git"

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Pull isofit and install it into a fresh virtualenv
if [ ! -d isofit ]; then
    git clone --depth 1 --branch "$REF" "$REPO" isofit
fi
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install ./isofit

# Fetch data and examples, then build the examples
isofit download all
isofit build

# Run the medium analytical line example, timed
EXAMPLE_DIR="$(isofit path examples)/image_cube/medium"
cd "$EXAMPLE_DIR"

START=$(date +%s)
bash analytical.sh
END=$(date +%s)
ELAPSED=$((END - START))

echo "----------------------------------------"
echo "Host:      $(hostname)"
echo "CPUs:      $(nproc 2>/dev/null || sysctl -n hw.ncpu)"
echo "Ref:       $REF"
echo "Elapsed:   ${ELAPSED}s"
echo "----------------------------------------"
