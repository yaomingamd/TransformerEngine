#!/usr/bin/env bash
set -euo pipefail
TE_DIR=/dockerx/TE_HD256
LOG_DIR=${TE_DIR}/logs
mkdir -p "${LOG_DIR}"
exec >> "${LOG_DIR}/wait_install_test.log" 2>&1
echo "[$(date -Is)] wait_then_install_test start"
# Wait for in-flight pip wheel / ninja (up to 6h)
for i in $(seq 1 360); do
  if ls ${TE_DIR}/dist/transformer_engine*.whl >/dev/null 2>&1; then
    echo "[$(date -Is)] wheel found"
    break
  fi
  if ! pgrep -f "pip wheel . --no-build-isolation" >/dev/null && ! pgrep -f "ninja -v -j" >/dev/null; then
    echo "[$(date -Is)] build processes ended, wheel check..."
    break
  fi
  echo "[$(date -Is)] still building... ($i)"
  sleep 60
done
if ! ls ${TE_DIR}/dist/transformer_engine*.whl >/dev/null 2>&1; then
  echo "[$(date -Is)] no wheel after wait; check build_wheel.log tail"
  tail -30 ${TE_DIR}/build_wheel.log || true
  exit 1
fi
bash ${TE_DIR}/scripts/build_install_test_hd256.sh install
bash ${TE_DIR}/scripts/build_install_test_hd256.sh test
echo "[$(date -Is)] wait_then_install_test done"
