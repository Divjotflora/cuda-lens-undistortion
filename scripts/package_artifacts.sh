#!/usr/bin/env bash
# Bundles logs, metrics and before/after images into artifacts.tar.gz
# (the "proof of execution" upload). Corrected PPMs are left out on purpose:
# they are hundreds of MB and fully reproducible with ./run.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

items=()
for p in output/logs output/metrics.csv output/samples output/samples_large \
         data/camera.txt; do
  [[ -e "$p" ]] && items+=("$p")
done
if [[ ${#items[@]} -eq 0 ]]; then
  echo "nothing to package - run ./run.sh first" >&2
  exit 1
fi
tar -czf artifacts.tar.gz "${items[@]}"
echo "wrote artifacts.tar.gz ($(du -h artifacts.tar.gz | cut -f1)):"
tar -tzf artifacts.tar.gz | sed 's/^/  /'
