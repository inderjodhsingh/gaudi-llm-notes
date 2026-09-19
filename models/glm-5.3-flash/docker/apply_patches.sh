#!/usr/bin/env bash
# Apply patches/NN-*.patch in version order. A patch whose first diff touches vllm/ goes to the vLLM tree; every
# other patch goes to vllm-gaudi. Usage: apply_patches.sh <patch dir> <vllm tree> <vllm-gaudi tree>
set -euo pipefail
P=$1; VLLM=$2; GAUDI=$3
n=0
for f in $(ls "$P"/*.patch | sort -V); do
  tgt=$(grep -m1 '^diff --git a/' "$f" | awk '{print $3}' | cut -c3- | cut -d/ -f1)
  if [ "$tgt" = "vllm" ]; then repo=$VLLM; else repo=$GAUDI; fi
  git -C "$repo" -c user.name=glm53-build -c user.email=build@local am --committer-date-is-author-date -q "$f"
  echo "applied $(basename "$f") -> $(basename "$repo")"
  n=$((n + 1))
done
echo "applied $n patch files; vllm HEAD $(git -C "$VLLM" rev-parse --short HEAD), vllm-gaudi HEAD $(git -C "$GAUDI" rev-parse --short HEAD)"
