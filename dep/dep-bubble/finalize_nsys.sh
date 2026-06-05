#!/usr/bin/env bash
# Finalize per-rank vLLM nsys captures: .qdstrm -> .nsys-rep -> .sqlite using the
# x86 Nsight 2025.3.1 host install. The dynamo vLLM container bind-mounts only the
# aarch64 *target* Nsight tree, so time-based captures stream raw .qdstrm but never
# finalize on-node (no host QdstrmImporter). Run this on the ptyche x86 login node.
#
# Usage: finalize_nsys.sh <profile_dir> <node_prefix> [rank ...]   (default ranks 0 1 2 3)
# e.g.   finalize_nsys.sh outputs/2191165/logs/profiles/agg ptyche0287
set -uo pipefail
DIR="${1%/}"; NODE="$2"; shift 2
RANKS=("$@"); [ ${#RANKS[@]} -eq 0 ] && RANKS=(0 1 2 3)
H=/lustre/fsw/coreai_dlfw_dev/connorc/tools/nsight-host-x64/host/opt/nvidia/nsight-systems/2025.3.1
IMP="$H/host-linux-x64/QdstrmImporter"; NSYS="$H/target-linux-x64/nsys"
for r in "${RANKS[@]}"; do
  b="$DIR/${NODE}_agg_w0_rank${r}_profile"
  echo "=== rank $r ==="
  if [ -f "$b.qdstrm" ]; then
    "$IMP" -f -i "$b.qdstrm" -o "$b.nsys-rep" 2>&1 | tail -1
  else
    echo "  no .qdstrm for rank $r"
  fi
  if [ -f "$b.nsys-rep" ]; then
    "$NSYS" export --type sqlite --force-overwrite true -o "$b.sqlite" "$b.nsys-rep" 2>&1 | tail -1
  fi
  ls -la "$b.sqlite" 2>/dev/null || echo "  NO sqlite produced for rank $r"
done
