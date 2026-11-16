#!/usr/bin/env bash
# Fetches the hnswlib baseline (header-only C++ library) into
# bench/third_party/hnswlib, pinned to one release tag and commit.
#
# Usage: bench/get_hnswlib.sh
#
# The tag is checked against the remote with git ls-remote before cloning, and
# the checkout is checked against HNSWLIB_COMMIT afterwards, so a moved tag or a
# wrong existing checkout fails instead of silently changing the baseline.
# bench/CMakeLists.txt reads HNSWLIB_COMMIT from this file.
set -euo pipefail

HNSWLIB_URL=https://github.com/nmslib/hnswlib.git
HNSWLIB_TAG=v0.9.0
HNSWLIB_COMMIT=d9b3608c83d83b46c96e25088cb1d729b29dcfe9

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dest="$here/third_party/hnswlib"

if [[ -d "$dest/.git" ]]; then
  have="$(git -C "$dest" rev-parse HEAD)"
  if [[ "$have" != "$HNSWLIB_COMMIT" ]]; then
    echo "error: $dest is at $have, expected $HNSWLIB_TAG ($HNSWLIB_COMMIT)." >&2
    echo "Remove that directory and rerun this script." >&2
    exit 1
  fi
  echo "hnswlib $HNSWLIB_TAG ($HNSWLIB_COMMIT) already present in $dest"
  exit 0
fi

remote="$(git ls-remote "$HNSWLIB_URL" "refs/tags/$HNSWLIB_TAG" "refs/tags/$HNSWLIB_TAG^{}")"
if [[ -z "$remote" ]]; then
  echo "error: tag $HNSWLIB_TAG not found at $HNSWLIB_URL (or the remote is unreachable)" >&2
  exit 1
fi
# An annotated tag lists the tag object and then the peeled commit (^{});
# a lightweight tag lists only the commit. The last line is the commit either way.
remote_commit="$(tail -n 1 <<<"$remote" | cut -f 1)"
if [[ "$remote_commit" != "$HNSWLIB_COMMIT" ]]; then
  echo "error: remote tag $HNSWLIB_TAG points to $remote_commit, expected $HNSWLIB_COMMIT" >&2
  exit 1
fi

mkdir -p "$here/third_party"
git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$HNSWLIB_TAG" \
  "$HNSWLIB_URL" "$dest"
have="$(git -C "$dest" rev-parse HEAD)"
if [[ "$have" != "$HNSWLIB_COMMIT" ]]; then
  echo "error: cloned $HNSWLIB_TAG is at $have, expected $HNSWLIB_COMMIT" >&2
  rm -rf "$dest"
  exit 1
fi
echo "hnswlib $HNSWLIB_TAG ($HNSWLIB_COMMIT) cloned into $dest"
