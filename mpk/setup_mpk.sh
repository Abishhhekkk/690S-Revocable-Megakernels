#!/bin/bash
# Build our patched MPK tree: upstream mirage @ MPK_BASE + mpk/patches/*.patch.
#
#   mpk/setup_mpk.sh [DEST]        default DEST=~/mirage-a3
#
# Needs an existing, already-built mirage checkout (MIRAGE_SRC, default ~/mirage,
# built with `pip install -e .`). DEST is a git worktree of it at MPK_BASE with
# the patches applied; it reuses MIRAGE_SRC's compiled core and deps through
# symlinks, so nothing is rebuilt (our patches only touch headers and Python,
# which MPK compiles per model at run time). Light enough for a login node.
#
# Use it by putting DEST/python first on PYTHONPATH (see profiling/a3_yield_test.sbatch).
set -euo pipefail

MPK_BASE="${MPK_BASE:-ed771bda}"
MIRAGE_SRC="${MIRAGE_SRC:-$HOME/mirage}"
DEST="${1:-$HOME/mirage-a3}"
PATCHES="$(cd "$(dirname "$0")" && pwd)/patches"

[ -d "$MIRAGE_SRC/.git" ] || { echo "no mirage checkout at $MIRAGE_SRC" >&2; exit 1; }
[ -e "$DEST" ] && { echo "$DEST already exists; remove it or pass another path" >&2; exit 1; }
ls "$MIRAGE_SRC"/python/mirage/core.cpython-*.so >/dev/null 2>&1 \
    || { echo "$MIRAGE_SRC is not built (no python/mirage/core*.so)" >&2; exit 1; }

git -C "$MIRAGE_SRC" fetch -q origin mpk
git -C "$MIRAGE_SRC" worktree add -q --detach "$DEST" "$MPK_BASE"
for p in "$PATCHES"/*.patch; do
    echo "applying $(basename "$p")"
    git -C "$DEST" apply --index "$p"
done

# reuse the compiled core and the dependency headers of MIRAGE_SRC
for f in "$MIRAGE_SRC"/python/mirage/core.cpython-*.so "$MIRAGE_SRC/python/mirage/lib"; do
    ln -s "$f" "$DEST/python/mirage/$(basename "$f")"
done
ln -s "$MIRAGE_SRC/deps/cutlass/include" "$DEST/deps/cutlass/include"
ln -s "$MIRAGE_SRC/deps/cutlass/tools"   "$DEST/deps/cutlass/tools"
ln -s "$MIRAGE_SRC/deps/json/include"    "$DEST/deps/json/include"

echo "done: $DEST  (base $MPK_BASE + $(ls "$PATCHES"/*.patch | wc -l) patches)"
echo "use:  export PYTHONPATH=$DEST/python"
