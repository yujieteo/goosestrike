#!/usr/bin/env bash
# Builds web/strike.mjs and web/strike.wasm from sim/strike.goose.
#
# Needs the Goose compiler (GOOSE, default: goose on PATH) and Emscripten's
# emcc on PATH. On NixOS: nix shell nixpkgs#emscripten --command ./build.sh
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
goose="${GOOSE:-goose}"
out="$root/build"
mkdir -p "$out"

# Goose -> one C file with the whole runtime and no main(); goose_init()
# and the export fn wrappers are declared in strike.h.
"$goose" --standalone --header "$out/strike.h" -o "$out/strike.c" "$root/sim/strike.goose"

# Every exported function of the header, for Emscripten.
exports="$(sed -n 's/^[a-z0-9_ *]*[ *]\([a-z_0-9]*\)(.*);$/_\1/p' "$out/strike.h" | paste -sd, -)"

# WebAssembly has no address-space reservation: the one Goose data stack
# gets a fixed region from Emscripten's mmap (which allocates it), and no
# guard gap traps an overrun. The solver keeps its state in static data.
# Only goose_init pushes to the data stack: one zeroed temporary per global
# array, popped at once. The largest is esig, f64[MAXE * 15] = 2.88 MB, so
# the region is 4 MiB. The time loop never touches the data stack (README).
emcc -O3 "$out/strike.c" -o "$root/web/strike.mjs" \
    -DGS_STACK_RESERVE='(4ull<<20)' -DGS_STACK_GAP=65536 \
    -sMODULARIZE -sEXPORT_ES6 -sENVIRONMENT=web,node \
    -sINITIAL_MEMORY=64MB -sALLOW_MEMORY_GROWTH \
    -sEXPORTED_FUNCTIONS="_malloc,_free,$exports" \
    -sEXPORTED_RUNTIME_METHODS=HEAPU8

# The tests compare this with sim/strike.goose, so a stale wasm fails CI.
sha256sum "$root/sim/strike.goose" | cut -d' ' -f1 > "$root/web/strike.goose.sha256"

echo "built web/strike.mjs and web/strike.wasm"
