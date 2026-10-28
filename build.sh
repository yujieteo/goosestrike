#!/usr/bin/env bash
# Builds gen/strike.c and gen/strike.h (C that Goose generates from
# sim/strike.goose), web/strike.mjs and web/strike.wasm from that C, and
# copies the source to web/strike.goose for the code view of the page.
# Then the same for the GPU solver: gen/gpu.c and gen/gpu.h from
# sim/gpu.goose (which imports strike.goose), and web/strike-gpu.mjs and
# web/strike-gpu.wasm from that C and sim/gpu.c, with Emscripten's WebGPU
# port (emdawnwebgpu, which Emscripten downloads from the Dawn releases).
#
# Needs the Goose compiler (GOOSE, default: goose on PATH) and Emscripten's
# emcc on PATH. On NixOS: nix shell nixpkgs#emscripten --command ./build.sh
set -euo pipefail

cd "$(dirname "$0")"
goose="${GOOSE:-goose}"
out=gen
mkdir -p "$out"

# Every exported function of a header, for Emscripten.
exports_of() { sed -n 's/^[a-z0-9_ *]*[ *]\([a-z_0-9]*\)(.*);$/_\1/p' "$1" | paste -sd, -; }

# WebAssembly has no address-space reservation: the one Goose data stack
# gets a fixed region from Emscripten's mmap (which allocates it), and no
# guard gap traps an overrun. The solver keeps its state in static data.
# Only goose_init pushes to the data stack: one zeroed temporary per global
# array, popped at once. The largest is esig, f64[MAXE * 15] = 2.88 MB, so
# the region is 4 MiB. The time loop never touches the data stack (README).
flags=(-O3 -DGS_STACK_RESERVE='(4ull<<20)' -DGS_STACK_GAP=65536
    -sMODULARIZE -sEXPORT_ES6 -sENVIRONMENT=web,node
    -sINITIAL_MEMORY=64MB -sALLOW_MEMORY_GROWTH -sEXPORTED_RUNTIME_METHODS=HEAPU8)

# Goose -> one C file with the whole runtime and no main(); goose_init()
# and the export fn wrappers are declared in the header. The relative source
# path keeps the path of this checkout out of the C.
"$goose" --standalone --header "$out/strike.h" -o "$out/strike.c" sim/strike.goose
emcc "$out/strike.c" -o web/strike.mjs "${flags[@]}" \
    -sEXPORTED_FUNCTIONS="_malloc,_free,$(exports_of "$out/strike.h")"

# The GPU solver. emdawnwebgpu is C++, so the link is a C++ link.
"$goose" --standalone --header "$out/gpu.h" -o "$out/gpu.c" sim/gpu.goose
emcc "$out/gpu.c" sim/gpu.c -o web/strike-gpu.mjs "${flags[@]}" \
    --use-port=emdawnwebgpu -sDEFAULT_TO_CXX \
    -sEXPORTED_FUNCTIONS="_malloc,_free,$(exports_of "$out/gpu.h")"

cp sim/strike.goose web/strike.goose
# The tests check these sums, so a stale or hand-edited C or wasm fails CI.
sha256sum sim/strike.goose gen/strike.c gen/strike.h web/strike.wasm \
    sim/gpu.goose sim/gpu.c gen/gpu.c gen/gpu.h web/strike-gpu.wasm > gen/sha256sums

echo "built web/strike.wasm and web/strike-gpu.wasm"
