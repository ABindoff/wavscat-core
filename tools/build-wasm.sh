#!/usr/bin/env sh
# Build the wasm module and its JavaScript glue for Node and for browsers.
#
# Requires the wasm32-unknown-unknown target and wasm-bindgen-cli at the same
# version as the wasm-bindgen crate:
#
#     rustup target add wasm32-unknown-unknown
#     cargo install wasm-bindgen-cli --version 0.2.129 --locked
#
# WebAssembly SIMD is on by default. It makes the dense products about a
# quarter to a half faster and changes no output bit (CI checks both builds
# against the golden records). Every current browser supports it; for older
# ones (Safari before 16.4, 2023), build with WAVSCAT_WASM_SIMD=0.
set -eu
cd "$(dirname "$0")/.."
if [ "${WAVSCAT_WASM_SIMD:-1}" = "0" ]; then
    flags=""
    dir=target
else
    flags="-C target-feature=+simd128"
    dir=target/simd
fi
RUSTFLAGS="$flags" cargo build --release --target wasm32-unknown-unknown -p wavscat-wasm --target-dir "$dir"
wasm="$dir/wasm32-unknown-unknown/release/wavscat_wasm.wasm"
wasm-bindgen --target nodejs --out-dir crates/wavscat-wasm/pkg-node "$wasm"
wasm-bindgen --target web --out-dir crates/wavscat-wasm/pkg-web "$wasm"
echo "Built crates/wavscat-wasm/pkg-node and pkg-web (SIMD: ${WAVSCAT_WASM_SIMD:-1})"
