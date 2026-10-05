#!/usr/bin/env sh
# Build the wasm module and its JavaScript glue for Node and for browsers.
#
# Requires the wasm32-unknown-unknown target and wasm-bindgen-cli at the same
# version as the wasm-bindgen crate:
#
#     rustup target add wasm32-unknown-unknown
#     cargo install wasm-bindgen-cli --version 0.2.129 --locked
set -eu
cd "$(dirname "$0")/.."
cargo build --release --target wasm32-unknown-unknown -p wavscat-wasm
wasm=target/wasm32-unknown-unknown/release/wavscat_wasm.wasm
wasm-bindgen --target nodejs --out-dir crates/wavscat-wasm/pkg-node "$wasm"
wasm-bindgen --target web --out-dir crates/wavscat-wasm/pkg-web "$wasm"
echo "Built crates/wavscat-wasm/pkg-node and pkg-web"
