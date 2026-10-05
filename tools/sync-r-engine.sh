#!/usr/bin/env sh
# Copy the crates the R package wavscatengine needs into it, so that the
# package is self-contained: remotes::install_github() and R CMD build both
# see only the package directory. The copies are committed; CI reruns this
# and fails if they differ, so they never drift from crates/.
set -eu
cd "$(dirname "$0")/.."
dest=r/wavscatengine/src/rust/crates
version=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
rm -rf "$dest"
for c in wavscat-core tapvid; do
    mkdir -p "$dest/$c"
    cp -r "crates/$c/src" "$dest/$c/"
    if [ -f "crates/$c/golden.tsv" ]; then cp "crates/$c/golden.tsv" "$dest/$c/"; fi
    # Resolve the fields these crates inherit from the workspace.
    sed -e "s/^version.workspace = true/version = \"$version\"/" \
        -e 's/^edition.workspace = true/edition = "2021"/' \
        -e 's/^license.workspace = true/license = "BSD-3-Clause"/' \
        -e '/^authors.workspace = true/d' \
        -e '/^repository.workspace = true/d' \
        -e '/^rust-version.workspace = true/d' \
        "crates/$c/Cargo.toml" > "$dest/$c/Cargo.toml"
done
printf 'Copied from crates/ by tools/sync-r-engine.sh. Do not edit here.\n' > "$dest/README"
echo "Synced $dest"
