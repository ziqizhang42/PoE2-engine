# Releasing

## Prepare and verify locally

1. Set the repository version in `CMakeLists.txt` and update `CHANGELOG.md`. The training tools read their version from `CMakeLists.txt`.

   - The middle number is for new capabilities in the engine or tools. The last is for a release without them: a bug fix or a new built-in net.
   - `poe2-train handoff --apply` already raised the last number when it installed a net ([nets.md](nets.md#the-built-in-net)), so two engines with different built-in nets never share a version. Prefer installing a net on a tagged release, so that the next release changes only the net.
   - The release section's heading names the version, the date, and the built-in net, as in `## 0.3.1 - 2026-11-17 · net run1/g001`.
   - A section describes the difference from the previous release. Edit and merge its entries as work lands instead of appending dated ones, and drop an entry when what it describes is removed again. How things work belongs in `docs/`, results in `eval/results.csv` and `models/`, and experiment findings in [research-log.md](research-log.md).

2. Run the native Debug and Release checks and build the packaged WebAssembly artifact:
   ```bash
   make release-verify
   ```

   The native build uses the host toolchain, and the isolated training-tool tests use its frozen uv environment. The package build uses the Emscripten version pinned by `.emscripten-version` inside Docker, runs the WebAssembly smoke test, and exports:

   ```text
   build/dist/v<version>/poe2-engine-wasm-<version>.tgz
   build/dist/v<version>/SHA256SUMS
   ```

3. Inspect the staged changes, package contents, and checksum before committing:

   ```bash
   git diff --check
   tar --list --file build/dist/vx.y.z/poe2-engine-wasm-vx.y.z.tgz
   cd build/dist/vx.y.z && sha256sum --check SHA256SUMS
   ```

Docker is only the reproducible WebAssembly build environment. This process does not build or publish a runtime image. Native Release builds enable host CPU tuning and are therefore not distributed as portable binaries.

## Publish later

After the release-preparation commit is on `main` and CI passes, create an annotated `vx.y.z` tag at that commit. Push the tag, create a GitHub release from it using the matching `CHANGELOG.md` section, and attach the `.tgz` and `SHA256SUMS` files. Publish the package to npm only if npm distribution is intended.

Tag creation, pushing, GitHub release creation, and npm publication are deliberately separate from `make release-verify`.
