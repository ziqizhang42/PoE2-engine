# Development Setup

Native C++ builds use CMake, Ninja, Clang, and Catch2 tests. The native presets use Apple Clang on macOS and prefer upstream Clang 22 on Linux.

LLVM 22 provides clang-format and clang-tidy.

## Linux

On Ubuntu 26.04, install the reusable host toolchain with:

```bash
sudo apt-get install --no-install-recommends \
  git build-essential cmake ninja-build \
  clang-22 llvm-22 clang-format-22 clang-tidy-22 clangd-22
```

No container is required. CMake fetches the pinned Catch2 source during the first native configure.

## Commands

```bash
make ready
make
make PRESET=release
make format
make tidy
make tidy-fix
make fix
make git-build PRESET=release
make git-test PRESET=release
make training-test
make wasm-test
make release-verify
make eval-smoke BASE=<baseline-build-id> NEW_ENGINE=<name> BASE_ENGINE=<name> PRESET=release
make eval-gate BASE=<baseline-build-id> NEW_ENGINE=<name> BASE_ENGINE=<name> PRESET=release
make clean
```

The WebAssembly build additionally requires the pinned Emscripten SDK described in [`wasm.md`](wasm.md). Native commands do not require or use Emscripten.

`make ready` is the standard post-edit command: it fixes formatting/tidy issues, then builds, tests, and checks Debug and Release.

## Release Optimization

Release builds use CMake's normal release flags. The project also enables IPO/LTO and native CPU tuning for release builds when the compiler supports them. Native CPU tuning builds binaries for the current machine's CPU.

## Commit-Specific Builds

Use `make git-build PRESET=release` to build from a clean checkout into a commit-specific directory:

```text
build/by-commit/<zero-padded-commit-count>-<short-git-sha>/release/
```

The commit count comes from `git rev-list --count HEAD`, so these directories sort in commit order as long as project history is not rewritten. `git-build` and `git-test` use the same CMake preset settings as normal builds, but override the build directory. They fail if the working tree is dirty.

## Building an Older Commit

From a clean working tree:

```bash
git log --oneline --decorate
git switch --detach <commit-sha>
make git-test PRESET=release
git switch main
```

The build output stays under `build/by-commit/<zero-padded-commit-count>-<short-git-sha>/release/`, so each checked commit keeps its own release build directory.

## Engine Evaluation

Use the standard recipe in `docs/evaluation.md` for a pair-aware sequential strength gate.
Evaluation runs are saved under `build/eval/runs/`, and summary rows are appended to `eval/results.csv`.
