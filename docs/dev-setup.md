# Development Setup

Native C++ builds use CMake, Ninja, Apple Clang, and Catch2 tests.

LLVM provides clang-format and clang-tidy.

## Commands

```bash
make
make PRESET=release
make format
make tidy
make tidy-fix
make fix
make git-build PRESET=release
make git-test PRESET=release
make clean
```

`make` defaults to the debug preset and runs configure, build, tests, format check, and clang-tidy. Use `PRESET=release` for the release preset.
`make format`, `make tidy-fix`, and `make fix` edit source files.

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
