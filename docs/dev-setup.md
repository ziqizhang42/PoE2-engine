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
make clean
```

`make` defaults to the debug preset and runs configure, build, tests, format check, and clang-tidy. Use `PRESET=release` for the release preset.
`make format`, `make tidy-fix`, and `make fix` edit source files.
