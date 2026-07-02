# Development Setup

Native C++ builds use CMake, Ninja, Apple Clang, and Catch2 tests.

Optionally LLVM.

## Commands

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug

cmake --build --preset debug --target format-check
cmake --build --preset debug --target format
```
