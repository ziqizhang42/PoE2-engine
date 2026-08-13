cmake_minimum_required(VERSION 3.25)

if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" AND NOT CMAKE_OSX_SYSROOT AND
   NOT CMAKE_TOOLCHAIN_FILE MATCHES "Emscripten\\.cmake$")
  execute_process(
    COMMAND xcrun --show-sdk-path
    OUTPUT_VARIABLE POE2_MACOS_SDK
    RESULT_VARIABLE POE2_MACOS_SDK_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )

  if(POE2_MACOS_SDK_RESULT EQUAL 0 AND POE2_MACOS_SDK)
    set(CMAKE_OSX_SYSROOT "${POE2_MACOS_SDK}" CACHE PATH "macOS SDK path" FORCE)
  endif()
endif()

project(
  PoE2
  VERSION 0.1.0
  DESCRIPTION "Powers of Exponent 2"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(POE2_ENABLE_IPO "Enable IPO/LTO for release builds when supported" ON)
option(POE2_ENABLE_NATIVE_ARCH "Tune release builds for the current CPU" ON)
option(POE2_BUILD_NATIVE_APPS "Build native stdio engines and the match runner" ON)
option(POE2_BUILD_WASM "Build the browser WebAssembly package" OFF)

if(POE2_BUILD_WASM)
  if(NOT EMSCRIPTEN)
    message(FATAL_ERROR "POE2_BUILD_WASM requires the Emscripten toolchain")
  endif()
  if(POE2_BUILD_NATIVE_APPS)
    message(FATAL_ERROR "POE2_BUILD_WASM requires POE2_BUILD_NATIVE_APPS=OFF")
  endif()

  file(STRINGS "${CMAKE_SOURCE_DIR}/.emscripten-version" POE2_EMSCRIPTEN_VERSION LIMIT_COUNT 1)
  if(NOT EMSCRIPTEN_VERSION STREQUAL POE2_EMSCRIPTEN_VERSION)
    message(FATAL_ERROR
      "Emscripten ${POE2_EMSCRIPTEN_VERSION} is required; found ${EMSCRIPTEN_VERSION}"
    )
  endif()
endif()

if(POE2_ENABLE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT POE2_IPO_SUPPORTED OUTPUT POE2_IPO_OUTPUT)
  if(POE2_IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  else()
    message(WARNING "IPO/LTO requested but not supported: ${POE2_IPO_OUTPUT}")
  endif()
endif()

if(POE2_ENABLE_NATIVE_ARCH AND NOT EMSCRIPTEN)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mcpu=native" POE2_HAS_MCPU_NATIVE)
  if(POE2_HAS_MCPU_NATIVE)
    add_compile_options($<$<CONFIG:Release>:-mcpu=native>)
  else()
    check_cxx_compiler_flag("-march=native" POE2_HAS_MARCH_NATIVE)
    if(POE2_HAS_MARCH_NATIVE)
      add_compile_options($<$<CONFIG:Release>:-march=native>)
    else()
      message(WARNING "Native CPU tuning requested but neither -mcpu=native nor -march=native is supported")
    endif()
  endif()
endif()

include(CTest)

add_subdirectory(game)
add_subdirectory(engines)
add_subdirectory(wasm)
if(POE2_BUILD_NATIVE_APPS)
  add_subdirectory(runner)
endif()
if(BUILD_TESTING AND POE2_BUILD_NATIVE_APPS)
  add_subdirectory(tests)
endif()

include(cmake/CodeTools.cmake)
