cmake_minimum_required(VERSION 3.25)

if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" AND NOT CMAKE_OSX_SYSROOT)
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

if(POE2_ENABLE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT POE2_IPO_SUPPORTED OUTPUT POE2_IPO_OUTPUT)
  if(POE2_IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  else()
    message(WARNING "IPO/LTO requested but not supported: ${POE2_IPO_OUTPUT}")
  endif()
endif()

if(POE2_ENABLE_NATIVE_ARCH)
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

enable_testing()

add_subdirectory(game)
add_subdirectory(engines)
add_subdirectory(runner)
add_subdirectory(tests)

include(cmake/CodeTools.cmake)
