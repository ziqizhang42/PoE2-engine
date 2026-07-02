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

enable_testing()

add_subdirectory(game)
add_subdirectory(tests)

include(cmake/CodeTools.cmake)
