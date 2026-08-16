set(POE2_LLVM_TOOL_VERSION "22" CACHE STRING "Preferred LLVM tool major version")

# Respect an explicitly selected compiler. Otherwise, use Apple Clang on macOS
# and prefer the matching upstream LLVM release on other hosts.
if(NOT CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
  if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    set(POE2_CLANGXX_NAMES clang++)
  else()
    set(POE2_CLANGXX_NAMES clang++-${POE2_LLVM_TOOL_VERSION} clang++)
  endif()

  find_program(POE2_CLANGXX NAMES ${POE2_CLANGXX_NAMES})
  if(NOT POE2_CLANGXX)
    message(FATAL_ERROR
      "Clang C++ compiler not found (tried: ${POE2_CLANGXX_NAMES})"
    )
  endif()

  set(CMAKE_CXX_COMPILER "${POE2_CLANGXX}" CACHE FILEPATH "C++ compiler")
endif()
