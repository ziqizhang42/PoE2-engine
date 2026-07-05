set(POE2_LLVM_TOOL_VERSION "22" CACHE STRING "Preferred LLVM tool major version")
set(POE2_LLVM_TOOL_HINTS
  "/opt/homebrew/opt/llvm/bin"
  "/opt/homebrew/opt/llvm@${POE2_LLVM_TOOL_VERSION}/bin"
  "/usr/lib/llvm-${POE2_LLVM_TOOL_VERSION}/bin"
)

find_program(
  POE2_CLANG_FORMAT
  NAMES clang-format-${POE2_LLVM_TOOL_VERSION} clang-format
  HINTS ${POE2_LLVM_TOOL_HINTS}
)

find_program(
  POE2_CLANG_TIDY
  NAMES clang-tidy-${POE2_LLVM_TOOL_VERSION} clang-tidy
  HINTS ${POE2_LLVM_TOOL_HINTS}
)

set(POE2_FORMAT_FILES)
set(POE2_TIDY_FILES)

foreach(POE2_FORMAT_DIR game engines runner tools tests)
  if(EXISTS "${PROJECT_SOURCE_DIR}/${POE2_FORMAT_DIR}")
    file(GLOB_RECURSE POE2_FORMAT_DIR_FILES CONFIGURE_DEPENDS
      "${PROJECT_SOURCE_DIR}/${POE2_FORMAT_DIR}/*.c"
      "${PROJECT_SOURCE_DIR}/${POE2_FORMAT_DIR}/*.cc"
      "${PROJECT_SOURCE_DIR}/${POE2_FORMAT_DIR}/*.cpp"
      "${PROJECT_SOURCE_DIR}/${POE2_FORMAT_DIR}/*.cxx"
      "${PROJECT_SOURCE_DIR}/${POE2_FORMAT_DIR}/*.h"
      "${PROJECT_SOURCE_DIR}/${POE2_FORMAT_DIR}/*.hh"
      "${PROJECT_SOURCE_DIR}/${POE2_FORMAT_DIR}/*.hpp"
      "${PROJECT_SOURCE_DIR}/${POE2_FORMAT_DIR}/*.hxx"
    )
    list(APPEND POE2_FORMAT_FILES ${POE2_FORMAT_DIR_FILES})

    file(GLOB_RECURSE POE2_TIDY_DIR_FILES CONFIGURE_DEPENDS
      "${PROJECT_SOURCE_DIR}/${POE2_FORMAT_DIR}/*.c"
      "${PROJECT_SOURCE_DIR}/${POE2_FORMAT_DIR}/*.cc"
      "${PROJECT_SOURCE_DIR}/${POE2_FORMAT_DIR}/*.cpp"
      "${PROJECT_SOURCE_DIR}/${POE2_FORMAT_DIR}/*.cxx"
    )
    list(APPEND POE2_TIDY_FILES ${POE2_TIDY_DIR_FILES})
  endif()
endforeach()

if(POE2_CLANG_FORMAT AND POE2_FORMAT_FILES)
  add_custom_target(format
    COMMAND ${POE2_CLANG_FORMAT} -i ${POE2_FORMAT_FILES}
    COMMENT "Formatting C++ sources"
    VERBATIM
  )

  add_custom_target(format-check
    COMMAND ${POE2_CLANG_FORMAT} --dry-run --Werror ${POE2_FORMAT_FILES}
    COMMENT "Checking C++ source formatting"
    VERBATIM
  )
endif()

if(POE2_CLANG_TIDY AND POE2_TIDY_FILES)
  add_custom_target(tidy
    COMMAND ${POE2_CLANG_TIDY} -p ${CMAKE_BINARY_DIR} ${POE2_TIDY_FILES}
    COMMENT "Running clang-tidy"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    VERBATIM
  )

  add_custom_target(tidy-fix
    COMMAND ${POE2_CLANG_TIDY} --fix -p ${CMAKE_BINARY_DIR} ${POE2_TIDY_FILES}
    COMMENT "Applying clang-tidy fixes"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    VERBATIM
  )
endif()
