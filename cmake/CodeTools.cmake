find_program(
  POE2_CLANG_FORMAT
  NAMES clang-format
  HINTS /opt/homebrew/opt/llvm/bin
)

set(POE2_FORMAT_FILES)

foreach(POE2_FORMAT_DIR game engines tools tests)
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
