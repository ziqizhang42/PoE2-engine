if(NOT DEFINED SOURCE_ROOT OR NOT DEFINED OUTPUT_ROOT OR NOT DEFINED GENERATOR OR
   NOT DEFINED CXX_COMPILER)
  message(FATAL_ERROR "external model-header test arguments are required")
endif()

file(REMOVE_RECURSE "${OUTPUT_ROOT}")
file(MAKE_DIRECTORY "${OUTPUT_ROOT}")
set(MODEL_HEADER "${OUTPUT_ROOT}/candidate_model.hpp")
configure_file(
  "${SOURCE_ROOT}/engines/minimax/src/frozen_pattern_gain_model.hpp"
  "${MODEL_HEADER}"
  COPYONLY
)
file(APPEND "${MODEL_HEADER}" "\n// External model-header override test.\n")

execute_process(
  COMMAND "${CMAKE_COMMAND}"
          -S "${SOURCE_ROOT}"
          -B "${OUTPUT_ROOT}/build"
          -G "${GENERATOR}"
          "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}"
          -DCMAKE_BUILD_TYPE=Debug
          -DBUILD_TESTING=OFF
          -DPOE2_ENABLE_IPO=OFF
          -DPOE2_ENABLE_NATIVE_ARCH=OFF
          "-DPOE2_MINIMAX_MODEL_HEADER=${MODEL_HEADER}"
  RESULT_VARIABLE CONFIGURE_RESULT
  OUTPUT_VARIABLE CONFIGURE_OUTPUT
  ERROR_VARIABLE CONFIGURE_ERROR
)
if(NOT CONFIGURE_RESULT EQUAL 0)
  message(FATAL_ERROR
    "external model-header configure failed:\n${CONFIGURE_OUTPUT}\n${CONFIGURE_ERROR}"
  )
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${OUTPUT_ROOT}/build" --target poe2_minimax_infer
  RESULT_VARIABLE BUILD_RESULT
  OUTPUT_VARIABLE BUILD_OUTPUT
  ERROR_VARIABLE BUILD_ERROR
)
if(NOT BUILD_RESULT EQUAL 0)
  message(FATAL_ERROR
    "external model-header build failed:\n${BUILD_OUTPUT}\n${BUILD_ERROR}"
  )
endif()

if(NOT EXISTS "${OUTPUT_ROOT}/build/engines/minimax/poe2_minimax_infer")
  message(FATAL_ERROR "external model-header build did not produce poe2_minimax_infer")
endif()
