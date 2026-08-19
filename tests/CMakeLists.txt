include(FetchContent)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(CATCH_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.15.1
)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

add_executable(poe2_core_tests
  board_test.cpp
  move_test.cpp
  symmetry_test.cpp
  transposition_table_test.cpp
)

target_link_libraries(poe2_core_tests
  PRIVATE
    poe2::game
    Catch2::Catch2WithMain
)

add_executable(poe2_engine_stdio_tests
  engine_stdio_test.cpp
)

target_link_libraries(poe2_engine_stdio_tests
  PRIVATE
    poe2::engine_stdio
    Catch2::Catch2WithMain
)

add_executable(poe2_minimax_tests
  minimax_evaluation_test.cpp
  minimax_feature_data_test.cpp
  minimax_labeling_test.cpp
  minimax_options_test.cpp
  minimax_position_source_test.cpp
  minimax_search_test.cpp
  wasm_adapter_test.cpp
)

target_link_libraries(poe2_minimax_tests
  PRIVATE
    poe2::minimax_labeling
    poe2::wasm_adapter
    Catch2::Catch2WithMain
)

target_compile_definitions(poe2_minimax_tests
  PRIVATE
    POE2_PROJECT_VERSION="${PROJECT_VERSION}"
)

add_executable(poe2_match_runner_tests
  match_runner_test.cpp
)

add_executable(poe2_malformed_test_engine
  malformed_engine.cpp
)

set_target_properties(poe2_malformed_test_engine PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/engines"
)

target_link_libraries(poe2_match_runner_tests
  PRIVATE
    poe2::match_runner
    Catch2::Catch2WithMain
)

target_compile_definitions(poe2_match_runner_tests
  PRIVATE
    POE2_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

include(Catch)

catch_discover_tests(poe2_core_tests)
catch_discover_tests(poe2_engine_stdio_tests)
catch_discover_tests(poe2_minimax_tests)
catch_discover_tests(poe2_match_runner_tests)

add_test(
  NAME minimax_label_data_reproducible
  COMMAND ${CMAKE_COMMAND}
          -DMINIMAX_DATA=$<TARGET_FILE:poe2_minimax_data>
          -DLABEL_INPUT=${PROJECT_SOURCE_DIR}/tests/fixtures/minimax-label-positions.txt
          -DOUTPUT_ROOT=${CMAKE_CURRENT_BINARY_DIR}/minimax-label-data-test
          -DPYTHON=${Python3_EXECUTABLE}
          -DLABEL_AUDITOR=${PROJECT_SOURCE_DIR}/tools/inspect_minimax_labels.py
          -P ${PROJECT_SOURCE_DIR}/tests/minimax_label_data_test.cmake
)
set_tests_properties(minimax_label_data_reproducible PROPERTIES TIMEOUT 30)

add_test(
  NAME minimax_position_source_reproducible
  COMMAND ${CMAKE_COMMAND}
          -DMINIMAX_DATA=$<TARGET_FILE:poe2_minimax_data>
          -DOUTPUT_ROOT=${CMAKE_CURRENT_BINARY_DIR}/minimax-position-source-test
          -DPYTHON=${Python3_EXECUTABLE}
          -DSOURCE_AUDITOR=${PROJECT_SOURCE_DIR}/tools/inspect_position_source.py
          -DLABEL_AUDITOR=${PROJECT_SOURCE_DIR}/tools/inspect_minimax_labels.py
          -P ${PROJECT_SOURCE_DIR}/tests/minimax_position_source_test.cmake
)
set_tests_properties(minimax_position_source_reproducible PROPERTIES TIMEOUT 30)

add_test(
  NAME minimax_feature_data_reproducible
  COMMAND ${CMAKE_COMMAND}
          -DMINIMAX_DATA=$<TARGET_FILE:poe2_minimax_data>
          -DOUTPUT_ROOT=${CMAKE_CURRENT_BINARY_DIR}/minimax-feature-data-test
          -DPYTHON=${Python3_EXECUTABLE}
          -DSOURCE_AUDITOR=${PROJECT_SOURCE_DIR}/tools/inspect_position_source.py
          -DLABEL_AUDITOR=${PROJECT_SOURCE_DIR}/tools/inspect_minimax_labels.py
          -DCORPUS_PREFLIGHT=${PROJECT_SOURCE_DIR}/tools/preflight_minimax_corpus.py
          -DFEATURE_AUDITOR=${PROJECT_SOURCE_DIR}/tools/inspect_minimax_features.py
          -P ${PROJECT_SOURCE_DIR}/tests/minimax_feature_data_test.cmake
)
set_tests_properties(minimax_feature_data_reproducible PROPERTIES TIMEOUT 60)

add_test(
  NAME minimax_label_auditor_golden
  COMMAND ${Python3_EXECUTABLE} -B
          ${PROJECT_SOURCE_DIR}/tests/minimax_label_auditor_test.py
          ${PROJECT_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}/minimax-label-auditor-test
)

add_test(
  NAME minimax_label_comparison_golden
  COMMAND ${Python3_EXECUTABLE} -B
          ${PROJECT_SOURCE_DIR}/tests/minimax_label_comparison_test.py
          ${PROJECT_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}/minimax-label-comparison-test
)

add_test(
  NAME minimax_external_model_header_build
  COMMAND ${CMAKE_COMMAND}
          -DSOURCE_ROOT=${PROJECT_SOURCE_DIR}
          -DOUTPUT_ROOT=${CMAKE_CURRENT_BINARY_DIR}/minimax-external-model-header-test
          -DGENERATOR=${CMAKE_GENERATOR}
          -DCXX_COMPILER=${CMAKE_CXX_COMPILER}
          -P ${PROJECT_SOURCE_DIR}/tests/minimax_external_model_header_test.cmake
)
set_tests_properties(minimax_external_model_header_build PROPERTIES TIMEOUT 120)

add_custom_target(poe2_tests
  DEPENDS
    poe2_core_tests
    poe2_engine_stdio_tests
    poe2_minimax_data
    poe2_minimax_tests
    poe2_match_runner_tests
    poe2_malformed_test_engine
)
