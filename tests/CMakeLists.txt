include(FetchContent)

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
  minimax_options_test.cpp
  minimax_search_test.cpp
  wasm_adapter_test.cpp
)

target_link_libraries(poe2_minimax_tests
  PRIVATE
    poe2::wasm_adapter
    Catch2::Catch2WithMain
)

add_executable(poe2_match_runner_tests
  match_runner_test.cpp
)

target_link_libraries(poe2_match_runner_tests
  PRIVATE
    poe2::match_runner
    Catch2::Catch2WithMain
)

include(Catch)

catch_discover_tests(poe2_core_tests)
catch_discover_tests(poe2_engine_stdio_tests)
catch_discover_tests(poe2_minimax_tests)
catch_discover_tests(poe2_match_runner_tests)

add_custom_target(poe2_tests
  DEPENDS
    poe2_core_tests
    poe2_engine_stdio_tests
    poe2_minimax_tests
    poe2_match_runner_tests
)
