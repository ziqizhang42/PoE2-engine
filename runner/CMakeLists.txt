add_library(poe2_match_runner STATIC
  src/match_runner.cpp
)

add_library(poe2::match_runner ALIAS poe2_match_runner)

target_include_directories(poe2_match_runner
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(poe2_match_runner
  PUBLIC
    poe2::engine_common
    poe2::game
  PRIVATE
    poe2::engine_stdio
)

target_compile_features(poe2_match_runner PUBLIC cxx_std_23)

add_executable(poe2_runner
  src/main.cpp
)

target_link_libraries(poe2_runner
  PRIVATE
    poe2::match_runner
)

target_compile_features(poe2_runner PRIVATE cxx_std_23)

add_test(
  NAME runner_first_legal_match
  COMMAND poe2_runner match
          --p1 $<TARGET_FILE:poe2_first_legal>
          --p2 $<TARGET_FILE:poe2_first_legal>
          --timeout-ms 1000
          --go-depth 1
          --go-movetime-ms 100
          --go-nodes 1000
)

add_test(
  NAME runner_first_vs_random_series
  COMMAND poe2_runner series
          --engine-one $<TARGET_FILE:poe2_first_legal>
          --engine-two "$<TARGET_FILE:poe2_random_legal> --seed 1"
          --games 4
          --sprt-stop
          --timeout-ms 1000
          --go-depth 1
          --go-movetime-ms 100
          --go-nodes 1000
)

add_test(
  NAME runner_eval_self_smoke
  COMMAND poe2_runner eval
          --new-build ${CMAKE_BINARY_DIR}
          --base ${CMAKE_BINARY_DIR}
          --engine poe2_greedy
          --games 2
          --timeout-ms 1000
          --go-depth 1
          --go-movetime-ms 100
          --run-root ${CMAKE_BINARY_DIR}/eval-test/runs
          --kind smoke
          --no-ledger
)
