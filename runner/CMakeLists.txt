find_package(Threads REQUIRED)

add_library(poe2_match_runner STATIC
  src/engine_process.cpp
  src/match_runner.cpp
  src/match_output.cpp
  src/opening_book.cpp
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
    Threads::Threads
)

target_compile_features(poe2_match_runner PUBLIC cxx_std_23)

add_executable(poe2_runner
  src/eval_cli.cpp
  src/main.cpp
  src/openings_cli.cpp
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
          --sequential-stop
          --timeout-ms 1000
          --go-depth 1
          --go-movetime-ms 100
          --go-nodes 1000
)

add_test(
  NAME runner_parallel_series_preserves_cutoff
  COMMAND ${CMAKE_COMMAND}
          -DRUNNER=$<TARGET_FILE:poe2_runner>
          -DGREEDY_ENGINE=$<TARGET_FILE:poe2_greedy>
          -DFIRST_LEGAL_ENGINE=$<TARGET_FILE:poe2_first_legal>
          -DOPENING_BOOK=${PROJECT_SOURCE_DIR}/eval/openings/systematic-2ply-v1.txt
          -P ${PROJECT_SOURCE_DIR}/tests/parallel_series_test.cmake
)

add_test(
  NAME runner_eval_self_smoke
  COMMAND poe2_runner eval
          --new-build ${CMAKE_BINARY_DIR}
          --base ${CMAKE_BINARY_DIR}
          --new-engine poe2_greedy
          --base-engine poe2_greedy
          --games 2
          --timeout-ms 1000
          --go-depth 1
          --go-movetime-ms 100
          --opening-book ${PROJECT_SOURCE_DIR}/eval/openings/systematic-2ply-v1.txt
          --shuffle-openings
          --run-root ${CMAKE_BINARY_DIR}/eval-test/runs
          --kind smoke
          --no-ledger
)

add_test(
  NAME runner_eval_cross_engine_smoke
  COMMAND poe2_runner eval
          --new-build ${CMAKE_BINARY_DIR}
          --base ${CMAKE_BINARY_DIR}
          --new-engine poe2_greedy
          --base-engine poe2_random_legal
          --base-engine-args "--seed 1"
          --games 2
          --timeout-ms 1000
          --go-depth 1
          --go-movetime-ms 100
          --opening-book ${PROJECT_SOURCE_DIR}/eval/openings/systematic-2ply-v1.txt
          --shuffle-openings
          --run-root ${CMAKE_BINARY_DIR}/eval-test/runs
          --kind smoke
          --no-ledger
)

set(POE2_GENERATED_CORPUS_DIR "${CMAKE_BINARY_DIR}/opening-corpus-test")
add_test(
  NAME runner_opening_corpus_generation
  COMMAND poe2_runner openings generate-corpus
          --development-out ${POE2_GENERATED_CORPUS_DIR}/development.txt
          --holdout-out ${POE2_GENERATED_CORPUS_DIR}/holdout.txt
          --count 20000
          --plies 2,4,6,8,10,12,14
          --seed 20260816
          --max-score-gap 4
)
set_tests_properties(runner_opening_corpus_generation PROPERTIES
  FIXTURES_SETUP opening_corpus_generation
)

add_test(
  NAME runner_opening_corpus_development_golden
  COMMAND ${CMAKE_COMMAND} -E compare_files
          ${POE2_GENERATED_CORPUS_DIR}/development.txt
          ${PROJECT_SOURCE_DIR}/eval/openings/development.txt
)
set_tests_properties(runner_opening_corpus_development_golden PROPERTIES
  FIXTURES_REQUIRED opening_corpus_generation
)

add_test(
  NAME runner_opening_corpus_holdout_golden
  COMMAND ${CMAKE_COMMAND} -E compare_files
          ${POE2_GENERATED_CORPUS_DIR}/holdout.txt
          ${PROJECT_SOURCE_DIR}/eval/openings/holdout.txt
)
set_tests_properties(runner_opening_corpus_holdout_golden PROPERTIES
  FIXTURES_REQUIRED opening_corpus_generation
)

add_test(
  NAME runner_rejects_opening_reuse
  COMMAND poe2_runner series
          --engine-one $<TARGET_FILE:poe2_greedy>
          --engine-two $<TARGET_FILE:poe2_greedy>
          --games 632
          --opening-book ${PROJECT_SOURCE_DIR}/eval/openings/systematic-2ply-v1.txt
)
set_tests_properties(runner_rejects_opening_reuse PROPERTIES WILL_FAIL TRUE)

add_test(
  NAME runner_eval_rejects_odd_game_budget
  COMMAND poe2_runner eval
          --new-build ${CMAKE_BINARY_DIR}
          --base ${CMAKE_BINARY_DIR}
          --new-engine poe2_greedy
          --base-engine poe2_greedy
          --games 3
          --opening-book ${PROJECT_SOURCE_DIR}/eval/openings/development.txt
          --shuffle-openings
          --no-ledger
)
set_tests_properties(runner_eval_rejects_odd_game_budget PROPERTIES WILL_FAIL TRUE)

add_test(
  NAME runner_eval_invalidates_abnormal_game
  COMMAND ${CMAKE_COMMAND}
          -DRUNNER=$<TARGET_FILE:poe2_runner>
          -DBUILD_DIR=${CMAKE_BINARY_DIR}
          -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
          -DTEST_ROOT=${CMAKE_BINARY_DIR}/eval-invalid-test
          -P ${PROJECT_SOURCE_DIR}/tests/eval_invalid_test.cmake
)

add_test(
  NAME runner_eval_migrates_legacy_ledger
  COMMAND ${CMAKE_COMMAND}
          -DRUNNER=$<TARGET_FILE:poe2_runner>
          -DBUILD_DIR=${CMAKE_BINARY_DIR}
          -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
          -DTEST_ROOT=${CMAKE_BINARY_DIR}/eval-ledger-migration-test
          -P ${PROJECT_SOURCE_DIR}/tests/eval_ledger_migration_test.cmake
)

add_test(
  NAME runner_eval_sampling_is_reproducible
  COMMAND ${CMAKE_COMMAND}
          -DRUNNER=$<TARGET_FILE:poe2_runner>
          -DBUILD_DIR=${CMAKE_BINARY_DIR}
          -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
          -DTEST_ROOT=${CMAKE_BINARY_DIR}/eval-sampling-test
          -P ${PROJECT_SOURCE_DIR}/tests/eval_sampling_test.cmake
)
