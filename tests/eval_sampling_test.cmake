if(NOT DEFINED RUNNER OR NOT DEFINED BUILD_DIR OR NOT DEFINED SOURCE_DIR OR
   NOT DEFINED TEST_ROOT)
  message(FATAL_ERROR "missing eval sampling-test path")
endif()

file(REMOVE_RECURSE "${TEST_ROOT}")
foreach(candidate candidate-a candidate-b)
  file(MAKE_DIRECTORY "${TEST_ROOT}/${candidate}/engines")
  file(COPY "${BUILD_DIR}/engines/poe2_greedy"
       DESTINATION "${TEST_ROOT}/${candidate}/engines")
endforeach()

function(run_sampling_eval label candidate kind book)
  set(run_root "${TEST_ROOT}/${label}")
  execute_process(
    COMMAND "${RUNNER}" eval
            --new-build "${candidate}"
            --base "${BUILD_DIR}"
            --new-engine poe2_greedy
            --base-engine poe2_greedy
            --games 2
            --timeout-ms 1000
            --go-depth 1
            --opening-book "${SOURCE_DIR}/eval/openings/${book}.txt"
            --shuffle-openings
            --run-root "${run_root}"
            --kind "${kind}"
            --no-ledger
    RESULT_VARIABLE result
    OUTPUT_VARIABLE runner_output
    ERROR_VARIABLE runner_error
  )
  if(NOT result EQUAL 0)
    message(FATAL_ERROR
      "sampling eval returned ${result}\n${runner_output}\n${runner_error}"
    )
  endif()
  file(GLOB manifests "${run_root}/*/manifest.json")
  file(GLOB games_files "${run_root}/*/games.csv")
  list(LENGTH manifests manifest_count)
  list(LENGTH games_files games_count)
  if(NOT manifest_count EQUAL 1 OR NOT games_count EQUAL 1)
    message(FATAL_ERROR "sampling eval did not write exactly one run")
  endif()
  list(GET manifests 0 manifest)
  list(GET games_files 0 games_file)
  file(READ "${manifest}" manifest_text)
  file(READ "${games_file}" games_text)
  string(REGEX MATCH [["opening_seed": ([0-9]+)]] seed_match "${manifest_text}")
  if(NOT seed_match)
    message(FATAL_ERROR "manifest has no resolved opening seed")
  endif()
  set(${label}_seed "${CMAKE_MATCH_1}" PARENT_SCOPE)
  set(${label}_games "${games_text}" PARENT_SCOPE)
endfunction()

run_sampling_eval(first "${TEST_ROOT}/candidate-a" smoke development)
run_sampling_eval(replay "${TEST_ROOT}/candidate-a" smoke development)
run_sampling_eval(candidate "${TEST_ROOT}/candidate-b" smoke development)
run_sampling_eval(kind "${TEST_ROOT}/candidate-a" gate development)
run_sampling_eval(book "${TEST_ROOT}/candidate-a" smoke holdout)

set(subpath_run_root "${TEST_ROOT}/engine-subpath")
execute_process(
  COMMAND "${RUNNER}" eval
          --new-build "${BUILD_DIR}"
          --base "${BUILD_DIR}"
          --new-engine minimax/poe2_minimax
          --base-engine minimax/poe2_minimax
          --games 2
          --timeout-ms 1000
          --go-depth 1
          --opening-book "${SOURCE_DIR}/eval/openings/development.txt"
          --shuffle-openings
          --run-root "${subpath_run_root}"
          --kind smoke
          --no-ledger
  RESULT_VARIABLE subpath_result
  OUTPUT_VARIABLE subpath_output
  ERROR_VARIABLE subpath_error
)
if(NOT subpath_result EQUAL 0)
  message(FATAL_ERROR
    "engine-subpath eval returned ${subpath_result}\n${subpath_output}\n${subpath_error}"
  )
endif()
file(GLOB subpath_manifests "${subpath_run_root}/*/manifest.json")
list(LENGTH subpath_manifests subpath_manifest_count)
if(NOT subpath_manifest_count EQUAL 1)
  message(FATAL_ERROR "an engine subpath did not produce one direct run directory")
endif()

if(NOT first_seed STREQUAL replay_seed OR NOT first_games STREQUAL replay_games)
  message(FATAL_ERROR "an identical matchup did not reproduce its opening selection")
endif()
if(first_seed STREQUAL candidate_seed)
  message(FATAL_ERROR "candidate build ID did not affect the derived opening seed")
endif()
if(first_seed STREQUAL kind_seed)
  message(FATAL_ERROR "eval kind did not affect the derived opening seed")
endif()
if(first_seed STREQUAL book_seed)
  message(FATAL_ERROR "opening-book digest did not affect the derived opening seed")
endif()
