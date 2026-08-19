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
  file(GLOB ledger_rows "${run_root}/*/ledger-row.csv")
  list(LENGTH manifests manifest_count)
  list(LENGTH games_files games_count)
  list(LENGTH ledger_rows ledger_row_count)
  if(NOT manifest_count EQUAL 1 OR NOT games_count EQUAL 1 OR NOT ledger_row_count EQUAL 1)
    message(FATAL_ERROR "sampling eval did not write one self-contained run")
  endif()
  list(GET manifests 0 manifest)
  list(GET games_files 0 games_file)
  list(GET ledger_rows 0 ledger_row)
  file(READ "${manifest}" manifest_text)
  file(READ "${games_file}" games_text)
  file(STRINGS "${ledger_row}" ledger_lines)
  list(LENGTH ledger_lines ledger_line_count)
  if(NOT ledger_line_count EQUAL 2)
    message(FATAL_ERROR "saved ledger row does not contain one header and one result")
  endif()
  list(GET ledger_lines 0 ledger_header)
  string(FIND "${ledger_header}" "run_id" ledger_run_id)
  if(ledger_run_id EQUAL -1)
    message(FATAL_ERROR "saved ledger row has no current ledger header")
  endif()
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
          --sequential-alpha 0.05123456789
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
file(GLOB subpath_summaries "${subpath_run_root}/*/summary.json")
list(LENGTH subpath_manifests subpath_manifest_count)
list(LENGTH subpath_summaries subpath_summary_count)
if(NOT subpath_manifest_count EQUAL 1 OR NOT subpath_summary_count EQUAL 1)
  message(FATAL_ERROR "an engine subpath did not produce one manifest and summary")
endif()
list(GET subpath_manifests 0 subpath_manifest)
list(GET subpath_summaries 0 subpath_summary)
file(READ "${subpath_manifest}" subpath_manifest_text)
file(READ "${subpath_summary}" subpath_summary_text)
string(REGEX MATCH [["sequential_alpha": 0\.051234567[0-9]+]]
       precise_manifest_alpha "${subpath_manifest_text}")
string(REGEX MATCH [["sequential_alpha": 0\.051234567[0-9]+]]
       precise_summary_alpha "${subpath_summary_text}")
if(NOT precise_manifest_alpha OR NOT precise_summary_alpha)
  message(FATAL_ERROR "the eval JSON artifacts did not retain round-trip floating-point precision")
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
