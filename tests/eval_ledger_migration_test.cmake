if(NOT DEFINED RUNNER OR NOT DEFINED BUILD_DIR OR NOT DEFINED SOURCE_DIR OR
   NOT DEFINED TEST_ROOT)
  message(FATAL_ERROR "missing ledger migration-test path")
endif()

file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}")
configure_file(
  "${SOURCE_DIR}/tests/fixtures/eval-results-v1.csv"
  "${TEST_ROOT}/results.csv"
  COPYONLY
)
execute_process(
  COMMAND "${RUNNER}" eval
          --new-build "${BUILD_DIR}"
          --base "${BUILD_DIR}"
          --new-engine poe2_greedy
          --base-engine poe2_greedy
          --games 2
          --workers 2
          --timeout-ms 1000
          --go-depth 1
          --opening-book "${SOURCE_DIR}/eval/openings/development.txt"
          --shuffle-openings
          --opening-seed 1
          --run-root "${TEST_ROOT}/runs"
          --kind smoke
          --ledger "${TEST_ROOT}/results.csv"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE runner_output
  ERROR_VARIABLE runner_error
)
if(NOT result EQUAL 0)
  message(FATAL_ERROR
    "ledger eval returned ${result}, expected 0\n${runner_output}\n${runner_error}"
  )
endif()

file(STRINGS "${TEST_ROOT}/results.csv" rows)
list(LENGTH rows row_count)
if(NOT row_count EQUAL 3)
  message(FATAL_ERROR "migrated ledger has ${row_count} rows, expected 3")
endif()
list(GET rows 0 header)
list(GET rows 1 historical)
list(GET rows 2 current)
foreach(column valid opening_seed sequential_model sequential_bound_unit sequential_llr normalized_elo
               workers_requested workers_used games_discarded)
  string(FIND "${header}" "${column}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "migrated ledger header is missing ${column}")
  endif()
endforeach()
string(FIND "${historical}" "historical-run" historical_id)
string(FIND "${historical}" "paired_betting_eprocess,score_rate" historical_model)
string(FIND "${current}" "paired_normalized_elo_gsprt,normalized_elo" current_model)
if(historical_id EQUAL -1 OR historical_model EQUAL -1 OR current_model EQUAL -1)
  message(FATAL_ERROR "ledger migration did not preserve and classify its rows")
endif()
if(NOT historical MATCHES ",1,1,0$" OR NOT current MATCHES ",2,1,0$")
  message(FATAL_ERROR "ledger migration did not record worker metadata")
endif()

string(REGEX REPLACE ",workers_requested,workers_used,games_discarded$" ""
       previous_header "${header}")
string(REGEX REPLACE ",1,1,0$" "" previous_historical "${historical}")
set(previous_ledger "${TEST_ROOT}/previous-results.csv")
file(WRITE "${previous_ledger}" "${previous_header}\n${previous_historical}\n")
execute_process(
  COMMAND "${RUNNER}" eval
          --new-build "${BUILD_DIR}"
          --base "${BUILD_DIR}"
          --new-engine poe2_greedy
          --base-engine poe2_greedy
          --games 2
          --workers 2
          --timeout-ms 1000
          --go-depth 1
          --opening-book "${SOURCE_DIR}/eval/openings/development.txt"
          --shuffle-openings
          --opening-seed 1
          --run-root "${TEST_ROOT}/previous-runs"
          --kind smoke
          --ledger "${previous_ledger}"
  RESULT_VARIABLE previous_result
  OUTPUT_VARIABLE previous_output
  ERROR_VARIABLE previous_error
)
if(NOT previous_result EQUAL 0)
  message(FATAL_ERROR
    "previous ledger eval returned ${previous_result}\n${previous_output}\n${previous_error}"
  )
endif()
file(STRINGS "${previous_ledger}" previous_rows)
list(LENGTH previous_rows previous_row_count)
list(GET previous_rows 0 migrated_previous_header)
list(GET previous_rows 1 migrated_previous_historical)
list(GET previous_rows 2 previous_current)
if(NOT previous_row_count EQUAL 3 OR NOT migrated_previous_header STREQUAL header OR
   NOT migrated_previous_historical MATCHES ",1,1,0$" OR
   NOT previous_current MATCHES ",2,1,0$")
  message(FATAL_ERROR "previous ledger schema was not migrated with worker metadata")
endif()
