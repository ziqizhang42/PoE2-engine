if(NOT DEFINED RUNNER OR NOT DEFINED BUILD_DIR OR NOT DEFINED SOURCE_DIR OR
   NOT DEFINED TEST_ROOT)
  message(FATAL_ERROR "missing ledger concurrency-test path")
endif()

file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}")
foreach(candidate candidate-a candidate-b)
  file(MAKE_DIRECTORY "${TEST_ROOT}/${candidate}/engines")
  file(COPY "${BUILD_DIR}/engines/poe2_greedy"
       DESTINATION "${TEST_ROOT}/${candidate}/engines")
endforeach()

file(STRINGS "${SOURCE_DIR}/eval/results.csv" ledger_header LIMIT_COUNT 1)
set(ledger "${TEST_ROOT}/results.csv")
file(WRITE "${ledger}" "${ledger_header}\n")
set(script "${TEST_ROOT}/run-concurrently.sh")
file(WRITE "${script}" "#!/bin/sh
set -eu
run_eval() {
  label=\"$1\"
  candidate=\"$2\"
  \"${RUNNER}\" eval \\
    --new-build \"$candidate\" \\
    --base \"${BUILD_DIR}\" \\
    --new-engine poe2_greedy \\
    --base-engine poe2_greedy \\
    --games 2 \\
    --timeout-ms 1000 \\
    --go-depth 1 \\
    --opening-book \"${SOURCE_DIR}/eval/openings/development.txt\" \\
    --shuffle-openings \\
    --run-root \"${TEST_ROOT}/$label\" \\
    --kind concurrency \\
    --ledger \"${ledger}\" \\
    >\"${TEST_ROOT}/$label.log\" 2>&1
}
run_eval one \"${TEST_ROOT}/candidate-a\" &
first_pid=$!
run_eval two \"${TEST_ROOT}/candidate-b\" &
second_pid=$!
wait \"$first_pid\"
wait \"$second_pid\"
")

execute_process(
  COMMAND /bin/sh "${script}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error
)
if(NOT result EQUAL 0)
  message(FATAL_ERROR
    "concurrent ledger evals returned ${result}\n${output}\n${error}"
  )
endif()

file(STRINGS "${ledger}" rows)
list(LENGTH rows row_count)
if(NOT row_count EQUAL 3)
  message(FATAL_ERROR "concurrent ledger has ${row_count} rows, expected 3")
endif()
string(REGEX REPLACE "[^,]" "" header_commas "${ledger_header}")
string(LENGTH "${header_commas}" header_comma_count)
foreach(index RANGE 1 2)
  list(GET rows ${index} row)
  string(REGEX REPLACE "[^,]" "" row_commas "${row}")
  string(LENGTH "${row_commas}" row_comma_count)
  if(NOT row_comma_count EQUAL header_comma_count)
    message(FATAL_ERROR "concurrent ledger row ${index} is malformed")
  endif()
endforeach()
if(NOT EXISTS "${ledger}.lock")
  message(FATAL_ERROR "concurrent ledger writers did not use the shared lock path")
endif()
