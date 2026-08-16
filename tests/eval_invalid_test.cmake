if(NOT DEFINED RUNNER OR NOT DEFINED BUILD_DIR OR NOT DEFINED SOURCE_DIR OR
   NOT DEFINED TEST_ROOT)
  message(FATAL_ERROR "missing eval invalid-test path")
endif()

file(REMOVE_RECURSE "${TEST_ROOT}")
execute_process(
  COMMAND "${RUNNER}" eval
          --new-build "${BUILD_DIR}"
          --base "${BUILD_DIR}"
          --new-engine poe2_malformed_test_engine
          --base-engine poe2_greedy
          --games 2
          --timeout-ms 1000
          --go-depth 1
          --opening-book "${SOURCE_DIR}/eval/openings/development.txt"
          --shuffle-openings
          --opening-seed 1
          --run-root "${TEST_ROOT}/runs"
          --kind smoke
          --no-ledger
  RESULT_VARIABLE result
  OUTPUT_VARIABLE runner_output
  ERROR_VARIABLE runner_error
)
if(NOT result EQUAL 3)
  message(FATAL_ERROR
    "invalid eval returned ${result}, expected 3\n${runner_output}\n${runner_error}"
  )
endif()

file(GLOB summaries "${TEST_ROOT}/runs/*/summary.json")
list(LENGTH summaries summary_count)
if(NOT summary_count EQUAL 1)
  message(FATAL_ERROR "invalid eval wrote ${summary_count} summaries, expected 1")
endif()
list(GET summaries 0 summary)
file(READ "${summary}" summary_text)
set(expected_values
  [["valid": false]]
  [["malformed_move": 1]]
  [["statistical_samples": 0]]
  [["sequential_decision": "invalid"]]
)
foreach(expected IN LISTS expected_values)
  string(FIND "${summary_text}" "${expected}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "summary is missing ${expected}:\n${summary_text}")
  endif()
endforeach()
