if(NOT DEFINED RUNNER OR NOT DEFINED ENGINE OR NOT DEFINED SOURCE_DIR OR NOT DEFINED TEST_ROOT)
  message(FATAL_ERROR "missing balanced-openings test path")
endif()

file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}")

function(run_runner)
  execute_process(
    COMMAND "${RUNNER}" openings ${ARGN}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE runner_output
    ERROR_VARIABLE runner_error
  )
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "openings ${ARGV0} returned ${result}\n${runner_output}\n${runner_error}")
  endif()
endfunction()

foreach(workers 1 3)
  run_runner(fit-balance
    --run "${SOURCE_DIR}/tests/fixtures/opening-balance-run"
    --engine "${ENGINE}"
    --nodes 2000
    --hash-mb 1
    --min-openings 4
    --workers ${workers}
    --out "${TEST_ROOT}/correction-${workers}.txt"
  )
endforeach()
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E compare_files
          "${TEST_ROOT}/correction-1.txt" "${TEST_ROOT}/correction-3.txt"
  RESULT_VARIABLE correction_differs
)
if(NOT correction_differs EQUAL 0)
  message(FATAL_ERROR "the fitted correction depends on the number of workers")
endif()
file(READ "${TEST_ROOT}/correction-1.txt" correction_text)
foreach(expected "fitted on 48 games of 24 openings from 1 runs" "nodes=2000" "hash_mb=1"
                 "correction_2=" "correction_4=" "correction_6=")
  string(FIND "${correction_text}" "${expected}" found)
  if(found EQUAL -1)
    message(FATAL_ERROR "the correction does not record ${expected}:\n${correction_text}")
  endif()
endforeach()

foreach(workers 1 3)
  run_runner(generate-balanced
    --development-out "${TEST_ROOT}/books-${workers}/development.txt"
    --holdout-out "${TEST_ROOT}/books-${workers}/holdout.txt"
    --correction "${TEST_ROOT}/correction-1.txt"
    --engine "${ENGINE}"
    --plies 2,4,6
    --per-length 6
    --threshold 30
    --seed 7
    --previous-development "${SOURCE_DIR}/eval/openings/development.txt"
    --previous-holdout "${SOURCE_DIR}/eval/openings/holdout.txt"
    --workers ${workers}
  )
endforeach()
foreach(book development holdout)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files
            "${TEST_ROOT}/books-1/${book}.txt" "${TEST_ROOT}/books-3/${book}.txt"
    RESULT_VARIABLE book_differs
  )
  if(NOT book_differs EQUAL 0)
    message(FATAL_ERROR "the ${book} book depends on the number of workers")
  endif()
  file(STRINGS "${TEST_ROOT}/books-1/${book}.txt" header REGEX "^# depth_count_[46]=6 ")
  list(LENGTH header filled_lengths)
  if(NOT filled_lengths EQUAL 2)
    message(FATAL_ERROR "the ${book} book does not hold the quota at 4 and 6 plies")
  endif()
endforeach()

execute_process(
  COMMAND "${RUNNER}" openings generate-balanced
          --development-out "${TEST_ROOT}/wrong-net/development.txt"
          --holdout-out "${TEST_ROOT}/wrong-net/holdout.txt"
          --correction "${TEST_ROOT}/correction-1.txt"
          --engine "${ENGINE}"
          --net "${SOURCE_DIR}/models/candidate-sha256-8e39040c15375119cc063c256367167f77e2b150c4e976eb70e953aa29821d60.poe2net"
          --plies 2
          --per-length 1
          --seed 7
  RESULT_VARIABLE wrong_net_result
  OUTPUT_QUIET
  ERROR_VARIABLE wrong_net_error
)
string(FIND "${wrong_net_error}" "but the correction was fitted for net" refused)
if(wrong_net_result EQUAL 0 OR refused EQUAL -1 OR EXISTS "${TEST_ROOT}/wrong-net")
  message(FATAL_ERROR "a correction was applied to another net's scores\n${wrong_net_error}")
endif()
