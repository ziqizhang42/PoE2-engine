if(NOT DEFINED RUNNER OR NOT DEFINED GREEDY_ENGINE OR NOT DEFINED FIRST_LEGAL_ENGINE OR
   NOT DEFINED OPENING_BOOK)
  message(FATAL_ERROR "missing parallel-series test path")
endif()

function(run_series workers output_name)
  execute_process(
    COMMAND "${RUNNER}" series
            --engine-one "${GREEDY_ENGINE}"
            --engine-two "${FIRST_LEGAL_ENGINE}"
            --games 120
            --workers "${workers}"
            --timeout-ms 1000
            --go-depth 1
            --opening-book "${OPENING_BOOK}"
            --shuffle-openings
            --opening-seed 7
            --sequential-stop
    RESULT_VARIABLE result
    OUTPUT_VARIABLE runner_output
    ERROR_VARIABLE runner_error
  )
  if(NOT result EQUAL 0)
    message(FATAL_ERROR
      "${workers}-worker series returned ${result}\n${runner_output}\n${runner_error}"
    )
  endif()
  set(${output_name} "${runner_output}" PARENT_SCOPE)
endfunction()

run_series(1 sequential_output)
run_series(3 parallel_output)

string(REGEX MATCHALL "game index=[^\n]*" sequential_games "${sequential_output}")
string(REGEX MATCHALL "game index=[^\n]*" parallel_games "${parallel_output}")
if(NOT sequential_games STREQUAL parallel_games)
  message(FATAL_ERROR "parallel series did not commit games in sequential order")
endif()

string(REGEX MATCH "series_pairs[^\n]*" sequential_pairs "${sequential_output}")
string(REGEX MATCH "series_pairs[^\n]*" parallel_pairs "${parallel_output}")
if(NOT sequential_pairs STREQUAL parallel_pairs)
  message(FATAL_ERROR "parallel series changed paired statistics")
endif()

string(REGEX MATCH "series_sequential[^\n]*" sequential_cutoff "${sequential_output}")
string(REGEX MATCH "series_sequential[^\n]*" parallel_cutoff "${parallel_output}")
if(NOT sequential_cutoff STREQUAL parallel_cutoff)
  message(FATAL_ERROR "parallel series changed the sequential cutoff")
endif()

string(FIND "${parallel_output}"
  "workers_requested=3 workers_used=3 games_discarded=4" worker_summary)
if(worker_summary EQUAL -1)
  message(FATAL_ERROR "parallel series did not report its workers and speculative games")
endif()
