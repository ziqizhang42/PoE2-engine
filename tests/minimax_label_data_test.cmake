if(NOT DEFINED MINIMAX_DATA OR NOT DEFINED LABEL_INPUT OR NOT DEFINED OUTPUT_ROOT)
  message(FATAL_ERROR "missing minimax-label-data test path")
endif()

set(first_binary "${OUTPUT_ROOT}/first.bin")
set(first_manifest "${OUTPUT_ROOT}/first.json")
set(second_binary "${OUTPUT_ROOT}/second.bin")
set(second_manifest "${OUTPUT_ROOT}/second.json")
set(incomplete_binary "${OUTPUT_ROOT}/incomplete.bin")
set(incomplete_manifest "${OUTPUT_ROOT}/incomplete.json")
set(output_files
  "${first_binary}"
  "${first_manifest}"
  "${second_binary}"
  "${second_manifest}"
  "${incomplete_binary}"
  "${incomplete_manifest}"
)
foreach(output_file IN LISTS output_files)
  file(REMOVE "${output_file}" "${output_file}.tmp")
endforeach()
file(MAKE_DIRECTORY "${OUTPUT_ROOT}")

function(run_exact output_binary output_manifest output_name)
  execute_process(
    COMMAND "${MINIMAX_DATA}" labels
            --input "${LABEL_INPUT}"
            --output "${output_binary}"
            --manifest "${output_manifest}"
            --mode exact
            --nodes 100000
            --hash-mb 1
            --workers 6
            --require-all
    RESULT_VARIABLE result
    OUTPUT_VARIABLE command_output
    ERROR_VARIABLE command_error
  )
  if(NOT result EQUAL 0)
    message(FATAL_ERROR
      "${output_name} label run returned ${result}\n${command_output}\n${command_error}"
    )
  endif()
endfunction()

run_exact("${first_binary}" "${first_manifest}" "first")
run_exact("${second_binary}" "${second_manifest}" "second")

file(READ "${first_manifest}" manifest_text)
string(FIND "${manifest_text}" "\"workers_requested\": 6" requested_workers)
string(FIND "${manifest_text}" "\"workers_used\": 2" used_workers)
if(requested_workers EQUAL -1 OR used_workers EQUAL -1)
  message(FATAL_ERROR "label manifest did not record requested and effective workers")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E compare_files "${first_binary}" "${second_binary}"
  RESULT_VARIABLE binary_comparison
)
if(NOT binary_comparison EQUAL 0)
  message(FATAL_ERROR "identical label runs produced different binary datasets")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E compare_files "${first_manifest}" "${second_manifest}"
  RESULT_VARIABLE manifest_comparison
)
if(NOT manifest_comparison EQUAL 0)
  message(FATAL_ERROR "identical label runs produced different manifests")
endif()

execute_process(
  COMMAND "${MINIMAX_DATA}" labels
          --input "${LABEL_INPUT}"
          --output "${first_binary}"
          --manifest "${first_manifest}"
          --mode exact
          --nodes 100000
          --hash-mb 1
          --workers 6
          --require-all
  RESULT_VARIABLE overwrite_result
  OUTPUT_QUIET
  ERROR_QUIET
)
if(overwrite_result EQUAL 0)
  message(FATAL_ERROR "label generation overwrote an existing dataset")
endif()

execute_process(
  COMMAND "${MINIMAX_DATA}" labels
          --input "${LABEL_INPUT}"
          --output "${incomplete_binary}"
          --manifest "${incomplete_manifest}"
          --mode exact
          --nodes 1
          --hash-mb 1
          --workers 6
          --require-all
  RESULT_VARIABLE incomplete_result
  OUTPUT_QUIET
  ERROR_QUIET
)
if(incomplete_result EQUAL 0)
  message(FATAL_ERROR "exact label generation accepted an insufficient node budget")
endif()
if(EXISTS "${incomplete_binary}" OR EXISTS "${incomplete_manifest}" OR
   EXISTS "${incomplete_binary}.tmp" OR EXISTS "${incomplete_manifest}.tmp")
  message(FATAL_ERROR "failed exact label generation left a partial dataset")
endif()
