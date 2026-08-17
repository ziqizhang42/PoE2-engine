if(NOT DEFINED MINIMAX_DATA OR NOT DEFINED LABEL_INPUT OR NOT DEFINED OUTPUT_ROOT OR
   NOT DEFINED PYTHON OR NOT DEFINED LABEL_AUDITOR)
  message(FATAL_ERROR "missing minimax-label-data test path")
endif()

set(first_output "${OUTPUT_ROOT}/first")
set(second_output "${OUTPUT_ROOT}/second")
set(incomplete_output "${OUTPUT_ROOT}/incomplete")
file(REMOVE_RECURSE "${OUTPUT_ROOT}")
file(MAKE_DIRECTORY "${OUTPUT_ROOT}")

function(run_exact output_directory output_name)
  execute_process(
    COMMAND "${MINIMAX_DATA}" labels
            --input "${LABEL_INPUT}"
            --output-dir "${output_directory}"
            --corpus-id "minimax-label-integration-test"
            --mode exact
            --nodes 100000
            --hash-mb 1
            --workers 6
            --progress-every 1
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
  execute_process(
    COMMAND "${PYTHON}" "${LABEL_AUDITOR}" "${output_directory}" --source "${LABEL_INPUT}"
    RESULT_VARIABLE audit_result
    OUTPUT_VARIABLE audit_output
    ERROR_VARIABLE audit_error
  )
  if(NOT audit_result EQUAL 0)
    message(FATAL_ERROR
      "${output_name} audit returned ${audit_result}\n${audit_output}\n${audit_error}"
    )
  endif()
endfunction()

run_exact("${first_output}" "first")
run_exact("${second_output}" "second")

file(READ "${first_output}/manifest.json" manifest_text)
string(FIND "${manifest_text}" "\"workers_requested\": 6" requested_workers)
string(FIND "${manifest_text}" "\"workers_used\": 2" used_workers)
string(FIND "${manifest_text}" "\"hash_bytes_effective\"" effective_hash)
string(FIND "${manifest_text}" "\"git_commit\"" git_commit)
if(requested_workers EQUAL -1 OR used_workers EQUAL -1 OR effective_hash EQUAL -1 OR
   git_commit EQUAL -1)
  message(FATAL_ERROR "label manifest omitted required operational provenance")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E compare_files
          "${first_output}/labels.bin" "${second_output}/labels.bin"
  RESULT_VARIABLE binary_comparison
)
if(NOT binary_comparison EQUAL 0)
  message(FATAL_ERROR "identical label runs produced different binary datasets")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E compare_files
          "${first_output}/manifest.json" "${second_output}/manifest.json"
  RESULT_VARIABLE manifest_comparison
)
if(NOT manifest_comparison EQUAL 0)
  message(FATAL_ERROR "identical label runs produced different manifests")
endif()

execute_process(
  COMMAND "${MINIMAX_DATA}" labels
          --input "${LABEL_INPUT}"
          --output-dir "${first_output}"
          --corpus-id "minimax-label-integration-test"
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
  message(FATAL_ERROR "label generation reused an existing dataset directory")
endif()

execute_process(
  COMMAND "${MINIMAX_DATA}" labels
          --input "${LABEL_INPUT}"
          --output-dir "${incomplete_output}"
          --corpus-id "minimax-label-integration-test"
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
if(NOT EXISTS "${incomplete_output}/INCOMPLETE" OR
   EXISTS "${incomplete_output}/COMPLETE" OR
   EXISTS "${incomplete_output}/labels.bin" OR
   EXISTS "${incomplete_output}/manifest.json")
  message(FATAL_ERROR "failed label generation was not left in an uncommitted state")
endif()

execute_process(
  COMMAND "${PYTHON}" "${LABEL_AUDITOR}" "${incomplete_output}"
  RESULT_VARIABLE incomplete_audit_result
  OUTPUT_QUIET
  ERROR_QUIET
)
if(incomplete_audit_result EQUAL 0)
  message(FATAL_ERROR "auditor accepted an incomplete dataset")
endif()
