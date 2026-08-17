if(NOT DEFINED MINIMAX_DATA OR NOT DEFINED OUTPUT_ROOT OR NOT DEFINED PYTHON OR
   NOT DEFINED SOURCE_AUDITOR OR NOT DEFINED LABEL_AUDITOR)
  message(FATAL_ERROR "missing position-source integration test path")
endif()

set(first_source "${OUTPUT_ROOT}/first-source")
set(second_source "${OUTPUT_ROOT}/second-source")
set(corrupt_source "${OUTPUT_ROOT}/corrupt-source")
set(label_output "${OUTPUT_ROOT}/labels")
file(REMOVE_RECURSE "${OUTPUT_ROOT}")
file(MAKE_DIRECTORY "${OUTPUT_ROOT}")

function(generate_source output_directory output_name)
  execute_process(
    COMMAND "${MINIMAX_DATA}" source
            --output-dir "${output_directory}"
            --corpus-id "position-source-integration-test"
            --seed 20260817
            --trajectories 8
            --samples-per-trajectory 2
            --shards 2
            --workers 6
            --search-nodes 50
            --search-hash-mb 1
            --random-weight 1
            --greedy-weight 0
            --opponent-weight 0
            --search-weight 0
            --progress-every 2
    RESULT_VARIABLE result
    OUTPUT_VARIABLE command_output
    ERROR_VARIABLE command_error
  )
  if(NOT result EQUAL 0)
    message(FATAL_ERROR
      "${output_name} source run returned ${result}\n${command_output}\n${command_error}"
    )
  endif()
  execute_process(
    COMMAND "${PYTHON}" "${SOURCE_AUDITOR}" "${output_directory}"
    RESULT_VARIABLE audit_result
    OUTPUT_VARIABLE audit_output
    ERROR_VARIABLE audit_error
  )
  if(NOT audit_result EQUAL 0)
    message(FATAL_ERROR
      "${output_name} source audit returned ${audit_result}\n${audit_output}\n${audit_error}"
    )
  endif()
endfunction()

generate_source("${first_source}" "first")
generate_source("${second_source}" "second")

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E compare_files
          "${first_source}/manifest.json" "${second_source}/manifest.json"
  RESULT_VARIABLE manifest_comparison
)
if(NOT manifest_comparison EQUAL 0)
  message(FATAL_ERROR "identical source runs produced different manifests")
endif()

file(GLOB first_shards "${first_source}/shards/shard-*.jsonl")
file(GLOB second_shards "${second_source}/shards/shard-*.jsonl")
list(LENGTH first_shards first_shard_count)
list(LENGTH second_shards second_shard_count)
if(NOT first_shard_count EQUAL 2 OR NOT second_shard_count EQUAL 2)
  message(FATAL_ERROR "source generator did not create exactly two shards")
endif()
list(SORT first_shards)
list(SORT second_shards)
foreach(index RANGE 0 1)
  list(GET first_shards ${index} first_shard)
  list(GET second_shards ${index} second_shard)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files "${first_shard}" "${second_shard}"
    RESULT_VARIABLE shard_comparison
  )
  if(NOT shard_comparison EQUAL 0)
    message(FATAL_ERROR "identical source runs produced a different shard ${index}")
  endif()
endforeach()

execute_process(
  COMMAND "${MINIMAX_DATA}" source
          --output-dir "${first_source}"
          --corpus-id "position-source-integration-test"
          --seed 20260817
          --trajectories 8
          --samples-per-trajectory 2
          --shards 2
          --random-weight 1
          --greedy-weight 0
          --opponent-weight 0
          --search-weight 0
  RESULT_VARIABLE overwrite_result
  OUTPUT_QUIET
  ERROR_QUIET
)
if(overwrite_result EQUAL 0)
  message(FATAL_ERROR "source generation reused an existing output directory")
endif()

list(GET first_shards 0 first_shard)
execute_process(
  COMMAND "${MINIMAX_DATA}" labels
          --input "${first_source}"
          --source-shard 0
          --output-dir "${label_output}"
          --mode teacher
          --nodes 100
          --hash-mb 1
          --workers 6
          --progress-every 2
          --require-all
  RESULT_VARIABLE label_result
  OUTPUT_VARIABLE label_stdout
  ERROR_VARIABLE label_stderr
)
if(NOT label_result EQUAL 0)
  message(FATAL_ERROR
    "source-backed label run returned ${label_result}\n${label_stdout}\n${label_stderr}"
  )
endif()
execute_process(
  COMMAND "${PYTHON}" "${LABEL_AUDITOR}" "${label_output}" --source "${first_shard}"
  RESULT_VARIABLE label_audit_result
  OUTPUT_VARIABLE label_audit_output
  ERROR_VARIABLE label_audit_error
)
if(NOT label_audit_result EQUAL 0)
  message(FATAL_ERROR
    "source-backed label audit returned ${label_audit_result}\n"
    "${label_audit_output}\n${label_audit_error}"
  )
endif()

file(MAKE_DIRECTORY "${corrupt_source}")
file(COPY "${first_source}/" DESTINATION "${corrupt_source}")
file(GLOB corrupt_shards "${corrupt_source}/shards/shard-*.jsonl")
list(SORT corrupt_shards)
list(GET corrupt_shards 0 corrupt_shard)
file(APPEND "${corrupt_shard}" "corruption\n")
execute_process(
  COMMAND "${PYTHON}" "${SOURCE_AUDITOR}" "${corrupt_source}"
  RESULT_VARIABLE corrupt_audit_result
  OUTPUT_QUIET
  ERROR_QUIET
)
if(corrupt_audit_result EQUAL 0)
  message(FATAL_ERROR "position source auditor accepted a corrupted shard")
endif()
