if(NOT DEFINED MINIMAX_DATA OR NOT DEFINED OUTPUT_ROOT OR NOT DEFINED PYTHON OR
   NOT DEFINED SOURCE_AUDITOR OR NOT DEFINED LABEL_AUDITOR OR
   NOT DEFINED CORPUS_PREFLIGHT OR NOT DEFINED FEATURE_AUDITOR)
  message(FATAL_ERROR "missing minimax feature integration test path")
endif()

set(source "${OUTPUT_ROOT}/source")
set(labels "${OUTPUT_ROOT}/labels")
set(first_features "${OUTPUT_ROOT}/first-features")
set(second_features "${OUTPUT_ROOT}/second-features")
set(corrupt_features "${OUTPUT_ROOT}/corrupt-features")
file(REMOVE_RECURSE "${OUTPUT_ROOT}")
file(MAKE_DIRECTORY "${OUTPUT_ROOT}" "${labels}/shards")

execute_process(
  COMMAND "${MINIMAX_DATA}" source
          --output-dir "${source}"
          --corpus-id "feature-integration-test"
          --seed 20260818
          --trajectories 8
          --samples-per-trajectory 2
          --shards 2
          --workers 2
          --search-nodes 50
          --search-hash-mb 1
          --random-weight 1
          --greedy-weight 0
          --opponent-weight 0
          --search-weight 0
          --progress-every 4
  RESULT_VARIABLE source_result
  OUTPUT_VARIABLE source_output
  ERROR_VARIABLE source_error
)
if(NOT source_result EQUAL 0)
  message(FATAL_ERROR "feature source failed\n${source_output}\n${source_error}")
endif()
execute_process(
  COMMAND "${PYTHON}" -B "${SOURCE_AUDITOR}" "${source}"
  RESULT_VARIABLE source_audit_result
  OUTPUT_VARIABLE source_audit_output
  ERROR_VARIABLE source_audit_error
)
if(NOT source_audit_result EQUAL 0)
  message(FATAL_ERROR "feature source audit failed\n${source_audit_output}\n${source_audit_error}")
endif()

foreach(shard RANGE 0 1)
  if(shard EQUAL 0)
    set(shard_name "000")
  else()
    set(shard_name "001")
  endif()
  execute_process(
    COMMAND "${MINIMAX_DATA}" labels
            --input "${source}"
            --source-shard ${shard}
            --output-dir "${labels}/shards/shard-${shard_name}"
            --mode teacher
            --nodes 5000
            --hash-mb 1
            --workers 2
            --progress-every 4
            --require-all
    RESULT_VARIABLE label_result
    OUTPUT_VARIABLE label_output
    ERROR_VARIABLE label_error
  )
  if(NOT label_result EQUAL 0)
    message(FATAL_ERROR "feature label shard ${shard} failed\n${label_output}\n${label_error}")
  endif()
endforeach()

execute_process(
  COMMAND "${PYTHON}" -B "${CORPUS_PREFLIGHT}" "${labels}" --source "${source}" --json
  RESULT_VARIABLE preflight_result
  OUTPUT_VARIABLE preflight_output
  ERROR_VARIABLE preflight_error
)
if(NOT preflight_result EQUAL 0)
  message(FATAL_ERROR "feature corpus preflight failed\n${preflight_output}\n${preflight_error}")
endif()

foreach(output IN ITEMS "${first_features}" "${second_features}")
  execute_process(
    COMMAND "${MINIMAX_DATA}" features
            --source "${source}"
            --labels "${labels}"
            --output-dir "${output}"
    RESULT_VARIABLE feature_result
    OUTPUT_VARIABLE feature_output
    ERROR_VARIABLE feature_error
  )
  if(NOT feature_result EQUAL 0)
    message(FATAL_ERROR "feature export failed\n${feature_output}\n${feature_error}")
  endif()
  execute_process(
    COMMAND "${PYTHON}" -B "${FEATURE_AUDITOR}" "${output}"
            --labels "${labels}" --gain-samples 16
    RESULT_VARIABLE feature_audit_result
    OUTPUT_VARIABLE feature_audit_output
    ERROR_VARIABLE feature_audit_error
  )
  if(NOT feature_audit_result EQUAL 0)
    message(FATAL_ERROR "feature audit failed\n${feature_audit_output}\n${feature_audit_error}")
  endif()
endforeach()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E compare_files
          "${first_features}/features.bin" "${second_features}/features.bin"
  RESULT_VARIABLE binary_comparison
)
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E compare_files
          "${first_features}/manifest.json" "${second_features}/manifest.json"
  RESULT_VARIABLE manifest_comparison
)
if(NOT binary_comparison EQUAL 0 OR NOT manifest_comparison EQUAL 0)
  message(FATAL_ERROR "identical feature exports differ")
endif()

execute_process(
  COMMAND "${MINIMAX_DATA}" features
          --source "${source}" --labels "${labels}" --output-dir "${first_features}"
  RESULT_VARIABLE overwrite_result
  OUTPUT_QUIET
  ERROR_QUIET
)
if(overwrite_result EQUAL 0)
  message(FATAL_ERROR "feature exporter overwrote an existing artifact")
endif()

file(MAKE_DIRECTORY "${corrupt_features}")
file(COPY "${first_features}/" DESTINATION "${corrupt_features}")
file(APPEND "${corrupt_features}/features.bin" "corruption")
execute_process(
  COMMAND "${PYTHON}" -B "${FEATURE_AUDITOR}" "${corrupt_features}"
  RESULT_VARIABLE corrupt_result
  OUTPUT_QUIET
  ERROR_QUIET
)
if(corrupt_result EQUAL 0)
  message(FATAL_ERROR "feature auditor accepted a corrupted binary")
endif()
