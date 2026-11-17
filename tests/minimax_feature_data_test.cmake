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

# A learned teacher drives both noisy-search sampling and labels, and its model digest must
# survive into the features. Shards labeled by different evaluators must not export together.
set(teacher_source "${OUTPUT_ROOT}/teacher-source")
set(teacher_labels "${OUTPUT_ROOT}/teacher-labels")
set(mixed_labels "${OUTPUT_ROOT}/mixed-labels")
set(teacher_features "${OUTPUT_ROOT}/teacher-features")
set(mixed_features "${OUTPUT_ROOT}/mixed-features")
file(MAKE_DIRECTORY "${teacher_labels}/shards" "${mixed_labels}/shards")

execute_process(
  COMMAND "${MINIMAX_DATA}" source
          --output-dir "${teacher_source}"
          --corpus-id "feature-teacher-test"
          --seed 20261114
          --trajectories 4
          --samples-per-trajectory 2
          --shards 2
          --workers 2
          --search-nodes 50
          --search-hash-mb 1
          --random-weight 0
          --greedy-weight 0
          --opponent-weight 0
          --search-weight 1
          --evaluator pattern-gain
  RESULT_VARIABLE teacher_source_result
  OUTPUT_VARIABLE teacher_source_output
  ERROR_VARIABLE teacher_source_error
)
if(NOT teacher_source_result EQUAL 0)
  message(FATAL_ERROR "teacher source failed\n${teacher_source_output}\n${teacher_source_error}")
endif()
file(READ "${teacher_source}/manifest.json" teacher_source_manifest)
if(NOT teacher_source_manifest MATCHES "\"evaluator\": \"pattern-gain\"" OR
   NOT teacher_source_manifest MATCHES "\"net_sha256\": \"[0-9a-f]+\"")
  message(FATAL_ERROR "teacher source manifest does not record its model")
endif()
execute_process(
  COMMAND "${PYTHON}" -B "${SOURCE_AUDITOR}" "${teacher_source}"
  RESULT_VARIABLE teacher_source_audit_result
  OUTPUT_VARIABLE teacher_source_audit_output
  ERROR_VARIABLE teacher_source_audit_error
)
if(NOT teacher_source_audit_result EQUAL 0)
  message(FATAL_ERROR
    "teacher source audit failed\n${teacher_source_audit_output}\n${teacher_source_audit_error}")
endif()

foreach(case IN ITEMS teacher-000 teacher-001 mixed-000 mixed-001)
  if(case MATCHES "000$")
    set(shard 0)
    set(shard_name "000")
  else()
    set(shard 1)
    set(shard_name "001")
  endif()
  if(case MATCHES "^teacher")
    set(output "${teacher_labels}/shards/shard-${shard_name}")
    set(evaluator "pattern-gain")
  else()
    set(output "${mixed_labels}/shards/shard-${shard_name}")
    if(shard EQUAL 0)
      set(evaluator "two-ply-closure")
    else()
      set(evaluator "pattern-gain")
    endif()
  endif()
  execute_process(
    COMMAND "${MINIMAX_DATA}" labels
            --input "${teacher_source}"
            --source-shard ${shard}
            --output-dir "${output}"
            --mode teacher
            --nodes 5000
            --hash-mb 1
            --workers 2
            --require-all
            --evaluator ${evaluator}
    RESULT_VARIABLE label_result
    OUTPUT_VARIABLE label_output
    ERROR_VARIABLE label_error
  )
  if(NOT label_result EQUAL 0)
    message(FATAL_ERROR "label case ${case} failed\n${label_output}\n${label_error}")
  endif()
  execute_process(
    COMMAND "${PYTHON}" -B "${LABEL_AUDITOR}" "${output}"
    RESULT_VARIABLE label_audit_result
    OUTPUT_VARIABLE label_audit_output
    ERROR_VARIABLE label_audit_error
  )
  if(NOT label_audit_result EQUAL 0)
    message(FATAL_ERROR
      "label case ${case} audit failed\n${label_audit_output}\n${label_audit_error}")
  endif()
endforeach()

execute_process(
  COMMAND "${MINIMAX_DATA}" features
          --source "${teacher_source}" --labels "${teacher_labels}"
          --output-dir "${teacher_features}"
  RESULT_VARIABLE teacher_feature_result
  OUTPUT_VARIABLE teacher_feature_output
  ERROR_VARIABLE teacher_feature_error
)
if(NOT teacher_feature_result EQUAL 0)
  message(FATAL_ERROR
    "teacher feature export failed\n${teacher_feature_output}\n${teacher_feature_error}")
endif()
file(READ "${teacher_features}/manifest.json" teacher_feature_manifest)
if(NOT teacher_feature_manifest MATCHES
   "\"evaluator\": \"pattern-gain\", \"net_sha256\": \"[0-9a-f]+\"")
  message(FATAL_ERROR "teacher features do not record the label model")
endif()
execute_process(
  COMMAND "${PYTHON}" -B "${FEATURE_AUDITOR}" "${teacher_features}"
          --labels "${teacher_labels}" --gain-samples 8
  RESULT_VARIABLE teacher_feature_audit_result
  OUTPUT_VARIABLE teacher_feature_audit_output
  ERROR_VARIABLE teacher_feature_audit_error
)
if(NOT teacher_feature_audit_result EQUAL 0)
  message(FATAL_ERROR
    "teacher feature audit failed\n${teacher_feature_audit_output}\n${teacher_feature_audit_error}")
endif()

execute_process(
  COMMAND "${MINIMAX_DATA}" features
          --source "${teacher_source}" --labels "${mixed_labels}"
          --output-dir "${mixed_features}"
  RESULT_VARIABLE mixed_feature_result
  OUTPUT_QUIET
  ERROR_QUIET
)
if(mixed_feature_result EQUAL 0)
  message(FATAL_ERROR "feature exporter accepted shards labeled by different evaluators")
endif()
