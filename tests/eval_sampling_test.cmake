if(NOT DEFINED RUNNER OR NOT DEFINED BUILD_DIR OR NOT DEFINED SOURCE_DIR OR
   NOT DEFINED TEST_ROOT)
  message(FATAL_ERROR "missing eval sampling-test path")
endif()

file(REMOVE_RECURSE "${TEST_ROOT}")
foreach(candidate candidate-a candidate-b)
  file(MAKE_DIRECTORY "${TEST_ROOT}/${candidate}/engines")
  file(COPY "${BUILD_DIR}/engines/poe2_greedy"
       DESTINATION "${TEST_ROOT}/${candidate}/engines")
endforeach()

function(run_sampling_eval label candidate kind book)
  set(run_root "${TEST_ROOT}/${label}")
  execute_process(
    COMMAND "${RUNNER}" eval
            --new-build "${candidate}"
            --base "${BUILD_DIR}"
            --new-engine poe2_greedy
            --base-engine poe2_greedy
            --games 2
            --timeout-ms 1000
            --go-depth 1
            --opening-book "${SOURCE_DIR}/eval/openings/${book}.txt"
            --shuffle-openings
            --run-root "${run_root}"
            --kind "${kind}"
            --no-ledger
    RESULT_VARIABLE result
    OUTPUT_VARIABLE runner_output
    ERROR_VARIABLE runner_error
  )
  if(NOT result EQUAL 0)
    message(FATAL_ERROR
      "sampling eval returned ${result}\n${runner_output}\n${runner_error}"
    )
  endif()
  file(GLOB manifests "${run_root}/*/manifest.json")
  file(GLOB games_files "${run_root}/*/games.csv")
  file(GLOB ledger_rows "${run_root}/*/ledger-row.csv")
  list(LENGTH manifests manifest_count)
  list(LENGTH games_files games_count)
  list(LENGTH ledger_rows ledger_row_count)
  if(NOT manifest_count EQUAL 1 OR NOT games_count EQUAL 1 OR NOT ledger_row_count EQUAL 1)
    message(FATAL_ERROR "sampling eval did not write one self-contained run")
  endif()
  list(GET manifests 0 manifest)
  list(GET games_files 0 games_file)
  list(GET ledger_rows 0 ledger_row)
  file(READ "${manifest}" manifest_text)
  file(READ "${games_file}" games_text)
  file(STRINGS "${ledger_row}" ledger_lines)
  list(LENGTH ledger_lines ledger_line_count)
  if(NOT ledger_line_count EQUAL 2)
    message(FATAL_ERROR "saved ledger row does not contain one header and one result")
  endif()
  list(GET ledger_lines 0 ledger_header)
  string(FIND "${ledger_header}" "run_id" ledger_run_id)
  if(ledger_run_id EQUAL -1)
    message(FATAL_ERROR "saved ledger row has no current ledger header")
  endif()
  string(REGEX MATCH [["opening_seed": ([0-9]+)]] seed_match "${manifest_text}")
  if(NOT seed_match)
    message(FATAL_ERROR "manifest has no resolved opening seed")
  endif()
  set(${label}_seed "${CMAKE_MATCH_1}" PARENT_SCOPE)
  set(${label}_games "${games_text}" PARENT_SCOPE)
endfunction()

run_sampling_eval(first "${TEST_ROOT}/candidate-a" smoke development)
run_sampling_eval(replay "${TEST_ROOT}/candidate-a" smoke development)
run_sampling_eval(candidate "${TEST_ROOT}/candidate-b" smoke development)
run_sampling_eval(kind "${TEST_ROOT}/candidate-a" gate development)
run_sampling_eval(book "${TEST_ROOT}/candidate-a" smoke holdout)

set(subpath_run_root "${TEST_ROOT}/engine-subpath")
execute_process(
  COMMAND "${RUNNER}" eval
          --new-build "${BUILD_DIR}"
          --base "${BUILD_DIR}"
          --new-engine minimax/poe2_minimax
          --base-engine minimax/poe2_minimax
          --games 2
          --timeout-ms 1000
          --go-depth 1
          --opening-book "${SOURCE_DIR}/eval/openings/development.txt"
          --shuffle-openings
          --sequential-alpha 0.05123456789
          --run-root "${subpath_run_root}"
          --kind smoke
          --no-ledger
  RESULT_VARIABLE subpath_result
  OUTPUT_VARIABLE subpath_output
  ERROR_VARIABLE subpath_error
)
if(NOT subpath_result EQUAL 0)
  message(FATAL_ERROR
    "engine-subpath eval returned ${subpath_result}\n${subpath_output}\n${subpath_error}"
  )
endif()
file(GLOB subpath_manifests "${subpath_run_root}/*/manifest.json")
file(GLOB subpath_summaries "${subpath_run_root}/*/summary.json")
list(LENGTH subpath_manifests subpath_manifest_count)
list(LENGTH subpath_summaries subpath_summary_count)
if(NOT subpath_manifest_count EQUAL 1 OR NOT subpath_summary_count EQUAL 1)
  message(FATAL_ERROR "an engine subpath did not produce one manifest and summary")
endif()
list(GET subpath_manifests 0 subpath_manifest)
list(GET subpath_summaries 0 subpath_summary)
file(READ "${subpath_manifest}" subpath_manifest_text)
file(READ "${subpath_summary}" subpath_summary_text)
string(REGEX MATCH [["sequential_alpha": 0\.051234567[0-9]+]]
       precise_manifest_alpha "${subpath_manifest_text}")
string(REGEX MATCH [["sequential_alpha": 0\.051234567[0-9]+]]
       precise_summary_alpha "${subpath_summary_text}")
if(NOT precise_manifest_alpha OR NOT precise_summary_alpha)
  message(FATAL_ERROR "the eval JSON artifacts did not retain round-trip floating-point precision")
endif()

if(NOT first_seed STREQUAL replay_seed OR NOT first_games STREQUAL replay_games)
  message(FATAL_ERROR "an identical matchup did not reproduce its opening selection")
endif()
if(first_seed STREQUAL candidate_seed)
  message(FATAL_ERROR "candidate build ID did not affect the derived opening seed")
endif()
if(first_seed STREQUAL kind_seed)
  message(FATAL_ERROR "eval kind did not affect the derived opening seed")
endif()
if(first_seed STREQUAL book_seed)
  message(FATAL_ERROR "opening-book digest did not affect the derived opening seed")
endif()

# Builds of archived models carry full SHA-256 IDs on both sides of a generation gate. Their run
# directory must still fit a 255-byte file name, while the manifest keeps both full IDs.
string(REPEAT "a" 64 new_digest)
string(REPEAT "b" 64 base_digest)
set(model_new_id "candidate-sha256-${new_digest}")
set(model_base_id "candidate-sha256-${base_digest}")
foreach(model_id IN ITEMS "${model_new_id}" "${model_base_id}")
  file(MAKE_DIRECTORY "${TEST_ROOT}/${model_id}/engines")
  file(COPY "${BUILD_DIR}/engines/poe2_greedy" DESTINATION "${TEST_ROOT}/${model_id}/engines")
endforeach()
set(model_run_root "${TEST_ROOT}/model-builds")
execute_process(
  COMMAND "${RUNNER}" eval
          --new-build "${TEST_ROOT}/${model_new_id}"
          --base "${TEST_ROOT}/${model_base_id}"
          --new-engine poe2_greedy
          --base-engine poe2_greedy
          --new-engine-args "--seed 1"
          --base-engine-args "--seed 2"
          --games 2
          --timeout-ms 1000
          --go-depth 1
          --opening-book "${SOURCE_DIR}/eval/openings/development.txt"
          --shuffle-openings
          --run-root "${model_run_root}"
          --kind training-gate
          --no-ledger
  RESULT_VARIABLE model_result
  OUTPUT_VARIABLE model_output
  ERROR_VARIABLE model_error
)
if(NOT model_result EQUAL 0)
  message(FATAL_ERROR "model-build eval returned ${model_result}\n${model_output}\n${model_error}")
endif()
file(GLOB model_manifests "${model_run_root}/*/manifest.json")
list(LENGTH model_manifests model_manifest_count)
if(NOT model_manifest_count EQUAL 1)
  message(FATAL_ERROR "model-build eval did not write one run")
endif()
list(GET model_manifests 0 model_manifest)
get_filename_component(model_run_dir "${model_manifest}" DIRECTORY)
get_filename_component(model_run_name "${model_run_dir}" NAME)
string(LENGTH "${model_run_name}" model_run_name_length)
string(FIND "${model_run_name}" "candidate-sha256-aaaaaaaaaaaa__" short_new)
string(FIND "${model_run_name}" "__vs__candidate-sha256-bbbbbbbbbbbb__" short_base)
if(model_run_name_length GREATER 255 OR short_new EQUAL -1 OR short_base EQUAL -1)
  message(FATAL_ERROR "model-build run name is not abbreviated: ${model_run_name}")
endif()
file(READ "${model_manifest}" model_manifest_text)
string(FIND "${model_manifest_text}" "\"new_id\": \"${model_new_id}\"" full_new)
string(FIND "${model_manifest_text}" "\"base_id\": \"${model_base_id}\"" full_base)
string(FIND "${model_manifest_text}" "\"run_id\": \"${model_run_name}\"" run_id_matches)
if(full_new EQUAL -1 OR full_base EQUAL -1 OR run_id_matches EQUAL -1)
  message(FATAL_ERROR "model-build manifest lost its full build IDs or run ID")
endif()

# Any other over-long run name keeps its readable start and ends with a digest of the full name.
string(REPEAT "e" 120 long_engine_dir)
set(long_engine "${long_engine_dir}/poe2_greedy")
file(MAKE_DIRECTORY "${TEST_ROOT}/long-engine-build/engines/${long_engine_dir}")
file(COPY "${BUILD_DIR}/engines/poe2_greedy"
     DESTINATION "${TEST_ROOT}/long-engine-build/engines/${long_engine_dir}")
set(long_run_root "${TEST_ROOT}/long-engine")
execute_process(
  COMMAND "${RUNNER}" eval
          --new-build "${TEST_ROOT}/long-engine-build"
          --base "${TEST_ROOT}/long-engine-build"
          --new-engine "${long_engine}"
          --base-engine "${long_engine}"
          --games 2
          --timeout-ms 1000
          --go-depth 1
          --opening-book "${SOURCE_DIR}/eval/openings/development.txt"
          --shuffle-openings
          --run-root "${long_run_root}"
          --kind smoke
          --no-ledger
  RESULT_VARIABLE long_result
  OUTPUT_VARIABLE long_output
  ERROR_VARIABLE long_error
)
if(NOT long_result EQUAL 0)
  message(FATAL_ERROR "long-engine eval returned ${long_result}\n${long_output}\n${long_error}")
endif()
file(GLOB long_manifests "${long_run_root}/*/manifest.json")
list(LENGTH long_manifests long_manifest_count)
if(NOT long_manifest_count EQUAL 1)
  message(FATAL_ERROR "long-engine eval did not write one run")
endif()
list(GET long_manifests 0 long_manifest)
get_filename_component(long_run_dir "${long_manifest}" DIRECTORY)
get_filename_component(long_run_name "${long_run_dir}" NAME)
string(LENGTH "${long_run_name}" long_run_name_length)
if(NOT long_run_name_length EQUAL 200 OR NOT long_run_name MATCHES "__[0-9a-f]+$")
  message(FATAL_ERROR "over-long run name was not bounded with a digest: ${long_run_name}")
endif()
file(READ "${long_manifest}" long_manifest_text)
string(FIND "${long_manifest_text}" "\"new_engine\": \"${long_engine}\"" full_engine)
string(FIND "${long_manifest_text}" "\"run_id\": \"${long_run_name}\"" long_run_id_matches)
if(full_engine EQUAL -1 OR long_run_id_matches EQUAL -1)
  message(FATAL_ERROR "bounded run lost its full engine name or run ID")
endif()
