if(NOT DEFINED RUNNER OR NOT DEFINED BUILD_DIR OR NOT DEFINED SOURCE_DIR OR
   NOT DEFINED TEST_ROOT)
  message(FATAL_ERROR "missing eval nets-test path")
endif()

file(REMOVE_RECURSE "${TEST_ROOT}")
set(net "${SOURCE_DIR}/tests/fixtures/nets/pattern-gain-v1/reference.poe2net")

function(run_net_eval label)
  execute_process(
    COMMAND "${RUNNER}" eval
            --new-build "${BUILD_DIR}"
            --base "${BUILD_DIR}"
            --new-engine minimax/poe2_minimax
            --base-engine minimax/poe2_minimax
            --new-engine-args "--hash-mb 1"
            --base-engine-args "--hash-mb 1"
            ${ARGN}
            --games 2
            --timeout-ms 5000
            --go-nodes 200
            --opening-book "${SOURCE_DIR}/eval/openings/development.txt"
            --shuffle-openings
            --run-root "${TEST_ROOT}/${label}"
            --kind smoke
            --ledger "${TEST_ROOT}/${label}-results.csv"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE runner_output
    ERROR_VARIABLE runner_error
  )
  set(${label}_result "${result}" PARENT_SCOPE)
  set(${label}_error "${runner_output}\n${runner_error}" PARENT_SCOPE)
  file(GLOB manifests "${TEST_ROOT}/${label}/*/manifest.json")
  set(${label}_manifests "${manifests}" PARENT_SCOPE)
endfunction()

run_net_eval(nets --new-net "${net}" --base-net "${net}")
if(NOT nets_result EQUAL 0)
  message(FATAL_ERROR "eval with nets returned ${nets_result}\n${nets_error}")
endif()
list(LENGTH nets_manifests manifest_count)
if(NOT manifest_count EQUAL 1)
  message(FATAL_ERROR "eval with nets did not write one run")
endif()
list(GET nets_manifests 0 manifest)
file(READ "${manifest}" manifest_text)
foreach(side new base)
  foreach(expected "\"${side}_net\": \"${net}\"" "\"${side}_net_name\": \"run1/g000\""
                   "\"${side}_net_id\": \"633309adb322\"")
    string(FIND "${manifest_text}" "${expected}" position)
    if(position EQUAL -1)
      message(FATAL_ERROR "manifest lacks ${expected}:\n${manifest_text}")
    endif()
  endforeach()
  if(NOT manifest_text MATCHES "\"${side}_engine_version\": \"[0-9]+\\.[0-9]+\\.[0-9]+\"")
    message(FATAL_ERROR "manifest lacks the ${side} engine version:\n${manifest_text}")
  endif()
  if(NOT manifest_text MATCHES "\"${side}_net_sha256\": \"[0-9a-f]+\"")
    message(FATAL_ERROR "manifest lacks the ${side} net hash:\n${manifest_text}")
  endif()
endforeach()
string(REGEX MATCH "\"run_id\": \"([^\"]+)\"" run_match "${manifest_text}")
set(run_id "${CMAKE_MATCH_1}")
string(LENGTH "${run_id}" run_id_length)
# Run IDs over 200 bytes, as with long build IDs, end in a digest of the full ID instead.
if(run_id_length LESS 200)
  if(NOT run_id MATCHES "__newnet_633309adb322__basenet_633309adb322$")
    message(FATAL_ERROR "run ID does not name both nets: ${run_id}")
  endif()
elseif(NOT run_id_length EQUAL 200 OR NOT run_id MATCHES "__newnet_.*__[0-9a-f]+$")
  message(FATAL_ERROR "long run ID is not bounded: ${run_id}")
endif()
file(STRINGS "${TEST_ROOT}/nets-results.csv" rows)
list(GET rows 1 row)
if(NOT row MATCHES ",run1/g000,633309adb322,[0-9a-f]+,[0-9.]+,run1/g000,633309adb322,[0-9a-f]+$")
  message(FATAL_ERROR "ledger row does not record both nets: ${row}")
endif()

run_net_eval(missing --new-net "${TEST_ROOT}/missing.poe2net")
if(missing_result EQUAL 0)
  message(FATAL_ERROR "eval accepted a missing net file")
endif()
string(FIND "${missing_error}" "missing.poe2net" missing_named)
if(missing_named EQUAL -1)
  message(FATAL_ERROR "eval did not name the missing net file:\n${missing_error}")
endif()
