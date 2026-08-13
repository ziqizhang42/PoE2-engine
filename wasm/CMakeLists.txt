set(POE2_WASM_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
configure_file(
  include/poe2/wasm/version.hpp.in
  "${POE2_WASM_GENERATED_INCLUDE_DIR}/poe2/wasm/version.hpp"
  @ONLY
)

add_library(poe2_wasm_adapter STATIC
  src/adapter.cpp
)

add_library(poe2::wasm_adapter ALIAS poe2_wasm_adapter)

target_include_directories(poe2_wasm_adapter
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${POE2_WASM_GENERATED_INCLUDE_DIR}
)

target_link_libraries(poe2_wasm_adapter
  PUBLIC
    poe2::minimax_engine
)

target_compile_features(poe2_wasm_adapter PUBLIC cxx_std_23)

if(NOT POE2_BUILD_WASM)
  return()
endif()

set(POE2_WASM_PACKAGE_DIR "${CMAKE_BINARY_DIR}/package")
file(MAKE_DIRECTORY "${POE2_WASM_PACKAGE_DIR}")

configure_file(
  package/package.json.in
  "${POE2_WASM_PACKAGE_DIR}/package.json"
  @ONLY
)
configure_file(
  package/index.d.ts.in
  "${POE2_WASM_PACKAGE_DIR}/index.d.ts"
  @ONLY
)
configure_file(
  package/index.mjs
  "${POE2_WASM_PACKAGE_DIR}/index.mjs"
  COPYONLY
)
configure_file(
  package/README.md
  "${POE2_WASM_PACKAGE_DIR}/README.md"
  COPYONLY
)

add_executable(poe2_engine_wasm
  src/bindings.cpp
)

target_link_libraries(poe2_engine_wasm
  PRIVATE
    poe2::wasm_adapter
)

target_compile_features(poe2_engine_wasm PRIVATE cxx_std_23)

target_link_options(poe2_engine_wasm
  PRIVATE
    --no-entry
    -lembind
    -sMODULARIZE=1
    -sEXPORT_ES6=1
    -sEXPORT_NAME=createPoe2EngineModule
    -sENVIRONMENT=web,worker,node
    -sFILESYSTEM=0
    -sALLOW_MEMORY_GROWTH=1
    -sINITIAL_HEAP=33554432
    -sSTACK_SIZE=1048576
    -sNO_EXIT_RUNTIME=1
    -sASSERTIONS=0
)

set_target_properties(poe2_engine_wasm PROPERTIES
  OUTPUT_NAME "poe2-engine"
  SUFFIX ".mjs"
  RUNTIME_OUTPUT_DIRECTORY "${POE2_WASM_PACKAGE_DIR}"
)

add_custom_target(poe2_engine_wasm_package
  DEPENDS poe2_engine_wasm
)

if(BUILD_TESTING)
  find_program(POE2_NODE_EXECUTABLE NAMES node nodejs REQUIRED)
  add_test(
    NAME wasm_smoke
    COMMAND
      ${POE2_NODE_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/smoke.mjs
      ${POE2_WASM_PACKAGE_DIR}/index.mjs
      ${POE2_WASM_PACKAGE_DIR}/package.json
      ${POE2_WASM_PACKAGE_DIR}/poe2-engine.wasm
  )
endif()
