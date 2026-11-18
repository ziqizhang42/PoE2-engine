add_library(poe2_minimax_engine STATIC
  src/engine.cpp
  src/evaluation.cpp
  src/evaluation_inputs.cpp
  src/evaluation_pattern_gain.cpp
  src/net.cpp
  src/net_file.cpp
  src/options.cpp
  src/search.cpp
  src/sha256.cpp
)

add_library(poe2::minimax_engine ALIAS poe2_minimax_engine)

# The built-in net is the net file nets/default.poe2net, compiled in byte for byte and loaded
# like any other net file. `poe2-train handoff --apply` replaces that file.
set(POE2_MINIMAX_DEFAULT_NET "${CMAKE_CURRENT_SOURCE_DIR}/nets/default.poe2net")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${POE2_MINIMAX_DEFAULT_NET}")
file(READ "${POE2_MINIMAX_DEFAULT_NET}" POE2_MINIMAX_DEFAULT_NET_HEX HEX)
string(LENGTH "${POE2_MINIMAX_DEFAULT_NET_HEX}" POE2_MINIMAX_DEFAULT_NET_HEX_LENGTH)
math(EXPR POE2_MINIMAX_DEFAULT_NET_SIZE "${POE2_MINIMAX_DEFAULT_NET_HEX_LENGTH} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," POE2_MINIMAX_DEFAULT_NET_BYTES
       "${POE2_MINIMAX_DEFAULT_NET_HEX}")
configure_file(
  src/default_net_bytes.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/poe2/minimax/default_net_bytes.hpp
  @ONLY
)

configure_file(
  src/engine_version.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/poe2/minimax/engine_version.hpp
  @ONLY
)

target_include_directories(poe2_minimax_engine
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)

target_link_libraries(poe2_minimax_engine
  PUBLIC
    poe2::engine_common
)

target_compile_features(poe2_minimax_engine PUBLIC cxx_std_23)

if(POE2_BUILD_NATIVE_APPS)
  find_package(Threads REQUIRED)

  set(POE2_LABEL_GIT_COMMIT "unknown")
  set(POE2_LABEL_GIT_DIRTY false)
  find_package(Git QUIET)
  if(Git_FOUND)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      OUTPUT_VARIABLE POE2_LABEL_GIT_COMMIT
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    execute_process(
      COMMAND ${GIT_EXECUTABLE} status --porcelain --untracked-files=normal
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      OUTPUT_VARIABLE POE2_LABEL_GIT_STATUS
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(POE2_LABEL_GIT_STATUS)
      set(POE2_LABEL_GIT_DIRTY true)
    endif()
  endif()
  if(POE2_ENABLE_NATIVE_ARCH)
    set(POE2_LABEL_NATIVE_ARCH true)
  else()
    set(POE2_LABEL_NATIVE_ARCH false)
  endif()

  configure_file(
    src/labeling_build.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/poe2/minimax/labeling_build.hpp
    @ONLY
  )

  add_library(poe2_minimax_labeling STATIC
    src/durable_file.cpp
    src/feature_data.cpp
    src/labeling.cpp
    src/position_source.cpp
  )

  add_library(poe2::minimax_labeling ALIAS poe2_minimax_labeling)

  target_include_directories(poe2_minimax_labeling
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
      ${CMAKE_CURRENT_BINARY_DIR}/generated
  )

  target_link_libraries(poe2_minimax_labeling
    PUBLIC
      poe2::minimax_engine
    PRIVATE
      Threads::Threads
  )

  target_compile_features(poe2_minimax_labeling PUBLIC cxx_std_23)

  add_executable(poe2_minimax
    src/main.cpp
  )

  target_link_libraries(poe2_minimax
    PRIVATE
      poe2::engine_stdio
      poe2::minimax_engine
  )

  target_compile_features(poe2_minimax PRIVATE cxx_std_23)

  add_executable(poe2_minimax_infer
    src/infer_main.cpp
  )

  target_link_libraries(poe2_minimax_infer
    PRIVATE
      poe2::minimax_engine
  )

  target_compile_features(poe2_minimax_infer PRIVATE cxx_std_23)
endif()
