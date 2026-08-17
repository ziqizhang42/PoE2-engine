add_library(poe2_minimax_engine STATIC
  src/engine.cpp
  src/evaluation.cpp
  src/options.cpp
  src/search.cpp
)

add_library(poe2::minimax_engine ALIAS poe2_minimax_engine)

target_include_directories(poe2_minimax_engine
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
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
    src/labeling.cpp
    src/position_source.cpp
    src/sha256.cpp
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
endif()
