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

  add_library(poe2_minimax_labeling STATIC
    src/labeling.cpp
  )

  add_library(poe2::minimax_labeling ALIAS poe2_minimax_labeling)

  target_include_directories(poe2_minimax_labeling
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include
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
