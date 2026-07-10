add_library(poe2_minimax_engine STATIC
  src/engine.cpp
  src/evaluation.cpp
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

add_executable(poe2_minimax
  src/main.cpp
)

target_link_libraries(poe2_minimax
  PRIVATE
    poe2::engine_stdio
    poe2::minimax_engine
)

target_compile_features(poe2_minimax PRIVATE cxx_std_23)
