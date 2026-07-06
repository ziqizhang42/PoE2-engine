add_library(poe2_engine_stdio STATIC
  src/engine_stdio.cpp
)

add_library(poe2::engine_stdio ALIAS poe2_engine_stdio)

target_include_directories(poe2_engine_stdio
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(poe2_engine_stdio
  PUBLIC
    poe2::game
)

target_compile_features(poe2_engine_stdio PUBLIC cxx_std_23)

add_executable(poe2_first_legal
  src/first_legal.cpp
)

target_link_libraries(poe2_first_legal
  PRIVATE
    poe2::engine_stdio
)

target_compile_features(poe2_first_legal PRIVATE cxx_std_23)

add_executable(poe2_greedy
  src/greedy.cpp
)

target_link_libraries(poe2_greedy
  PRIVATE
    poe2::engine_stdio
)

target_compile_features(poe2_greedy PRIVATE cxx_std_23)

add_executable(poe2_random_legal
  src/random_legal.cpp
)

target_link_libraries(poe2_random_legal
  PRIVATE
    poe2::engine_stdio
)

target_compile_features(poe2_random_legal PRIVATE cxx_std_23)
