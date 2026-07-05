add_library(poe2_match_runner STATIC
  src/match_runner.cpp
)

add_library(poe2::match_runner ALIAS poe2_match_runner)

target_include_directories(poe2_match_runner
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(poe2_match_runner
  PUBLIC
    poe2::game
  PRIVATE
    poe2::engine_stdio
)

target_compile_features(poe2_match_runner PUBLIC cxx_std_23)

add_executable(poe2_runner
  src/main.cpp
)

target_link_libraries(poe2_runner
  PRIVATE
    poe2::match_runner
)

target_compile_features(poe2_runner PRIVATE cxx_std_23)

add_test(
  NAME runner_first_legal_match
  COMMAND poe2_runner match
          --p1 $<TARGET_FILE:poe2_first_legal>
          --p2 $<TARGET_FILE:poe2_first_legal>
          --timeout-ms 1000
)
