add_library(poe2_game STATIC
  src/board.cpp
  src/transposition_table.cpp
)

add_library(poe2::game ALIAS poe2_game)

target_include_directories(poe2_game
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(poe2_game PUBLIC cxx_std_23)
