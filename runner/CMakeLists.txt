add_executable(poe2_runner
  src/main.cpp
)

target_link_libraries(poe2_runner
  PRIVATE
    poe2::game
)

target_compile_features(poe2_runner PRIVATE cxx_std_23)
