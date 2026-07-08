include(FetchContent)

set(CATCH_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.15.1
)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

add_executable(poe2_tests
  board_test.cpp
  engine_stdio_test.cpp
  match_runner_test.cpp
  move_test.cpp
  symmetry_test.cpp
)

target_link_libraries(poe2_tests
  PRIVATE
    poe2::engine_stdio
    poe2::match_runner
    Catch2::Catch2WithMain
)

include(Catch)

catch_discover_tests(poe2_tests)
