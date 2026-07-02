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

add_executable(poe2_smoke
  smoke.cpp
)

target_link_libraries(poe2_smoke
  PRIVATE
    poe2::game
    Catch2::Catch2WithMain
)

include(Catch)

catch_discover_tests(poe2_smoke)
