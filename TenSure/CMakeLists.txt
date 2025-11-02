cmake_minimum_required(VERSION 3.28)
project(TacoKernelGenerator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SRC_FILES src/*.cpp)

include(FetchContent)

# === nlohmann_json ===
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# === TACO ===
FetchContent_Declare(
    taco
    GIT_REPOSITORY https://github.com/tensor-compiler/taco.git
    GIT_TAG first-taco
)
FetchContent_MakeAvailable(taco)

# === Executable ===
add_executable(${PROJECT_NAME} ${SRC_FILES})
target_include_directories(${PROJECT_NAME} PRIVATE include)
target_link_libraries(${PROJECT_NAME} PRIVATE taco nlohmann_json::nlohmann_json)
