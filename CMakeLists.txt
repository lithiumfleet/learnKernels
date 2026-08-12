cmake_minimum_required(VERSION 3.25)

project(learn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

add_executable(learn ${CMAKE_SOURCE_DIR}/src/main.cu)

set_target_properties(learn PROPERTIES
    CXX_STANDARD 20
    CUDA_STANDARD 20
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(learn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -lineinfo
        -enable-tile
    >
)