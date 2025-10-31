# Specify the minimum version of CMake required to build the project
cmake_minimum_required(VERSION 3.21)

project(lsCOMP
        VERSION 1.0.0
        DESCRIPTION "A fast GPU lossy/lossless compressor for light source data and unsigned integers."
        )
set(namespace "lsCOMP")
enable_language(CXX)
enable_language(CUDA)

# find_package(CUDAToolkit REQUIRED)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA host compiler
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 80 86)
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set default build type to Release if not specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY VALUE Release)
endif()

# Disable specific warnings (adjust as needed)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-unused-result")


add_library(${PROJECT_NAME} STATIC)

target_sources(${PROJECT_NAME}
        PRIVATE
        src/lsCOMP_kernel.cu
        src/lsCOMP_utility.cu
        src/lsCOMP_timer.cu
        src/lsCOMP_entry.cu
        )

target_include_directories(${PROJECT_NAME}
        PRIVATE
        # where the library itself will look for its internal headers
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        PUBLIC
        # where top-level project will look for the library's public headers
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        # where external projects will look for the library's public headers
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )

#target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)

set(public_headers
        include/lsCOMP_kernel.h
        include/lsCOMP_utility.h
        include/lsCOMP_timer.h
        include/lsCOMP_entry.h
        )

# set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# include(Installing)

option(lsCOMP_BUILD_EXAMPLES "Option to enable building example programs" ON)
if (lsCOMP_BUILD_EXAMPLES)
    add_subdirectory(example)
endif ()