# Find CUDA package
find_package(CUDA REQUIRED)

set(install_dir ${PROJECT_BINARY_DIR})
set(lsCOMP_cssi "lsCOMP_uint32")
set(lsCOMP_xpcs "lsCOMP_uint16")
set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

# Add include and library directories
include_directories(${INCLUDE_DIR})

# Compile headers as a library
cuda_add_library(lsCOMP_libs STATIC ${SRC_DIR}/lsCOMP_kernel.cu
                                   ${SRC_DIR}/lsCOMP_utility.cu
                                   ${SRC_DIR}/lsCOMP_timer.cu
                                   ${SRC_DIR}/lsCOMP_entry.cu)

# Compile executable binary
cuda_add_executable(${lsCOMP_cssi} lsCOMP_uint32.cpp)
cuda_add_executable(${lsCOMP_xpcs} lsCOMP_uint16.cpp)

# Link with headers
target_link_libraries(${lsCOMP_cssi} lsCOMP_libs)
target_link_libraries(${lsCOMP_xpcs} lsCOMP_libs)

# Set output paths for the compiled binary
set_target_properties(${lsCOMP_cssi} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})
set_target_properties(${lsCOMP_xpcs} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})

# Set installation paths for the compiled binary.
install(TARGETS ${lsCOMP_cssi} DESTINATION bin)
install(TARGETS ${lsCOMP_xpcs} DESTINATION bin)