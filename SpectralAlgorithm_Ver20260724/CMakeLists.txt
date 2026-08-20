cmake_minimum_required(VERSION 3.16)
project (main)                  #name of project

# ==========================================
# Set default build type to Release for optimization
# ==========================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE not specified. Using Release mode.")
else()
    # FORCE update cache even if already set
    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ==========================================
# Compiler optimization flags
# ==========================================
# Release mode: Aggressive optimizations for maximum speed
# -O3: Highest optimization level
# -march=native: CPU-specific instructions (SSE4.2, AVX2, AVX-512)
# -ffast-math: IEEE 754 relaxation for faster math (exp, pow, sin, etc.)
# -DNDEBUG: Remove all assert() calls
# -flto: Link-Time Optimization for inter-procedural optimization (+5-10% perf)
# -funroll-loops: Unroll small loops for better cache utilization
# -fno-signed-zeros: Optimization for floating point
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -funroll-loops -fno-signed-zeros" CACHE STRING "C++ Release flags" FORCE)
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -DNDEBUG  -funroll-loops -fno-signed-zeros" CACHE STRING "C Release flags" FORCE)

# Enable LTO for linker
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-O3" CACHE STRING "Linker Release flags" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "-O3" CACHE STRING "Shared linker Release flags" FORCE)

# For Debug builds: add debug symbols but keep optimization for debugging
set(CMAKE_CXX_FLAGS_DEBUG "-O2 -g -march=native")
set(CMAKE_C_FLAGS_DEBUG "-O2 -g -march=native")

if(CMAKE_BUILD_TYPE STREQUAL Release)
    message(STATUS "Optimization flags: -O3 -march=native -funroll-loops")
    message(STATUS "Expected speedup: 8-20× vs unoptimized")
elseif(CMAKE_BUILD_TYPE STREQUAL Debug)
    message(STATUS "Debug flags: -O2 -g -march=native")
endif()

#parallel computing
OPTION (USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
    FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()
ENDIF()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -fopenmp")  #set C++ standard to C++20

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")


# Automatically include src directory and all its subdirectories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
file(GLOB_RECURSE ALL_DIRS LIST_DIRECTORIES true "${CMAKE_CURRENT_SOURCE_DIR}/src/*")
foreach(dir ${ALL_DIRS})
    if(IS_DIRECTORY ${dir})
        include_directories(${dir})
    endif()
endforeach()

# Find HDF5 (C and C++ interfaces). Optional for HDF5 I/O (H5Cpp.h).
find_package(HDF5 COMPONENTS C CXX)
if(HDF5_FOUND)
    message(STATUS "HDF5 found: ${HDF5_LIBRARIES}")
else()
    message(WARNING "HDF5 not found. HDF5 I/O features will be disabled.")
endif()

# Find FFTW3 library. Optional for FFT operations.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(FFTW3 fftw3 fftw3f fftw3l)
endif()
if(NOT FFTW3_FOUND)
    find_package(FFTW3)
endif()
if(FFTW3_FOUND)
    message(STATUS "FFTW3 found: ${FFTW3_LIBRARIES}")
else()
    message(WARNING "FFTW3 not found. FFT operations may fail to link.")
endif()

# Collect all .cpp files from src and its subdirectories
file(GLOB_RECURSE SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*/*.cpp"
)

#set(SOURCES ../Src/utilities.cpp)
set(MAIN ${CMAKE_CURRENT_SOURCE_DIR}/main/main.cpp)         #set file main.cpp 
add_executable(main ${MAIN} ${SOURCES})  #name of executable file

set(TEST ${CMAKE_CURRENT_SOURCE_DIR}/test/test.cpp)         #set file test.cpp
add_executable(test ${TEST} ${SOURCES})  #name of executable file

# Link HDF5 to the Main target if available
if(HDF5_FOUND)
    target_include_directories(main PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(main PRIVATE ${HDF5_LIBRARIES})

    target_include_directories(test PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(test PRIVATE ${HDF5_LIBRARIES})
endif()

# Link FFTW3 to the Main target if available
if(FFTW3_FOUND)
    target_include_directories(main PRIVATE ${FFTW3_INCLUDE_DIRS})
    target_link_libraries(main PRIVATE ${FFTW3_LIBRARIES} fftw3_threads)

    target_include_directories(test PRIVATE ${FFTW3_INCLUDE_DIRS})
    target_link_libraries(test PRIVATE ${FFTW3_LIBRARIES} fftw3_threads)
else()
    # Fallback: link against FFTW3 if find_package didn't work
    target_link_libraries(main PRIVATE fftw3 fftw3_threads m)
    target_link_libraries(test PRIVATE fftw3 fftw3_threads m)
endif()

add_custom_target(run_main
            COMMAND ./main
            DEPENDS main
            )

add_custom_target(run_test
            COMMAND ./test
            DEPENDS test
            )

add_custom_target(clean_build
            COMMAND rm -rf *
            )

add_custom_target(clean_output
            COMMAND rm -f ${CMAKE_CURRENT_SOURCE_DIR}/data/output/Output_0/*
            COMMAND rm -f ${CMAKE_CURRENT_SOURCE_DIR}/data/output/Output_1/*
            COMMAND rm -f ${CMAKE_CURRENT_SOURCE_DIR}/data/output/Output_2/*
            COMMAND rm -f ${CMAKE_CURRENT_SOURCE_DIR}/data/output/Output_3/*
            COMMAND rm -f ${CMAKE_CURRENT_SOURCE_DIR}/data/output/Output_4/*
            COMMAND rm -f ${CMAKE_CURRENT_SOURCE_DIR}/data/output/Output_5/*
            COMMAND rm -f ${CMAKE_CURRENT_SOURCE_DIR}/data/output/Output_6/*
)