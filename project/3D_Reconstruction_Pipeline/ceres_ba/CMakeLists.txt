cmake_minimum_required(VERSION 3.10)
project(ceres_ba_runner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point CMake to the custom Ceres you just built+installed
# (the one with no SuiteSparse / no OpenMP headaches)
list(PREPEND CMAKE_PREFIX_PATH "$ENV{HOME}/ceres_clean_install")

# Eigen should already be available (conda-forge eigen or system Eigen)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

message(STATUS "Using Ceres from: ${CERES_INCLUDE_DIRS}")
message(STATUS "Ceres libs: ${CERES_LIBRARIES}")
message(STATUS "Eigen3 include: ${EIGEN3_INCLUDE_DIR}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(ceres_ba_runner
    ba_ceres_runner.cc
)

target_link_libraries(ceres_ba_runner
    PRIVATE
    ${CERES_LIBRARIES}
    Eigen3::Eigen
)
