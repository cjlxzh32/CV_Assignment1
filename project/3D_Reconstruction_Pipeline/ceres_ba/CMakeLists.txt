cmake_minimum_required(VERSION 3.10)
project(ceres_ba_runner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(glog QUIET)

add_executable(ceres_ba_runner ba_ceres_runner.cc)
target_include_directories(ceres_ba_runner PRIVATE
    ${CERES_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
if (glog_FOUND)
    target_link_libraries(ceres_ba_runner PRIVATE ${CERES_LIBRARIES} glog::glog)
else()
    message(STATUS "glog not found explicitly; relying on CERES transitively.")
    target_link_libraries(ceres_ba_runner PRIVATE ${CERES_LIBRARIES})
endif()
