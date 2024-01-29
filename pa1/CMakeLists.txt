cmake_minimum_required(VERSION 3.26)
project(pi_calc)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)

add_executable(pi_calc main.cpp)

target_link_libraries(pi_calc PRIVATE MPI::MPI_CXX)