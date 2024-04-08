cmake_minimum_required(VERSION 3.27)
project(spmat)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)

add_executable(spmat main.cpp)

target_link_libraries(spmat PRIVATE MPI::MPI_CXX)