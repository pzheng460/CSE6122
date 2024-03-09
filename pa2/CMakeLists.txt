cmake_minimum_required(VERSION 3.27)
project(transpose)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)

add_executable(transpose main.cpp)

target_link_libraries(transpose PRIVATE MPI::MPI_CXX)
