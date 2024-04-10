cmake_minimum_required(VERSION 3.27)
project(spmatBonus)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)

add_executable(spmatBonus main.cpp)

target_link_libraries(spmatBonus PRIVATE MPI::MPI_CXX)