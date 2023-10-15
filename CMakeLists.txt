cmake_minimum_required(VERSION 3.23)
project(untitled10 C)

set(CMAKE_C_STANDARD 11)

add_executable(untitled10 main.c)
find_package(PostgreSQL REQUIRED)
include_directories(${PostgreSQL_INCLUDE_DIRS})
target_link_libraries(untitled10 ${PostgreSQL_LIBRARIES})
