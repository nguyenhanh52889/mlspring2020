cmake_minimum_required(VERSION 3.10.2)
project(ML)
set(CMAKE_BUILD_TYPE Release)

include_directories(include)
file(GLOB SOURCES "src/*.cpp")

add_executable(mnist ${SOURCES})

