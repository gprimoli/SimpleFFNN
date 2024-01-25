cmake_minimum_required(VERSION 3.27)
project(SimpleFFNN)

set(CMAKE_CXX_STANDARD 23)

add_executable(SimpleFFNN Net.h Net.cpp main.cpp)
