cmake_minimum_required(VERSION 3.7.2)

project(projet VERSION 1.0 LANGUAGES CXX)

# compilation c++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(CMAKE_BUILD_TYPE Release)

add_executable(MNISTNN
  src/main.cpp
  )

install (TARGETS MNISTNN DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/bin)
