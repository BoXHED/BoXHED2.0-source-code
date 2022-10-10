cmake_minimum_required(VERSION 3.18.2)


PROJECT(boxhed_prep LANGUAGES CXX C VERSION 1.0)

set(CMAKE_POLICY_DEFAULT_CMP0054 NEW)
cmake_policy(SET CMP0054 NEW)

set (CMAKE_CXX_STANDARD 11)
if ( CMAKE_COMPILER_IS_GNUCC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wextra")
endif()
if ( MSVC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -openmp:llvm")
endif()


add_library(_boxhed_prep MODULE boxhed_prep.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
        target_link_libraries(_boxhed_prep OpenMP::OpenMP_CXX)
endif()
