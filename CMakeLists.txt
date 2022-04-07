cmake_minimum_required(VERSION 3.21)
project(mc859)

set(CMAKE_CXX_STANDARD 14)

# Set a default build type if none was specified
set(default_build_type "Release")

# Headers directories
include_directories("include")
include_directories($ENV{GUROBI_HOME}/include)
include_directories("lemon/lemon-1.3.1/include")

# Gurobi lib directory
link_directories($ENV{GUROBI_HOME}/lib)

# Lemon lib directory
link_directories(${CMAKE_SOURCE_DIR}/lemon/lemon-1.3.1/lib)

# Compiler flags
# set(CMAKE_CXX_FLAGS "-Ofast -Wall -std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0 -Wunused-local-typedefs -lemon")

# Target executable and linking
add_executable(mc859 src/main.cpp)
target_link_libraries(mc859 m)
target_link_libraries(mc859 m gurobi_c++ gurobi95)
