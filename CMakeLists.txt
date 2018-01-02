# Specify the minimum version for CMake

cmake_minimum_required(VERSION 3.4.1)

add_compile_options(-std=c++14)

project (TrackingView VERSION 0.1 LANGUAGES CXX)
#Bring the headers into the project
include_directories(include)
 
#Bring sources into the project:
file(GLOB SOURCES "src/*.cpp")
file(GLOB HEADERS "include/*.h" "include/*.hpp")

source_group("Headers" FILES ${HEADERS})
source_group("Sources" FILES ${SOURCES})

find_package( OpenCV REQUIRED )
add_executable(TrackingView ${SOURCES} ${HEADERS})
target_link_libraries( TrackingView ${OpenCV_LIBS} )
