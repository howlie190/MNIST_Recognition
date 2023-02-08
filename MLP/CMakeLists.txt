cmake_minimum_required(VERSION 3.20)
project(MLP)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(MLP main.cpp Define.h MLP.cpp MLP.h Define.cpp BmpDigitRecognition.cpp BmpDigitRecognition.h)
target_link_libraries(MLP ${OpenCV_LIBS})