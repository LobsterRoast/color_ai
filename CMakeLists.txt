cmake_minimum_required(VERSION 3.31)
project(color_ai)

add_executable(color_ai main.c ai.c ai_utils.c)

target_link_libraries(color_ai m)