include(bin2h.cmake)

message("Embedding following files into header file - ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib - ggml-metal-shader.h")
bin2h(SOURCE_FILE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib HEADER_FILE ggml-metal-shader.h VARIABLE_NAME default_metallib)




