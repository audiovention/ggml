include(${CMAKE_CURRENT_LIST_DIR}/bin2h.cmake)

message("Embedding following files into header file - ${MY_INPUT} - ${MY_OUTPUT}")
bin2h(SOURCE_FILE ${MY_INPUT} HEADER_FILE ${MY_OUTPUT} VARIABLE_NAME default_metallib)




