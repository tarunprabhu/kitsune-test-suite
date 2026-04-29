include(KitsuneTestSuite)

string(APPEND CMAKE_C_FLAGS " -std=c23")
file(COPY lit.local.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

kitsune_singlesource()
