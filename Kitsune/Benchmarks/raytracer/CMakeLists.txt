include(KitsuneTestSuite)

file(COPY lit.local.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# samples    128
# width      1280
# height     1024
kitsune_singlesource(CMDARGS -j 128 1280 1024)
