include(KitsuneTestSuite)

file(COPY lit.local.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# iterations = 2000
# rows       = 6400
# cols       = 6400
# y1         = 0
# y2         = 127
# x1         = 0
# x2         = 127
# lambda     = 0.5
kitsune_singlesource(CMDARGS -j 2000 6400 6400 0 127 0 127 0.5)
