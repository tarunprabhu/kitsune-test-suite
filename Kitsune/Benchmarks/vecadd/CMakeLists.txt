include(KitsuneTestSuite)

file(COPY lit.local.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# 268435456 = 1024*1024*256
kitsune_singlesource(CMDARGS -j 268435456)
