include(KitsuneTestSuite)

file(COPY lit.local.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

if (KITSUNE_BENCHMARK)
  kitsune_singlesource(CMDARGS -j 268435456) # 268435456 = 1024*1024*256
else()
  kitsune_singlesource(CMDARGS 8192)
endif()
