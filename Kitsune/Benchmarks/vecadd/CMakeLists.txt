include(KitsuneTestSuite)

file(COPY lit.local.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

if (KITSUNE_TEST_INPUT_SIZE STREQUAL "small")
  kitsune_singlesource(CMDARGS -j 8192)
elseif (KITSUNE_TEST_INPUT_SIZE STREQUAL "medium")
  kitsune_singlesource(CMDARGS -j 268435456) # 268435456 = 1024*1024*256
else()
  kitsune_singlesource()
endif()
