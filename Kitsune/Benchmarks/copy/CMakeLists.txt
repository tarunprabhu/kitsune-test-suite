include(KitsuneTestSuite)

if (KITSUNE_TEST_INPUT_SIZE STREQUAL "small")
  kitsune_singlesource(CMDARGS 8192)
elseif (KITSUNE_TEST_INPUT_SIZE STREQUAL "medium")
  kitsune_singlesource(CMDARGS 268435456) # 268435456 = 1024*1024*256 = 1 << 28
else()
  kitsune_singlesource()
endif()
