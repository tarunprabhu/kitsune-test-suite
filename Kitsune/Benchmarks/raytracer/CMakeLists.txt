include(KitsuneTestSuite)

if (KITSUNE_TEST_INPUT_SIZE STREQUAL "default")
  # samples  128
  # width    1280
  # height   1024
  kitsune_singlesource(CMDARGS 128 1280 1024)
elseif (KITSUNE_TEST_INPUT_SIZE STREQUAL "small")
  # samples  64
  # width    320
  # height   200
  kitsune_singlesource(CMDARGS 16 320 200)
else()
  kitsune_singlesource()
endif()
